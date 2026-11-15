"""Tests for the EdgeInferenceDetector, its preprocessing, post-processing and backend selection."""

import asyncio
import struct
import sys
import threading
import time
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from is_goat_burning.config import Accelerator
from is_goat_burning.config import EdgeSettings
from is_goat_burning.fire_detection import edge_detector
from is_goat_burning.fire_detection.edge_detector import CONSECUTIVE_ERRORS_BEFORE_CRITICAL
from is_goat_burning.fire_detection.edge_detector import CvDnnBackend
from is_goat_burning.fire_detection.edge_detector import EdgeInferenceDetector
from is_goat_burning.fire_detection.edge_detector import EdgeModelError
from is_goat_burning.fire_detection.edge_detector import OnnxRuntimeBackend
from is_goat_burning.fire_detection.edge_detector import classifier_fire_probability
from is_goat_burning.fire_detection.edge_detector import preprocess
from is_goat_burning.fire_detection.edge_detector import select_backend

# --- backend selection ---


@pytest.mark.parametrize(
    ("accelerator", "cuda", "opencl", "engine", "resolved"),
    [
        (Accelerator.CPU, False, False, "cv2dnn", Accelerator.CPU),
        (Accelerator.CPU, True, True, "cv2dnn", Accelerator.CPU),
        (Accelerator.CUDA, True, False, "onnxruntime", Accelerator.CUDA),
        (Accelerator.OPENCL, False, True, "cv2dnn", Accelerator.OPENCL),
        (Accelerator.AUTO, True, True, "onnxruntime", Accelerator.CUDA),
        (Accelerator.AUTO, False, True, "cv2dnn", Accelerator.CPU),  # opencl is never auto-selected
        (Accelerator.AUTO, False, False, "cv2dnn", Accelerator.CPU),
    ],
)
def test_select_backend(accelerator, cuda, opencl, engine, resolved):
    plan = select_backend(accelerator, cuda_available=cuda, opencl_available=opencl)
    assert (plan.engine, plan.accelerator) == (engine, resolved)


def test_select_cuda_unavailable_raises():
    with pytest.raises(EdgeModelError, match="cuda"):
        select_backend(Accelerator.CUDA, cuda_available=False, opencl_available=True)


def test_select_opencl_unavailable_raises():
    with pytest.raises(EdgeModelError, match="opencl"):
        select_backend(Accelerator.OPENCL, cuda_available=True, opencl_available=False)


def test_select_invalid_accelerator_raises():
    with pytest.raises(EdgeModelError, match="Unsupported"):
        select_backend("ncs2", cuda_available=True, opencl_available=True)


def test_ncs2_is_rejected_by_settings():
    with pytest.raises(ValueError):
        EdgeSettings(accelerator="ncs2")


def test_constructor_cuda_unavailable_raises(tmp_path, monkeypatch):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    monkeypatch.setattr(edge_detector, "onnxruntime_has_cuda", lambda: False)
    with pytest.raises(EdgeModelError, match="cuda"):
        EdgeInferenceDetector(EdgeSettings(model_path=str(model), accelerator=Accelerator.CUDA))


def test_constructor_missing_model_raises(tmp_path):
    with pytest.raises(EdgeModelError, match="not found"):
        EdgeInferenceDetector(EdgeSettings(model_path=str(tmp_path / "nope.onnx"), accelerator=Accelerator.CPU))


def test_constructor_no_model_path_raises():
    with pytest.raises(EdgeModelError, match="EDGE__MODEL_PATH"):
        EdgeInferenceDetector(EdgeSettings(accelerator=Accelerator.CPU))


def test_constructor_invalid_model_raises(tmp_path):
    model = tmp_path / "bad.onnx"
    model.write_bytes(b"this is not an onnx model")
    with pytest.raises(EdgeModelError, match="Failed to load"):
        EdgeInferenceDetector(EdgeSettings(model_path=str(model), accelerator=Accelerator.CPU))


def test_input_size_must_be_positive():
    with pytest.raises(ValueError):
        EdgeSettings(input_size=0)


# --- preprocessing ---


def test_preprocess_values_and_layout():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[..., 1] = 255  # G
    frame[..., 2] = 255  # R (BGR order: B stays 0)
    blob = preprocess(frame, 4)
    assert blob.shape == (1, 3, 4, 4)
    assert blob.dtype == np.float32
    assert blob.flags["C_CONTIGUOUS"]
    # Channel order is RGB: R=1, G=1, B=0 after scaling.
    expected = [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (0 - 0.406) / 0.225]
    for c, value in enumerate(expected):
        np.testing.assert_allclose(blob[0, c], value, rtol=1e-5)


def test_preprocess_squash_resize_no_crop():
    frame = np.zeros((6, 12, 3), dtype=np.uint8)
    frame[:, 6:] = 255  # right half white
    blob = preprocess(frame, 4)
    # Squash keeps both halves visible: left columns black, right columns white.
    assert blob[0, 0, 0, 0] == pytest.approx((0 - 0.485) / 0.229)
    assert blob[0, 0, 0, 3] == pytest.approx((1 - 0.485) / 0.229)


def test_preprocess_rejects_non_color_frame():
    with pytest.raises(ValueError):
        preprocess(np.zeros((8, 8), dtype=np.uint8), 4)


# --- post-processing ---


def test_softmax_probability():
    assert classifier_fire_probability(np.array([[0.0, 0.0]])) == pytest.approx(0.5)
    assert classifier_fire_probability(np.array([[0.0, np.log(3.0)]])) == pytest.approx(0.75)
    assert classifier_fire_probability(np.array([[1000.0, -1000.0]])) == pytest.approx(0.0)  # numerically stable


def test_single_logit_sigmoid():
    assert classifier_fire_probability(np.array([[0.0]])) == pytest.approx(0.5)
    assert classifier_fire_probability(np.array([2.0])) == pytest.approx(1 / (1 + np.exp(-2.0)))


def test_unsupported_output_raises():
    with pytest.raises(ValueError):
        classifier_fire_probability(np.zeros((1, 5)))


# --- detect ---


def _detector(backend, threshold=0.5, **kwargs):
    return EdgeInferenceDetector(EdgeSettings(confidence_threshold=threshold, input_size=8), backend=backend, **kwargs)


@pytest.mark.asyncio
async def test_detect_end_to_end_with_mock_backend():
    backend = MagicMock()
    backend.predict.return_value = np.array([[0.0, np.log(3.0)]], dtype=np.float32)  # P = 0.75
    frame = np.full((16, 16, 3), 7, dtype=np.uint8)
    is_fire, out = await _detector(backend).detect(frame)
    assert is_fire is True
    assert out is frame
    blob = backend.predict.call_args.args[0]
    assert blob.shape == (1, 3, 8, 8)
    assert blob.dtype == np.float32


@pytest.mark.asyncio
@pytest.mark.parametrize(("threshold", "expected"), [(0.75, True), (0.7500001, False), (0.5, True)])
async def test_threshold_edge_is_inclusive(threshold, expected):
    backend = MagicMock()
    backend.predict.return_value = np.array([[0.0, np.log(3.0)]])
    is_fire, _ = await _detector(backend, threshold).detect(np.zeros((8, 8, 3), np.uint8))
    assert is_fire is expected


@pytest.mark.asyncio
async def test_detect_uses_custom_postprocess():
    detector = _detector(MagicMock(), postprocess=lambda _out: 0.9)
    is_fire, _ = await detector.detect(np.zeros((8, 8, 3), np.uint8))
    assert is_fire is True


@pytest.mark.asyncio
async def test_detect_never_raises_and_escalates_critical_once(monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr(edge_detector, "logger", mock_logger)
    backend = MagicMock()
    backend.predict.side_effect = RuntimeError("boom")
    detector = _detector(backend)
    frame = np.zeros((8, 8, 3), np.uint8)
    for _ in range(CONSECUTIVE_ERRORS_BEFORE_CRITICAL + 2):
        is_fire, out = await detector.detect(frame)
        assert is_fire is False
        assert out is frame
    assert mock_logger.exception.call_count == CONSECUTIVE_ERRORS_BEFORE_CRITICAL + 2
    assert mock_logger.critical.call_count == 1


@pytest.mark.asyncio
async def test_success_resets_error_counter():
    backend = MagicMock()
    backend.predict.side_effect = [RuntimeError("x")] * 3 + [np.array([[0.0, 0.0]])]
    detector = _detector(backend)
    frame = np.zeros((8, 8, 3), np.uint8)
    for _ in range(4):
        await detector.detect(frame)
    assert detector._consecutive_errors == 0


@pytest.mark.asyncio
async def test_bad_frame_does_not_raise():
    is_fire, _ = await _detector(MagicMock()).detect(np.zeros((8, 8), np.uint8))
    assert is_fire is False


# --- real cv2.dnn end-to-end with a hand-built ONNX classifier (raw protobuf, no onnx dependency) ---


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(num: int, payload: bytes) -> bytes:
    return _varint((num << 3) | 2) + _varint(len(payload)) + payload


def _vi(num: int, value: int) -> bytes:
    return _varint(num << 3) + _varint(value)


def _value_info(name: str, dims: list[int]) -> bytes:
    shape = b"".join(_ld(1, _vi(1, d)) for d in dims)
    tensor_type = _vi(1, 1) + _ld(2, shape)
    return _ld(1, name.encode()) + _ld(2, _ld(1, tensor_type))


def _initializer(name: str, dims: list[int], values: list[float]) -> bytes:
    raw = struct.pack(f"<{len(values)}f", *values)
    return b"".join(_vi(1, d) for d in dims) + _vi(2, 1) + _ld(8, name.encode()) + _ld(9, raw)


def _node(op: str, inputs: list[str], output: str) -> bytes:
    return b"".join(_ld(1, i.encode()) for i in inputs) + _ld(2, output.encode()) + _ld(4, op.encode())


def _build_classifier_onnx(size: int) -> bytes:
    """GlobalAveragePool -> Flatten -> MatMul -> Add; logit[1] = 10 * mean(R channel), logit[0] = 0."""
    nodes = [
        _node("GlobalAveragePool", ["x"], "p"),
        _node("Flatten", ["p"], "f"),
        _node("MatMul", ["f", "w"], "m"),
        _node("Add", ["m", "b"], "logits"),
    ]
    graph = b"".join(_ld(1, n) for n in nodes) + _ld(2, b"tiny")
    graph += _ld(5, _initializer("w", [3, 2], [0, 10, 0, 0, 0, 0]))
    graph += _ld(5, _initializer("b", [2], [0, 0]))
    graph += _ld(11, _value_info("x", [1, 3, size, size])) + _ld(12, _value_info("logits", [1, 2]))
    return _vi(1, 8) + _ld(8, _vi(2, 12)) + _ld(7, graph)


def _build_flatten_onnx(size: int) -> bytes:
    """Flatten -> MatMul classifier whose weight shape depends on the input size (wrong sizes fail to run)."""
    n = 3 * size * size
    nodes = [_node("Flatten", ["x"], "f"), _node("MatMul", ["f", "w"], "logits")]
    graph = b"".join(_ld(1, x) for x in nodes) + _ld(2, b"flat")
    graph += _ld(5, _initializer("w", [n, 2], [0.0] * (n * 2)))
    graph += _ld(11, _value_info("x", [1, 3, size, size])) + _ld(12, _value_info("logits", [1, 2]))
    return _vi(1, 8) + _ld(8, _vi(2, 12)) + _ld(7, graph)


@pytest.mark.asyncio
async def test_real_cv2_dnn_tiny_onnx_classifier(tmp_path):
    model = tmp_path / "tiny.onnx"
    model.write_bytes(_build_classifier_onnx(8))
    detector = EdgeInferenceDetector(
        EdgeSettings(model_path=str(model), accelerator=Accelerator.CPU, input_size=8, confidence_threshold=0.5)
    )

    red = np.zeros((20, 30, 3), np.uint8)
    red[..., 2] = 255  # BGR red -> high R channel -> fire
    blue = np.zeros((20, 30, 3), np.uint8)
    blue[..., 0] = 255  # BGR blue -> R == 0 -> no fire

    assert (await detector.detect(red))[0] is True
    assert (await detector.detect(blue))[0] is False


def test_input_size_mismatch_fails_at_construction(tmp_path):
    model = tmp_path / "flat.onnx"
    model.write_bytes(_build_flatten_onnx(8))
    with pytest.raises(EdgeModelError, match="EDGE__INPUT_SIZE"):
        EdgeInferenceDetector(EdgeSettings(model_path=str(model), accelerator=Accelerator.CPU, input_size=16))


def test_warm_up_rejects_unusable_output():
    backend = MagicMock()
    backend.predict.return_value = np.zeros((1, 5))
    detector = _detector(backend)  # injected backends skip the warm-up; call it explicitly
    with pytest.raises(EdgeModelError, match="EDGE__INPUT_SIZE"):
        detector._warm_up("m.onnx")


# --- fake onnxruntime ---


def _fake_ort(monkeypatch, providers, active=None, shape=(1, 3, 8, 8)):
    ort = types.ModuleType("onnxruntime")
    calls = {}

    class Session:
        def __init__(self, path, providers=None):
            calls["providers"] = providers

        def get_providers(self):
            return active if active is not None else calls["providers"]

        def get_inputs(self):
            return [types.SimpleNamespace(name="x", shape=list(shape))]

        def run(self, _names, feed):
            calls["feed"] = feed
            return [np.array([[0.0, 1.0]], dtype=np.float32)]

    ort.InferenceSession = Session
    ort.get_available_providers = lambda: providers
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return ort, calls


def test_onnxruntime_backend_cpu_loads_and_predicts(monkeypatch):
    _, calls = _fake_ort(monkeypatch, ["CPUExecutionProvider"])
    backend = OnnxRuntimeBackend("m.onnx", Accelerator.CPU, input_size=8)
    assert calls["providers"] == ["CPUExecutionProvider"]
    blob = np.zeros((1, 3, 8, 8), np.float32)
    assert backend.predict(blob).shape == (1, 2)
    assert calls["feed"]["x"] is blob


def test_onnxruntime_backend_cuda_provider_order(monkeypatch):
    _, calls = _fake_ort(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    backend = OnnxRuntimeBackend("m.onnx", Accelerator.CUDA)
    assert calls["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert backend.accelerator == Accelerator.CUDA


def test_onnxruntime_backend_cuda_provider_missing(monkeypatch):
    _fake_ort(monkeypatch, ["CPUExecutionProvider"])
    with pytest.raises(EdgeModelError, match="CUDAExecutionProvider"):
        OnnxRuntimeBackend("m.onnx", Accelerator.CUDA)


def test_onnxruntime_cuda_not_active_strict_raises(monkeypatch):
    _fake_ort(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"], active=["CPUExecutionProvider"])
    with pytest.raises(EdgeModelError, match="not the active"):
        OnnxRuntimeBackend("m.onnx", Accelerator.CUDA, strict=True)


def test_onnxruntime_cuda_not_active_auto_warns_and_reports_cpu(monkeypatch):
    _fake_ort(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"], active=["CPUExecutionProvider"])
    mock_logger = MagicMock()
    monkeypatch.setattr(edge_detector, "logger", mock_logger)
    backend = OnnxRuntimeBackend("m.onnx", Accelerator.CUDA, strict=False)
    assert backend.accelerator == Accelerator.CPU
    mock_logger.warning.assert_called_once()


def test_onnxruntime_input_shape_mismatch(monkeypatch):
    _fake_ort(monkeypatch, ["CPUExecutionProvider"], shape=(1, 3, 224, 224))
    with pytest.raises(EdgeModelError, match="EDGE__INPUT_SIZE"):
        OnnxRuntimeBackend("m.onnx", Accelerator.CPU, input_size=8)


def test_onnxruntime_dynamic_shape_is_accepted(monkeypatch):
    _fake_ort(monkeypatch, ["CPUExecutionProvider"], shape=(1, 3, "h", "w"))
    OnnxRuntimeBackend("m.onnx", Accelerator.CPU, input_size=8)


def test_onnxruntime_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(EdgeModelError, match="onnxruntime"):
        OnnxRuntimeBackend("m.onnx", Accelerator.CPU)


def test_onnxruntime_session_failure(monkeypatch):
    ort, _ = _fake_ort(monkeypatch, ["CPUExecutionProvider"])
    ort.InferenceSession = MagicMock(side_effect=RuntimeError("bad model"))
    with pytest.raises(EdgeModelError, match="Failed to load"):
        OnnxRuntimeBackend("m.onnx", Accelerator.CPU)


def test_onnxruntime_has_cuda_probe(monkeypatch):
    _fake_ort(monkeypatch, ["CUDAExecutionProvider"])
    assert edge_detector.onnxruntime_has_cuda() is True
    _fake_ort(monkeypatch, ["CPUExecutionProvider"])
    assert edge_detector.onnxruntime_has_cuda() is False
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert edge_detector.onnxruntime_has_cuda() is False


def test_opencl_available_probe(monkeypatch):
    monkeypatch.setattr(edge_detector.cv2.ocl, "haveOpenCL", lambda: True)
    monkeypatch.setattr(edge_detector.cv2.ocl, "useOpenCL", lambda: True)
    assert edge_detector.opencl_available() is True
    monkeypatch.setattr(edge_detector.cv2.ocl, "useOpenCL", lambda: False)
    assert edge_detector.opencl_available() is False
    monkeypatch.setattr(edge_detector.cv2.ocl, "haveOpenCL", MagicMock(side_effect=RuntimeError))
    assert edge_detector.opencl_available() is False


def test_cvdnn_opencl_branch_warns_and_sets_target(monkeypatch):
    net = MagicMock()
    monkeypatch.setattr(edge_detector.cv2.dnn, "readNetFromONNX", lambda _p: net)
    mock_logger = MagicMock()
    monkeypatch.setattr(edge_detector, "logger", mock_logger)
    CvDnnBackend("m.onnx", Accelerator.OPENCL)
    mock_logger.warning.assert_called_once()
    net.setPreferableTarget.assert_called_once_with(edge_detector.cv2.dnn.DNN_TARGET_OPENCL)


# --- hardening ---


@pytest.mark.asyncio
async def test_critical_fires_again_on_second_outage(monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr(edge_detector, "logger", mock_logger)
    n = CONSECUTIVE_ERRORS_BEFORE_CRITICAL
    backend = MagicMock()
    backend.predict.side_effect = [RuntimeError("x")] * n + [np.array([[0.0, 0.0]])] + [RuntimeError("x")] * n
    detector = _detector(backend)
    frame = np.zeros((8, 8, 3), np.uint8)
    for _ in range(2 * n + 1):
        await detector.detect(frame)
    assert mock_logger.critical.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
async def test_non_finite_probability_counts_as_error(bad):
    detector = _detector(MagicMock(), postprocess=lambda _o: bad)
    is_fire, _ = await detector.detect(np.zeros((8, 8, 3), np.uint8))
    assert is_fire is False
    assert detector._consecutive_errors == 1


@pytest.mark.asyncio
async def test_backend_inference_is_serialized():
    state = {"active": 0, "max": 0}

    class Slow:
        def predict(self, blob):
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            time.sleep(0.02)
            state["active"] -= 1
            return np.array([[0.0, 0.0]])

    detector = _detector(Slow())
    assert isinstance(detector._lock, type(threading.Lock()))
    frame = np.zeros((8, 8, 3), np.uint8)
    await asyncio.gather(*(detector.detect(frame) for _ in range(4)))
    assert state["max"] == 1
