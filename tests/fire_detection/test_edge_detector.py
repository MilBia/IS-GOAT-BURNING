"""Tests for the EdgeInferenceDetector, its preprocessing, post-processing and backend selection."""

import struct
from unittest.mock import MagicMock

import numpy as np
import pytest

from is_goat_burning.config import Accelerator
from is_goat_burning.config import EdgeSettings
from is_goat_burning.fire_detection import edge_detector
from is_goat_burning.fire_detection.edge_detector import CONSECUTIVE_ERRORS_BEFORE_CRITICAL
from is_goat_burning.fire_detection.edge_detector import EdgeInferenceDetector
from is_goat_burning.fire_detection.edge_detector import EdgeModelError
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


@pytest.mark.asyncio
async def test_real_cv2_dnn_tiny_onnx_classifier(tmp_path):
    model = tmp_path / "tiny.onnx"
    model.write_bytes(_build_classifier_onnx(8))
    try:
        detector = EdgeInferenceDetector(
            EdgeSettings(model_path=str(model), accelerator=Accelerator.CPU, input_size=8, confidence_threshold=0.5)
        )
    except EdgeModelError as e:  # pragma: no cover - depends on the OpenCV build
        pytest.skip(f"cv2.dnn could not load the hand-built ONNX model: {e}")

    red = np.zeros((20, 30, 3), np.uint8)
    red[..., 2] = 255  # BGR red -> high R channel -> fire
    blue = np.zeros((20, 30, 3), np.uint8)
    blue[..., 0] = 255  # BGR blue -> R == 0 -> no fire

    assert (await detector.detect(red))[0] is True
    assert (await detector.detect(blue))[0] is False
