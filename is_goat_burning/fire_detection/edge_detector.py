"""Offline fire verification with a local image classifier (``hybrid_edge`` building block).

Backends: ``cv2.dnn`` (default, CPU; NVIDIA CUDA with an OpenCV built with CUDA; experimental OpenCL) and OpenVINO
(CPU or Intel iGPU, only when explicitly requested). ``auto`` uses ``cv2.dnn`` CUDA when OpenCV offers the dnn CUDA target
and otherwise the CPU; an explicit ``cuda`` never falls back.
"""

import asyncio
from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import threading
from typing import Protocol

import cv2
import numpy as np

from is_goat_burning.config import Accelerator
from is_goat_burning.config import EdgeSettings
from is_goat_burning.fire_detection.verifier_health import VerifierError
from is_goat_burning.logger import get_logger

logger = get_logger("EdgeInferenceDetector")

# Number of consecutive failed inferences after which a CRITICAL message is logged (once per outage).
CONSECUTIVE_ERRORS_BEFORE_CRITICAL = 5

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)


class EdgeModelError(RuntimeError):
    """Raised when the edge model cannot be loaded or the requested backend is unavailable."""


class EdgeBackend(Protocol):
    """Runs a preprocessed NCHW blob through the model."""

    def predict(self, blob: np.ndarray) -> np.ndarray:
        """Runs inference.

        Args:
            blob: ``float32`` array of shape ``(1, 3, H, W)``.

        Returns:
            The raw model output.
        """
        ...


PostProcessor = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class BackendPlan:
    """Resolved inference engine selection.

    Attributes:
        engine: ``"cv2dnn"`` or ``"openvino"``.
        accelerator: The concrete (never ``auto``) accelerator.
    """

    engine: str
    accelerator: Accelerator


def cuda_available() -> bool:
    """Probes whether ``cv2.dnn`` can run on an NVIDIA GPU.

    Checks the DNN CUDA target rather than ``cv2.cuda.getCudaEnabledDeviceCount()``: an OpenCV built with CUDA but
    without cuDNN / ``OPENCV_DNN_CUDA`` sees the device, yet ``cv2.dnn`` would silently run on the CPU. The target list
    is empty in the pip wheels and without a usable device or driver.

    Returns:
        ``True`` if ``DNN_TARGET_CUDA`` is available for ``DNN_BACKEND_CUDA``.
    """
    try:
        return cv2.dnn.DNN_TARGET_CUDA in cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA)
    except Exception:  # cv2.dnn without CUDA constants or a driver error
        return False


def opencl_available() -> bool:
    """Probes whether OpenCV reports a usable OpenCL runtime.

    Returns:
        ``True`` if ``cv2.ocl.haveOpenCL()`` and ``cv2.ocl.useOpenCL()`` are true.
    """
    try:
        return bool(cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL())
    except Exception:
        return False


def select_backend(accelerator: Accelerator, *, cuda_available: bool, opencl_available: bool) -> BackendPlan:
    """Maps the configured accelerator to an inference engine (pure function).

    ``auto`` selects CUDA when available and the CPU otherwise (both ``cv2.dnn``). OpenCL is never
    auto-selected because ``cv2.dnn`` OpenCL gave no gain and was unreliable in the #73 research. OpenVINO is never
    auto-selected either (optional package; its availability, including the iGPU, is checked by ``OpenVinoBackend``).

    Args:
        accelerator: The configured accelerator.
        cuda_available: Whether ``cv2.dnn`` offers the CUDA target (``cuda_available()``).
        opencl_available: Whether OpenCV reports an OpenCL runtime.

    Returns:
        The resolved plan.

    Raises:
        EdgeModelError: If an explicitly requested accelerator is unavailable or unsupported.
    """
    if accelerator == Accelerator.AUTO:
        accelerator = Accelerator.CUDA if cuda_available else Accelerator.CPU
    if accelerator == Accelerator.CPU:
        return BackendPlan("cv2dnn", Accelerator.CPU)
    if accelerator == Accelerator.CUDA:
        if not cuda_available:
            raise EdgeModelError(
                "EDGE__ACCELERATOR=cuda requested but OpenCV reports no cv2.dnn CUDA target (needs OpenCV built with "
                "CUDA, cuDNN and OPENCV_DNN_CUDA, e.g. the Docker gpu image, an NVIDIA GPU and its driver)."
            )
        return BackendPlan("cv2dnn", Accelerator.CUDA)
    if accelerator == Accelerator.OPENCL:
        if not opencl_available:
            raise EdgeModelError("EDGE__ACCELERATOR=opencl requested but OpenCV reports no usable OpenCL runtime.")
        return BackendPlan("cv2dnn", Accelerator.OPENCL)
    if accelerator in (Accelerator.OPENVINO, Accelerator.OPENVINO_GPU):
        return BackendPlan("openvino", accelerator)
    raise EdgeModelError(f"Unsupported EDGE__ACCELERATOR value: {accelerator!r}")


def preprocess(frame: np.ndarray, input_size: int) -> np.ndarray:
    """Converts a BGR frame to the classifier input blob.

    BGR -> RGB, squash-resize (no crop) to ``input_size``, scale to 0-1, ImageNet mean/std, NCHW ``float32``.

    Args:
        frame: BGR ``uint8`` image of shape ``(H, W, 3)``.
        input_size: Side of the square model input.

    Returns:
        Array of shape ``(1, 3, input_size, input_size)`` and dtype ``float32``.

    Raises:
        ValueError: If the frame is not a 3-channel image.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR frame, got shape {frame.shape}.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    return np.ascontiguousarray((blob - _MEAN) / _STD, dtype=np.float32)


def classifier_fire_probability(output: np.ndarray) -> float:
    """Converts classifier output to P(fire).

    Two logits -> ``softmax(logits)[1]``; a single logit -> sigmoid.

    Args:
        output: Raw model output holding one or two logits.

    Returns:
        P(fire) in ``[0, 1]``.

    Raises:
        ValueError: If the output does not hold one or two values.
    """
    logits = np.asarray(output, dtype=np.float64).reshape(-1)
    if logits.size == 1:
        return float(1.0 / (1.0 + np.exp(-logits[0])))
    if logits.size == 2:
        exp = np.exp(logits - logits.max())
        return float(exp[1] / exp.sum())
    raise ValueError(f"Unsupported classifier output with {logits.size} values (expected 1 or 2).")


def _read_varint(buf: bytes | memoryview, pos: int) -> tuple[int, int]:
    """Decodes a protobuf varint.

    Args:
        buf: The buffer.
        pos: Start offset.

    Returns:
        ``(value, next_offset)``.

    Raises:
        ValueError: If the varint is truncated or longer than 10 bytes.
    """
    result = 0
    for i in range(10):
        if pos + i >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[pos + i]
        result |= (byte & 0x7F) << (7 * i)
        if not byte & 0x80:
            return result & ((1 << 64) - 1), pos + i + 1
    raise ValueError("varint too long")


def _pb_fields(buf: bytes | memoryview) -> Iterator[tuple[int, int | memoryview]]:
    """Iterates over the top-level fields of a protobuf message.

    Args:
        buf: The serialized message.

    Yields:
        ``(field_number, value)``: ``int`` for varints, ``memoryview`` (zero-copy) for length-delimited
        fields (fixed-width fields are skipped).

    Raises:
        ValueError: If the message is malformed or truncated.
    """
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
            yield number, value
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            if pos + length > len(buf):
                raise ValueError("truncated field")
            yield number, buf[pos : pos + length]
            pos += length
        elif wire in (1, 5):
            pos += 8 if wire == 1 else 4
            if pos > len(buf):
                raise ValueError("truncated field")
        else:
            raise ValueError(f"unsupported wire type {wire}")


def _first_field(buf: bytes | memoryview, number: int) -> memoryview | None:
    """Returns the first length-delimited field ``number`` of a message, or ``None``."""
    for num, value in _pb_fields(buf):
        if num == number and isinstance(value, memoryview):
            return value
    return None


def read_onnx_input_shape(model_path: Path | str) -> list[int | str] | None:
    """Reads the declared shape of the first graph input of an ONNX model without the ``onnx`` package.

    Walks the protobuf wire format by hand: ``ModelProto.graph`` (7) -> ``GraphProto.input`` (11) ->
    ``ValueInfoProto.type`` (2) -> ``TypeProto.tensor_type`` (1) -> ``Tensor.shape`` (2) -> ``TensorShapeProto.dim`` (1)
    -> ``Dimension.dim_value`` (1) / ``dim_param`` (2). The first input is used as is (initializers listed among the
    inputs by old IR versions are not skipped); our classifiers take the image first.

    Args:
        model_path: Path to the ``.onnx`` file.

    Returns:
        One entry per dimension (``int`` for static, ``str`` for dynamic or unnamed dims), or ``None`` if the file cannot
        be read or parsed or declares no shape. Never raises.
    """
    try:
        graph = _first_field(memoryview(Path(model_path).read_bytes()), 7)
        value_info = _first_field(graph, 11) if graph is not None else None
        type_proto = _first_field(value_info, 2) if value_info is not None else None
        tensor_type = _first_field(type_proto, 1) if type_proto is not None else None
        shape = _first_field(tensor_type, 2) if tensor_type is not None else None
        if shape is None:
            return None
        dims: list[int | str] = []
        for num, dim in _pb_fields(shape):
            if num != 1 or not isinstance(dim, memoryview):
                continue
            value: int | str = "?"
            for dnum, dvalue in _pb_fields(dim):
                if dnum == 1 and isinstance(dvalue, int):
                    # dim_value is int64: -1 arrives as a 10-byte varint (unsigned 2**64 - 1).
                    value = dvalue - (1 << 64) if dvalue >= 1 << 63 else dvalue
                elif dnum == 2 and isinstance(dvalue, memoryview):
                    value = bytes(dvalue).decode("utf-8", errors="replace")
            dims.append(value)
        return dims
    except Exception:  # best-effort probe: must never break model loading
        return None


def _check_declared_input_size(model_path: Path | str, shape: list[int | str] | None, input_size: int | None) -> None:
    """Compares a model's declared NCHW input with ``EDGE__INPUT_SIZE``.

    Unknown, non-4-D, dynamic or non-positive (``-1`` / ``0``) shapes are skipped (the warm-up remains the safety net).

    Args:
        model_path: Path to the model (for the message).
        shape: Declared input shape, or ``None`` if unknown.
        input_size: Configured square input side; ``None`` disables the check.

    Raises:
        EdgeModelError: If the declared static height or width differs from ``input_size``.
    """
    if input_size is None or shape is None or len(shape) != 4:
        return
    n_dims = [d if isinstance(d, int) and d > 0 else None for d in shape]
    if n_dims[1] is not None and n_dims[2] is not None and n_dims[3] == 3 and n_dims[1] != 3:
        raise EdgeModelError(
            f"Model '{model_path}' declares the input {shape} which looks like an NHWC model; "
            "only NCHW is supported by the preprocessing contract."
        )
    height, width = n_dims[2], n_dims[3]
    if height is None or width is None:  # unknown, dynamic or non-positive (-1 / 0) dims
        return
    if (height, width) != (input_size, input_size):
        raise EdgeModelError(f"Model '{model_path}' expects a {height}x{width} input but EDGE__INPUT_SIZE={input_size}.")


class CvDnnBackend:
    """``cv2.dnn`` backend (default; CPU, NVIDIA CUDA or experimental OpenCL)."""

    def __init__(self, model_path: Path, accelerator: Accelerator, *, input_size: int | None = None) -> None:
        """Loads the model.

        Args:
            model_path: Path to the ``.onnx`` model.
            accelerator: ``CPU``, ``CUDA`` or ``OPENCL``.
            input_size: Expected square input side; compared with the model's declared static input dims when given
                (``cv2.dnn`` does not expose them, so the ONNX file is parsed).

        Raises:
            EdgeModelError: If the model cannot be loaded or its declared input size does not match ``input_size``.
        """
        try:
            self._net = cv2.dnn.readNetFromONNX(str(model_path))
            if accelerator == Accelerator.CUDA:
                # Layers without a CUDA implementation run on the CPU (OPENCV_LOG_LEVEL=INFO lists the per-layer choice).
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            elif accelerator == Accelerator.OPENCL:
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                logger.warning(
                    "cv2.dnn OpenCL is experimental and unreliable in recent OpenCV releases (see #73); "
                    "inference may silently run on the CPU."
                )
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            else:
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as e:
            raise EdgeModelError(f"Failed to load ONNX model '{model_path}' with cv2.dnn: {e}") from e
        if input_size is not None:
            _check_declared_input_size(model_path, read_onnx_input_shape(model_path), input_size)

    def predict(self, blob: np.ndarray) -> np.ndarray:
        """Runs a forward pass.

        Args:
            blob: ``float32`` NCHW input.

        Returns:
            The raw network output.
        """
        self._net.setInput(blob)
        return np.asarray(self._net.forward())


def _openvino_input_shape(partial_shape) -> list[int | str] | None:
    """Converts an OpenVINO ``PartialShape`` to the shape list used by ``_check_declared_input_size``.

    Args:
        partial_shape: The model input's ``PartialShape``.

    Returns:
        One entry per dimension (``int`` for static, ``"?"`` for dynamic), or ``None`` if the rank is dynamic.
    """
    if partial_shape.rank.is_dynamic:
        return None
    return [dim.get_length() if dim.is_static else "?" for dim in partial_shape]


def _block_openvino_telemetry() -> None:
    """Makes ``openvino_telemetry`` unimportable so ``import openvino`` sends nothing (#207).

    ``openvino`` depends on ``openvino-telemetry`` and, unless the user opted out in ``~/intel``, ``import openvino``
    sends a usage event to Google Analytics from a ``multiprocessing`` child (a resident forkserver on Python 3.14).
    Every ``openvino`` import of that package is wrapped in ``try/except ImportError`` and falls back to its own no-op
    ``openvino.tools.ovc.telemetry_stub``, so a ``None`` entry in ``sys.modules`` disables it without touching the
    environment or the home directory. It must run before the first ``import openvino`` and is left in place for the
    process lifetime (later lazy imports inside ``openvino`` must hit the stub too).
    """
    sys.modules["openvino_telemetry"] = None


class OpenVinoBackend:
    """OpenVINO backend (CPU or Intel iGPU); ``openvino`` is imported lazily (optional ``openvino`` extra).

    Compiled with ``PERFORMANCE_HINT=LATENCY`` (single request); on the CPU also ``NUM_STREAMS=1`` and OpenVINO's
    default thread count. Each device runs its default inference precision (f32 on most CPUs, f16 on the iGPU).

    OpenVINO telemetry is disabled: see ``_block_openvino_telemetry``.
    """

    def __init__(self, model_path: Path, accelerator: Accelerator, *, input_size: int | None = None) -> None:
        """Loads and compiles the model.

        Args:
            model_path: Path to the ``.onnx`` model or OpenVINO IR ``.xml`` (with its ``.bin`` next to it).
            accelerator: ``OPENVINO`` (CPU) or ``OPENVINO_GPU`` (Intel iGPU).
            input_size: Expected square input side; compared with the model's static input dims when given.

        Raises:
            EdgeModelError: If openvino is missing, the iGPU is not available, the model cannot be loaded or compiled
                or its input size does not match ``input_size``.
        """
        _block_openvino_telemetry()
        try:
            import openvino as ov
        except ImportError as e:
            raise EdgeModelError(
                f"The 'openvino' package is required for EDGE__ACCELERATOR={accelerator} "
                "(pip install --no-deps -r requirements-openvino.txt)."
            ) from e
        device = "GPU" if accelerator == Accelerator.OPENVINO_GPU else "CPU"
        config: dict[str, object] = {"PERFORMANCE_HINT": "LATENCY"}
        if device == "CPU":
            config["NUM_STREAMS"] = 1
        try:
            core = ov.Core()
            if device == "GPU" and not any(d == "GPU" or d.startswith("GPU.") for d in core.available_devices):
                raise EdgeModelError(
                    f"EDGE__ACCELERATOR=openvino_gpu requested but OpenVINO reports no GPU device "
                    f"(available: {core.available_devices}); check the Intel compute runtime and /dev/dri access."
                )
            model = core.read_model(str(model_path))
            shape = _openvino_input_shape(model.input(0).get_partial_shape())
            _check_declared_input_size(model_path, shape, input_size)
            compiled = core.compile_model(model, device, config)
        except EdgeModelError:
            raise
        except Exception as e:
            raise EdgeModelError(f"Failed to load model '{model_path}' with OpenVINO ({device}): {e}") from e
        self._request = compiled.create_infer_request()
        self._output = compiled.output(0)

    def predict(self, blob: np.ndarray) -> np.ndarray:
        """Runs inference.

        Args:
            blob: ``float32`` NCHW input.

        Returns:
            A copy of the first model output (the request's output buffer is reused by the next call).
        """
        return np.array(self._request.infer({0: blob})[self._output], copy=True)


def _create_backend(plan: BackendPlan, model_path: Path, input_size: int) -> EdgeBackend:
    """Instantiates the backend for a plan.

    Args:
        plan: The selected plan.
        model_path: Path to the model.
        input_size: Expected square input side.

    Returns:
        The backend.

    Raises:
        EdgeModelError: If the backend cannot be created.
    """
    if plan.engine == "openvino":
        return OpenVinoBackend(model_path, plan.accelerator, input_size=input_size)
    return CvDnnBackend(model_path, plan.accelerator, input_size=input_size)


class EdgeInferenceDetector:
    """Confirms fire with a local image classifier (fire / no-fire) running fully offline.

    Inference failures are logged at ERROR and counted (``needs_rebuild``); after
    ``CONSECUTIVE_ERRORS_BEFORE_CRITICAL`` consecutive failures a CRITICAL message is logged once.
    ``verify()`` (used by ``HybridFireDetector``) raises ``VerifierError`` on a failure so the
    ``VERIFIER_FAILURE_POLICY`` applies; ``detect()`` never raises and reports a failure as "no fire".

    With ``EDGE__ACCELERATOR=auto`` ``cv2.dnn`` CUDA is preferred when OpenCV offers the dnn CUDA target; if the model
    cannot be loaded or warmed up on it, the detector falls back to ``cv2.dnn`` on the CPU (WARNING). An explicit
    ``cuda`` fails with ``EdgeModelError`` instead, as do ``openvino`` / ``openvino_gpu`` when the package or the iGPU
    is missing.
    """

    def __init__(
        self,
        settings: EdgeSettings,
        *,
        backend: EdgeBackend | None = None,
        postprocess: PostProcessor = classifier_fire_probability,
    ) -> None:
        """Initializes the detector and loads the model.

        Args:
            settings: Edge configuration.
            backend: Optional pre-built backend (mainly for tests); skips model loading and backend selection.
            postprocess: Converts the raw model output to P(fire); replaceable for other model heads.

        Raises:
            EdgeModelError: If the model is missing/invalid or the requested backend is unavailable.
        """
        self.threshold = settings.confidence_threshold
        self.input_size = settings.input_size
        self._postprocess = postprocess
        self._consecutive_errors = 0
        #: ``P(fire)`` of the last successful ``verify()`` (None before it and after a failed call).
        self.last_probability: float | None = None
        self._lock = threading.Lock()  # backends are not thread-safe; a cancelled to_thread keeps running
        if backend is not None:
            self._backend = backend
            return
        if not settings.model_path:
            raise EdgeModelError("EDGE__MODEL_PATH must be set to use the edge detector.")
        model_path = Path(settings.model_path)
        if not model_path.is_file():
            raise EdgeModelError(f"Edge model file not found: {model_path}")
        plan = select_backend(
            settings.accelerator,
            cuda_available=cuda_available() if settings.accelerator in (Accelerator.AUTO, Accelerator.CUDA) else False,
            opencl_available=opencl_available() if settings.accelerator == Accelerator.OPENCL else False,
        )
        try:
            self._backend = _create_backend(plan, model_path, self.input_size)
            self._warm_up(model_path)
        except EdgeModelError as e:
            if plan.accelerator != Accelerator.CUDA:
                raise
            if settings.accelerator != Accelerator.AUTO:
                raise EdgeModelError(f"EDGE__ACCELERATOR=cuda requested but unusable: {e}") from e
            logger.warning("cv2.dnn CUDA could not be used (%s); falling back to cv2.dnn on the CPU.", e)
            plan = BackendPlan("cv2dnn", Accelerator.CPU)
            self._backend = _create_backend(plan, model_path, self.input_size)
            self._warm_up(model_path)
        logger.info("Edge detector ready: engine=%s accelerator=%s model=%s", plan.engine, plan.accelerator, model_path)

    def _warm_up(self, model_path: Path | str) -> None:
        """Runs one zero-frame inference to fail fast on a wrong input size or unusable output.

        Args:
            model_path: Model path, used in the error message.

        Raises:
            EdgeModelError: If the inference fails or yields an unusable output.
        """
        try:
            self._infer(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))
        except Exception as e:
            raise EdgeModelError(
                f"Test inference with model '{model_path}' failed ({e}); check that EDGE__INPUT_SIZE={self.input_size} "
                "matches the model and that it is a fire classifier with 1 or 2 outputs."
            ) from e

    @property
    def needs_rebuild(self) -> bool:
        """Whether the last inference failed, i.e. the backend session may be broken and should be rebuilt."""
        return self._consecutive_errors > 0

    def _record_error(self) -> None:
        """Counts a failed inference and escalates to CRITICAL once per outage."""
        self._consecutive_errors += 1
        if self._consecutive_errors == CONSECUTIVE_ERRORS_BEFORE_CRITICAL:
            logger.critical(
                "Edge inference failed %d times in a row; fire verification is effectively disabled until it recovers.",
                self._consecutive_errors,
            )

    def _infer(self, frame: np.ndarray) -> float:
        """Runs preprocessing, inference and post-processing (blocking)."""
        blob = preprocess(frame, self.input_size)
        with self._lock:
            output = self._backend.predict(blob)
        probability = float(self._postprocess(output))
        if not math.isfinite(probability):
            raise ValueError(f"Model produced a non-finite probability: {probability}.")
        return probability

    async def verify(self, frame: np.ndarray) -> bool:
        """Classifies a frame with the local model.

        Args:
            frame: The input video frame. Only BGR ``np.ndarray`` frames are accepted; GPU/UMat frames must be
                converted beforehand (``HybridFireDetector`` does this).

        Returns:
            The verdict: True if ``P(fire) >= threshold``.

        Raises:
            VerifierError: If the inference failed (the failure is counted, so ``needs_rebuild`` becomes True).
        """
        self.last_probability = None
        try:
            probability = await asyncio.to_thread(self._infer, frame)
        except Exception as e:
            logger.exception("Edge inference failed.")
            self._record_error()
            raise VerifierError(f"Edge inference failed: {type(e).__name__}") from e

        if self._consecutive_errors:
            logger.info("Edge inference recovered after %d consecutive errors.", self._consecutive_errors)
            self._consecutive_errors = 0

        self.last_probability = probability
        is_fire = probability >= self.threshold
        logger.debug("Edge result: P(fire)=%.3f threshold=%.2f is_fire=%s", probability, self.threshold, is_fire)
        return is_fire

    async def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Analyzes a frame for fire with the local model; never raises.

        Args:
            frame: The input video frame (BGR ``np.ndarray``).

        Returns:
            A tuple containing:
            - A boolean indicating if fire was detected (``False`` on any error).
            - The original, unmodified frame.
        """
        try:
            return await self.verify(frame), frame
        except VerifierError:
            return False, frame
