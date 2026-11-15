"""Offline fire verification with a local image classifier (``hybrid_edge`` building block).

Backends: ``cv2.dnn`` (default, CPU; experimental OpenCL) and ONNX Runtime (CUDA). ``auto`` uses ONNX Runtime CUDA
only when it is actually active and otherwise ``cv2.dnn`` on the CPU; an explicit ``cuda`` never falls back.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import Protocol

import cv2
import numpy as np

from is_goat_burning.config import Accelerator
from is_goat_burning.config import EdgeSettings
from is_goat_burning.logger import get_logger

logger = get_logger("EdgeInferenceDetector")

# Number of consecutive failed inferences after which a CRITICAL message is logged (once per outage).
CONSECUTIVE_ERRORS_BEFORE_CRITICAL = 5

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

_CUDA_PROVIDER = "CUDAExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


class EdgeModelError(RuntimeError):
    """Raised when the edge model cannot be loaded or the requested backend is unavailable."""


class CudaNotActiveError(EdgeModelError):
    """Raised when onnxruntime lists the CUDA provider but the session does not run on it."""


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
        engine: ``"cv2dnn"`` or ``"onnxruntime"``.
        accelerator: The concrete (never ``auto``) accelerator.
    """

    engine: str
    accelerator: Accelerator


def onnxruntime_has_cuda() -> bool:
    """Probes whether ONNX Runtime is installed and offers the CUDA execution provider.

    Returns:
        ``True`` if ``CUDAExecutionProvider`` is available.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return _CUDA_PROVIDER in ort.get_available_providers()


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

    ``auto`` selects CUDA (ONNX Runtime) when available and CPU (``cv2.dnn``) otherwise. OpenCL is never
    auto-selected because ``cv2.dnn`` OpenCL gave no gain and was unreliable in the #73 research.

    Args:
        accelerator: The configured accelerator.
        cuda_available: Whether the ONNX Runtime CUDA provider is available.
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
                "EDGE__ACCELERATOR=cuda requires onnxruntime-gpu with CUDAExecutionProvider, which is not available."
            )
        return BackendPlan("onnxruntime", Accelerator.CUDA)
    if accelerator == Accelerator.OPENCL:
        if not opencl_available:
            raise EdgeModelError("EDGE__ACCELERATOR=opencl requested but OpenCV reports no usable OpenCL runtime.")
        return BackendPlan("cv2dnn", Accelerator.OPENCL)
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


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
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
            return result, pos + i + 1
    raise ValueError("varint too long")


def _pb_fields(buf: bytes):
    """Iterates over the top-level fields of a protobuf message.

    Args:
        buf: The serialized message.

    Yields:
        ``(field_number, value)``: ``int`` for varints, ``bytes`` for length-delimited fields (fixed-width fields are
        skipped).

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


def _first_field(buf: bytes, number: int) -> bytes | None:
    """Returns the first length-delimited field ``number`` of a message, or ``None``."""
    for num, value in _pb_fields(buf):
        if num == number and isinstance(value, bytes):
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
        graph = _first_field(Path(model_path).read_bytes(), 7)
        value_info = _first_field(graph, 11) if graph is not None else None
        type_proto = _first_field(value_info, 2) if value_info is not None else None
        tensor_type = _first_field(type_proto, 1) if type_proto is not None else None
        shape = _first_field(tensor_type, 2) if tensor_type is not None else None
        if shape is None:
            return None
        dims: list[int | str] = []
        for num, dim in _pb_fields(shape):
            if num != 1 or not isinstance(dim, bytes):
                continue
            value: int | str = "?"
            for dnum, dvalue in _pb_fields(dim):
                if dnum == 1 and isinstance(dvalue, int):
                    value = dvalue
                elif dnum == 2 and isinstance(dvalue, bytes):
                    value = dvalue.decode("utf-8", errors="replace")
            dims.append(value)
        return dims
    except Exception:  # best-effort probe: must never break model loading
        return None


def _check_declared_input_size(model_path: Path | str, shape: list[int | str] | None, input_size: int | None) -> None:
    """Compares a model's declared NCHW input with ``EDGE__INPUT_SIZE``.

    Unknown, non-4-D or dynamic shapes are skipped (the warm-up inference remains the safety net).

    Args:
        model_path: Path to the model (for the message).
        shape: Declared input shape, or ``None`` if unknown.
        input_size: Configured square input side; ``None`` disables the check.

    Raises:
        EdgeModelError: If the declared static height or width differs from ``input_size``.
    """
    if input_size is None or shape is None or len(shape) != 4:
        return
    height, width = shape[2], shape[3]
    if not (isinstance(height, int) and isinstance(width, int)):
        return
    if (height, width) != (input_size, input_size):
        raise EdgeModelError(f"Model '{model_path}' expects a {height}x{width} input but EDGE__INPUT_SIZE={input_size}.")


class CvDnnBackend:
    """``cv2.dnn`` backend (default; CPU or experimental OpenCL)."""

    def __init__(self, model_path: Path, accelerator: Accelerator, *, input_size: int | None = None) -> None:
        """Loads the model.

        Args:
            model_path: Path to the ``.onnx`` model.
            accelerator: ``CPU`` or ``OPENCL``.
            input_size: Expected square input side; compared with the model's declared static input dims when given
                (``cv2.dnn`` does not expose them, so the ONNX file is parsed).

        Raises:
            EdgeModelError: If the model cannot be loaded or its declared input size does not match ``input_size``.
        """
        try:
            self._net = cv2.dnn.readNetFromONNX(str(model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            if accelerator == Accelerator.OPENCL:
                logger.warning(
                    "cv2.dnn OpenCL is experimental and unreliable in recent OpenCV releases (see #73); "
                    "inference may silently run on the CPU."
                )
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            else:
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


class OnnxRuntimeBackend:
    """ONNX Runtime backend (CPU or CUDA execution provider); ``onnxruntime`` is imported lazily."""

    def __init__(self, model_path: Path, accelerator: Accelerator, *, input_size: int | None = None) -> None:
        """Loads the model.

        Args:
            model_path: Path to the ``.onnx`` model.
            accelerator: ``CPU`` or ``CUDA``.
            input_size: Expected square input side; compared with the model's static input dims when given.

        Raises:
            EdgeModelError: If onnxruntime is missing, CUDA is unavailable, the model cannot be loaded or its input
                size does not match ``input_size``.
            EdgeModelError: (``CudaNotActiveError``) If CUDA was requested and is listed but is not the active provider.
        """
        self.accelerator = accelerator
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise EdgeModelError(
                "The 'onnxruntime' package is required for this accelerator (install onnxruntime or onnxruntime-gpu)."
            ) from e
        providers = [_CPU_PROVIDER]
        if accelerator == Accelerator.CUDA:
            if _CUDA_PROVIDER not in ort.get_available_providers():
                raise EdgeModelError("onnxruntime has no CUDAExecutionProvider (install onnxruntime-gpu).")
            providers = [_CUDA_PROVIDER, _CPU_PROVIDER]
        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as e:
            raise EdgeModelError(f"Failed to load ONNX model '{model_path}' with onnxruntime: {e}") from e
        if accelerator == Accelerator.CUDA and self._session.get_providers()[0] != _CUDA_PROVIDER:
            raise CudaNotActiveError(
                f"CUDA is not the active onnxruntime provider ({self._session.get_providers()}); "
                "check the CUDA/cuDNN installation."
            )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        _check_declared_input_size(model_path, list(model_input.shape), input_size)

    def predict(self, blob: np.ndarray) -> np.ndarray:
        """Runs inference.

        Args:
            blob: ``float32`` NCHW input.

        Returns:
            The first model output.
        """
        return np.asarray(self._session.run(None, {self._input_name: blob})[0])


def _create_backend(
    plan: BackendPlan, model_path: Path, input_size: int, *, cpu_fallback: bool
) -> tuple[EdgeBackend, BackendPlan]:
    """Instantiates the backend for a plan.

    Args:
        plan: The selected plan.
        model_path: Path to the model.
        input_size: Expected square input side.
        cpu_fallback: If ``True`` (``auto``) and the ONNX Runtime CUDA backend cannot be activated, log a WARNING and
            build the (faster on CPU, see #73) ``cv2.dnn`` CPU backend instead of failing.

    Returns:
        The backend and the plan it actually implements.

    Raises:
        EdgeModelError: If the backend cannot be created (and no fallback applies).
    """
    if plan.engine == "onnxruntime":
        try:
            return OnnxRuntimeBackend(model_path, plan.accelerator, input_size=input_size), plan
        except EdgeModelError as e:
            if not cpu_fallback:
                raise EdgeModelError(f"EDGE__ACCELERATOR=cuda requested but unusable: {e}") from e
            logger.warning("ONNX Runtime CUDA could not be activated (%s); falling back to cv2.dnn on the CPU.", e)
            plan = BackendPlan("cv2dnn", Accelerator.CPU)
    return CvDnnBackend(model_path, plan.accelerator, input_size=input_size), plan


class EdgeInferenceDetector:
    """Confirms fire with a local image classifier (fire / no-fire) running fully offline.

    ``detect()`` never raises (an exception would abort the detection loop): failures are logged at ERROR,
    treated as "no fire" and counted; after ``CONSECUTIVE_ERRORS_BEFORE_CRITICAL`` consecutive failures a
    CRITICAL message is logged once.

    With ``EDGE__ACCELERATOR=auto`` the ONNX Runtime CUDA backend is preferred; if CUDA is listed but does not
    activate, the detector falls back to ``cv2.dnn`` on the CPU (WARNING). An explicit ``cuda`` fails with
    ``EdgeModelError`` instead.
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
        self._lock = threading.Lock()  # cv2.dnn Net is not thread-safe; a cancelled to_thread keeps running
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
            cuda_available=onnxruntime_has_cuda() if settings.accelerator in (Accelerator.AUTO, Accelerator.CUDA) else False,
            opencl_available=opencl_available() if settings.accelerator == Accelerator.OPENCL else False,
        )
        self._backend, plan = _create_backend(
            plan, model_path, self.input_size, cpu_fallback=settings.accelerator == Accelerator.AUTO
        )
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

    async def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Analyzes a frame for fire with the local model.

        Args:
            frame: The input video frame. Only BGR ``np.ndarray`` frames are accepted; GPU/UMat frames must be
                converted beforehand (``HybridFireDetector`` does this).

        Returns:
            A tuple containing:
            - A boolean indicating if fire was detected (``False`` on any error).
            - The original, unmodified frame.
        """
        try:
            probability = await asyncio.to_thread(self._infer, frame)
        except Exception:
            logger.exception("Edge inference failed.")
            self._record_error()
            return False, frame

        if self._consecutive_errors:
            logger.info("Edge inference recovered after %d consecutive errors.", self._consecutive_errors)
            self._consecutive_errors = 0

        is_fire = probability >= self.threshold
        logger.debug("Edge result: P(fire)=%.3f threshold=%.2f is_fire=%s", probability, self.threshold, is_fire)
        return is_fire, frame
