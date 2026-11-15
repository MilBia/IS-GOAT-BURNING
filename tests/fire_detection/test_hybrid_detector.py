"""Tests for the HybridFireDetector class.

This module tests the two-stage hybrid detection strategy that uses a local
CV detector as a gatekeeper and a pluggable verifier (Gemini, Decisions API, edge model) for verification.
"""

from collections.abc import Generator
import logging
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import cv2
import numpy as np
from pydantic import SecretStr
import pytest

from is_goat_burning.config import EdgeSettings
from is_goat_burning.config import settings
from is_goat_burning.fire_detection.detectors import CPUFireDetector
from is_goat_burning.fire_detection.detectors import HybridFireDetector
from is_goat_burning.fire_detection.detectors import create_fire_detector
from is_goat_burning.fire_detection.edge_detector import EdgeModelError
from is_goat_burning.fire_detection.gemini_detector import GeminiFireDetector

# --- Test Constants ---

LOWER_HSV = np.array([18, 50, 50], dtype="uint8")
UPPER_HSV = np.array([35, 255, 255], dtype="uint8")
TEST_FRAME_HEIGHT = 100
TEST_FRAME_WIDTH = 100
TEST_FRAME_CHANNELS = 3
TEST_MARGIN = 100


# --- Fixtures ---


@pytest.fixture
def mock_local_detector() -> MagicMock:
    """Creates a mock local detector."""
    mock = MagicMock(spec=CPUFireDetector)
    mock.detect = AsyncMock()
    return mock


@pytest.fixture
def mock_verifier() -> MagicMock:
    """Creates a mock verifier (shaped like the Gemini detector)."""
    mock = MagicMock(spec=GeminiFireDetector)
    mock.detect = AsyncMock()
    return mock


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Creates a sample video frame for testing."""
    return np.zeros((TEST_FRAME_HEIGHT, TEST_FRAME_WIDTH, TEST_FRAME_CHANNELS), dtype=np.uint8)


# --- Test Cases ---


@pytest.mark.asyncio
async def test_hybrid_case_a_local_false_no_api_call(
    mock_local_detector: MagicMock,
    mock_verifier: MagicMock,
    sample_frame: np.ndarray,
) -> None:
    """Case A: Local(False) -> verifier not called -> Result(False)."""
    # Setup: local detector returns False
    mock_local_detector.detect.return_value = (False, sample_frame)

    detector = HybridFireDetector(mock_local_detector, mock_verifier)
    is_fire, returned_frame = await detector.detect(sample_frame)

    # Verify
    assert is_fire is False
    assert returned_frame is sample_frame
    mock_local_detector.detect.assert_called_once_with(sample_frame)
    mock_verifier.detect.assert_not_called()  # Critical: verifier NOT called


@pytest.mark.asyncio
async def test_hybrid_case_b_local_true_gemini_false(
    mock_local_detector: MagicMock,
    mock_verifier: MagicMock,
    sample_frame: np.ndarray,
) -> None:
    """Case B: Local(True) -> verifier(False) -> Result(False) [False positive filtered]."""
    # Setup: local detector returns True, verifier returns False
    mock_local_detector.detect.return_value = (True, sample_frame)
    mock_verifier.detect.return_value = (False, sample_frame)

    detector = HybridFireDetector(mock_local_detector, mock_verifier)
    is_fire, returned_frame = await detector.detect(sample_frame)

    # Verify
    assert is_fire is False  # False positive filtered by the verifier
    assert returned_frame is sample_frame
    mock_local_detector.detect.assert_called_once_with(sample_frame)
    mock_verifier.detect.assert_called_once_with(sample_frame)


@pytest.mark.asyncio
async def test_hybrid_case_c_local_true_gemini_true(
    mock_local_detector: MagicMock,
    mock_verifier: MagicMock,
    sample_frame: np.ndarray,
) -> None:
    """Case C: Local(True) -> verifier(True) -> Result(True) [Confirmed fire]."""
    # Setup: both detectors return True
    mock_local_detector.detect.return_value = (True, sample_frame)
    mock_verifier.detect.return_value = (True, sample_frame)

    detector = HybridFireDetector(mock_local_detector, mock_verifier)
    is_fire, returned_frame = await detector.detect(sample_frame)

    # Verify
    assert is_fire is True  # Fire confirmed by both
    assert returned_frame is sample_frame
    mock_local_detector.detect.assert_called_once_with(sample_frame)
    mock_verifier.detect.assert_called_once_with(sample_frame)


@pytest.fixture
def mock_settings_hybrid() -> Generator[MagicMock]:
    """Mocks settings to use hybrid strategy with a fake API key."""
    with patch("is_goat_burning.fire_detection.detectors.settings") as mock:
        mock.detection_strategy = "hybrid"
        mock.open_cl = False
        mock.cuda = False
        yield mock


@pytest.fixture
def mock_genai_client() -> Generator[MagicMock]:
    """Mocks the google.genai.Client."""
    with patch("is_goat_burning.fire_detection.gemini_detector.genai.Client") as mock:
        mock.return_value.aio.models.generate_content = AsyncMock()
        yield mock


def test_create_fire_detector_factory_returns_hybrid_when_requested(
    mock_settings_hybrid: MagicMock,  # noqa: ARG001
    mock_genai_client: MagicMock,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the factory returns a HybridFireDetector when strategy is 'hybrid'."""
    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("fake-key"))

    detector = create_fire_detector(TEST_MARGIN, LOWER_HSV, UPPER_HSV, use_open_cl=False, use_cuda=False, strategy="hybrid")

    assert isinstance(detector, HybridFireDetector)
    assert isinstance(detector.local_detector, CPUFireDetector)
    assert isinstance(detector.verifier, GeminiFireDetector)


@pytest.mark.parametrize("strategy", ["hybrid_cloud", "hybrid"])
def test_create_fire_detector_factory_returns_hybrid_for_cloud_strategies(
    strategy: str,
    mock_genai_client: MagicMock,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the factory returns a HybridFireDetector for `hybrid_cloud` and legacy `hybrid`."""
    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("fake-key"))

    detector = create_fire_detector(TEST_MARGIN, LOWER_HSV, UPPER_HSV, use_open_cl=False, use_cuda=False, strategy=strategy)

    assert isinstance(detector, HybridFireDetector)
    assert isinstance(detector.local_detector, CPUFireDetector)
    assert isinstance(detector.verifier, GeminiFireDetector)


class _FakeVerifier:
    """A minimal object satisfying the FireVerifier protocol."""

    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.calls = 0

    async def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        self.calls += 1
        return self.verdict, frame


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", [True, False])
async def test_hybrid_accepts_any_protocol_verifier(
    mock_local_detector: MagicMock, sample_frame: np.ndarray, verdict: bool
) -> None:
    """Any FireVerifier-compatible object decides the final result."""
    mock_local_detector.detect.return_value = (True, sample_frame)
    verifier = _FakeVerifier(verdict)

    detector = HybridFireDetector(mock_local_detector, verifier)
    result, _ = await detector.detect(sample_frame)

    assert result is verdict
    assert verifier.calls == 1


@pytest.mark.asyncio
async def test_hybrid_fake_verifier_not_called_when_local_negative(
    mock_local_detector: MagicMock, sample_frame: np.ndarray
) -> None:
    """The verifier is never called when the local gatekeeper says no."""
    mock_local_detector.detect.return_value = (False, sample_frame)
    verifier = _FakeVerifier(True)

    detector = HybridFireDetector(mock_local_detector, verifier)
    result, _ = await detector.detect(sample_frame)

    assert result is False
    assert verifier.calls == 0


def test_hybrid_logs_verifier_class_name(mock_local_detector: MagicMock) -> None:
    """Construction logs the active verifier backend by class name."""
    with patch("is_goat_burning.fire_detection.detectors.get_logger") as get_logger:
        HybridFireDetector(mock_local_detector, _FakeVerifier(True))
    get_logger.return_value.info.assert_called_once_with("Initialized hybrid detector with verifier _FakeVerifier")


@pytest.mark.asyncio
async def test_hybrid_logs_rejection_with_verifier_class_name(mock_local_detector: MagicMock) -> None:
    """A rejected detection is logged with the verifier's class name."""
    mock_local_detector.detect.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
    with patch("is_goat_burning.fire_detection.detectors.get_logger") as get_logger:
        detector = HybridFireDetector(mock_local_detector, _FakeVerifier(False))
        get_logger.return_value.info.reset_mock()
        is_fire, _ = await detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert is_fire is False
    message = get_logger.return_value.info.call_args.args[0]
    assert "_FakeVerifier" in message
    assert "false positive filtered" in message


# --- hybrid_edge factory wiring ---

EDGE_STRATEGY = "hybrid_edge"
FIRE_HSV = (25, 200, 200)


def _fire_frame(value: int) -> np.ndarray:
    """Builds a uniformly fire-coloured BGR frame; differing ``value`` between frames simulates motion."""
    hsv = np.zeros((TEST_FRAME_HEIGHT, TEST_FRAME_WIDTH, 3), dtype=np.uint8)
    hsv[:] = (FIRE_HSV[0], FIRE_HSV[1], value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _edge_factory(fire_probability: float) -> MagicMock:
    """Returns a stand-in for ``EdgeInferenceDetector`` built on a fake backend with a fixed P(fire).

    The returned mock records its call arguments and exposes the fake backend as ``.backend``.
    """
    from is_goat_burning.fire_detection.edge_detector import EdgeInferenceDetector

    backend = MagicMock()
    backend.predict.return_value = np.array([[np.log(1 - fire_probability), np.log(fire_probability)]], dtype=np.float32)

    def build(_settings: object) -> EdgeInferenceDetector:
        return EdgeInferenceDetector(EdgeSettings(confidence_threshold=0.5, input_size=8), backend=backend)

    factory = MagicMock(side_effect=build)
    factory.backend = backend
    return factory


def _create_edge():
    return create_fire_detector(TEST_MARGIN, LOWER_HSV, UPPER_HSV, use_open_cl=False, use_cuda=False, strategy=EDGE_STRATEGY)


def _critical_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.CRITICAL]


def test_factory_builds_hybrid_edge() -> None:
    """The factory wraps the local detector with an EdgeInferenceDetector built from settings.edge."""
    from is_goat_burning.fire_detection.edge_detector import EdgeInferenceDetector

    factory = _edge_factory(0.9)
    with patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", factory):
        detector = _create_edge()

    assert isinstance(detector, HybridFireDetector)
    assert isinstance(detector.local_detector, CPUFireDetector)
    assert isinstance(detector.verifier, EdgeInferenceDetector)
    factory.assert_called_once_with(settings.edge)


@pytest.mark.asyncio
@pytest.mark.parametrize(("probability", "expected"), [(0.9, True), (0.1, False)])
async def test_hybrid_edge_detect_end_to_end(probability: float, expected: bool) -> None:
    """Local CV fires on a moving fire-coloured pair; the edge verifier confirms or rejects it."""
    factory = _edge_factory(probability)
    with patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", factory):
        detector = _create_edge()

    await detector.detect(_fire_frame(150))  # first frame: motion baseline only
    factory.backend.predict.assert_not_called()
    is_fire, _ = await detector.detect(_fire_frame(250))

    assert is_fire is expected
    factory.backend.predict.assert_called_once()  # verifier consulted in both the fire and no-fire cases


@pytest.mark.parametrize(
    "error",
    [EdgeModelError("model missing"), RuntimeError("onnxruntime exploded")],
    ids=["edge_model_error", "unexpected_error"],
)
def test_hybrid_edge_falls_back_to_local_detector_on_load_error(error: Exception, caplog: pytest.LogCaptureFixture) -> None:
    """Any load failure falls back to the plain local detector and logs CRITICAL."""
    with patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", side_effect=error):
        detector = _create_edge()

    assert isinstance(detector, CPUFireDetector)
    messages = _critical_messages(caplog)
    assert len(messages) == 1
    assert "hybrid_edge verification is unavailable" in messages[0]
    assert str(error) in messages[0]
    critical = next(r for r in caplog.records if r.levelno == logging.CRITICAL)
    assert bool(critical.exc_info) == (not isinstance(error, EdgeModelError))  # traceback only for unexpected errors


def test_hybrid_edge_base_exception_propagates() -> None:
    """KeyboardInterrupt/CancelledError must not be swallowed by the fallback."""
    with (
        patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _create_edge()


def test_hybrid_edge_fallback_with_corrupt_model(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A real model file with garbage bytes triggers the fallback without patching the detector."""
    model = tmp_path / "corrupt.onnx"
    model.write_bytes(b"this is not an onnx model")
    with patch.object(settings, "edge", EdgeSettings(model_path=str(model), accelerator="cpu")):
        detector = _create_edge()

    assert isinstance(detector, CPUFireDetector)
    assert len(_critical_messages(caplog)) == 1


def test_hybrid_cloud_strategy_unchanged() -> None:
    """hybrid_cloud still pairs the local detector with Gemini."""
    with patch("is_goat_burning.fire_detection.gemini_detector.GeminiFireDetector") as gemini:
        detector = create_fire_detector(
            TEST_MARGIN, LOWER_HSV, UPPER_HSV, use_open_cl=False, use_cuda=False, strategy="hybrid_cloud"
        )

    assert isinstance(detector, HybridFireDetector)
    assert isinstance(detector.local_detector, CPUFireDetector)
    assert detector.verifier is gemini.return_value
