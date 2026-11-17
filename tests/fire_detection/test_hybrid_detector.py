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
from is_goat_burning.fire_detection.detectors import clear_edge_verifier_cache
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
    mock.verify = AsyncMock()
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
    mock_verifier.verify.assert_not_called()  # Critical: verifier NOT called


@pytest.mark.asyncio
async def test_hybrid_case_b_local_true_gemini_false(
    mock_local_detector: MagicMock,
    mock_verifier: MagicMock,
    sample_frame: np.ndarray,
) -> None:
    """Case B: Local(True) -> verifier(False) -> Result(False) [False positive filtered]."""
    # Setup: local detector returns True, verifier returns False
    mock_local_detector.detect.return_value = (True, sample_frame)
    mock_verifier.verify.return_value = False

    detector = HybridFireDetector(mock_local_detector, mock_verifier)
    is_fire, returned_frame = await detector.detect(sample_frame)

    # Verify
    assert is_fire is False  # False positive filtered by the verifier
    assert returned_frame is sample_frame
    mock_local_detector.detect.assert_called_once_with(sample_frame)
    mock_verifier.verify.assert_called_once_with(sample_frame)


@pytest.mark.asyncio
async def test_hybrid_case_c_local_true_gemini_true(
    mock_local_detector: MagicMock,
    mock_verifier: MagicMock,
    sample_frame: np.ndarray,
) -> None:
    """Case C: Local(True) -> verifier(True) -> Result(True) [Confirmed fire]."""
    # Setup: both detectors return True
    mock_local_detector.detect.return_value = (True, sample_frame)
    mock_verifier.verify.return_value = True

    detector = HybridFireDetector(mock_local_detector, mock_verifier)
    is_fire, returned_frame = await detector.detect(sample_frame)

    # Verify
    assert is_fire is True  # Fire confirmed by both
    assert returned_frame is sample_frame
    mock_local_detector.detect.assert_called_once_with(sample_frame)
    mock_verifier.verify.assert_called_once_with(sample_frame)


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

    async def verify(self, frame: np.ndarray) -> bool:  # noqa: ARG002
        self.calls += 1
        return self.verdict


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


# --- edge verifier caching across reconnects (#119) ---


def test_edge_verifier_is_cached_across_factory_calls() -> None:
    """Two consecutive factory calls build the verifier once but return fresh detectors sharing it."""
    factory = _edge_factory(0.9)
    with patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", factory):
        first = _create_edge()
        second = _create_edge()

    factory.assert_called_once()
    assert first is not second
    assert first.local_detector is not second.local_detector  # motion baseline is never shared
    assert first.verifier is second.verifier


@pytest.mark.parametrize(
    "changed",
    [{"confidence_threshold": 0.9}, {"input_size": 16}, {"accelerator": "opencl"}, {"model_path": "OTHER"}],
    ids=["confidence_threshold", "input_size", "accelerator", "model_path"],
)
def test_edge_verifier_rebuilt_when_settings_change(changed: dict, tmp_path: Path) -> None:
    """Every field of the cache key invalidates the cached verifier when it changes."""
    model_a = tmp_path / "a.onnx"
    model_a.write_bytes(b"model")
    model_b = tmp_path / "b.onnx"
    model_b.write_bytes(b"model")  # same size and content: only the path differs
    base = {"model_path": str(model_a), "accelerator": "cpu", "input_size": 8, "confidence_threshold": 0.5}
    changed = {k: (str(model_b) if v == "OTHER" else v) for k, v in changed.items()}
    factory = _edge_factory(0.9)
    with patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", factory):
        with patch.object(settings, "edge", EdgeSettings(**base)):
            _create_edge()
            _create_edge()
            assert factory.call_count == 1
        with patch.object(settings, "edge", EdgeSettings(**{**base, **changed})):
            _create_edge()

    assert factory.call_count == 2


def test_edge_verifier_rebuilt_when_model_file_changes(tmp_path: Path) -> None:
    """Touching or replacing the model file (mtime/size) invalidates the cached verifier."""
    import os

    model = tmp_path / "m.onnx"
    model.write_bytes(b"v1")
    factory = _edge_factory(0.9)
    with (
        patch.object(settings, "edge", EdgeSettings(model_path=str(model), accelerator="cpu")),
        patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", factory),
    ):
        _create_edge()
        _create_edge()
        assert factory.call_count == 1

        stat = model.stat()
        os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))  # touch: same size, new mtime
        _create_edge()
        assert factory.call_count == 2

        model.write_bytes(b"version two")  # replaced: new size
        _create_edge()
        assert factory.call_count == 3


@pytest.mark.asyncio
async def test_unhealthy_edge_verifier_is_rebuilt(caplog: pytest.LogCaptureFixture) -> None:
    """A verifier that failed inference is replaced on the next call; a healthy one is reused."""
    caplog.set_level(logging.WARNING, logger="create_fire_detector")
    factory = _edge_factory(0.9)
    with patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", factory):
        first = _create_edge()
        assert _create_edge().verifier is first.verifier  # healthy: reused
        factory.backend.predict.side_effect = RuntimeError("CUDA context lost")
        await first.verifier.detect(_fire_frame(150))
        assert first.verifier.needs_rebuild
        factory.backend.predict.side_effect = None
        third = _create_edge()

    assert factory.call_count == 2
    assert third.verifier is not first.verifier
    assert not third.verifier.needs_rebuild
    assert any("Rebuilding the edge verifier" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_failed_rebuild_of_unhealthy_verifier_falls_back_uncached(caplog: pytest.LogCaptureFixture) -> None:
    """If rebuilding after errors fails, the CRITICAL fallback applies and nothing stays cached."""
    factory = _edge_factory(0.9)
    with patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", factory):
        first = _create_edge()
        factory.backend.predict.side_effect = RuntimeError("boom")
        await first.verifier.detect(_fire_frame(150))
        factory.side_effect = EdgeModelError("cannot rebuild")
        fallback = _create_edge()
        factory.side_effect = None
        factory.backend.predict.side_effect = None
        again = _create_edge()

    assert isinstance(fallback, CPUFireDetector)
    assert len(_critical_messages(caplog)) == 1
    assert isinstance(again, HybridFireDetector)
    assert again.verifier is not first.verifier


def test_clear_edge_verifier_cache_forces_rebuild() -> None:
    """clear_edge_verifier_cache() makes the next call rebuild the verifier."""
    factory = _edge_factory(0.9)
    with patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", factory):
        _create_edge()
        clear_edge_verifier_cache()
        _create_edge()

    assert factory.call_count == 2


def test_failed_edge_build_is_not_cached(caplog: pytest.LogCaptureFixture) -> None:
    """A failing build is retried (and re-logged CRITICAL) on each call; the first success is then cached."""
    from is_goat_burning.fire_detection.edge_detector import EdgeInferenceDetector

    backend = MagicMock()
    good = EdgeInferenceDetector(EdgeSettings(input_size=8), backend=backend)
    factory = MagicMock(side_effect=[EdgeModelError("broken"), EdgeModelError("still broken"), good])
    with patch("is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector", factory):
        first = _create_edge()
        second = _create_edge()
        third = _create_edge()
        fourth = _create_edge()

    assert isinstance(first, CPUFireDetector)
    assert isinstance(second, CPUFireDetector)
    assert len(_critical_messages(caplog)) == 2
    assert isinstance(third, HybridFireDetector)
    assert third.verifier is good
    assert fourth.verifier is good
    assert factory.call_count == 3


def test_corrupt_model_replaced_by_valid_one_is_picked_up(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """With real files: a corrupt model falls back, then a valid replacement is loaded and reused."""
    from tests.fire_detection.test_edge_detector import _build_classifier_onnx

    model = tmp_path / "m.onnx"
    model.write_bytes(b"this is not an onnx model")
    with patch.object(settings, "edge", EdgeSettings(model_path=str(model), accelerator="cpu", input_size=8)):
        assert isinstance(_create_edge(), CPUFireDetector)
        assert isinstance(_create_edge(), CPUFireDetector)
        assert len(_critical_messages(caplog)) == 2

        model.write_bytes(_build_classifier_onnx(8))
        first = _create_edge()
        second = _create_edge()

    assert isinstance(first, HybridFireDetector)
    assert first.verifier is second.verifier
    assert len(_critical_messages(caplog)) == 2


def test_hybrid_cloud_strategy_unchanged() -> None:
    """hybrid_cloud still pairs the local detector with Gemini."""
    with patch("is_goat_burning.fire_detection.gemini_detector.GeminiFireDetector") as gemini:
        detector = create_fire_detector(
            TEST_MARGIN, LOWER_HSV, UPPER_HSV, use_open_cl=False, use_cuda=False, strategy="hybrid_cloud"
        )

    assert isinstance(detector, HybridFireDetector)
    assert isinstance(detector.local_detector, CPUFireDetector)
    assert detector.verifier is gemini.return_value


def test_input_size_mismatch_after_model_swap_falls_back_and_drops_cache(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Real files: a model replaced by one declaring another input size falls back (CRITICAL) and leaves no cache entry."""
    import os

    from is_goat_burning.fire_detection import detectors
    from tests.fire_detection.test_edge_detector import _build_classifier_onnx

    model = tmp_path / "m.onnx"
    model.write_bytes(_build_classifier_onnx(8))
    with patch.object(settings, "edge", EdgeSettings(model_path=str(model), accelerator="cpu", input_size=8)):
        assert isinstance(_create_edge(), HybridFireDetector)
        assert len(detectors._edge_verifier_cache) == 1

        model.write_bytes(_build_classifier_onnx(16))  # declares 16x16 while EDGE__INPUT_SIZE stays 8
        stat = model.stat()
        os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        detector = _create_edge()

    assert isinstance(detector, CPUFireDetector)
    messages = _critical_messages(caplog)
    assert len(messages) == 1
    assert "expects a 16x16 input" in messages[0]
    assert len(detectors._edge_verifier_cache) == 0
