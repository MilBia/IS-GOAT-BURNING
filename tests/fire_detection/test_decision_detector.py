"""Tests for the DecisionFireDetector class and its factory wiring."""

import asyncio
from collections.abc import Generator
import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.genai import errors as genai_errors
import numpy as np
from pydantic import SecretStr
import pytest

from is_goat_burning.config import settings
from is_goat_burning.fire_detection.decision_detector import CONSECUTIVE_ERRORS_BEFORE_CRITICAL
from is_goat_burning.fire_detection.decision_detector import DecisionFireDetector
from is_goat_burning.fire_detection.detectors import CPUFireDetector
from is_goat_burning.fire_detection.detectors import HybridFireDetector
from is_goat_burning.fire_detection.detectors import create_fire_detector

LOWER_HSV = np.array([18, 50, 50], dtype="uint8")
UPPER_HSV = np.array([35, 255, 255], dtype="uint8")
TEST_MARGIN = 100
LOGGER_NAME = "DecisionFireDetector"


@pytest.fixture
def mock_genai_client(monkeypatch: pytest.MonkeyPatch) -> Generator[MagicMock]:
    """Mocks google.genai.Client and sets a fake API key."""
    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("fake-key"))
    with patch("is_goat_burning.fire_detection.decision_detector.genai.Client") as mock:
        mock.return_value.aio.decisions.create = AsyncMock()
        yield mock


@pytest.fixture
def frame() -> np.ndarray:
    """Creates a sample video frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _response(probability: float) -> MagicMock:
    """Builds a mock Decisions API response with the given P(fire)."""
    response = MagicMock()
    response.answers = {"fire": MagicMock(noul=probability, confidence=max(probability, 1 - probability))}
    return response


def test_init_raises_error_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies initialization fails if the API key is missing."""
    monkeypatch.setattr(settings.gemini, "api_key", None)
    with pytest.raises(ValueError, match="GEMINI__API_KEY must be set"):
        DecisionFireDetector()


def test_init_uses_v1beta_api_version(mock_genai_client: MagicMock) -> None:
    """Verifies the client is created for the v1beta API (the only one serving the Decisions API)."""
    DecisionFireDetector()
    http_options = mock_genai_client.call_args.kwargs["http_options"]
    assert http_options.api_version == "v1beta"


@pytest.mark.asyncio
@pytest.mark.parametrize(("probability", "expected"), [(0.914, True), (0.5, True), (0.499, False), (0.086, False)])
async def test_detect_applies_threshold(
    mock_genai_client: MagicMock, frame: np.ndarray, probability: float, expected: bool
) -> None:
    """Verifies P(fire) is compared against the configured threshold (inclusive)."""
    mock_genai_client.return_value.aio.decisions.create.return_value = _response(probability)

    is_fire, returned_frame = await DecisionFireDetector().detect(frame)

    assert is_fire is expected
    assert returned_frame is frame


@pytest.mark.asyncio
async def test_detect_sends_image_and_noul_question(mock_genai_client: MagicMock, frame: np.ndarray) -> None:
    """Verifies the request carries the model, a base64 JPEG image block and a noul question."""
    create = mock_genai_client.return_value.aio.decisions.create
    create.return_value = _response(0.9)

    await DecisionFireDetector().detect(frame)

    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == settings.decision.model
    assert kwargs["input"][0]["type"] == "image"
    assert kwargs["input"][0]["mime_type"] == "image/jpeg"
    assert kwargs["input"][0]["data"]
    assert kwargs["questions"] == {"fire": {"type": "noul", "instructions": settings.decision.instructions}}


@pytest.mark.asyncio
async def test_custom_threshold_is_respected(
    mock_genai_client: MagicMock, frame: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies a stricter threshold turns a borderline probability into 'no fire'."""
    monkeypatch.setattr(settings.decision, "threshold", 0.95)
    mock_genai_client.return_value.aio.decisions.create.return_value = _response(0.914)

    is_fire, _ = await DecisionFireDetector().detect(frame)

    assert is_fire is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        genai_errors.ServerError(503, {"error": {"message": "high demand", "status": "UNAVAILABLE"}}),
        genai_errors.ClientError(404, {"error": {"message": "not found", "status": "NOT_FOUND"}}),
        RuntimeError("boom"),
    ],
)
async def test_detect_swallows_errors_but_logs_error(
    mock_genai_client: MagicMock, frame: np.ndarray, error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies API/unexpected errors return 'no fire' (never raise) and are logged at ERROR."""
    mock_genai_client.return_value.aio.decisions.create.side_effect = error
    logging.getLogger(LOGGER_NAME).propagate = True

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        is_fire, returned_frame = await DecisionFireDetector().detect(frame)

    assert is_fire is False
    assert returned_frame is frame
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


@pytest.mark.asyncio
async def test_detect_times_out(
    mock_genai_client: MagicMock, frame: np.ndarray, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies a request exceeding the timeout is treated as an error ('no fire')."""
    monkeypatch.setattr(settings.decision, "timeout_seconds", 0.01)

    async def _hang(**_kwargs: object) -> None:
        await asyncio.sleep(5)

    mock_genai_client.return_value.aio.decisions.create.side_effect = _hang
    logging.getLogger(LOGGER_NAME).propagate = True

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        is_fire, _ = await DecisionFireDetector().detect(frame)

    assert is_fire is False
    assert "timed out" in caplog.text


@pytest.mark.asyncio
async def test_critical_logged_once_after_consecutive_errors_and_recovers(
    mock_genai_client: MagicMock, frame: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies CRITICAL is logged exactly once per outage and the counter resets on success."""
    create = mock_genai_client.return_value.aio.decisions.create
    detector = DecisionFireDetector()
    logging.getLogger(LOGGER_NAME).propagate = True

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        create.side_effect = RuntimeError("down")
        for _ in range(CONSECUTIVE_ERRORS_BEFORE_CRITICAL + 3):
            await detector.detect(frame)
        criticals = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(criticals) == 1

        create.side_effect = None
        create.return_value = _response(0.9)
        is_fire, _ = await detector.detect(frame)

    assert is_fire is True
    assert "recovered" in caplog.text


@pytest.mark.parametrize("strategy", ["gemini_decision", "hybrid_decision"])
def test_factory_builds_decision_detectors(mock_genai_client: MagicMock, strategy: str) -> None:  # noqa: ARG001
    """Verifies the factory returns the right detector for each Decisions strategy."""
    detector = create_fire_detector(TEST_MARGIN, LOWER_HSV, UPPER_HSV, use_open_cl=False, use_cuda=False, strategy=strategy)

    if strategy == "gemini_decision":
        assert isinstance(detector, DecisionFireDetector)
    else:
        assert isinstance(detector, HybridFireDetector)
        assert isinstance(detector.local_detector, CPUFireDetector)
        assert isinstance(detector.verifier, DecisionFireDetector)


# --- verifier contract (#169): errors are distinguishable from "no fire" ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [genai_errors.ServerError(503, {"error": {"message": "x", "status": "UNAVAILABLE"}}), RuntimeError("boom")],
)
async def test_verify_raises_verifier_error(mock_genai_client: MagicMock, frame: np.ndarray, error: Exception) -> None:
    """``verify`` reports API failures as VerifierError (counted like ``detect`` failures)."""
    from is_goat_burning.fire_detection.verifier_health import VerifierError

    mock_genai_client.return_value.aio.decisions.create.side_effect = error
    detector = DecisionFireDetector()

    with pytest.raises(VerifierError):
        await detector.verify(frame)
    assert detector._consecutive_errors == 1


@pytest.mark.asyncio
async def test_verify_returns_thresholded_verdict(mock_genai_client: MagicMock, frame: np.ndarray) -> None:
    """A successful call returns a plain bool verdict."""
    mock_genai_client.return_value.aio.decisions.create.return_value = _response(0.2)
    assert await DecisionFireDetector().verify(frame) is False


# --- standalone gemini_decision publishes verifier health (#171) ---


@pytest.mark.asyncio
async def test_standalone_detect_publishes_outage_and_recovery_once(mock_genai_client: MagicMock, frame: np.ndarray) -> None:
    """``detect()`` publishes the outage at the CRITICAL threshold and the recovery once; ``verify()`` publishes nothing."""
    from is_goat_burning.fire_detection.verifier_health import STANDALONE_POLICY
    from is_goat_burning.fire_detection.verifier_health import VerifierError
    from is_goat_burning.fire_detection.verifier_health import add_verifier_status_listener
    from is_goat_burning.fire_detection.verifier_health import remove_verifier_status_listener

    create = mock_genai_client.return_value.aio.decisions.create
    detector = DecisionFireDetector()
    statuses: list = []
    listener = statuses.append
    add_verifier_status_listener(listener)
    try:
        create.side_effect = RuntimeError("down")
        for _ in range(CONSECUTIVE_ERRORS_BEFORE_CRITICAL + 2):
            with pytest.raises(VerifierError):
                await detector.verify(frame)  # the hybrid path: HybridFireDetector publishes, not the verifier
        assert statuses == []

        detector._consecutive_errors = 0
        for _ in range(CONSECUTIVE_ERRORS_BEFORE_CRITICAL - 1):
            await detector.detect(frame)
        assert statuses == []
        for _ in range(3):
            await detector.detect(frame)

        create.side_effect = None
        create.return_value = _response(0.1)
        await detector.detect(frame)
        await detector.detect(frame)
    finally:
        remove_verifier_status_listener(listener)

    assert [s.available for s in statuses] == [False, True]
    assert {s.verifier for s in statuses} == {"DecisionFireDetector"}
    assert {s.policy for s in statuses} == {STANDALONE_POLICY}
    assert f"{CONSECUTIVE_ERRORS_BEFORE_CRITICAL} consecutive errors" in statuses[0].reason
