"""Tests for the GeminiFireDetector class."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from pydantic import SecretStr
import pytest

from is_goat_burning.config import settings
from is_goat_burning.fire_detection.gemini_detector import FireDetectionResponse
from is_goat_burning.fire_detection.gemini_detector import GeminiFireDetector


@pytest.fixture
def mock_genai_client():
    """Mocks the google.genai.Client."""
    with patch("is_goat_burning.fire_detection.gemini_detector.genai.Client") as mock:
        # Mock the async generate_content method
        mock.return_value.aio.models.generate_content = AsyncMock()
        yield mock


def test_init_raises_error_without_api_key(monkeypatch):  # noqa: ARG001
    """Verifies that initialization fails if API key is missing."""
    monkeypatch.setattr(settings.gemini, "api_key", None)
    with pytest.raises(ValueError, match="GEMINI__API_KEY must be set"):
        GeminiFireDetector()


@pytest.mark.asyncio
async def test_detect_fire_positive(mock_genai_client, monkeypatch) -> None:
    """Verifies positive fire detection."""
    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("fake-key"))

    # Setup mock response
    mock_response = MagicMock()
    mock_response.parsed = FireDetectionResponse(is_fire=True, confidence=0.95)

    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.aio.models.generate_content.return_value = mock_response

    detector = GeminiFireDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    is_fire, returned_frame = await detector.detect(frame)

    assert is_fire is True
    assert returned_frame is frame
    mock_client_instance.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_detect_fire_negative(mock_genai_client, monkeypatch) -> None:
    """Verifies negative fire detection."""
    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("fake-key"))

    # Setup mock response
    mock_response = MagicMock()
    mock_response.parsed = FireDetectionResponse(is_fire=False, confidence=0.1)

    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.aio.models.generate_content.return_value = mock_response

    detector = GeminiFireDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    is_fire, returned_frame = await detector.detect(frame)

    assert is_fire is False
    assert returned_frame is frame


@pytest.mark.asyncio
async def test_detect_api_failure(mock_genai_client, monkeypatch) -> None:
    """Verifies that API failure returns False (safe fallback)."""
    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("fake-key"))

    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.aio.models.generate_content.side_effect = Exception("API Error")

    detector = GeminiFireDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    is_fire, returned_frame = await detector.detect(frame)

    assert is_fire is False
    assert returned_frame is frame


# --- verifier contract (#169): errors are distinguishable from "no fire" ---


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [Exception("API Error"), ValueError("bad frame")])
async def test_verify_raises_verifier_error_on_failure(mock_genai_client, monkeypatch, error) -> None:
    """``verify`` reports a failed call as VerifierError, not as "no fire"."""
    from is_goat_burning.fire_detection.verifier_health import VerifierError

    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("fake-key"))
    mock_genai_client.return_value.aio.models.generate_content.side_effect = error

    with pytest.raises(VerifierError):
        await GeminiFireDetector().verify(np.zeros((10, 10, 3), dtype=np.uint8))


@pytest.mark.asyncio
async def test_verify_raises_when_response_is_not_parsed(mock_genai_client, monkeypatch) -> None:
    """An unparseable answer is an error, not a "no fire" verdict."""
    from is_goat_burning.fire_detection.verifier_health import VerifierError

    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("fake-key"))
    mock_genai_client.return_value.aio.models.generate_content.return_value = MagicMock(parsed=None)

    with pytest.raises(VerifierError):
        await GeminiFireDetector().verify(np.zeros((10, 10, 3), dtype=np.uint8))


@pytest.mark.asyncio
async def test_verify_returns_the_verdict(mock_genai_client, monkeypatch) -> None:
    """A parsed answer is returned as a plain bool."""
    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("fake-key"))
    mock_genai_client.return_value.aio.models.generate_content.return_value = MagicMock(
        parsed=FireDetectionResponse(is_fire=True, confidence=0.9)
    )

    assert await GeminiFireDetector().verify(np.zeros((10, 10, 3), dtype=np.uint8)) is True


@pytest.mark.asyncio
async def test_call_times_out_after_configured_seconds(mock_genai_client, monkeypatch) -> None:
    """A hanging request is cut off at GEMINI__TIMEOUT_SECONDS and counts as a verifier error."""
    import asyncio

    from is_goat_burning.fire_detection.verifier_health import VerifierError

    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("fake-key"))
    monkeypatch.setattr(settings.gemini, "timeout_seconds", 0.01)

    async def _hang(**_kwargs: object) -> None:
        await asyncio.sleep(5)

    mock_genai_client.return_value.aio.models.generate_content.side_effect = _hang
    detector = GeminiFireDetector()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(VerifierError, match="timed out"):
        await detector.verify(frame)
    assert await detector.detect(frame) == (False, frame)  # standalone use still never raises
