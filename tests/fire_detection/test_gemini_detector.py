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
