"""Tests for the GeminiFireDetector class."""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from pydantic import SecretStr
import pytest

from is_goat_burning.config import settings
from is_goat_burning.fire_detection.gemini_detector import FireDetectionResponse
from is_goat_burning.fire_detection.gemini_detector import GeminiFireDetector


@pytest.fixture
def mock_settings():
    """Mocks the application settings to enable Gemini strategy."""
    settings.gemini.api_key = SecretStr("fake-api-key")
    settings.gemini.model = "gemini-2.5-flash"
    settings.gemini.prompt = "Test prompt"
    return settings


@pytest.fixture
def mock_genai_client():
    """Mocks the google.genai.Client."""
    with patch("is_goat_burning.fire_detection.gemini_detector.genai.Client") as mock:
        yield mock


def test_init_raises_error_without_api_key():
    """Verifies that initialization fails if API key is missing."""
    settings.gemini.api_key = None
    with pytest.raises(ValueError, match="GEMINI__API_KEY must be set"):
        GeminiFireDetector()


def test_detect_fire_positive(mock_settings, mock_genai_client):  # noqa: ARG001
    """Verifies positive fire detection."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.parsed = FireDetectionResponse(is_fire=True, confidence=0.95)

    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.models.generate_content.return_value = mock_response

    detector = GeminiFireDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    is_fire, returned_frame = detector.detect(frame)

    assert is_fire is True
    assert returned_frame is frame
    mock_client_instance.models.generate_content.assert_called_once()


def test_detect_fire_negative(mock_settings, mock_genai_client):  # noqa: ARG001
    """Verifies negative fire detection."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.parsed = FireDetectionResponse(is_fire=False, confidence=0.1)

    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.models.generate_content.return_value = mock_response

    detector = GeminiFireDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    is_fire, returned_frame = detector.detect(frame)

    assert is_fire is False
    assert returned_frame is frame


def test_detect_api_failure(mock_settings, mock_genai_client):  # noqa: ARG001
    """Verifies that API failure returns False (safe fallback)."""
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.models.generate_content.side_effect = Exception("API Error")

    detector = GeminiFireDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    is_fire, returned_frame = detector.detect(frame)

    assert is_fire is False
    assert returned_frame is frame
