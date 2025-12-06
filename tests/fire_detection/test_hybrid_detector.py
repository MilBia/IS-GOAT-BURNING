"""Tests for the HybridFireDetector class.

This module tests the two-stage hybrid detection strategy that uses a local
CV detector as a gatekeeper and the Gemini API for verification.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from pydantic import SecretStr
import pytest

from is_goat_burning.config import settings
from is_goat_burning.fire_detection.detectors import CPUFireDetector
from is_goat_burning.fire_detection.detectors import HybridFireDetector
from is_goat_burning.fire_detection.detectors import create_fire_detector
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
def mock_gemini_detector() -> MagicMock:
    """Creates a mock Gemini detector."""
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
    mock_gemini_detector: MagicMock,
    sample_frame: np.ndarray,
) -> None:
    """Case A: Local(False) -> API not called -> Result(False)."""
    # Setup: local detector returns False
    mock_local_detector.detect.return_value = (False, sample_frame)

    detector = HybridFireDetector(mock_local_detector, mock_gemini_detector)
    is_fire, returned_frame = await detector.detect(sample_frame)

    # Verify
    assert is_fire is False
    assert returned_frame is sample_frame
    mock_local_detector.detect.assert_called_once_with(sample_frame)
    mock_gemini_detector.detect.assert_not_called()  # Critical: API NOT called


@pytest.mark.asyncio
async def test_hybrid_case_b_local_true_gemini_false(
    mock_local_detector: MagicMock,
    mock_gemini_detector: MagicMock,
    sample_frame: np.ndarray,
) -> None:
    """Case B: Local(True) -> API(False) -> Result(False) [False positive filtered]."""
    # Setup: local detector returns True, Gemini returns False
    mock_local_detector.detect.return_value = (True, sample_frame)
    mock_gemini_detector.detect.return_value = (False, sample_frame)

    detector = HybridFireDetector(mock_local_detector, mock_gemini_detector)
    is_fire, returned_frame = await detector.detect(sample_frame)

    # Verify
    assert is_fire is False  # False positive filtered by Gemini
    assert returned_frame is sample_frame
    mock_local_detector.detect.assert_called_once_with(sample_frame)
    mock_gemini_detector.detect.assert_called_once_with(sample_frame)


@pytest.mark.asyncio
async def test_hybrid_case_c_local_true_gemini_true(
    mock_local_detector: MagicMock,
    mock_gemini_detector: MagicMock,
    sample_frame: np.ndarray,
) -> None:
    """Case C: Local(True) -> API(True) -> Result(True) [Confirmed fire]."""
    # Setup: both detectors return True
    mock_local_detector.detect.return_value = (True, sample_frame)
    mock_gemini_detector.detect.return_value = (True, sample_frame)

    detector = HybridFireDetector(mock_local_detector, mock_gemini_detector)
    is_fire, returned_frame = await detector.detect(sample_frame)

    # Verify
    assert is_fire is True  # Fire confirmed by both
    assert returned_frame is sample_frame
    mock_local_detector.detect.assert_called_once_with(sample_frame)
    mock_gemini_detector.detect.assert_called_once_with(sample_frame)


@pytest.fixture
def mock_settings_hybrid() -> Generator[MagicMock, None, None]:
    """Mocks settings to use hybrid strategy with a fake API key."""
    with patch("is_goat_burning.fire_detection.detectors.settings") as mock:
        mock.detection_strategy = "hybrid"
        mock.open_cl = False
        mock.cuda = False
        yield mock


@pytest.fixture
def mock_genai_client() -> Generator[MagicMock, None, None]:
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
    assert isinstance(detector.gemini_detector, GeminiFireDetector)
