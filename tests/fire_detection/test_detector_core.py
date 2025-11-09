"""Unit tests for the StreamFireDetector class.

These tests focus on the debouncing and state transition logic within the
detector's core, ensuring that signals are emitted correctly based on the
configured time delays.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from is_goat_burning.fire_detection.detector_core import StreamFireDetector


@pytest.fixture
def mock_settings() -> Generator[MagicMock, None, None]:
    """Fixture to mock the global settings object for debouncing."""
    with patch("is_goat_burning.fire_detection.detector_core.settings") as mock:
        mock.fire_detected_debounce_seconds = 1.0
        mock.fire_extinguished_debounce_seconds = 2.0
        yield mock


@pytest.fixture
def detector(mock_settings: MagicMock) -> StreamFireDetector:  # noqa: ARG001
    """Provides a StreamFireDetector instance with mocked dependencies."""
    mock_on_fire_action = AsyncMock()
    detector = StreamFireDetector(
        on_fire_action=mock_on_fire_action,
        video_output=False,
        checks_per_second=1.0,
    )
    detector.signal_handler = MagicMock()
    return detector


@pytest.mark.asyncio
async def test_fire_detected_debounce_prevents_signal_on_brief_detection(detector: StreamFireDetector) -> None:
    """Verify that a brief fire detection does not trigger a signal if it's shorter than the debounce time."""
    with patch("time.monotonic", side_effect=[100, 100.5]):  # 0.5s elapsed
        await detector._handle_fire_detection(fire_in_frame=True)  # Potential start
        await detector._handle_fire_detection(fire_in_frame=False)  # Fire gone before debounce ends

    detector.signal_handler.fire_detected.assert_not_called()
    assert detector.fire_is_currently_detected is False


@pytest.mark.asyncio
async def test_fire_detected_debounce_triggers_signal_after_delay(detector: StreamFireDetector) -> None:
    """Verify that a sustained fire detection triggers a signal after the debounce delay."""
    with patch("time.monotonic", side_effect=[100, 101.1]):  # 1.1s elapsed
        await detector._handle_fire_detection(fire_in_frame=True)  # Potential start
        await detector._handle_fire_detection(fire_in_frame=True)  # Debounce time exceeded

    detector.signal_handler.fire_detected.assert_called_once()
    assert detector.fire_is_currently_detected is True
    assert detector.on_fire_action.called is True


@pytest.mark.asyncio
async def test_fire_extinguished_debounce_prevents_signal_on_brief_absence(detector: StreamFireDetector) -> None:
    """Verify that a brief absence of fire does not trigger the extinguished signal."""
    detector.fire_is_currently_detected = True  # Assume fire is ongoing

    with patch("time.monotonic", side_effect=[200, 201.0]):  # 1.0s elapsed
        await detector._handle_fire_detection(fire_in_frame=False)  # Potential end
        await detector._handle_fire_detection(fire_in_frame=True)  # Fire returns before debounce

    detector.signal_handler.fire_extinguished.assert_not_called()
    assert detector.fire_is_currently_detected is True


@pytest.mark.asyncio
async def test_fire_extinguished_debounce_triggers_signal_after_delay(detector: StreamFireDetector) -> None:
    """Verify that a sustained absence of fire triggers the extinguished signal after the debounce delay."""
    detector.fire_is_currently_detected = True  # Assume fire is ongoing

    with patch("time.monotonic", side_effect=[200, 202.1]):  # 2.1s elapsed
        await detector._handle_fire_detection(fire_in_frame=False)  # Potential end
        await detector._handle_fire_detection(fire_in_frame=False)  # Debounce time exceeded

    detector.signal_handler.fire_extinguished.assert_called_once()
    assert detector.fire_is_currently_detected is False


@pytest.mark.asyncio
async def test_fire_detected_with_zero_debounce_triggers_immediately(
    detector: StreamFireDetector, mock_settings: MagicMock
) -> None:
    """Verify that a fire detection with zero debounce triggers on the first frame."""
    mock_settings.fire_detected_debounce_seconds = 0.0

    with patch("time.monotonic", return_value=100):
        await detector._handle_fire_detection(fire_in_frame=True)

    detector.signal_handler.fire_detected.assert_called_once()
    assert detector.fire_is_currently_detected is True


@pytest.mark.asyncio
async def test_fire_extinguished_with_zero_debounce_triggers_immediately(
    detector: StreamFireDetector, mock_settings: MagicMock
) -> None:
    """Verify that a fire extinguished signal with zero debounce triggers on the first frame."""
    mock_settings.fire_extinguished_debounce_seconds = 0.0
    detector.fire_is_currently_detected = True  # Assume fire is ongoing

    with patch("time.monotonic", return_value=200):
        await detector._handle_fire_detection(fire_in_frame=False)

    detector.signal_handler.fire_extinguished.assert_called_once()
    assert detector.fire_is_currently_detected is False


# ==============================================================================
# == Tests for _frame_generator (New in Stage 3)
# ==============================================================================


async def mock_frame_source(num_frames: int) -> AsyncGenerator[np.ndarray, None]:
    """A helper async generator to simulate a video stream source."""
    for i in range(num_frames):
        yield np.array([i], dtype=np.uint8)


@pytest.mark.asyncio
async def test_frame_generator_yields_all_frames_without_throttling(detector: StreamFireDetector) -> None:
    """
    Arrange: Configure detector for no throttling. Mock the stream source.
    Act: Consume the _frame_generator.
    Assert: All frames from the source are yielded.
    """
    # Arrange
    detector.checks_per_second = None  # No throttling
    detector.stream = MagicMock()
    detector.stream.framerate = 30.0  # Set a value to avoid TypeError
    detector.stream.frames.return_value = mock_frame_source(10)

    # Act
    yielded_frames = [frame async for frame in detector._frame_generator()]

    # Assert
    assert len(yielded_frames) == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream_fps, checks_per_sec, total_frames, expected_yields",
    [
        (30.0, 1.0, 100, 3),  # 100 frames at 30fps = 3.33s -> 3 checks
        (60.0, 2.0, 60, 2),  # 60 frames at 60fps = 1.0s -> 2 checks
        (10.0, 10.0, 50, 50),  # Checks per second >= framerate -> all frames
        (25.0, 0.5, 100, 2),  # 1 check every 2 seconds
    ],
)
async def test_frame_generator_throttles_frames_correctly(
    detector: StreamFireDetector, stream_fps: float, checks_per_sec: float, total_frames: int, expected_yields: int
) -> None:
    """
    Arrange: Configure detector with specific FPS and check rates. Mock stream source.
    Act: Consume the _frame_generator.
    Assert: The correct, throttled number of frames are yielded.
    """
    # Arrange
    detector.checks_per_second = checks_per_sec
    detector.stream = MagicMock()
    detector.stream.framerate = stream_fps
    detector.stream.frames.return_value = mock_frame_source(total_frames)

    # Act
    yielded_frames = [frame async for frame in detector._frame_generator()]

    # Assert
    assert len(yielded_frames) == expected_yields


@pytest.mark.asyncio
async def test_frame_generator_handles_empty_stream(detector: StreamFireDetector) -> None:
    """

    Arrange: Mock an empty stream source.
    Act: Consume the _frame_generator.
    Assert: No frames are yielded and the generator terminates cleanly.
    """
    # Arrange
    detector.stream = MagicMock()
    detector.stream.framerate = 30.0
    detector.stream.frames.return_value = mock_frame_source(0)

    # Act
    yielded_frames = [frame async for frame in detector._frame_generator()]

    # Assert
    assert len(yielded_frames) == 0
