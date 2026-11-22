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
def mock_settings() -> Generator[MagicMock]:
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
    detector.fire_detector = MagicMock()
    detector.fire_detector.detect = AsyncMock(return_value=(False, np.zeros((10, 10, 3))))
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
async def test_fire_extinguished_invokes_on_extinguished_action(detector: StreamFireDetector) -> None:
    """The optional extinguished callback is awaited once the extinguish debounce elapses (re-arms alerts)."""
    detector.on_extinguished_action = AsyncMock()
    detector.fire_is_currently_detected = True

    with patch("time.monotonic", side_effect=[200, 201.0, 202.1]):
        await detector._handle_fire_detection(fire_in_frame=False)  # Potential end
        await detector._handle_fire_detection(fire_in_frame=False)  # Still debouncing
        detector.on_extinguished_action.assert_not_awaited()
        await detector._handle_fire_detection(fire_in_frame=False)  # Debounce time exceeded

    detector.on_extinguished_action.assert_awaited_once()


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


async def mock_frame_source(num_frames: int) -> AsyncGenerator[np.ndarray]:
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
    detector: StreamFireDetector,
    stream_fps: float,
    checks_per_sec: float,
    total_frames: int,
    expected_yields: int,
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


# --- frame in the alert (#177) ---


@pytest.mark.asyncio
async def test_fire_signal_passes_a_copy_of_the_triggering_frame(detector: StreamFireDetector) -> None:
    """The alert gets a copy of the frame that completed the debounce, not the live buffer."""
    frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    with patch("time.monotonic", side_effect=[100, 101.1]):
        await detector._handle_fire_detection(True, frame=frame)
        await detector._handle_fire_detection(True, frame=frame)
    sent = detector.on_fire_action.await_args.kwargs["frame"]
    assert np.array_equal(sent, frame)
    assert sent is not frame
    frame[:] = 0
    assert sent[0, 0, 0] == 7


@pytest.mark.asyncio
async def test_fire_signal_converts_umat_frames(detector: StreamFireDetector) -> None:
    """GPU/UMat frames are downloaded to numpy before being queued."""
    import cv2  # noqa: PLC0415

    host = np.full((4, 4, 3), 5, dtype=np.uint8)
    umat = cv2.UMat(host)
    with patch("time.monotonic", side_effect=[100, 101.1]):
        await detector._handle_fire_detection(True, frame=umat)
        await detector._handle_fire_detection(True, frame=umat)
    sent = detector.on_fire_action.await_args.kwargs["frame"]
    assert isinstance(sent, np.ndarray)
    assert np.array_equal(sent, host)


@pytest.mark.asyncio
async def test_fire_signal_without_frame_passes_no_frame_kwarg(detector: StreamFireDetector) -> None:
    """No frame (or an unconvertible one) leaves the call as before."""
    with patch("time.monotonic", side_effect=[100, 101.1]):
        await detector._handle_fire_detection(True)
        await detector._handle_fire_detection(True, frame=object())
    detector.on_fire_action.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_on_first_frame_is_called_once_after_the_first_analysed_frame(detector: StreamFireDetector) -> None:
    """The first-frame callback fires after the first frame was analysed, and only once."""
    order: list[str] = []
    detector.on_first_frame = lambda: order.append("first_frame")
    detector.fire_detector.detect = AsyncMock(side_effect=lambda _f: order.append("detect") or (False, np.zeros((1, 1, 3))))
    detector.stream = MagicMock(stop=AsyncMock())

    async def frames() -> AsyncGenerator[np.ndarray]:
        for _ in range(3):
            yield np.zeros((1, 1, 3))

    detector._frame_generator = frames  # type: ignore[method-assign]
    await detector()

    assert order == ["detect", "first_frame", "detect", "detect"]


# --- the unverified label covers the whole fire-debounce window (#171) ---


async def _debounced_fire(detector: StreamFireDetector, unverified_flags: list[bool]) -> None:
    """Runs one fire check per flag (one second apart); each check reports ``last_result_unverified`` = flag."""
    times = [100.0 + i for i in range(len(unverified_flags))]
    with patch("time.monotonic", side_effect=times):
        for flag in unverified_flags:
            detector.fire_detector.last_result_unverified = flag
            await detector._handle_fire_detection(fire_in_frame=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("flags", "expected_unverified"),
    [
        ([True, True], True),  # every check in the window unverified
        ([False, True], False),  # verified at the start, the closing check unverified
        ([True, False], False),  # the closing check verified
    ],
    ids=["all_unverified", "verified_then_unverified", "unverified_then_verified"],
)
async def test_alert_is_unverified_only_if_no_check_in_the_window_was_verified(
    detector: StreamFireDetector, flags: list[bool], expected_unverified: bool
) -> None:
    """The alert carries ``unverified=True`` only when no fire check of the debounce window was verified."""
    await _debounced_fire(detector, flags)

    detector.on_fire_action.assert_awaited_once()
    assert detector.on_fire_action.await_args.kwargs.get("unverified", False) is expected_unverified


@pytest.mark.asyncio
async def test_verified_check_does_not_leak_into_the_next_window(detector: StreamFireDetector) -> None:
    """A window reset by a no-fire check forgets an earlier verified check."""
    with patch("time.monotonic", side_effect=[100.0, 100.5, 101.0, 102.0]):
        detector.fire_detector.last_result_unverified = False
        await detector._handle_fire_detection(fire_in_frame=True)  # verified, window starts
        await detector._handle_fire_detection(fire_in_frame=False)  # window reset before the debounce ends
        detector.fire_detector.last_result_unverified = True
        await detector._handle_fire_detection(fire_in_frame=True)  # new window, unverified
        await detector._handle_fire_detection(fire_in_frame=True)  # debounce reached, still unverified

    detector.on_fire_action.assert_awaited_once()
    assert detector.on_fire_action.await_args.kwargs.get("unverified") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [True, False])
async def test_zero_debounce_uses_the_single_check(detector: StreamFireDetector, mock_settings: MagicMock, flag: bool) -> None:
    """With no debounce the alert's label is that of the one triggering check (unchanged behaviour)."""
    mock_settings.fire_detected_debounce_seconds = 0.0
    await _debounced_fire(detector, [flag])

    assert detector.on_fire_action.await_args.kwargs.get("unverified", False) is flag
