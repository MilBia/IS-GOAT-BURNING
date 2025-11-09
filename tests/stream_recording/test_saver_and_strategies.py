"""
Unit and integration tests for the AsyncVideoChunkSaver and its buffer strategies.

This module provides a comprehensive suite of tests for the video recording
and archiving logic, which is one of the most complex parts of the application.
It uses extensive mocking for time, file system operations, and the OpenCV
VideoWriter to ensure tests are fast, deterministic, and isolated.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from is_goat_burning.fire_detection.signal_handler import SignalHandler
from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver
from is_goat_burning.stream_recording.strategies import DiskBufferStrategy
from is_goat_burning.stream_recording.strategies import MemoryBufferStrategy

# ==============================================================================
# == Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_frame() -> np.ndarray:
    """Provides a standard numpy array representing a video frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_settings_base(mocker: MockerFixture) -> MagicMock:
    """Provides a base mock for settings, shared across strategies."""
    # Patch settings in the save_stream_to_file module
    save_stream_settings = {
        "video.buffer_mode": "disk",  # Default to disk for saver tests
        "video.record_during_fire": False,
        "video.chunks_to_keep_after_fire": 2,
        "video.chunk_length_seconds": 1,
    }
    mocker.patch(
        "is_goat_burning.stream_recording.save_stream_to_file.settings",
        **save_stream_settings,
    )

    # Patch settings in the strategies module
    strategies_settings = {"video.memory_buffer_seconds": 10, "video.fps": 30.0}
    return mocker.patch(
        "is_goat_burning.stream_recording.strategies.settings",
        **strategies_settings,
    )


@pytest.fixture
def mock_saver_context() -> MagicMock:
    """Mocks the AsyncVideoChunkSaver context for strategy testing."""
    saver_context = MagicMock(spec=AsyncVideoChunkSaver)
    saver_context.frame_queue = asyncio.Queue()
    saver_context.archive_queue = asyncio.Queue()
    saver_context.signal_handler = MagicMock()
    saver_context._write_frame_blocking = MagicMock(return_value=None)
    saver_context._handle_fire_event_async = AsyncMock()
    saver_context._fire_handling_lock = asyncio.Lock()
    saver_context.FRAME_QUEUE_POLL_TIMEOUT = 0.01
    return saver_context


# ==============================================================================
# == Stage 1: DiskBufferStrategy Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_disk_strategy_continuously_writes_frames(mock_saver_context: MagicMock, mock_frame: np.ndarray) -> None:
    """
    Arrange: Initialize DiskBufferStrategy and add a frame.
    Act: Let the strategy's main loop run once by consuming the frame.
    Assert: The context's `_write_frame_blocking` method is called correctly.
    """
    # Arrange
    strategy = DiskBufferStrategy(context=mock_saver_context)
    strategy.add_frame(mock_frame)
    strategy.add_frame(None)  # Sentinel to stop the loop

    # Act
    await strategy._run_main_loop()

    # Assert
    mock_saver_context._write_frame_blocking.assert_called_once_with(mock_frame)


@pytest.mark.asyncio
async def test_disk_strategy_triggers_fire_event_handling(mock_saver_context: MagicMock, mocker: MockerFixture) -> None:
    """
    Arrange: Initialize DiskBufferStrategy, mock the fire signal to be active.
    Act: Let the strategy's main loop run until it stops.
    Assert: The context's `_handle_fire_event_async` method is called.
    """
    # Arrange
    strategy = DiskBufferStrategy(context=mock_saver_context)
    mock_saver_context.signal_handler.is_fire_detected.return_value = True

    async def stop_loop_after_call(*args, **kwargs):  # noqa: ARG001
        mock_saver_context.signal_handler.is_fire_detected.return_value = False

    mock_saver_context._handle_fire_event_async.side_effect = stop_loop_after_call
    mocker.patch.object(mock_saver_context.frame_queue, "get", side_effect=[asyncio.TimeoutError, None])

    # Act
    await strategy._run_main_loop()

    # Assert
    mock_saver_context._handle_fire_event_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_disk_strategy_archives_pre_fire_chunks_on_event(mocker: MockerFixture, mock_settings_base: MagicMock) -> None:
    """
    Arrange: Create a saver with Disk strategy and populate its pre-fire buffer.
    Act: Call the strategy's fire event handler.
    Assert: All but the last chunk from the buffer are queued for archiving.
    """
    # Arrange
    mock_settings_base.video.buffer_mode = "disk"
    mocker.patch("os.makedirs")
    mock_finalize = mocker.patch.object(AsyncVideoChunkSaver, "_finalize_active_chunk_async", new_callable=AsyncMock)

    saver = AsyncVideoChunkSaver(
        enabled=True, output_dir="test", chunk_length_seconds=1, max_chunks=5, chunks_to_keep_after_fire=1, fps=30
    )
    saver.pre_fire_buffer.extend(["chunk1.mp4", "chunk2.mp4", "active_chunk3.mp4"])

    # Act
    await saver.strategy.handle_fire_event(event_dir="event_dir_archive")

    # Assert
    assert saver.archive_queue.qsize() == 2
    assert await saver.archive_queue.get() == ("chunk1.mp4", "event_dir_archive")
    assert await saver.archive_queue.get() == ("chunk2.mp4", "event_dir_archive")
    mock_finalize.assert_awaited_once_with("event_dir_archive")


# ==============================================================================
# == Stage 2: MemoryBufferStrategy Tests
# ==============================================================================


@pytest.fixture
def memory_strategy(mock_saver_context: MagicMock) -> MemoryBufferStrategy:
    """Provides a MemoryBufferStrategy with a mocked context for isolated testing."""
    mock_saver_context.fps = 30.0
    return MemoryBufferStrategy(context=mock_saver_context)


def test_memory_strategy_buffers_frames_in_ram_by_default(
    memory_strategy: MemoryBufferStrategy, mock_frame: np.ndarray, mocker: MockerFixture
) -> None:
    """
    Arrange: Initialize MemoryBufferStrategy.
    Act: Add a frame.
    Assert: The frame is encoded and added to the internal memory buffer, and
            the disk queue is not used.
    """
    # Arrange
    mock_imencode = mocker.patch("cv2.imencode", return_value=(True, np.array([1, 2, 3])))

    # Act
    memory_strategy.add_frame(mock_frame)

    # Assert
    assert len(memory_strategy.memory_buffer) == 1
    mock_imencode.assert_called_once()
    assert memory_strategy.context.frame_queue.qsize() == 0


@pytest.mark.asyncio
async def test_memory_strategy_flushes_buffer_to_disk_on_fire_event(
    memory_strategy: MemoryBufferStrategy, mocker: MockerFixture
) -> None:
    """
    Arrange: Add some frames to the memory buffer.
    Act: Trigger the fire event handler.
    Assert: The context's flush method is called with the buffered frames.
    """
    # Arrange
    memory_strategy.memory_buffer.extend([b"frame1", b"frame2"])
    mock_flush = mocker.patch.object(memory_strategy.context, "_flush_buffer_to_disk_blocking")
    mocker.patch("is_goat_burning.stream_recording.strategies.datetime")

    # Act
    await memory_strategy._flush_and_archive_memory_buffer("event_dir")

    # Assert
    assert mock_flush.call_count == 1
    flushed_frames_arg = mock_flush.call_args[0][0]
    assert len(flushed_frames_arg) == 2
    assert list(flushed_frames_arg) == [b"frame1", b"frame2"]
    assert len(memory_strategy.memory_buffer) == 0


@pytest.mark.asyncio
async def test_memory_strategy_switches_to_queueing_after_fire(
    memory_strategy: MemoryBufferStrategy, mock_frame: np.ndarray
) -> None:
    """
    Arrange: Trigger the fire event to switch the strategy's mode.
    Act: Add a new frame after the event.
    Assert: The frame is added to the disk queue, not the memory buffer.
    """
    # Arrange
    memory_strategy._is_post_fire_recording = True
    assert memory_strategy.context.frame_queue.qsize() == 0

    # Act
    memory_strategy.add_frame(mock_frame)

    # Assert
    assert len(memory_strategy.memory_buffer) == 0
    assert memory_strategy.context.frame_queue.qsize() == 1
    assert await memory_strategy.context.frame_queue.get() is mock_frame


# ==============================================================================
# == Stage 3: Shared Post-Fire Recording Logic Tests
# ==============================================================================


@pytest.fixture
def saver(mocker: MockerFixture, mock_settings_base) -> AsyncVideoChunkSaver:
    """Provides a fully initialized AsyncVideoChunkSaver with mocked dependencies."""
    mock_settings_base.video.buffer_mode = "disk"
    mocker.patch("os.makedirs")
    mocker.patch.object(AsyncVideoChunkSaver, "_write_frame_blocking")
    return AsyncVideoChunkSaver(
        enabled=True, output_dir=".", chunk_length_seconds=1, max_chunks=1, chunks_to_keep_after_fire=2, fps=10
    )


@pytest.mark.asyncio
async def test_post_fire_loop_records_for_configured_duration(saver: AsyncVideoChunkSaver, mock_frame: np.ndarray) -> None:
    """
    Arrange: Configure the saver for a specific post-fire duration and populate the queue.
    Act: Run the final chunks recording loop.
    Assert: The write method is called the exact number of times corresponding to
            the configured duration.
    """
    # Arrange
    expected_frames = 20  # chunks_to_keep * chunk_length * fps = 2 * 1 * 10
    for _ in range(expected_frames + 5):
        await saver.frame_queue.put(mock_frame)

    # Act
    await saver._record_final_chunks_loop("event_dir")

    # Assert
    assert saver._write_frame_blocking.call_count == expected_frames
    assert saver.frame_queue.qsize() == 5


@pytest.mark.asyncio
async def test_record_during_fire_loop_continues_until_extinguished_signal(
    saver: AsyncVideoChunkSaver, mock_frame: np.ndarray, mocker: MockerFixture
) -> None:
    """
    Arrange: Configure the saver to record during fire and mock the extinguished signal.
    Act: Run the "record during fire" loop.
    Assert: The loop writes frames until the signal is set.
    """
    # Arrange
    # FIX: Correctly patch the method on the SignalHandler CLASS since it's a singleton.
    mock_is_extinguished = mocker.patch.object(SignalHandler, "is_fire_extinguished", side_effect=[False, False, False, True])
    for _ in range(5):
        await saver.frame_queue.put(mock_frame)

    # Act
    await saver._record_during_fire_loop("event_dir")

    # Assert
    # The loop should run 3 times before the signal becomes True
    assert saver._write_frame_blocking.call_count == 3
    assert mock_is_extinguished.call_count == 4
    assert saver.frame_queue.qsize() == 2
