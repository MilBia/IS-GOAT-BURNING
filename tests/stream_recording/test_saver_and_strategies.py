"""
Unit and integration tests for the AsyncVideoChunkSaver and its buffer strategies.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import call

import numpy as np
import pytest
from pytest_mock import MockerFixture

from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver
from is_goat_burning.stream_recording.strategies import DiskBufferStrategy
from is_goat_burning.stream_recording.strategies import MemoryBufferStrategy

TEST_CHUNK_LENGTH_S = 1
TEST_MAX_CHUNKS = 1
TEST_CHUNKS_TO_KEEP = 2
TEST_FPS = 10


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
    save_stream_settings = {
        "video.buffer_mode": "disk",
        "video.record_during_fire": False,
        "video.chunks_to_keep_after_fire": 2,
        "video.chunk_length_seconds": 1,
    }
    mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings", **save_stream_settings)
    strategies_settings = {"video.memory_buffer_seconds": 10, "video.fps": 30.0}
    return mocker.patch("is_goat_burning.stream_recording.strategies.settings", **strategies_settings)


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
    saver_context.FRAME_WRITE_BATCH_SIZE = 30
    return saver_context


# ==============================================================================
# == Stage 1: DiskBufferStrategy Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_disk_strategy_continuously_writes_frames(
    mock_saver_context: MagicMock, mock_frame: np.ndarray, mocker: MockerFixture
) -> None:
    mocker.patch("cv2.imencode", return_value=(True, np.array([1, 2, 3], dtype=np.uint8)))
    strategy = DiskBufferStrategy(context=mock_saver_context)
    strategy.add_frame(mock_frame)
    await mock_saver_context.frame_queue.put(None)  # Sentinel to stop the loop
    await strategy._run_main_loop()
    # The strategy now puts bytes into the queue and writes in batch
    mock_saver_context._write_frames_blocking.assert_called_once_with([b"\x01\x02\x03"])


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

    async def stop_loop(*args, **kwargs) -> None:  # noqa: ARG001
        mock_saver_context.signal_handler.is_fire_detected.return_value = False

    mock_saver_context._handle_fire_event_async.side_effect = stop_loop
    mocker.patch.object(mock_saver_context.frame_queue, "get", side_effect=[asyncio.TimeoutError, None])
    await strategy._run_main_loop()
    mock_saver_context._handle_fire_event_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_disk_strategy_archives_pre_fire_chunks_on_event(mocker: MockerFixture, mock_settings_base: MagicMock) -> None:
    mock_settings_base.video.buffer_mode = "disk"
    mocker.patch("os.makedirs")
    mock_finalize = mocker.patch.object(AsyncVideoChunkSaver, "_finalize_active_chunk_async", new_callable=AsyncMock)
    saver = AsyncVideoChunkSaver(
        enabled=True, output_dir="test", chunk_length_seconds=1, max_chunks=5, chunks_to_keep_after_fire=1, fps=30
    )
    saver.pre_fire_buffer.extend(["chunk1.mp4", "chunk2.mp4", "active_chunk3.mp4"])
    await saver.strategy.handle_fire_event(event_dir="event_dir_archive")
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
    memory_strategy: MemoryBufferStrategy,
    mock_frame: np.ndarray,
    mocker: MockerFixture,
) -> None:
    """
    Arrange: Initialize MemoryBufferStrategy.
    Act: Add a frame.
    Assert: The frame is encoded and added to the internal memory buffer, and
            the disk queue is not used.
    """
    # Arrange
    mock_imencode = mocker.patch("cv2.imencode", return_value=(True, np.array([1, 2, 3])))
    memory_strategy.add_frame(mock_frame)
    assert len(memory_strategy.memory_buffer) == 1
    mock_imencode.assert_called_once()
    assert memory_strategy.context.frame_queue.qsize() == 0


@pytest.mark.asyncio
async def test_memory_strategy_flushes_buffer_to_disk_on_fire_event(
    memory_strategy: MemoryBufferStrategy, mocker: MockerFixture
) -> None:
    memory_strategy.memory_buffer.extend([b"frame1", b"frame2"])
    mock_flush = mocker.patch.object(memory_strategy.context, "_flush_buffer_to_disk_blocking")
    mocker.patch("is_goat_burning.stream_recording.strategies.datetime")
    await memory_strategy._flush_and_archive_memory_buffer("event_dir")
    assert mock_flush.call_count == 1
    flushed_frames_arg = mock_flush.call_args[0][0]
    assert len(flushed_frames_arg) == 2
    assert list(flushed_frames_arg) == [b"frame1", b"frame2"]
    assert len(memory_strategy.memory_buffer) == 0


@pytest.mark.asyncio
async def test_memory_strategy_switches_to_queueing_after_fire(
    memory_strategy: MemoryBufferStrategy, mocker: MockerFixture
) -> None:
    memory_strategy._is_post_fire_recording = True
    # Verify that add_frame now queues instead of buffering
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    mocker.patch(
        "cv2.imencode",
        return_value=(True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    # Mock put_nowait to assert its call
    memory_strategy.context.frame_queue.put_nowait = MagicMock()
    memory_strategy.add_frame(frame)
    assert len(memory_strategy.memory_buffer) == 0
    memory_strategy.context.frame_queue.put_nowait.assert_called_once_with(b"\x01\x02\x03")


# ==============================================================================
# == Stage 3 & 4: Shared Logic and Helpers Tests
# ==============================================================================


@pytest.fixture
def saver(mocker: MockerFixture, mock_settings_base: MagicMock) -> AsyncVideoChunkSaver:
    mock_settings_base.video.buffer_mode = "disk"
    mocker.patch("os.makedirs")
    return AsyncVideoChunkSaver(
        enabled=True,
        output_dir=".",
        chunk_length_seconds=TEST_CHUNK_LENGTH_S,
        max_chunks=TEST_MAX_CHUNKS,
        chunks_to_keep_after_fire=TEST_CHUNKS_TO_KEEP,
        fps=TEST_FPS,
    )


@pytest.mark.asyncio
async def test_save_post_fire_chunks_records_final_duration(
    saver: AsyncVideoChunkSaver, mock_frame: np.ndarray, mocker: MockerFixture
) -> None:
    """Tests that post-fire recording saves the correctly configured duration using the ffmpeg strategy."""
    # Arrange
    mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings.video.record_during_fire", False)
    # chunks_to_keep_after_fire=2, chunk_length_seconds=1, fps=10 -> 20 frames total
    expected_frames = TEST_CHUNKS_TO_KEEP * TEST_CHUNK_LENGTH_S * TEST_FPS
    extra_frames = 5

    mock_ffmpeg_flush = mocker.patch(
        "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._flush_buffer_to_disk_ffmpeg"
    )

    # Pre-fill the queue
    for _ in range(expected_frames + extra_frames):
        await saver.frame_queue.put(mock_frame)
    await saver.frame_queue.put(None)  # Add sentinel to prevent waiting forever if logic is flawed

    # Act
    await saver._save_post_fire_chunks_async("event_dir")

    # Assert
    assert mock_ffmpeg_flush.call_count > 0
    total_written = sum(len(call.args[0]) for call in mock_ffmpeg_flush.call_args_list)
    assert total_written == expected_frames


@pytest.mark.asyncio
async def test_save_post_fire_chunks_records_during_fire_until_signal(
    saver: AsyncVideoChunkSaver, mock_frame: np.ndarray, mocker: MockerFixture
) -> None:
    """Tests that 'record_during_fire' mode records until the extinguish signal using the ffmpeg strategy."""
    # Arrange
    mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings.video.record_during_fire", True)
    mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings.video.chunks_to_keep_after_fire", 0)

    mock_ffmpeg_flush = mocker.patch(
        "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._flush_buffer_to_disk_ffmpeg"
    )

    frames_before_extinguish = 15  # Should create one full chunk and one partial
    frames_per_chunk = TEST_CHUNK_LENGTH_S * TEST_FPS  # 10

    # Mock the signal to return False a few times, then True
    mocker.patch.object(saver.signal_handler, "is_fire_extinguished", side_effect=([False] * frames_before_extinguish) + [True])

    # Mock the queue to feed frames and then stop
    queue_effects = ([mock_frame] * frames_before_extinguish) + [asyncio.TimeoutError]
    mocker.patch.object(saver.frame_queue, "get", AsyncMock(side_effect=queue_effects))

    # Act
    await saver._save_post_fire_chunks_async("event_dir")

    # Assert
    # One full chunk of 10, one partial of 5
    assert mock_ffmpeg_flush.call_count == 2

    total_written = sum(len(call.args[0]) for call in mock_ffmpeg_flush.call_args_list)
    assert total_written == frames_before_extinguish

    # Check that the chunks were correct
    assert len(mock_ffmpeg_flush.call_args_list[0].args[0]) == frames_per_chunk
    assert len(mock_ffmpeg_flush.call_args_list[1].args[0]) == frames_before_extinguish - frames_per_chunk


def test_enforce_chunk_limit_deletes_oldest_files(mocker: MockerFixture, saver: AsyncVideoChunkSaver) -> None:
    saver.max_chunks = 3
    file_list = ["goat-cam_2.mp4", "goat-cam_3.mp4", "goat-cam_1.mp4", "goat-cam_4.mp4"]
    mock_listdir = mocker.patch("os.listdir", return_value=file_list)
    mock_remove = mocker.patch("os.remove")
    mocker.patch("os.path.isfile", return_value=True)
    saver._enforce_chunk_limit_blocking()
    mock_listdir.assert_called_once_with(saver.output_dir)
    expected_calls = [
        call(os.path.join(saver.output_dir, "goat-cam_1.mp4")),
        call(os.path.join(saver.output_dir, "goat-cam_2.mp4")),
    ]
    mock_remove.assert_has_calls(expected_calls, any_order=True)
    assert mock_remove.call_count == 2


def test_write_frame_blocking_rotates_chunk_after_timeout(
    mocker: MockerFixture, saver: AsyncVideoChunkSaver, mock_frame: np.ndarray
) -> None:
    # This test now uses the REAL _write_frame_blocking method from the saver fixture.
    saver.chunk_length_seconds = 5
    mock_video_writer = mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.cv2.VideoWriter")
    mock_time = mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.time.time")

    mock_time.return_value = 1000.0
    saver._write_frame_blocking(mock_frame)
    assert mock_video_writer.call_count == 1

    mock_time.return_value = 1004.0
    saver._write_frame_blocking(mock_frame)
    assert mock_video_writer.call_count == 1

    mock_time.return_value = 1006.0
    saver._write_frame_blocking(mock_frame)
    assert mock_video_writer.call_count == 2


@pytest.mark.asyncio
async def test_create_event_directory_handles_os_error(saver: AsyncVideoChunkSaver, mocker: MockerFixture) -> None:
    """
    Arrange: Mock the saver's makedirs callable to raise an OSError.
    Act: Call the _create_event_directory method.
    Assert: The method returns None and does not raise an exception.
    """
    # Arrange
    mocker.patch.object(saver, "makedirs_callable", side_effect=OSError("Permission denied"))

    # Act
    result = await saver._create_event_directory()

    # Assert
    assert result is None
