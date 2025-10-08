"""Unit tests for the AsyncVideoChunkSaver and its buffer strategies."""

from unittest.mock import MagicMock
from unittest.mock import call

import numpy as np
import pytest
from pytest_mock import MockerFixture

from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver
from is_goat_burning.stream_recording.strategies import DiskBufferStrategy
from is_goat_burning.stream_recording.strategies import MemoryBufferStrategy

# --- Test Constants ---
TEST_NUM_PRE_FIRE_FRAMES = 5
TEST_NUM_POST_FIRE_FRAMES_TO_QUEUE = 3
DURING_FIRE_FRAMES = 2  # Frames to simulate during the fire event
FINAL_CHUNKS_TO_KEEP = 2  # Should match chunks_to_keep_after_fire


@pytest.fixture
def mock_settings(mocker: MockerFixture) -> MagicMock:
    """Fixture to provide a mock of the global settings object."""
    return mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings")


@pytest.fixture
def mock_strategy_settings(mocker: MockerFixture) -> MagicMock:
    """Fixture to provide a mock of settings specifically for the strategies module."""
    return mocker.patch("is_goat_burning.stream_recording.strategies.settings")


# ==============================================================================
# == AsyncVideoChunkSaver (Context) Tests
# ==============================================================================


def test_saver_initializes_disk_strategy_by_default(mock_settings: MagicMock, mocker: MockerFixture) -> None:
    """Verify the saver selects DiskBufferStrategy based on settings."""
    mock_settings.video.buffer_mode = "disk"
    mocker.patch("os.makedirs")
    saver = AsyncVideoChunkSaver(
        enabled=True, output_dir="test", chunk_length_seconds=1, max_chunks=1, chunks_to_keep_after_fire=1
    )
    assert isinstance(saver.strategy, DiskBufferStrategy)


def test_saver_initializes_memory_strategy_when_configured(
    mock_settings: MagicMock, mock_strategy_settings, mocker: MockerFixture
) -> None:
    """Verify the saver selects MemoryBufferStrategy based on settings."""
    mock_settings.video.buffer_mode = "memory"
    mock_strategy_settings.video.memory_buffer_seconds = 10
    mock_strategy_settings.video.fps = 30
    mocker.patch("os.makedirs")
    saver = AsyncVideoChunkSaver(
        enabled=True, output_dir="test", chunk_length_seconds=1, max_chunks=1, chunks_to_keep_after_fire=1
    )
    assert isinstance(saver.strategy, MemoryBufferStrategy)


def test_write_frame_blocking_creates_and_rotates_chunks(mocker: MockerFixture) -> None:
    """Tests the synchronous, low-level logic for creating and rotating video files."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_settings = mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings")
    mock_settings.video.buffer_mode = "disk"  # Strategy doesn't matter for this low-level test
    mocker.patch("os.makedirs")

    saver = AsyncVideoChunkSaver(
        enabled=True, output_dir="test_output", chunk_length_seconds=60, max_chunks=3, chunks_to_keep_after_fire=1, fps=30
    )

    mock_writers = [MagicMock() for _ in range(4)]
    listdir_side_effect = [
        [],  # Before chunk 1
        ["goat-cam_f1.mp4"],  # Before chunk 2
        ["goat-cam_f1.mp4", "goat-cam_f2.mp4"],  # Before chunk 3
        ["goat-cam_f1.mp4", "goat-cam_f2.mp4", "goat-cam_f3.mp4"],  # Before chunk 4
    ]

    mock_vw = mocker.patch("cv2.VideoWriter", side_effect=mock_writers)
    mock_time = mocker.patch("time.time")
    mock_remove = mocker.patch("os.remove")
    mocker.patch("os.listdir", side_effect=listdir_side_effect)
    mocker.patch("os.path.isfile", return_value=True)

    # --- Create first 3 chunks (no cleanup expected) ---
    mock_time.return_value = 1000.0
    saver._write_frame_blocking(frame)  # Chunk 1
    mock_time.return_value = 1061.0
    saver._write_frame_blocking(frame)  # Chunk 2
    mock_time.return_value = 1122.0
    saver._write_frame_blocking(frame)  # Chunk 3
    mock_remove.assert_not_called()
    assert mock_vw.call_count == 3

    # --- Create 4th chunk (triggers cleanup) ---
    mock_time.return_value = 1183.0
    saver._write_frame_blocking(frame)
    assert mock_vw.call_count == 4
    mock_remove.assert_called_once_with("test_output/goat-cam_f1.mp4")


# ==============================================================================
# == DiskBufferStrategy Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_disk_strategy_fire_event_archives_chunks(mocker: MockerFixture) -> None:
    """Tests the DiskBufferStrategy's fire event orchestration."""
    mock_context = mocker.MagicMock(spec=AsyncVideoChunkSaver)
    mock_context.pre_fire_buffer = ["/path/pre_fire_1.mp4", "/path/pre_fire_2.mp4", "/path/active_3.mp4"]
    mock_context.archive_queue.put_nowait = mocker.MagicMock()
    mock_context._finalize_active_chunk_async = mocker.AsyncMock()

    strategy = DiskBufferStrategy(context=mock_context)
    await strategy.handle_fire_event(event_dir="event_dir")

    # Assert that pre-fire chunks (not the active one) were queued
    expected_archive_calls = [
        call(("/path/pre_fire_1.mp4", "event_dir")),
        call(("/path/pre_fire_2.mp4", "event_dir")),
    ]
    mock_context.archive_queue.put_nowait.assert_has_calls(expected_archive_calls)
    assert mock_context.archive_queue.put_nowait.call_count == 2

    # Assert that the finalization of the active chunk was called
    mock_context._finalize_active_chunk_async.assert_awaited_once_with("event_dir")


# ==============================================================================
# == MemoryBufferStrategy Tests
# ==============================================================================


@pytest.fixture
def memory_strategy(mocker: MockerFixture) -> MemoryBufferStrategy:
    """Provides a MemoryBufferStrategy with a mocked context for isolated testing."""
    mocker.patch("is_goat_burning.stream_recording.strategies.settings")
    mock_context = mocker.MagicMock(spec=AsyncVideoChunkSaver)
    mock_context.fps = 1.0  # Simplify buffer size calculation
    return MemoryBufferStrategy(context=mock_context)


def test_memory_strategy_add_frame_buffers_by_default(memory_strategy: MemoryBufferStrategy) -> None:
    """Tests that the memory strategy adds frames to its internal buffer."""
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    memory_strategy.add_frame(frame)
    assert len(memory_strategy.memory_buffer) == 1
    memory_strategy.context.frame_queue.put_nowait.assert_not_called()


@pytest.mark.asyncio
async def test_memory_strategy_fire_event_flushes_and_switches_mode(
    memory_strategy: MemoryBufferStrategy, mocker: MockerFixture
) -> None:
    """Tests that the fire event flushes the buffer and changes the `add_frame` behavior."""
    # Setup
    mock_flush = mocker.patch.object(memory_strategy, "_flush_and_archive_memory_buffer", new_callable=mocker.AsyncMock)
    assert memory_strategy._is_post_fire_recording is False

    # Execute
    await memory_strategy.handle_fire_event("event_dir")

    # Assert flush
    mock_flush.assert_awaited_once_with("event_dir")

    # Assert mode switch
    assert memory_strategy._is_post_fire_recording is True

    # Verify that add_frame now queues instead of buffering
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    memory_strategy.add_frame(frame)
    assert len(memory_strategy.memory_buffer) == 0
    memory_strategy.context.frame_queue.put_nowait.assert_called_once_with(frame)


def test_memory_strategy_reset_restores_initial_state(memory_strategy: MemoryBufferStrategy) -> None:
    """Tests that the reset method correctly reverts the strategy's state."""
    # Change state
    memory_strategy._is_post_fire_recording = True
    memory_strategy.memory_buffer.append(np.zeros((1, 1, 3), dtype=np.uint8))

    # Reset
    memory_strategy.reset()

    # Assert initial state
    assert memory_strategy._is_post_fire_recording is False
    assert len(memory_strategy.memory_buffer) == 0


# ==============================================================================
# == Full Integration Tests (Saver + Strategy)
# ==============================================================================


@pytest.mark.asyncio
async def test_saver_with_memory_strategy_full_flow(mocker: MockerFixture) -> None:
    """An integration test of the saver and memory strategy working together."""
    # --- Mocks and Setup ---
    mocker.patch("os.makedirs")
    mock_settings = mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings")
    mock_settings.video.buffer_mode = "memory"
    mock_settings.video.chunks_to_keep_after_fire = FINAL_CHUNKS_TO_KEEP
    mock_settings.video.record_during_fire = False
    mocker.patch(
        "is_goat_burning.stream_recording.strategies.settings",
        **{"video.memory_buffer_seconds": 10, "video.fps": 1.0},
    )

    saver = AsyncVideoChunkSaver(
        enabled=True, output_dir=".", chunk_length_seconds=1, max_chunks=1, chunks_to_keep_after_fire=2, fps=1
    )

    # --- FIX: Patch the class methods using string paths, not the instance ---
    mock_flush = mocker.patch(
        "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._flush_buffer_to_disk_blocking"
    )
    mock_write = mocker.patch(
        "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._write_frame_blocking",
        side_effect=[None, "chunk1.mp4", "chunk2.mp4"],
    )
    mock_create_dir = mocker.patch(
        "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._create_event_directory",
        new_callable=mocker.AsyncMock,
        return_value="mock_event_dir",
    )

    mocker.patch.object(saver.archive_queue, "put_nowait")

    # --- Simulation ---
    # 1. Pre-fire: Add frames to memory buffer
    pre_fire_frames = [np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(5)]
    for frame in pre_fire_frames:
        saver(frame)
    assert len(saver.strategy.memory_buffer) == 5

    # 2. Fire Event Triggered
    # (Populate frame queue for post-fire recording)
    post_fire_frames = [np.ones((1, 1, 3), dtype=np.uint8) for _ in range(3)]
    for frame in post_fire_frames:
        await saver.frame_queue.put(frame)

    await saver._handle_fire_event_async()

    # --- Assertions ---
    # 1. Assert event directory was created
    mock_create_dir.assert_awaited_once()

    # 2. Assert pre-fire buffer was flushed
    mock_flush.assert_called_once()
    flushed_frames_arg = mock_flush.call_args[0][0]
    assert len(flushed_frames_arg) == 5

    # 3. Assert post-fire chunks were written
    assert mock_write.call_count == 3
    expected_calls = [call(frame, "mock_event_dir") for frame in post_fire_frames]
    # Use assert_has_calls because the exact call order might be complex
    mock_write.assert_has_calls(expected_calls)
