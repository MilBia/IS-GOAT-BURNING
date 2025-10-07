"""Unit tests for the AsyncVideoChunkSaver class.

These tests cover the synchronous (blocking) logic of file I/O and the
asynchronous orchestration of the fire event handling mechanism.
"""

from collections.abc import Generator
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import numpy as np
import pytest
from pytest_mock import MockerFixture

from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver

# --- Test Constants ---
TEST_NUM_PRE_FIRE_FRAMES = 5
TEST_NUM_POST_FIRE_FRAMES_TO_QUEUE = 3
DURING_FIRE_FRAMES = 2  # Frames to simulate during the fire event
FINAL_CHUNKS_TO_KEEP = 2  # Should match chunks_to_keep_after_fire


@pytest.fixture
def saver() -> Generator[tuple[AsyncVideoChunkSaver, MagicMock], None, None]:
    """Provides a clean, enabled AsyncVideoChunkSaver instance for each test.

    This fixture also patches `os.makedirs` to prevent actual directory
    creation during tests.

    Yields:
        A tuple containing the configured `AsyncVideoChunkSaver` instance and
        the mock for `os.makedirs`.
    """
    with (
        patch("os.makedirs") as mock_mkdir,
        patch("is_goat_burning.stream_recording.save_stream_to_file.settings") as mock_settings,
    ):
        # Default to 'disk' mode for existing tests
        mock_settings.video.buffer_mode = "disk"
        mock_settings.video.record_during_fire = False

        saver_instance = AsyncVideoChunkSaver(
            enabled=True,
            output_dir="test_output",
            chunk_length_seconds=60,
            max_chunks=3,
            chunks_to_keep_after_fire=FINAL_CHUNKS_TO_KEEP,
            fps=30,
        )
        yield saver_instance, mock_mkdir


@pytest.fixture
def memory_mode_saver(mocker: MockerFixture) -> tuple[AsyncVideoChunkSaver, MagicMock, MagicMock, MagicMock]:
    """Provides a fully configured and mocked saver for memory mode testing."""
    mock_settings = mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings")
    mock_settings.video.buffer_mode = "memory"
    mock_settings.video.chunks_to_keep_after_fire = FINAL_CHUNKS_TO_KEEP
    mock_settings.video.memory_buffer_seconds = TEST_NUM_PRE_FIRE_FRAMES
    mock_settings.video.record_during_fire = False

    # Patch os.makedirs BEFORE the saver is instantiated
    mock_makedirs = mocker.patch("os.makedirs")

    flush_mock = mocker.patch(
        "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._flush_buffer_to_disk_blocking"
    )
    write_mock = mocker.patch(
        "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._write_frame_blocking",
        side_effect=[None, "/path/post_fire_1.mp4", "/path/post_fire_2.mp4"],
    )

    saver = AsyncVideoChunkSaver(
        enabled=True, output_dir="test_output", chunk_length_seconds=1, max_chunks=3, chunks_to_keep_after_fire=2, fps=1
    )
    mocker.patch.object(saver.archive_queue, "put_nowait")

    return saver, flush_mock, write_mock, mock_makedirs


@pytest.fixture
def duration_mode_saver(mocker: MockerFixture) -> tuple[AsyncVideoChunkSaver, MagicMock, MagicMock]:
    """Provides a saver configured to record for the fire's duration."""
    mock_settings = mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings")
    mock_settings.video.buffer_mode = "memory"
    mock_settings.video.chunks_to_keep_after_fire = FINAL_CHUNKS_TO_KEEP
    mock_settings.video.record_during_fire = True

    mock_makedirs = mocker.patch("os.makedirs")
    write_mock = mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._write_frame_blocking")

    # Make chunk length short to force chunk rotation in tests
    saver = AsyncVideoChunkSaver(
        enabled=True,
        output_dir="test_output",
        chunk_length_seconds=1,  # 1 second per chunk
        max_chunks=3,
        chunks_to_keep_after_fire=FINAL_CHUNKS_TO_KEEP,
        fps=1,  # 1 frame per second
    )
    return saver, write_mock, mock_makedirs


def test_write_frame_blocking_creates_and_rotates_chunks() -> None:
    """Tests the synchronous logic for creating, rotating, and cleaning up chunks.

    This test uses time and filesystem mocks to deterministically verify that
    `_write_frame_blocking` creates new video files when the time limit is
    exceeded and deletes the oldest file when the `max_chunks` limit is reached.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("os.makedirs"), patch("is_goat_burning.stream_recording.save_stream_to_file.settings") as mock_settings:
        mock_settings.video.buffer_mode = "disk"
        mock_settings.video.record_during_fire = False
        saver = AsyncVideoChunkSaver(
            enabled=True,
            output_dir="test_output",
            chunk_length_seconds=60,
            max_chunks=3,
            chunks_to_keep_after_fire=FINAL_CHUNKS_TO_KEEP,
            fps=30,
        )

    mock_writers = [MagicMock() for _ in range(4)]
    listdir_side_effect = [
        [],  # Before chunk 1
        ["goat-cam_f1.mp4"],  # Before chunk 2
        ["goat-cam_f1.mp4", "goat-cam_f2.mp4"],  # Before chunk 3
        ["goat-cam_f1.mp4", "goat-cam_f2.mp4", "goat-cam_f3.mp4"],  # Before chunk 4
    ]

    with (
        patch("cv2.VideoWriter", side_effect=mock_writers) as mock_vw,
        patch("time.time") as mock_time,
        patch("os.remove") as mock_remove,
        patch("os.listdir", side_effect=listdir_side_effect),
        patch("os.path.isfile", return_value=True),
    ):
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


@pytest.mark.asyncio
async def test_fire_event_archives_chunks_and_records_new_ones(saver: tuple[AsyncVideoChunkSaver, MagicMock]) -> None:
    """Tests the async fire event orchestration for 'disk' mode."""
    saver_instance, mock_mkdir = saver
    # --- Setup ---
    saver_instance.pre_fire_buffer.extend(["/path/pre_fire_chunk_1.mp4", "/path/active_chunk_2.mp4"])
    saver_instance.current_video_path = "/path/active_chunk_2.mp4"
    saver_instance.writer = MagicMock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Put enough frames for finalize_active + save_post_fire
    for _ in range(saver_instance.chunks_to_keep_after_fire + 1):
        await saver_instance.frame_queue.put(frame)

    write_blocking_mock = MagicMock(side_effect=["/path/active_chunk_2.mp4", "/path/post_fire_1.mp4", "/path/post_fire_2.mp4"])
    patch_path = "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._write_frame_blocking"

    with patch(patch_path, write_blocking_mock), patch.object(saver_instance.signal_handler, "reset_fire_event"):
        # --- Execution ---
        await saver_instance._handle_fire_event_async()

        # --- Assertions ---
        # 1. Verify event directory was created
        assert mock_mkdir.call_count == 2
        last_call_args, _ = mock_mkdir.call_args
        event_dir = last_call_args[0]
        assert "test_output/event_" in event_dir

        # 2. Verify only pre-fire and active chunks were queued for archiving
        assert saver_instance.archive_queue.qsize() == 2
        item1 = await saver_instance.archive_queue.get()
        assert item1 == ("/path/pre_fire_chunk_1.mp4", event_dir)
        item2 = await saver_instance.archive_queue.get()
        assert item2 == ("/path/active_chunk_2.mp4", event_dir)

        # 3. Verify all chunks (active and post-fire) were written directly to the event directory
        assert write_blocking_mock.call_count == 3
        expected_calls = [
            call(frame, event_dir),  # From _finalize_active_chunk_async
            call(frame, event_dir),  # From _save_post_fire_chunks_async
            call(frame, event_dir),  # From _save_post_fire_chunks_async
        ]
        assert write_blocking_mock.call_args_list == expected_calls


@pytest.mark.asyncio
async def test_memory_mode_fire_handler_flushes_and_switches_mode(
    memory_mode_saver: tuple[AsyncVideoChunkSaver, MagicMock, MagicMock, MagicMock],
) -> None:
    """Tests that the memory mode fire handler flushes, switches mode, and saves post-fire chunks."""
    saver, flush_mock, write_mock, mock_makedirs = memory_mode_saver

    # --- Setup ---
    # Populate the memory buffer and assert the initial state
    pre_fire_frames = [np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(TEST_NUM_PRE_FIRE_FRAMES)]
    for frame in pre_fire_frames:
        saver(frame)
    assert saver.memory_buffer is not None and len(saver.memory_buffer) == TEST_NUM_PRE_FIRE_FRAMES
    assert saver.__call__.__func__.__name__ == "_add_frame_to_memory_buffer"

    # Pre-populate the frame queue for the post-fire recording part
    post_fire_frames = [np.ones((1, 1, 3), dtype=np.uint8) for _ in range(TEST_NUM_POST_FIRE_FRAMES_TO_QUEUE)]
    for frame in post_fire_frames:
        await saver.frame_queue.put(frame)

    # --- Execution ---
    await saver._handle_fire_event_async()

    # --- Assertions ---
    # 1. Assert pre-fire buffer was flushed with the correct frames
    flush_mock.assert_called_once()
    flushed_frames = flush_mock.call_args[0][0]
    assert len(flushed_frames) == TEST_NUM_PRE_FIRE_FRAMES
    assert saver.memory_buffer is not None and len(saver.memory_buffer) == 0

    # 2. Assert the saver's mode was switched to queue frames for disk writing
    assert saver.__call__.__func__.__name__ == "_queue_frame"

    # 3. Assert post-fire chunks were written (by checking the mock)
    assert write_mock.call_count == TEST_NUM_POST_FIRE_FRAMES_TO_QUEUE

    # 4. Assert post-fire chunks were written directly to the event directory
    # The last call to makedirs is the one for the event directory
    event_dir = mock_makedirs.call_args_list[-1][0][0]
    expected_calls = [call(frame, event_dir) for frame in post_fire_frames]
    write_mock.assert_has_calls(expected_calls)


@pytest.mark.asyncio
async def test_duration_mode_records_until_fire_is_extinguished(
    duration_mode_saver: tuple[AsyncVideoChunkSaver, MagicMock, MagicMock],
) -> None:
    """Tests that `record_during_fire` mode records until the signal is set."""
    saver, write_mock, mock_makedirs = duration_mode_saver

    # --- Setup ---
    # Simulate the fire ending after 2 frames have been processed
    is_extinguished_side_effect = [False] * DURING_FIRE_FRAMES + [True]
    saver.signal_handler.is_fire_extinguished = MagicMock(side_effect=is_extinguished_side_effect)

    # Mock _write_frame_blocking to return a path every time to count chunks
    write_mock.side_effect = lambda frame, target_dir: f"/path/{target_dir}/{frame.sum()}.mp4"

    # Queue enough frames for the "during fire" and "after fire" loops
    total_frames_needed = DURING_FIRE_FRAMES + FINAL_CHUNKS_TO_KEEP
    frames = [np.ones((1, 1, 3), dtype=np.uint8) * (i + 1) for i in range(total_frames_needed)]
    for frame in frames:
        await saver.frame_queue.put(frame)

    # --- Execution ---
    await saver._handle_fire_event_async()

    # --- Assertions ---
    # 1. Verify that write was called for all frames
    assert write_mock.call_count == total_frames_needed

    # 2. Verify all calls were made with the correct event directory
    assert mock_makedirs.call_count == 2  # Once for root, once for event
    # The last call to makedirs is the one for the event directory
    event_dir = mock_makedirs.call_args_list[-1][0][0]
    expected_calls = [call(frame, event_dir) for frame in frames]
    assert write_mock.call_args_list == expected_calls

    # 3. Verify the extinguished signal was checked the correct number of times
    # (once per frame during the fire, plus the final check that returns True)
    assert saver.signal_handler.is_fire_extinguished.call_count == DURING_FIRE_FRAMES + 1
