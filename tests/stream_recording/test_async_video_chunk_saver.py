"""Unit tests for the AsyncVideoChunkSaver class.

These tests cover the synchronous (blocking) logic of file I/O and the
asynchronous orchestration of the fire event handling mechanism.
"""

from collections.abc import Generator
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from pytest_mock import MockerFixture

from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver


@pytest.fixture
def saver() -> Generator[tuple[AsyncVideoChunkSaver, MagicMock], None, None]:
    """Provides a clean, enabled AsyncVideoChunkSaver instance for each test.

    This fixture also patches `os.makedirs` to prevent actual directory
    creation during tests.

    Yields:
        A tuple containing the configured `AsyncVideoChunkSaver` instance and
        the mock for `os.makedirs`.
    """
    with patch("os.makedirs") as mock_mkdir:
        saver_instance = AsyncVideoChunkSaver(
            enabled=True,
            output_dir="test_output",
            chunk_length_seconds=60,
            max_chunks=3,
            chunks_to_keep_after_fire=2,
            fps=30,
        )
        yield saver_instance, mock_mkdir


def test_write_frame_blocking_creates_and_rotates_chunks() -> None:
    """Tests the synchronous logic for creating, rotating, and cleaning up chunks.

    This test uses time and filesystem mocks to deterministically verify that
    `_write_frame_blocking` creates new video files when the time limit is
    exceeded and deletes the oldest file when the `max_chunks` limit is reached.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("os.makedirs"):
        saver = AsyncVideoChunkSaver(
            enabled=True,
            output_dir="test_output",
            chunk_length_seconds=60,
            max_chunks=3,
            chunks_to_keep_after_fire=2,
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
    """Tests the async fire event orchestration for 'disk' mode.

    This test directly invokes `_handle_fire_event_async` and verifies that it
    correctly queues the pre-fire, active, and post-fire video chunks for
    archiving by placing the correct items onto the `archive_queue`.

    Args:
        saver: The fixture providing the `AsyncVideoChunkSaver` instance.
    """
    saver_instance, mock_mkdir = saver
    # --- Setup ---
    saver_instance.pre_fire_buffer.extend(["/path/pre_fire_chunk_1.mp4", "/path/active_chunk_2.mp4"])
    saver_instance.current_video_path = "/path/active_chunk_2.mp4"
    saver_instance.writer = MagicMock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
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

        # 2. Verify all expected chunks were queued for archiving
        assert saver_instance.archive_queue.qsize() == 4
        item1 = await saver_instance.archive_queue.get()
        assert item1 == ("/path/pre_fire_chunk_1.mp4", event_dir)
        item2 = await saver_instance.archive_queue.get()
        assert item2 == ("/path/active_chunk_2.mp4", event_dir)
        item3 = await saver_instance.archive_queue.get()
        assert item3 == ("/path/post_fire_1.mp4", event_dir)
        item4 = await saver_instance.archive_queue.get()
        assert item4 == ("/path/post_fire_2.mp4", event_dir)


@pytest.mark.asyncio
async def test_memory_mode_fire_handler_flushes_and_switches_mode(mocker: MockerFixture) -> None:
    """Tests that the memory mode fire handler flushes, switches mode, and saves post-fire chunks."""
    # --- Setup ---
    mock_settings = mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings")
    mock_settings.video.buffer_mode = "memory"
    mock_settings.video.chunks_to_keep_after_fire = 2

    flush_mock = mocker.patch(
        "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._flush_buffer_to_disk_blocking"
    )
    write_mock = mocker.patch(
        "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._write_frame_blocking",
        side_effect=[None, "/path/post_fire_1.mp4", "/path/post_fire_2.mp4"],
    )
    mocker.patch("os.makedirs")

    saver = AsyncVideoChunkSaver(
        enabled=True, output_dir="test_output", chunk_length_seconds=1, max_chunks=3, chunks_to_keep_after_fire=2, fps=30
    )

    # Populate the memory buffer and assert the initial state
    pre_fire_frames = [np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(5)]
    for frame in pre_fire_frames:
        saver(frame)
    assert saver.memory_buffer is not None and len(saver.memory_buffer) == 5
    # CORRECTED ASSERTION: Check the name of the underlying function.
    assert saver.__call__.__func__.__name__ == "_add_frame_to_memory_buffer"

    # Pre-populate the frame queue for the post-fire recording part
    for _ in range(3):
        await saver.frame_queue.put(np.ones((1, 1, 3), dtype=np.uint8))

    mock_archive_put = mocker.patch.object(saver.archive_queue, "put_nowait")

    # --- Execution ---
    # Directly call the handler, not the entire run loop
    await saver._handle_fire_event_async()

    # --- Assertions ---
    # 1. Assert pre-fire buffer was flushed with the correct frames
    flush_mock.assert_called_once()
    flushed_frames = flush_mock.call_args[0][0]
    assert len(flushed_frames) == 5
    assert saver.memory_buffer is not None and len(saver.memory_buffer) == 0

    # 2. Assert the saver's mode was switched to queue frames for disk writing
    assert saver.__call__.__func__.__name__ == "_queue_frame"

    # 3. Assert post-fire chunks were written (by checking the mock)
    assert write_mock.call_count == 3

    # 4. Assert the completed post-fire chunks were queued for archiving
    assert mock_archive_put.call_count == 2
    mock_archive_put.assert_any_call(("/path/post_fire_1.mp4", mocker.ANY))
    mock_archive_put.assert_any_call(("/path/post_fire_2.mp4", mocker.ANY))
