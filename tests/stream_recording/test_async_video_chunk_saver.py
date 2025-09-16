"""Unit tests for the AsyncVideoChunkSaver class.

These tests cover the synchronous (blocking) logic of file I/O and the
asynchronous orchestration of the fire event handling mechanism.
"""

from collections.abc import Generator
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

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
    """Tests the async fire event orchestration.

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
