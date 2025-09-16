from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver


@pytest.fixture
def saver():
    """
    Fixture to create a clean AsyncVideoChunkSaver instance for each test.
    `os.makedirs` is patched here to ensure the mock is active before the
    saver's __init__ captures it in a partial function.
    """
    with patch("os.makedirs") as mock_mkdir:
        saver_instance = AsyncVideoChunkSaver(
            enabled=True, output_dir="test_output", chunk_length_seconds=60, max_chunks=3, chunks_to_keep_after_fire=2, fps=30
        )
        # Yield both the saver and the mock for use in tests
        yield saver_instance, mock_mkdir


def test_write_frame_blocking_creates_and_rotates_chunks():
    """
    Tests the core synchronous logic of creating, rotating, and cleaning up video chunks.
    This test uses time mocking to be instantaneous and deterministic.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # We patch makedirs here because this test doesn't use the fixture
    with patch("os.makedirs"):
        saver = AsyncVideoChunkSaver(
            enabled=True, output_dir="test_output", chunk_length_seconds=60, max_chunks=3, chunks_to_keep_after_fire=2, fps=30
        )

    mock_writer_1, mock_writer_2, mock_writer_3, mock_writer_4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()

    # This side_effect now correctly simulates the directory growing
    listdir_side_effect = [
        [],  # Before chunk 1 is created
        ["goat-cam_f1.mp4"],  # Before chunk 2
        ["goat-cam_f1.mp4", "goat-cam_f2.mp4"],  # Before chunk 3
        ["goat-cam_f1.mp4", "goat-cam_f2.mp4", "goat-cam_f3.mp4"],  # Before chunk 4 (triggers delete)
    ]
    with (
        patch("cv2.VideoWriter", side_effect=[mock_writer_1, mock_writer_2, mock_writer_3, mock_writer_4]) as mock_vw,
        patch("time.time") as mock_time,
        patch("os.remove") as mock_remove,
        patch("os.listdir", side_effect=listdir_side_effect),
        patch("os.path.isfile", return_value=True),
    ):
        # --- Create first 3 chunks (no cleanup) ---
        mock_time.return_value = 1000.0
        saver._write_frame_blocking(frame)  # Chunk 1
        mock_time.return_value = 1061.0
        saver._write_frame_blocking(frame)  # Chunk 2
        mock_time.return_value = 1122.0
        saver._write_frame_blocking(frame)  # Chunk 3

        mock_remove.assert_not_called()
        assert mock_vw.call_count == 3

        # --- Create 4th chunk (triggers cleanup) ---
        # `max_chunks` is 3. `listdir` now returns 3 files.
        # num_to_delete = 3 - 3 + 1 = 1.
        mock_time.return_value = 1183.0
        saver._write_frame_blocking(frame)

        assert mock_vw.call_count == 4
        # Assert that cleanup was finally called, and only once.
        mock_remove.assert_called_once_with("test_output/goat-cam_f1.mp4")


@pytest.mark.asyncio
async def test_fire_event_archives_chunks_and_records_new_ones(saver):
    """
    Tests the asynchronous fire event orchestration logic directly.
    This test confirms the correct items are placed on the archive_queue.
    """
    saver_instance, mock_mkdir = saver  # Unpack from fixture

    # --- Setup ---
    saver_instance.pre_fire_buffer.extend(["/path/pre_fire_chunk_1.mp4", "/path/active_chunk_2.mp4"])
    saver_instance.current_video_path = "/path/active_chunk_2.mp4"
    saver_instance.writer = MagicMock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        await saver_instance.frame_queue.put(frame)

    write_blocking_mock = MagicMock(side_effect=["/path/active_chunk_2.mp4", "/path/post_fire_1.mp4", "/path/post_fire_2.mp4"])
    patch_path = "is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._write_frame_blocking"

    # Patch the method *before* calling the function under test, as suggested.
    with patch(patch_path, write_blocking_mock), patch.object(saver_instance.signal_handler, "reset_fire_event") as mock_reset:
        # --- Execution ---
        await saver_instance._handle_fire_event_async()

        # --- Assertions ---
        # 1. Was the event directory created?
        assert mock_mkdir.call_count == 2
        last_call_args, _ = mock_mkdir.call_args
        event_dir = last_call_args[0]
        assert "test_output/event_" in event_dir

        # 2. Check the contents of the archive queue
        assert saver_instance.archive_queue.qsize() == 4

        # 3. Verify items were queued correctly
        item1 = await saver_instance.archive_queue.get()
        assert item1 == ("/path/pre_fire_chunk_1.mp4", event_dir)
        item2 = await saver_instance.archive_queue.get()
        assert item2 == ("/path/active_chunk_2.mp4", event_dir)
        item3 = await saver_instance.archive_queue.get()
        assert item3 == ("/path/post_fire_1.mp4", event_dir)
        item4 = await saver_instance.archive_queue.get()
        assert item4 == ("/path/post_fire_2.mp4", event_dir)

        # 4. Verify state that IS handled by this function
        # The buffer's state should be unchanged by this specific method.
        assert len(saver_instance.pre_fire_buffer) == 2
        # Correctly assert that the mock was not called during execution.
        mock_reset.assert_not_called()
