import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver


@pytest.fixture
def saver():
    with patch("os.makedirs"):
        return AsyncVideoChunkSaver(
            enabled=True, output_dir="test_output", chunk_length_seconds=1, max_chunks=2, chunks_to_keep_after_fire=1, fps=30
        )


@pytest.mark.asyncio
async def test_add_frame_and_start_new_chunk(saver):
    with patch("cv2.VideoWriter") as mock_video_writer, patch("os.remove"), patch("os.listdir", return_value=[]):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        saver.start()
        saver(frame)
        for _ in range(10):  # Poll for up to 0.1 seconds
            if mock_video_writer.called:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("The mock_video_writer was not called within the timeout.")

        assert saver.writer is not None
        mock_video_writer.assert_called_once()
        saver.writer.write.assert_called_once_with(frame)

        await saver.stop()


@pytest.mark.asyncio
async def test_chunk_rotation(saver):
    mock_writer_1 = MagicMock()
    mock_writer_2 = MagicMock()
    with (
        patch("cv2.VideoWriter", side_effect=[mock_writer_1, mock_writer_2]) as mock_video_writer,
        patch("os.remove"),
        patch("os.listdir", return_value=[]),
    ):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        saver.start()

        # First frame starts a new chunk
        saver(frame)
        for _ in range(10):  # Poll for up to 0.1 seconds
            if mock_video_writer.called:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("The mock_video_writer was not called within the timeout.")

        assert mock_video_writer.call_count == 1
        first_writer_instance = saver.writer

        # Simulate time passing to trigger a new chunk
        saver.chunk_start_time = saver.chunk_start_time - 2
        saver(frame)
        for _ in range(10):  # Poll for up to 0.1 seconds
            if mock_video_writer.call_count == 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("The mock_video_writer was not called within the timeout.")

        assert mock_video_writer.call_count == 2
        assert saver.writer != first_writer_instance
        first_writer_instance.release.assert_called_once()

        await saver.stop()


def test_cleanup_old_chunks(saver):
    with (
        patch("cv2.VideoWriter"),
        patch("os.remove") as mock_remove,
        patch("os.path.isfile", return_value=True),
        patch("os.listdir", return_value=["goat-cam_f1.mp4", "goat-cam_f2.mp4", "goat-cam_f3.mp4"]),
    ):
        saver.current_video_path = "f3"

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        saver._write_frame_blocking(frame)

        assert mock_remove.call_count == 2


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_archive_chunks_on_fire_event():
    with (
        patch("cv2.VideoWriter") as mock_video_writer,
        patch("shutil.move") as mock_move,
        patch("os.makedirs") as mock_mkdir,
        patch("os.path.exists", return_value=True),
        patch("os.listdir", return_value=["f1", "f2"]),
    ):
        saver = AsyncVideoChunkSaver(
            enabled=True, output_dir="test_output", chunk_length_seconds=1, max_chunks=2, chunks_to_keep_after_fire=1, fps=30
        )

        saver.pre_fire_buffer.append("f1")
        saver.pre_fire_buffer.append("f2")
        saver.signal_handler.fire_detected_event.set()
        saver.start()

        # Add a frame to be processed
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(5):
            saver(frame)
            for _ in range(5):
                if saver.frame_queue.empty():
                    break
                await asyncio.sleep(0.5)
            else:
                pytest.fail(
                    f"{i}. The on_fire_action_handler was not called within the timeout. {mock_video_writer.call_count}"
                )

        await saver.stop()
        assert mock_mkdir.call_count == 2
        # one for pre-fire, one for active chunk, one for post-fire
        assert mock_move.call_count == 3
