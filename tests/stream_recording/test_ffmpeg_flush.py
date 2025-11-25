import subprocess
from collections import deque
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver


@pytest.fixture
def saver():
    saver = AsyncVideoChunkSaver(
        enabled=True,
        output_dir="/tmp/test_output",
        chunk_length_seconds=10,
        max_chunks=5,
        chunks_to_keep_after_fire=2,
        fps=30.0,
    )
    return saver


def test_flush_buffer_to_disk_ffmpeg_success(saver):
    """Test that ffmpeg is called with correct arguments and data is piped."""
    frames = deque([b"frame1", b"frame2", b"frame3"])
    output_path = "/tmp/test_output/test.mp4"

    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        saver._flush_buffer_to_disk_ffmpeg(frames, output_path)

        # Verify ffmpeg command
        expected_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-r",
            "30.0",
            "-i",
            "-",
            "-c",
            "copy",
            output_path,
        ]
        mock_popen.assert_called_once_with(
            expected_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

        # Verify data written to stdin
        assert mock_process.stdin.write.call_count == 3
        mock_process.stdin.write.assert_any_call(b"frame1")
        mock_process.stdin.write.assert_any_call(b"frame2")
        mock_process.stdin.write.assert_any_call(b"frame3")





def test_flush_buffer_to_disk_blocking_fallback(saver):
    """Test fallback to OpenCV if ffmpeg fails (simulated by exception)."""
    frames = deque([b"frame1"])
    output_path = "/tmp/test_output/test.mp4"

    # Patch the class method instead of the instance method because of __slots__
    with patch("is_goat_burning.stream_recording.save_stream_to_file.AsyncVideoChunkSaver._flush_buffer_to_disk_ffmpeg", side_effect=OSError("ffmpeg not found")):
        with patch("cv2.VideoWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer.isOpened.return_value = True
            mock_writer_cls.return_value = mock_writer
            
            # Mock imdecode to return a dummy frame
            with patch("cv2.imdecode", return_value=MagicMock(shape=(100, 100, 3))):
                 saver._flush_buffer_to_disk_blocking(frames, output_path)

            # Verify that OpenCV writer was used
            mock_writer_cls.assert_called()
