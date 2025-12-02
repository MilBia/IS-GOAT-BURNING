# tests/test_stream.py

"""Unit tests for the video streaming module (is_goat_burning/stream.py).

This module tests the YouTube URL resolution and the VideoStreamer class,
ensuring they handle successes, failures, and different backend configurations
correctly. Mocks are used to isolate the tests from network and hardware
dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import call

import cv2
import numpy as np
import pytest
from pytest_mock import MockerFixture
from yt_dlp.utils import DownloadError

from is_goat_burning.stream import VideoStreamer
from is_goat_burning.stream import YouTubeStream

# --- Test Constants ---
TEST_URL = "https://youtube.com/watch?v=test"
TEST_STREAM_URL = "https://some.stream/url"
TEST_FPS = 30.0
TEST_WIDTH = 1920
TEST_HEIGHT = 1080
SMALL_TEST_WIDTH = 100
SMALL_TEST_HEIGHT = 100
TINY_FRAME_WIDTH = 1
TINY_FRAME_HEIGHT = 1
TEST_FRAME_SHAPE = (10, 10, 3)
NUM_TEST_FRAMES = 2

# ==============================================================================
# == Tests for YouTubeStream
# ==============================================================================


@pytest.fixture
def mock_ytdlp(mocker: MockerFixture) -> MagicMock:
    """Fixture to mock the yt_dlp.YoutubeDL class."""
    return mocker.patch("yt_dlp.YoutubeDL")


@pytest.mark.asyncio
async def test_resolve_url_succeeds_on_first_attempt(mock_ytdlp: MagicMock) -> None:
    """
    Arrange: Mock yt-dlp to successfully return stream info.
    Act: Call resolve_url.
    Assert: The correct stream URL is returned without any fallback.
    """
    # Arrange
    mock_ytdlp.return_value.__enter__.return_value.extract_info.return_value = {"url": "https://best.stream/url"}
    resolver = YouTubeStream(url=TEST_URL)

    # Act
    resolved_url = await resolver.resolve_url()

    # Assert
    assert resolved_url == "https://best.stream/url"
    mock_ytdlp.return_value.__enter__.return_value.extract_info.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_url_uses_fallback_when_preferred_format_fails(mock_ytdlp: MagicMock) -> None:
    """
    Arrange: Mock yt-dlp to first raise a format error, then succeed.
    Act: Call resolve_url.
    Assert: The fallback stream URL is returned after the initial failure.
    """
    # Arrange
    mock_ytdlp.return_value.__enter__.return_value.extract_info.side_effect = [
        DownloadError("Requested format is not available"),
        {"url": "https://fallback.stream/url"},
    ]
    resolver = YouTubeStream(url=TEST_URL)

    # Act
    resolved_url = await resolver.resolve_url()

    # Assert
    assert resolved_url == "https://fallback.stream/url"
    assert mock_ytdlp.return_value.__enter__.return_value.extract_info.call_count == 2


@pytest.mark.asyncio
async def test_resolve_url_raises_value_error_on_download_error(mock_ytdlp: MagicMock) -> None:
    """
    Arrange: Mock yt-dlp to raise a generic DownloadError.
    Act: Call resolve_url.
    Assert: A ValueError is raised.
    """
    # Arrange
    mock_ytdlp.return_value.__enter__.return_value.extract_info.side_effect = DownloadError("Generic download error")
    resolver = YouTubeStream(url=TEST_URL)

    # Act & Assert
    with pytest.raises(ValueError, match="Failed to resolve stream URL."):
        await resolver.resolve_url()


@pytest.mark.asyncio
async def test_resolve_url_raises_value_error_when_fallback_also_fails(mock_ytdlp: MagicMock) -> None:
    """
    Arrange: Mock yt-dlp to raise DownloadError on both primary and fallback attempts.
    Act: Call resolve_url.
    Assert: A ValueError is raised.
    """
    # Arrange
    mock_ytdlp.return_value.__enter__.return_value.extract_info.side_effect = [
        DownloadError("Requested format is not available"),
        DownloadError("Fallback also failed"),
    ]
    resolver = YouTubeStream(url=TEST_URL)

    # Act & Assert
    with pytest.raises(ValueError, match="Failed to resolve stream URL."):
        await resolver.resolve_url()


# ==============================================================================
# == Tests for VideoStreamer
# ==============================================================================


@pytest.fixture
def mock_cv2_video_capture(mocker: MockerFixture) -> MagicMock:
    """Fixture to mock the cv2.VideoCapture class."""
    return mocker.patch("cv2.VideoCapture")


@pytest.fixture
def mock_settings(mocker: MockerFixture) -> MagicMock:
    """Fixture to mock the settings object in the stream module."""
    return mocker.patch("is_goat_burning.stream.settings")


@pytest.fixture
def mock_video_saver(mocker: MockerFixture) -> MagicMock:
    """
    Mocks the AsyncVideoChunkSaver class to prevent real task creation
    and returns the mock instance for assertion.
    """
    mock_saver_class = mocker.patch("is_goat_burning.stream.AsyncVideoChunkSaver", autospec=True)
    return mock_saver_class.return_value


@pytest.mark.asyncio
async def test_create_video_streamer_succeeds_with_valid_url(
    mock_cv2_video_capture: MagicMock,
    mock_settings: MagicMock,  # noqa: ARG001
    mock_video_saver: MagicMock,
) -> None:
    """
    Arrange: Mock cv2.VideoCapture to return an "opened" capture object.
    Act: Call VideoStreamer.create.
    Assert: A VideoStreamer instance is created successfully and the saver is started.
    """
    # Arrange
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.get.side_effect = [TEST_FPS, TEST_WIDTH, TEST_HEIGHT]
    mock_cv2_video_capture.return_value = mock_capture

    # Act
    streamer = await VideoStreamer.create(url=TEST_STREAM_URL)

    # Assert
    assert isinstance(streamer, VideoStreamer)
    assert streamer.framerate == TEST_FPS
    assert streamer.frame_shape == (TEST_WIDTH, TEST_HEIGHT)
    mock_cv2_video_capture.assert_called_once_with(TEST_STREAM_URL, cv2.CAP_FFMPEG)
    mock_video_saver.start.assert_called_once()


@pytest.mark.asyncio
async def test_create_video_streamer_raises_runtime_error_on_open_failure(
    mock_cv2_video_capture: MagicMock, mock_video_saver: MagicMock
) -> None:
    """
    Arrange: Mock cv2.VideoCapture to return a "closed" capture object.
    Act: Call VideoStreamer.create.
    Assert: A RuntimeError is raised and saver is not started.
    """
    # Arrange
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = False
    mock_cv2_video_capture.return_value = mock_capture

    # Act & Assert
    with pytest.raises(RuntimeError, match="Could not open video stream"):
        await VideoStreamer.create(url=TEST_STREAM_URL)
    mock_video_saver.start.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend, cuda_setting, opencl_setting",
    [
        ("cpu", False, False),
        ("cuda", True, False),
        ("opencl", False, True),
    ],
)
async def test_process_frame_selects_correct_backend(
    backend: str,
    cuda_setting: bool,
    opencl_setting: bool,
    mock_cv2_video_capture: MagicMock,
    mock_settings: MagicMock,
    mock_video_saver: MagicMock,  # noqa: ARG001
    mocker: MockerFixture,
) -> None:
    """
    Arrange: Configure settings for a specific backend (CPU, CUDA, or OpenCL).
    Act: Call the internal _process_frame method.
    Assert: The returned frame is of the correct type for the specified backend.
    """
    # Arrange
    mock_settings.cuda = cuda_setting
    mock_settings.open_cl = opencl_setting

    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.get.side_effect = [TEST_FPS, SMALL_TEST_WIDTH, SMALL_TEST_HEIGHT]
    mock_cv2_video_capture.return_value = mock_capture

    streamer = await VideoStreamer.create(url=TEST_STREAM_URL)
    raw_frame = np.zeros(TEST_FRAME_SHAPE, dtype=np.uint8)

    # Act & Assert
    if backend == "cpu":
        processed_frame = streamer._process_frame(raw_frame)
        assert isinstance(processed_frame, np.ndarray)
    elif backend == "cuda":
        mock_gpumat = mocker.patch("cv2.cuda.GpuMat", return_value=MagicMock())
        processed_frame = streamer._process_frame(raw_frame)
        mock_gpumat.assert_called_once()
        mock_gpumat.return_value.upload.assert_called_once_with(raw_frame)
        assert isinstance(processed_frame, MagicMock)
    elif backend == "opencl":
        mock_umat = mocker.patch("cv2.UMat", return_value=MagicMock())
        processed_frame = streamer._process_frame(raw_frame)
        mock_umat.assert_called_once_with(raw_frame)
        assert isinstance(processed_frame, MagicMock)


@pytest.mark.asyncio
async def test_frames_generator_yields_frames_and_stops(
    mock_cv2_video_capture: MagicMock,
    mock_settings: MagicMock,
    mock_video_saver: MagicMock,
) -> None:
    """
    Arrange: Mock VideoCapture to return a few frames then indicate the stream has ended.
    Act: Iterate through the frames() async generator.
    Assert: The correct number of frames are yielded and the loop terminates gracefully.
    """
    # Arrange
    frame1 = np.ones((TINY_FRAME_HEIGHT, TINY_FRAME_WIDTH, 3), dtype=np.uint8)
    frame2 = np.ones((TINY_FRAME_HEIGHT, TINY_FRAME_WIDTH, 3), dtype=np.uint8) * 2

    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.get.side_effect = [TEST_FPS, TINY_FRAME_WIDTH, TINY_FRAME_HEIGHT]
    mock_capture.read.side_effect = [(True, frame1), (True, frame2), (False, None)]
    mock_cv2_video_capture.return_value = mock_capture

    mock_settings.cuda = False
    mock_settings.open_cl = False

    streamer = await VideoStreamer.create(url=TEST_STREAM_URL)

    # Act
    yielded_frames = [frame async for frame in streamer.frames()]
    await streamer.stop()

    # Assert
    assert len(yielded_frames) == NUM_TEST_FRAMES
    np.testing.assert_array_equal(yielded_frames[0], frame1)
    np.testing.assert_array_equal(yielded_frames[1], frame2)

    # Verify that the raw frames were passed to the video saver
    mock_video_saver.assert_has_calls([call(frame1), call(frame2)])
    assert mock_video_saver.call_count == NUM_TEST_FRAMES

    mock_video_saver.stop.assert_awaited_once()
