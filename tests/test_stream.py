# tests/test_stream.py

"""Unit tests for the video streaming module (is_goat_burning/stream.py).

This module tests the YouTube URL resolution and the VideoStreamer class,
ensuring they handle successes, failures, and different backend configurations
correctly. Mocks are used to isolate the tests from network and hardware
dependencies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import call

import numpy as np
import pytest
from pytest_mock import MockerFixture
from yt_dlp.utils import DownloadError

from is_goat_burning.stream import VideoStreamer
from is_goat_burning.stream import YouTubeStream

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
    resolver = YouTubeStream(url="https://youtube.com/watch?v=test")

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
    resolver = YouTubeStream(url="https://youtube.com/watch?v=test")

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
    resolver = YouTubeStream(url="https://youtube.com/watch?v=test")

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
    resolver = YouTubeStream(url="https://youtube.com/watch?v=test")

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


@pytest.mark.asyncio
async def test_create_video_streamer_succeeds_with_valid_url(mock_cv2_video_capture: MagicMock) -> None:
    """
    Arrange: Mock cv2.VideoCapture to return an "opened" capture object.
    Act: Call VideoStreamer.create.
    Assert: A VideoStreamer instance is created successfully.
    """
    # Arrange
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.get.side_effect = [30.0, 1920, 1080]  # FPS, width, height
    mock_cv2_video_capture.return_value = mock_capture

    # Act
    streamer = await VideoStreamer.create(url="https://some.stream/url")

    # Assert
    assert isinstance(streamer, VideoStreamer)
    assert streamer.framerate == 30.0
    assert streamer.frame_shape == (1920, 1080)
    mock_cv2_video_capture.assert_called_once_with("https://some.stream/url")


@pytest.mark.asyncio
async def test_create_video_streamer_raises_runtime_error_on_open_failure(mock_cv2_video_capture: MagicMock) -> None:
    """
    Arrange: Mock cv2.VideoCapture to return a "closed" capture object.
    Act: Call VideoStreamer.create.
    Assert: A RuntimeError is raised.
    """
    # Arrange
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = False
    mock_cv2_video_capture.return_value = mock_capture

    # Act & Assert
    with pytest.raises(RuntimeError, match="Could not open video stream"):
        await VideoStreamer.create(url="https://some.stream/url")


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
    mock_capture.get.side_effect = [30.0, 100, 100]
    mock_cv2_video_capture.return_value = mock_capture

    streamer = await VideoStreamer.create(url="dummy")
    raw_frame = np.zeros((10, 10, 3), dtype=np.uint8)

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
async def test_frames_generator_yields_frames_and_stops(mock_cv2_video_capture: MagicMock, mock_settings: MagicMock) -> None:
    """
    Arrange: Mock VideoCapture to return a few frames then indicate the stream has ended.
    Act: Iterate through the frames() async generator.
    Assert: The correct number of frames are yielded and the loop terminates gracefully.
    """
    # Arrange
    frame1 = np.ones((1, 1, 3), dtype=np.uint8)
    frame2 = np.ones((1, 1, 3), dtype=np.uint8) * 2

    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.get.side_effect = [30.0, 1, 1]
    mock_capture.read.side_effect = [(True, frame1), (True, frame2), (False, None)]
    mock_cv2_video_capture.return_value = mock_capture

    mock_settings.cuda = False
    mock_settings.open_cl = False

    streamer = await VideoStreamer.create(url="dummy")
    # FIX: Use AsyncMock for objects with async methods
    streamer.video_saver = AsyncMock()

    # Act
    yielded_frames = [frame async for frame in streamer.frames()]
    await streamer.stop()

    # Assert
    assert len(yielded_frames) == 2
    np.testing.assert_array_equal(yielded_frames[0], frame1)
    np.testing.assert_array_equal(yielded_frames[1], frame2)

    # Verify that the raw frames were passed to the video saver
    streamer.video_saver.assert_has_calls([call(frame1), call(frame2)])
    assert streamer.video_saver.call_count == 2

    streamer.video_saver.stop.assert_awaited_once()
