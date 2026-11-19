# tests/test_stream.py

"""Unit tests for the video streaming module (is_goat_burning/stream.py).

This module tests the YouTube URL resolution and the VideoStreamer class,
ensuring they handle successes, failures, and different backend configurations
correctly. Mocks are used to isolate the tests from network and hardware
dependencies.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
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


# --- yt-dlp log routing (#193) ---
SECRET_URL = "https://x.googlevideo.com/videoplayback?sig=SECRET"


def test_ytdlp_options_route_messages_to_app_logger_at_debug(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every yt-dlp message level lands in the app logger at DEBUG with URLs masked."""
    import is_goat_burning.stream as stream_module

    # The app logger does not propagate, so attach caplog's handler directly.
    monkeypatch.setattr(stream_module.logger, "level", logging.DEBUG)
    stream_module.logger.addHandler(caplog.handler)
    caplog.set_level("DEBUG")
    ytdlp_logger = YouTubeStream(TEST_URL).ytdlp_options["logger"]

    try:
        for method in ("debug", "info", "warning", "error"):
            getattr(ytdlp_logger, method)(f"msg {method} {SECRET_URL} end")
    finally:
        stream_module.logger.removeHandler(caplog.handler)

    records = [r for r in caplog.records if "msg " in r.getMessage()]
    assert len(records) == 4
    assert {r.levelname for r in records} == {"DEBUG"}
    for record in records:
        assert "<url>" in record.getMessage()
        assert "SECRET" not in record.getMessage()


def test_real_ytdlp_writes_nothing_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    """A real YoutubeDL built with the app's options never prints warnings/errors itself."""
    import yt_dlp

    ydl = yt_dlp.YoutubeDL(YouTubeStream(TEST_URL).ytdlp_options)
    ydl.report_warning("x warning")
    ydl.to_stderr("y stderr")
    with contextlib.suppress(DownloadError):
        ydl.report_error("y error")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


# --- Capture release serialized after in-flight reads (#201) ---
WAIT_SECONDS = 5.0


class _BlockingCapture:
    """Fake capture whose read()/grab() block on an event; records a release() during a read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.unblock = threading.Event()
        self.read_started = threading.Event()
        self.released = threading.Event()
        self.in_read = False
        self.overlap = False
        self.release_calls = 0

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return True

    def get(self, prop: int) -> float:
        return TEST_FPS if prop == cv2.CAP_PROP_FPS else float(TINY_FRAME_WIDTH)

    def _block(self) -> None:
        with self._lock:
            self.in_read = True
        self.read_started.set()
        self.unblock.wait(timeout=WAIT_SECONDS)
        with self._lock:
            self.in_read = False

    def read(self) -> tuple[bool, None]:
        self._block()
        return False, None

    def grab(self) -> bool:
        self._block()
        return False

    def retrieve(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        with self._lock:
            if self.in_read:
                self.overlap = True
            self.release_calls += 1
        self.released.set()


async def _streamer_with_blocked_read(
    mock_cv2_video_capture: MagicMock, mock_settings: MagicMock, sampled: bool
) -> tuple[VideoStreamer, _BlockingCapture]:
    """Creates a streamer, starts consuming frames until a read blocks, then cancels the consumer."""
    mock_settings.cuda = False
    mock_settings.open_cl = False
    cap = _BlockingCapture()
    mock_cv2_video_capture.return_value = cap
    streamer = await VideoStreamer.create(url=TEST_STREAM_URL)
    streamer.skips_decode = sampled

    async def consume() -> None:
        async for _ in streamer.frames(every=3 if sampled else 1):
            pass

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(cap.read_started.wait, WAIT_SECONDS)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return streamer, cap


@pytest.mark.asyncio
@pytest.mark.parametrize("sampled", [False, True], ids=["read", "sampled_grab"])
async def test_stop_releases_capture_only_after_in_flight_read_returns(
    sampled: bool,
    mock_cv2_video_capture: MagicMock,
    mock_settings: MagicMock,
    mock_video_saver: MagicMock,
) -> None:
    """A cancelled consumer leaves a read in its worker thread; stop() must not release under it."""
    streamer, cap = await _streamer_with_blocked_read(mock_cv2_video_capture, mock_settings, sampled)
    try:
        stop_task = asyncio.create_task(streamer.stop())
        await asyncio.sleep(0.05)
        assert cap.release_calls == 0  # waits for the in-flight read

        cap.unblock.set()
        await asyncio.wait_for(stop_task, WAIT_SECONDS)
    finally:
        cap.unblock.set()

    assert cap.release_calls == 1
    assert not cap.overlap
    mock_video_saver.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_times_out_on_blocked_read_and_still_stops_saver(
    mock_cv2_video_capture: MagicMock,
    mock_settings: MagicMock,
    mock_video_saver: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A read that never returns: stop() gives up after the timeout, warns, stops the saver, release is deferred."""
    import is_goat_burning.stream as stream_module

    monkeypatch.setattr(stream_module, "_RELEASE_TIMEOUT_SECONDS", 0.1)
    # The app logger does not propagate and is CRITICAL in tests: lower it (setLevel clears the level cache) and
    # attach caplog's handler directly.
    caplog.set_level(logging.DEBUG, logger=stream_module.logger.name)
    streamer, cap = await _streamer_with_blocked_read(mock_cv2_video_capture, mock_settings, sampled=False)
    stream_module.logger.addHandler(caplog.handler)
    try:
        started = time.monotonic()
        await asyncio.wait_for(streamer.stop(), WAIT_SECONDS)
        elapsed = time.monotonic() - started

        assert elapsed < 1.0
        assert cap.release_calls == 0
        mock_video_saver.stop.assert_awaited_once()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "release" in r.getMessage()]
        assert len(warnings) == 1
    finally:
        stream_module.logger.removeHandler(caplog.handler)
        cap.unblock.set()

    # The deferred release runs on the capture's worker once the read finally returns.
    assert await asyncio.to_thread(cap.released.wait, WAIT_SECONDS)
    assert cap.release_calls == 1
    assert not cap.overlap


@pytest.mark.asyncio
async def test_stop_twice_releases_once(
    mock_cv2_video_capture: MagicMock,
    mock_settings: MagicMock,
    mock_video_saver: MagicMock,  # noqa: ARG001
) -> None:
    """stop() is idempotent: the second call neither releases again nor raises."""
    mock_settings.cuda = False
    mock_settings.open_cl = False
    cap = _BlockingCapture()
    mock_cv2_video_capture.return_value = cap
    streamer = await VideoStreamer.create(url=TEST_STREAM_URL)

    await asyncio.wait_for(streamer.stop(), WAIT_SECONDS)
    await asyncio.wait_for(streamer.stop(), WAIT_SECONDS)

    assert cap.release_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("read_returns", [True, False], ids=["read_returns", "read_stays_blocked"])
async def test_stop_in_finally_of_cancelled_consumer(
    read_returns: bool,
    mock_cv2_video_capture: MagicMock,
    mock_settings: MagicMock,
    mock_video_saver: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Production path: stop() runs in the `finally` of the cancelled task that consumed frames (detector_core)."""
    import is_goat_burning.stream as stream_module

    monkeypatch.setattr(stream_module, "_RELEASE_TIMEOUT_SECONDS", 0.2)
    caplog.set_level(logging.DEBUG, logger=stream_module.logger.name)
    mock_settings.cuda = False
    mock_settings.open_cl = False
    cap = _BlockingCapture()
    mock_cv2_video_capture.return_value = cap
    streamer = await VideoStreamer.create(url=TEST_STREAM_URL)

    async def consume() -> None:
        try:
            async for _ in streamer.frames():
                pass
        finally:
            await streamer.stop()

    stream_module.logger.addHandler(caplog.handler)
    try:
        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(cap.read_started.wait, WAIT_SECONDS)
        task.cancel()
        await asyncio.sleep(0.05)
        assert cap.release_calls == 0  # queued behind the in-flight read
        if read_returns:
            cap.unblock.set()
        started = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, WAIT_SECONDS)
        assert time.monotonic() - started < 1.0
        mock_video_saver.stop.assert_awaited_once()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "release" in r.getMessage()]
        assert len(warnings) == (0 if read_returns else 1)
    finally:
        stream_module.logger.removeHandler(caplog.handler)
        cap.unblock.set()

    assert await asyncio.to_thread(cap.released.wait, WAIT_SECONDS)
    assert cap.release_calls == 1
    assert not cap.overlap
