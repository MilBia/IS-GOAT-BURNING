"""Provides the core video streaming and processing pipeline.

This module replaces the previous `vidgear`-based implementation with a more
direct pipeline using `yt-dlp` to resolve stream URLs and `OpenCV` for
frame grabbing and processing.
"""

import asyncio
from collections.abc import AsyncGenerator
from functools import partial
from typing import Any
from typing import ClassVar
from typing import Literal

import cv2
import numpy as np
import yt_dlp

from is_goat_burning.config import settings
from is_goat_burning.logger import get_logger
from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver

logger = get_logger("Stream")

# --- Constants ---
DEFAULT_FRAMERATE = 30.0
Backend = Literal["cpu", "cuda", "opencl"]


class YouTubeStream:
    """Resolves a YouTube URL to a direct, playable video stream URL."""

    YTDLP_OPTIONS: ClassVar[dict[str, Any]] = {
        "format": "bestvideo/best",  # TODO: Add option to chose an format in .env file
        "quiet": True,
    }

    def __init__(self, url: str) -> None:
        """Initializes the YouTubeStream resolver.

        Args:
            url: The public URL of the YouTube video or live stream.
        """
        self.url = url

    async def resolve_url(self) -> str:
        """Resolves the YouTube URL to a direct, playable video stream URL.

        This method uses `yt-dlp` to extract the manifest URL for the best
        available video stream. It runs the blocking I/O operation in a thread
        pool to avoid stalling the asyncio event loop.

        Returns:
            The direct URL to the video stream.

        Raises:
            ValueError: If the stream URL cannot be resolved.
        """
        loop = asyncio.get_running_loop()
        try:
            # Use functools.partial to pass arguments to the executor function
            extractor = partial(yt_dlp.YoutubeDL, self.YTDLP_OPTIONS)
            with await loop.run_in_executor(None, extractor) as ydl:
                info = await loop.run_in_executor(None, ydl.extract_info, self.url, False)
                if not info or "url" not in info:
                    raise ValueError("Could not extract stream URL.")
                logger.info(f"Successfully resolved stream URL for {self.url}")
                return info["url"]
        except Exception as e:
            logger.error(f"Failed to resolve stream URL for {self.url}: {e}")
            raise ValueError("Failed to resolve stream URL.") from e


class VideoStreamer:
    """Opens a video stream and yields frames asynchronously.

    This class wraps `cv2.VideoCapture` to provide a non-blocking, asynchronous
    interface for reading video frames. It handles backend-specific frame
    preparation for CPU, CUDA, and OpenCL processing.

    Attributes:
        url (str): The direct video stream URL.
        cap (cv2.VideoCapture): The underlying OpenCV VideoCapture object.
        video_saver (AsyncVideoChunkSaver): The instance for saving video chunks.
        backend (Backend): The determined processing backend ("cpu", "cuda", "opencl").
        framerate (float): The framerate of the stream.
        frame_shape (tuple[int, int]): The (width, height) of the video frames.
    """

    def __init__(self, url: str) -> None:
        """Initializes the VideoStreamer.

        Args:
            url: The direct, playable video stream URL.

        Raises:
            RuntimeError: If the video stream cannot be opened.
        """
        self.url = url
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video stream at {url}")

        self.framerate = self.cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FRAMERATE
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_shape = (width, height)
        self._running = False

        # Determine backend
        self.backend: Backend = "cpu"
        if settings.cuda:
            self.backend = "cuda"
        elif settings.open_cl:
            self.backend = "opencl"

        # Initialize video saver
        self.video_saver = AsyncVideoChunkSaver(
            fps=self.framerate,
            enabled=settings.video.save_video_chunks,
            output_dir=settings.video.video_output_directory,
            chunk_length_seconds=settings.video.video_chunk_length_seconds,
            max_chunks=settings.video.max_video_chunks,
            chunks_to_keep_after_fire=settings.video.chunks_to_keep_after_fire,
        )
        self.video_saver.start()
        logger.info(f"VideoStreamer initialized for backend '{self.backend}' at {self.framerate} FPS.")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray | cv2.UMat | cv2.cuda.GpuMat:
        """Prepares a raw frame for the selected processing backend.

        - For CUDA, uploads the frame to the GPU.
        - For OpenCL, converts the frame to a UMat.
        - For CPU, returns the frame as is.

        Args:
            frame: The raw NumPy ndarray frame from the video capture.

        Returns:
            The frame in the format expected by the corresponding FireDetector.
        """
        if self.backend == "cuda":
            gpu_frame = cv2.cuda.GpuMat()
            gpu_frame.upload(frame)
            return gpu_frame
        if self.backend == "opencl":
            return cv2.UMat(frame)
        return frame

    async def frames(self) -> AsyncGenerator[np.ndarray | cv2.UMat | cv2.cuda.GpuMat, None]:
        """An async generator that yields processed frames from the stream.

        This is the main loop that reads from the camera, queues the raw frame
        for saving, processes it for the target backend, and yields it.
        """
        loop = asyncio.get_running_loop()
        self._running = True
        while self._running:
            # Run the blocking cap.read() in an executor
            grabbed, frame = await loop.run_in_executor(None, self.cap.read)

            if not grabbed:
                logger.warning("Stream ended or frame could not be grabbed.")
                break

            # Queue the raw frame for the video saver
            self.video_saver(frame)

            # Process and yield the frame for the detector
            processed_frame = self._process_frame(frame)
            yield processed_frame

    async def stop(self) -> None:
        """Stops the frame grabbing loop and cleans up resources."""
        logger.info("Stopping video stream...")
        self._running = False
        if self.cap.isOpened():
            self.cap.release()
        await self.video_saver.stop()
        logger.info("Video stream stopped.")
