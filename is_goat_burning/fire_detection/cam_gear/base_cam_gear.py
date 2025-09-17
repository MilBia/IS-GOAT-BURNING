"""Provides a base class for custom YTCamGear implementations.

This module defines `BaseYTCamGear`, which extends `vidgear.CamGear` with
asynchronous frame reading, processing, and integration with the
`AsyncVideoChunkSaver` for recording.
"""

import asyncio
from collections.abc import AsyncGenerator
import logging as log
from typing import Any

import cv2
import numpy as np
from vidgear.gears import CamGear
from vidgear.gears.helper import logger_handler

from is_goat_burning.config import settings
from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver

logger = log.getLogger("YTCamGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class BaseYTCamGear(CamGear):
    """An extended `CamGear` with async processing and video saving.

    This base class handles the common setup for all `YTCamGear` variants,
    including initializing the `AsyncVideoChunkSaver` and providing a custom
    asynchronous `read` method that processes frames in a thread pool executor.

    Attributes:
        video_saver (AsyncVideoChunkSaver): An instance to handle saving video
            chunks to disk.
        framerate (int): The framerate of the video stream.
    """

    video_saver: AsyncVideoChunkSaver

    def _pre__init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Sets required kwargs before the parent `CamGear` is initialized.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments for the `CamGear` constructor.
        """
        kwargs["THREADED_QUEUE_MODE"] = False

    def _post__init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Initializes the video saver after the parent `CamGear` is set up.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).
        """
        self.video_saver = AsyncVideoChunkSaver(
            fps=self.framerate,
            enabled=settings.video.save_video_chunks,
            output_dir=settings.video.video_output_directory,
            chunk_length_seconds=settings.video.video_chunk_length_seconds,
            max_chunks=settings.video.max_video_chunks,
            chunks_to_keep_after_fire=settings.video.chunks_to_keep_after_fire,
        )
        self.video_saver.start()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the custom `YTCamGear` instance.

        Args:
            *args: Positional arguments passed to the parent `CamGear`.
            **kwargs: Keyword arguments passed to the parent `CamGear`.
        """
        self._pre__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.framerate = self.ytv_metadata.get("fps", 30)
        self._post__init__(*args, **kwargs)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray | cv2.UMat | cv2.cuda.GpuMat:
        """A placeholder for frame processing logic in subclasses.

        This method must be implemented by subclasses to handle the specific
        processing for their backend (e.g., color space conversion on the CPU,
        GPU, etc.).

        Args:
            frame: The raw input frame read from the video stream.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError

    async def read(self) -> AsyncGenerator[np.ndarray | cv2.UMat | cv2.cuda.GpuMat, None]:
        """Asynchronously reads and processes frames from the video stream.

        This method reads frames from the stream in a blocking manner within an
        executor, queues them for saving, processes them, and then yields the
        processed frame.

        Yields:
            The processed video frame, in a format dependent on the subclass
            implementation (e.g., `np.ndarray`, `cv2.cuda.GpuMat`).
        """
        loop = asyncio.get_running_loop()
        try:
            while True:
                self._CamGear__stream_read.clear()
                # Read the next frame in an executor to avoid blocking the loop
                (grabbed, frame) = await loop.run_in_executor(None, self.stream.read)
                self._CamGear__stream_read.set()

                if not grabbed:
                    break

                # Queue the raw frame for saving
                self.video_saver(frame)

                # Process the frame in an executor
                processed_frame = await loop.run_in_executor(None, self._process_frame, frame)
                yield processed_frame
        finally:
            await self.video_saver.stop()
