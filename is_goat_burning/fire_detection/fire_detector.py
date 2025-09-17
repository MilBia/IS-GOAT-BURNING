"""The main fire detector class orchestrating video streaming and analysis.

This module contains the `YTCamGearFireDetector` class, which is the primary
entry point for the fire detection logic. It uses `vidgear` to stream video
from a YouTube source, processes frames at a configurable rate, and uses a
selected detector implementation (CPU, CUDA, or OpenCL) to check for fire.
"""

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from is_goat_burning.fire_detection.cam_gear import YTCamGear
from is_goat_burning.fire_detection.detectors import create_fire_detector
from is_goat_burning.fire_detection.signal_handler import SignalHandler

# --- Constants ---
# Default HSV color range for fire detection, optimized for yellow/orange hues.
# These values can be overridden during the detector's initialization.
DEFAULT_LOWER_HSV_FIRE = np.array([18, 50, 50], dtype="uint8")
DEFAULT_UPPER_HSV_FIRE = np.array([35, 255, 255], dtype="uint8")


class YTCamGearFireDetector:
    """Orchestrates video streaming and fire detection.

    This class wraps `vidgear.YTCamGear` to handle video stream setup from a
    source URL. It provides an asynchronous frame generator that can be
    throttled using the `checks_per_second` parameter. For each relevant frame,
    it invokes a `FireDetector` instance to perform the actual analysis.

    Attributes:
        on_fire_action (Callable): A callback function to be executed when fire
            is detected.
        video_output (bool): If True, displays the annotated video stream in an
            OpenCV window.
        signal_handler (SignalHandler): A singleton instance to handle global
            events like fire detection signals.
        lower_hsv (np.ndarray): The lower bound of the HSV color range for fire.
        upper_hsv (np.ndarray): The upper bound of the HSV color range for fire.
        stream (YTCamGear): The underlying VidGear video stream instance.
        fire_detector (FireDetector): The detector instance (CPU, CUDA, or
            OpenCL) used for frame analysis.
        frame_generator (AsyncGenerator): The asynchronous generator that
            yields frames to be processed.
    """

    DEFAULT_OPTIONS: dict[str, Any] = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS": 30}

    def __init__(
        self,
        src: str,
        on_fire_action: Callable[[], Any],
        threshold: float = 0.05,
        logging: bool = False,
        video_output: bool = False,
        checks_per_second: float | None = None,
        lower_hsv: np.ndarray | None = None,
        upper_hsv: np.ndarray | None = None,
        **yt_cam_gear_options: Any,
    ) -> None:
        """Initializes the YTCamGearFireDetector.

        Args:
            src: The source URL of the video stream (e.g., a YouTube URL).
            on_fire_action: An awaitable callback to execute on fire detection.
            threshold: The percentage of the frame's pixels that must match
                the fire color to trigger a detection.
            logging: Enables or disables VidGear's internal logging.
            video_output: If True, displays a window with the video feed and
                detection mask.
            checks_per_second: The number of frames per second to analyze. If
                None, every frame is analyzed.
            lower_hsv: An optional override for the lower HSV fire color bound.
            upper_hsv: An optional override for the upper HSV fire color bound.
            **yt_cam_gear_options: Additional keyword arguments passed directly
                to the `vidgear.YTCamGear` constructor.
        """
        self.on_fire_action = on_fire_action
        self.video_output = video_output
        self.signal_handler = SignalHandler()
        self.lower_hsv = lower_hsv if lower_hsv is not None else DEFAULT_LOWER_HSV_FIRE
        self.upper_hsv = upper_hsv if upper_hsv is not None else DEFAULT_UPPER_HSV_FIRE
        options = {**self.DEFAULT_OPTIONS, **yt_cam_gear_options}
        self.stream = YTCamGear(source=src, stream_mode=True, logging=logging, **options)
        fire_threshold = int(self.stream.frame.shape[0] * self.stream.frame.shape[1] * threshold)
        self.fire_detector = create_fire_detector(
            margin=fire_threshold,
            lower=self.lower_hsv,
            upper=self.upper_hsv,
        )
        self.frames_between_step: float = 0.0
        self.check_iterator: AsyncGenerator[bool, None] | None = None

        if checks_per_second and checks_per_second < self.stream.framerate > 0:
            self.frames_between_step = self.stream.framerate / checks_per_second
            self.check_iterator = self.checkout_generator()
            self.frame_generator = self._frame_gen_with_iterator
        else:
            self.frame_generator = self._frame_gen

    async def checkout_generator(self) -> AsyncGenerator[bool, None]:
        """Yields True for frames that should be processed.

        This asynchronous generator implements the logic for throttling frame
        analysis based on the `checks_per_second` setting.

        Yields:
            True if the current frame should be analyzed, False otherwise.
        """
        frame_counter: float = 0.0
        while True:
            frame_counter += 1.0
            if frame_counter >= self.frames_between_step:
                yield True
                frame_counter -= self.frames_between_step
            else:
                yield False

    async def _frame_gen(self) -> AsyncGenerator[np.ndarray | cv2.UMat | cv2.cuda.GpuMat, None]:
        """An async generator that yields all frames from the stream."""
        async for frame in self.stream.read():
            if frame is None:
                break
            yield frame

    async def _frame_gen_with_iterator(self) -> AsyncGenerator[np.ndarray | cv2.UMat | cv2.cuda.GpuMat, None]:
        """An async generator that yields frames filtered by the iterator."""
        assert self.check_iterator is not None
        async for frame in self.stream.read():
            if frame is None:
                break
            if await anext(self.check_iterator):
                yield frame

    async def __call__(self) -> None:
        """Starts the fire detection loop.

        This method continuously reads frames from the configured frame
        generator, passes them to the detector, and triggers the on-fire action
        and signal if fire is detected. It also handles displaying the video
        output if enabled.
        """
        try:
            loop = asyncio.get_running_loop()
            async for frame in self.frame_generator():
                fire, annotated_frame = await loop.run_in_executor(None, self.fire_detector.detect, frame)
                if fire:
                    self.signal_handler.fire_detected()
                    await self.on_fire_action()

                if self.video_output:
                    cv2.imshow("output", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if self.video_output:
                cv2.destroyAllWindows()
            self.stream.stop()
