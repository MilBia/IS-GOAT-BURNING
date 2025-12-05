"""The main fire detector class orchestrating video streaming and analysis.

This module contains the `StreamFireDetector` class, which is the primary
entry point for the fire detection logic. It uses the `VideoStreamer` to handle
video streaming, processes frames at a configurable rate, and uses a
selected detector implementation (CPU, CUDA, or OpenCL) to check for fire.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Callable
import time
from typing import Any

import cv2
import numpy as np

from is_goat_burning.config import settings
from is_goat_burning.fire_detection.detectors import FireDetector
from is_goat_burning.fire_detection.detectors import create_fire_detector
from is_goat_burning.fire_detection.signal_handler import SignalHandler
from is_goat_burning.logger import get_logger
from is_goat_burning.stream import VideoStreamer
from is_goat_burning.stream import YouTubeStream

# --- Constants ---
# Default HSV color range for fire detection, optimized for yellow/orange hues.
# These values can be overridden during the detector's initialization.
DEFAULT_DETECTION_THRESHOLD = 0.05
DEFAULT_LOWER_HSV_FIRE = np.array([18, 50, 50], dtype="uint8")
DEFAULT_UPPER_HSV_FIRE = np.array([35, 255, 255], dtype="uint8")

logger = get_logger("FireDetector")


class StreamFireDetector:
    """Orchestrates video streaming and fire detection.

    This class should be instantiated via the `create` async factory method.
    """

    def __init__(
        self,
        on_fire_action: Callable[[], Any],
        video_output: bool,
        checks_per_second: float | None,
    ) -> None:
        """Initializes the StreamFireDetector. Private, use `create`."""
        self.on_fire_action = on_fire_action
        self.video_output = video_output
        self.checks_per_second = checks_per_second
        self.signal_handler = SignalHandler()
        self.stream: VideoStreamer | None = None
        self.fire_detector: FireDetector | None = None
        self.fire_is_currently_detected = False
        self._potential_fire_start_time: float | None = None
        self._potential_fire_end_time: float | None = None

    @classmethod
    async def create(
        cls,
        src: str,
        on_fire_action: Callable[[], Any],
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        video_output: bool = False,
        checks_per_second: float | None = None,
        lower_hsv: np.ndarray | None = None,
        upper_hsv: np.ndarray | None = None,
    ) -> StreamFireDetector:
        """Creates and asynchronously initializes a StreamFireDetector instance.

        Args:
            src: The source URL of the video stream (e.g., a YouTube URL).
            on_fire_action: An awaitable callback to execute on fire detection.
            threshold: The percentage of the frame's pixels that must match
                the fire color to trigger a detection.
            video_output: If True, displays a window with the video feed and
                detection mask.
            checks_per_second: The number of frames per second to analyze. If
                None, every frame is analyzed.
            lower_hsv: An optional override for the lower HSV fire color bound.
            upper_hsv: An optional override for the upper HSV fire color bound.

        Returns:
            A fully initialized StreamFireDetector instance.
        """
        instance = cls(on_fire_action, video_output, checks_per_second)

        # Resolve the stream URL
        resolver = YouTubeStream(url=src)
        stream_url = await resolver.resolve_url()
        instance.stream = await VideoStreamer.create(url=stream_url)

        # This assertion helps static analysis tools understand that `instance.stream`
        # is guaranteed to be initialized before it is used below.
        assert instance.stream is not None

        # Create the appropriate detector
        lower = lower_hsv if lower_hsv is not None else DEFAULT_LOWER_HSV_FIRE
        upper = upper_hsv if upper_hsv is not None else DEFAULT_UPPER_HSV_FIRE
        fire_pixel_margin = int(instance.stream.frame_shape[0] * instance.stream.frame_shape[1] * threshold)
        instance.fire_detector = create_fire_detector(
            margin=fire_pixel_margin,
            lower=lower,
            upper=upper,
            strategy=settings.detection_strategy,
        )
        return instance

    async def _frame_generator(self) -> AsyncGenerator[np.ndarray | cv2.UMat | cv2.cuda.GpuMat, None]:
        """An async generator that yields frames, potentially throttled."""
        assert self.stream is not None
        frame_counter: float = 0.0
        frames_between_step: float = 0.0
        if self.checks_per_second and self.checks_per_second < self.stream.framerate > 0:
            frames_between_step = self.stream.framerate / self.checks_per_second

        async for frame in self.stream.frames():
            if frame is None:
                break

            if frames_between_step == 0:  # No throttling
                yield frame
                continue

            frame_counter += 1.0
            if frame_counter >= frames_between_step:
                yield frame
                frame_counter -= frames_between_step

    async def _handle_fire_detection(self, fire_in_frame: bool) -> None:
        """Handles the state logic for fire detection with debouncing.

        Args:
            fire_in_frame: A boolean indicating if fire was detected in the current frame.
        """
        now = time.monotonic()

        if fire_in_frame:
            self._potential_fire_end_time = None  # Reset any pending extinguish signal
            if self.fire_is_currently_detected:
                return  # Already in fire state, no change needed

            # Start or continue debounce for fire detection
            if self._potential_fire_start_time is None:
                self._potential_fire_start_time = now

            if now - self._potential_fire_start_time >= settings.fire_detected_debounce_seconds:
                self.fire_is_currently_detected = True
                self._potential_fire_start_time = None  # Clear timer
                self.signal_handler.fire_detected()
                await self.on_fire_action()
        else:
            self._potential_fire_start_time = None  # Reset any pending fire signal
            if not self.fire_is_currently_detected:
                return  # Already in no-fire state, no change needed

            # Start or continue debounce for fire extinguished
            if self._potential_fire_end_time is None:
                self._potential_fire_end_time = now

            if now - self._potential_fire_end_time >= settings.fire_extinguished_debounce_seconds:
                self.fire_is_currently_detected = False
                self._potential_fire_end_time = None  # Clear timer
                logger.info("Fire is no longer detected (after debounce).")
                self.signal_handler.fire_extinguished()

    async def __call__(self) -> None:
        """Starts the fire detection loop.

        This method continuously reads frames from the configured frame
        generator, passes them to the detector, and triggers the on-fire action
        and signal if fire is detected. It also handles displaying the video
        output if enabled.
        """
        assert self.stream is not None and self.fire_detector is not None

        if self.video_output:
            logger.warning(
                "VIDEO_OUTPUT is enabled. This is a debug-only feature that runs blocking GUI "
                "code in the main asyncio event loop. It will severely degrade performance and "
                "may cause other concurrent tasks (e.g., notifications) to fail. "
                "Use only for local debugging."
            )

        try:
            async for frame in self._frame_generator():
                fire_in_frame, annotated_frame = await self.fire_detector.detect(frame)
                await self._handle_fire_detection(fire_in_frame)

                if self.video_output:
                    cv2.imshow("output", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User pressed 'q' in debug window. Initiating shutdown.")
                        self.signal_handler.exit_gracefully()
                        break
        finally:
            if self.video_output:
                cv2.destroyAllWindows()
            if self.stream:
                await self.stream.stop()
