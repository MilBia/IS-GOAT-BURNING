import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Callable

import cv2
import numpy as np

from is_goat_burning.fire_detection.cam_gear import YTCamGear
from is_goat_burning.fire_detection.detectors import create_fire_detector
from is_goat_burning.fire_detection.signal_handler import SignalHandler


class YTCamGearFireDetector:
    DEFAULT_OPTIONS = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS": 30}
    DEFAULT_LOWER_HSV = np.array([18, 50, 50], dtype="uint8")
    DEFAULT_UPPER_HSV = np.array([35, 255, 255], dtype="uint8")

    def __init__(
        self,
        src: str,
        on_fire_action: Callable,
        threshold: float = 0.05,
        logging: bool = False,
        video_output: bool = False,
        checks_per_second: float | None = None,
        lower_hsv: np.ndarray | None = None,
        upper_hsv: np.ndarray | None = None,
        **yt_cam_gear_options,
    ):
        """
        Initializes the color detector.

        Args:
            src (str): The source of the video stream.
            on_fire_action (Callable): The action to perform when fire is detected.
            threshold (float): The threshold for fire detection.
            logging (bool): Whether to log the video stream.
            video_output (bool): Whether to output the video stream.
            checks_per_second (float): The number of checks per second.
            lower_hsv (np.array): The lower bound of the HSV color range.
                                  The default values are optimized for detecting yellow.
            upper_hsv (np.array): The upper bound of the HSV color range.
                                  The default values are optimized for detecting yellow.
            **yt_cam_gear_options: Additional options for YTCamGear.
        """
        self.on_fire_action = on_fire_action
        self.video_output = video_output
        self.lower_hsv = lower_hsv if lower_hsv is not None else self.DEFAULT_LOWER_HSV
        self.upper_hsv = upper_hsv if upper_hsv is not None else self.DEFAULT_UPPER_HSV
        options = {**self.DEFAULT_OPTIONS, **yt_cam_gear_options}
        self.stream = YTCamGear(source=src, stream_mode=True, logging=logging, **options)
        fire_threshold = int(self.stream.frame.shape[0] * self.stream.frame.shape[1] * threshold)
        self.fire_detector = create_fire_detector(fire_threshold, self.lower_hsv, self.upper_hsv)

        if checks_per_second and checks_per_second < self.stream.framerate > 0:
            self.frames_between_step = self.stream.framerate / checks_per_second
            self.check_iterator = self.checkout_generator()
            self.frame_generator = self._frame_gen_with_iterator
        else:
            self.frame_generator = self._frame_gen

    async def checkout_generator(self) -> AsyncGenerator[bool, None]:
        """
        Asynchronous generator that yields True for frames that are destined for processing,
        based on the `checks_per_second` parameter.
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
        async for frame in self.stream.read():
            if frame is None:
                break
            yield frame

    async def _frame_gen_with_iterator(self) -> AsyncGenerator[np.ndarray | cv2.UMat | cv2.cuda.GpuMat, None]:
        async for frame in self.stream.read():
            if frame is None:
                break
            if await anext(self.check_iterator):
                yield frame

    async def __call__(self):
        try:
            loop = asyncio.get_running_loop()
            signal_handler = SignalHandler()
            async for frame in self.frame_generator():
                fire, annotated_frame = await loop.run_in_executor(None, self.fire_detector.detect, frame)
                if fire:
                    signal_handler.fire_detected()
                    await self.on_fire_action()

                if self.video_output:
                    cv2.imshow("output", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except asyncio.CancelledError:
            pass
        finally:
            if self.video_output:
                cv2.destroyAllWindows()
            self.stream.stop()
