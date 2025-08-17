from collections.abc import AsyncGenerator
from collections.abc import Callable

import cv2
import numpy as np

from fire_detection.cam_gear import YTCamGear
from fire_detection.detectors import create_fire_detector
from fire_detection.signal_handler import SignalHandler

options = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS": 30}
_lower = [18, 50, 50]
_upper = [35, 255, 255]
lower = np.array(_lower, dtype="uint8")
upper = np.array(_upper, dtype="uint8")


class YTCamGearFireDetector:
    def __init__(
        self,
        src: str,
        on_fire_action: Callable,
        threshold: float = 0.05,
        logging: bool = False,
        video_output: bool = False,
        checks_per_second: int | None = None,
    ):
        self.on_fire_action = on_fire_action
        self.video_output = video_output
        self.stream = YTCamGear(source=src, stream_mode=True, logging=logging, **options)
        fire_threshold = self.stream.frame.shape[0] * self.stream.frame.shape[1] * threshold
        self.fire_detector = create_fire_detector(fire_threshold, lower, upper)
        self.signal_handler = SignalHandler()

        if checks_per_second and checks_per_second < self.stream.framerate:
            self.step = self.stream.framerate / checks_per_second
            self.check_iterator = self.checkout_generator()
            self.frame_generator = self._frame_gen_with_iterator
        else:
            self.frame_generator = self._frame_gen

    async def checkout_generator(self):
        """
        Generator increasing the counter each execution and yield information if current frame is destin to check

        :return: if current frame is destin to check
        """
        frame: int = 0
        while True:
            frame += 1
            if frame >= self.step:
                yield True
                frame -= self.step
            else:
                yield False

    async def _frame_gen(self):
        async for frame in self.stream.read():
            if frame is None:
                break
            yield frame

    async def _frame_gen_with_iterator(self) -> AsyncGenerator[np.ndarray, np.ndarray]:
        async for frame in self.stream.read():
            if frame is None:
                break
            if await anext(self.check_iterator):
                yield frame

    async def __call__(self, *args, **kwargs):
        async for frame in self.frame_generator():
            if self.fire_detector.detect(frame):
                self.signal_handler.fire_detected()
                await self.on_fire_action()

            if self.video_output:
                cv2.imshow("output", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if self.video_output:
            cv2.destroyAllWindows()
        self.stream.stop()
