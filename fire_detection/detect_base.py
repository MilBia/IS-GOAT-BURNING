from collections.abc import Callable
from functools import partial

import numpy as np

from setting import CUDA
from setting import OPEN_CL

from .base_fire_detection import _detect_loop as base_detect_fire
from .base_fire_detection import _detect_loop_with_frequency as base_detect_fire_with_frequency

if OPEN_CL:
    from fire_detection.cam_gear_opencl import YTCamGear
elif CUDA:
    from fire_detection.cam_gear_cuda import YTCamGear
else:
    from fire_detection.cam_gear import YTCamGear

from .showcase_fire_detection import _detect_loop as showcase_detect_fire
from .showcase_fire_detection import _detect_loop_with_frequency as showcase_detect_fire_with_frequency

options = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS": 30}
_lower = [18, 50, 50]
_upper = [35, 255, 255]
lower = np.array(_lower, dtype="uint8")
upper = np.array(_upper, dtype="uint8")


async def checkout_generator(framerate, duration):
    """
    Generator increasing the counter each execution and yield information if current frame is destin to check

    :param framerate: detected stream framerate
    :param duration: how many check will be performed per second
    :return: if current frame is destin to check
    """
    step = framerate / duration
    frame = 0.0
    while True:
        frame += 1
        if frame >= step:
            yield True
            frame -= step
        else:
            yield False


async def detect_fire(
    src: str,
    on_fire_action: Callable,
    threshold: float = 0.05,
    logging: bool = False,
    video_output: bool = False,
    checks_per_second: int | None = None,
) -> None:
    """
     Fire detection in output

    :param src: path to file or URL to video or stream
    :param on_fire_action: action which will be taken on case of fire detected
    :param threshold: percentage threshold to fire like elements in frame to assume the fire detection
    :param logging: log to standard output
    :param video_output: showing the frames after fire detection
    :param checks_per_second: number of checks per second, if greater than framerate then not taken into consideration
    """

    stream = YTCamGear(source=src, stream_mode=True, logging=logging, **options)

    detect_loop = showcase_detect_fire if video_output else base_detect_fire

    if checks_per_second:
        if checks_per_second >= stream.framerate:
            pass
        else:
            check_iterator = checkout_generator(stream.framerate, checks_per_second)
            if video_output:
                detect_loop = partial(showcase_detect_fire_with_frequency, iterator=check_iterator)
            else:
                detect_loop = partial(base_detect_fire_with_frequency, iterator=check_iterator)

    fire_threshold = stream.frame.shape[0] * stream.frame.shape[1] * threshold
    await detect_loop(stream, fire_threshold, lower, upper, on_fire_action)
    stream.stop()
