from functools import partial
from typing import Callable, Optional

import numpy as np
from vidgear.gears import CamGear

from .base_fire_detection import _detect_loop as base_detect_fire
from .base_fire_detection import _detect_loop_with_frequency as base_detect_fire_with_frequency
from .showcase_fire_detection import _detect_loop as showcase_detect_fire
from .showcase_fire_detection import _detect_loop_with_frequency as showcase_detect_fire_with_frequency


options = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS": 30}
_lower = [18, 50, 50]
_upper = [35, 255, 255]
lower = np.array(_lower, dtype="uint8")
upper = np.array(_upper, dtype="uint8")


def checkout_generator(framerate, duration):
    """
    Generator increasing the counter each execution and yield information if current frame is destin to check

    :param framerate: detected stream framerate, if higher than 144 it assumes detection error and fix framerate to 25fps
    :param duration: how many check will be performed per second
    :return: if current frame is destin to check
    """
    if framerate > 144:
        framerate = 25
    step = framerate / duration
    value = 0
    frame = 0
    while True:
        frame += 1
        if frame > int(value):
            value += step
        yield int(value) == frame


async def detect_fire(
    src: str,
    on_fire_action: Callable,
    threshold: float = 0.05,
    logging: bool = False,
    video_output: bool = False,
    checks_per_second: Optional[int] = None,
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

    stream = CamGear(source=src, stream_mode=True, logging=logging, **options).start()

    frame = stream.read()
    if frame is None:
        return

    if video_output:
        detect_loop = showcase_detect_fire
    else:
        detect_loop = base_detect_fire

    if checks_per_second:
        if checks_per_second >= stream.framerate:
            pass
        else:
            check_iterator = iter(checkout_generator(stream.framerate, checks_per_second))
            if video_output:
                detect_loop = partial(showcase_detect_fire_with_frequency, iterator=check_iterator)
            else:
                detect_loop = partial(base_detect_fire_with_frequency, iterator=check_iterator)

    fire_threshold = frame.shape[0] * frame.shape[1] * threshold
    await detect_loop(stream, fire_threshold, lower, upper, on_fire_action)
    stream.stop()
