from functools import partial
from typing import Callable, Optional

import cv2
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
    fire_border: float = 0.05,
    logging: bool = False,
    video_output: bool = False,
    checks_per_second: Optional[int] = None,
) -> None:
    """

    :param src:
    :param on_fire_action:
    :param fire_border:
    :param logging:
    :param video_output:
    :param checks_per_second: number of checks per second, if greater than framerate then not taken into consideration
    :return:
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

    margin = frame.shape[0] * frame.shape[1] * fire_border
    await detect_loop(stream, margin, lower, upper, on_fire_action)
    stream.stop()
