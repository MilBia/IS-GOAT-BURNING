from typing import Callable

import numpy as np
from vidgear.gears import CamGear

from .base_fire_detection import _detect_loop as base_detect_fire
from .showcase_fire_detection import _detect_loop as showcase_detect_fire


options = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS": 30}
_lower = [18, 50, 50]
_upper = [35, 255, 255]
lower = np.array(_lower, dtype="uint8")
upper = np.array(_upper, dtype="uint8")


async def detect_fire(
    src: str, on_fire_action: Callable, fire_border: float = 0.05, logging: bool = False, video_output: bool = False
):
    if video_output:
        detect_loop = showcase_detect_fire
    else:
        detect_loop = base_detect_fire

    stream = CamGear(source=src, stream_mode=True, logging=logging, **options).start()

    frame = stream.read()
    if frame is None:
        return
    margin = frame.shape[0] * frame.shape[1] * fire_border
    await detect_loop(stream, margin, lower, upper, on_fire_action)
    stream.stop()
