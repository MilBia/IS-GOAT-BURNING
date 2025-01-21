from typing import Callable, Generator

import cv2
import numpy as np

from fire_detection.async_frame_generator import frame_gen_with_iterator, frame_gen
from fire_detection.cam_gear import YTCamGear


async def _detect_fire(frame: np.ndarray, fire_border: int, border_lower: np.ndarray, border_upper: np.ndarray) -> bool:
    """
    Detecting fire in gaven frame.

    :param frame: received frame
    :param fire_border: minimal number of pixels assumed as fire, needed to flag frame as containing fire
    :param border_lower: lower border of HSV values for fire color
    :param border_upper: upper border of HSV values for fire color
    :return: is frame containing fire
    """
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, border_lower, border_upper)

    no_red = cv2.countNonZero(mask)

    return int(no_red) > fire_border


async def _detect_loop(stream: YTCamGear, margin: int, lower: np.ndarray, upper: np.ndarray, on_fire_action: Callable) -> None:
    """
    Main loop with reading frames from stream and detecting fire on them.

    :param stream: stream to fource
    :param margin: minimal number of pixels assumed as fire, needed to flag frame as containing fire
    :param lower: lower border of HSV values for fire color
    :param upper: upper border of HSV values for fire color
    :param on_fire_action: action to perform on case of fire detection
    """
    async for frame in frame_gen(stream):
        fire = await _detect_fire(frame, margin, lower, upper)

        if fire:
            await on_fire_action()


async def _detect_loop_with_frequency(
    stream: YTCamGear, margin: int, lower: np.ndarray, upper: np.ndarray, on_fire_action: Callable, iterator: Generator
) -> None:
    """
    Main loop with reading frames from stream and detecting fire on them.

    :param stream: stream to fource
    :param margin: minimal number of pixels assumed as fire, needed to flag frame as containing fire
    :param lower: lower border of HSV values for fire color
    :param upper: upper border of HSV values for fire color
    :param on_fire_action: action to perform on case of fire detection
    :param iterator: generating which frame is destine to frame
    """
    async for frame in frame_gen_with_iterator(stream, iterator):
        fire = await _detect_fire(frame, margin, lower, upper)

        if fire:
            await on_fire_action()
