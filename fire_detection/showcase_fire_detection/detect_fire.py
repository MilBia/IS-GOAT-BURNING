from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generator

import cv2
import numpy as np

from setting import OPEN_CL

if OPEN_CL:
    from fire_detection.async_frame_generator import (
        frame_gen_with_iterator_cl as frame_gen_with_iterator,
        frame_gen_cl as frame_gen,
    )
else:
    from fire_detection.async_frame_generator import frame_gen_with_iterator, frame_gen
from fire_detection.cam_gear import YTCamGear


async def _detect_fire(
    frame: np.ndarray, fire_border: int, border_lower: np.ndarray, border_upper: np.ndarray
) -> [bool, np.ndarray]:
    """
    Detecting fire in gaven frame.

    :param frame: received frame
    :param fire_border: minimal number of pixels assumed as fire, needed to flag frame as containing fire
    :param border_lower: lower border of HSV values for fire color
    :param border_upper: upper border of HSV values for fire color
    :return: is frame containing fire, frame in HVS with mask showing only fire
    """
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, border_lower, border_upper)

    no_red = cv2.countNonZero(mask)

    return int(no_red) > fire_border, cv2.bitwise_and(frame, hsv, mask=mask)


async def _detect_loop(stream: YTCamGear, margin: int, lower: np.ndarray, upper: np.ndarray, on_fire_action: Callable) -> None:
    """
    Main loop with reading frames from stream and detecting fire on them.

    :param stream: stream to fource
    :param margin: minimal number of pixels assumed as fire, needed to flag frame as containing fire
    :param lower: lower border of HSV values for fire color
    :param upper: upper border of HSV values for fire color
    :param on_fire_action: action to perform on case of fire detection
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        async for frame in frame_gen(stream):
            fire, output = await _detect_fire(frame, margin, lower, upper)

            cv2.imshow("output", output)

            if fire:
                executor.submit(on_fire_action)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


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
    with ThreadPoolExecutor(max_workers=1) as executor:
        async for frame in frame_gen_with_iterator(stream, iterator):
            fire, output = await _detect_fire(frame, margin, lower, upper)

            cv2.imshow("output", output)

            if fire:
                executor.submit(on_fire_action)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
