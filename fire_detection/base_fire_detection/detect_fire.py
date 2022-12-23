from typing import Callable, Generator

import cv2
import numpy as np
from vidgear.gears import CamGear

from fire_detection.signal_handler import SignalHandler


async def _detect_fire(frame: np.ndarray, fire_border: int, border_lower: np.ndarray, border_upper: np.ndarray) -> bool:
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, border_lower, border_upper)

    no_red = cv2.countNonZero(mask)

    return int(no_red) > fire_border


async def _detect_loop(stream: CamGear, margin: int, lower: np.ndarray, upper: np.ndarray, on_fire_action: Callable) -> None:
    signal_handler = SignalHandler()
    while signal_handler.KEEP_PROCESSING:
        frame = stream.read()

        if frame is None:
            break

        fire = await _detect_fire(frame, margin, lower, upper)

        if fire:
            on_fire_action()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


async def _detect_loop_with_frequency(
    stream: CamGear, margin: int, lower: np.ndarray, upper: np.ndarray, on_fire_action: Callable, iterator: Generator
) -> None:
    signal_handler = SignalHandler()
    while signal_handler.KEEP_PROCESSING:
        frame = stream.read()

        if frame is None:
            break

        if next(iterator):
            fire = await _detect_fire(frame, margin, lower, upper)

            if fire:
                on_fire_action()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
