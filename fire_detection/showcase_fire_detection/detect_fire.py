import cv2
import numpy as np

from fire_detection.signal_handler import SignalHandler


async def _detect_fire(
    frame: np.ndarray, fire_border: int, border_lower: np.ndarray, border_upper: np.ndarray
) -> [bool, np.ndarray]:
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, border_lower, border_upper)

    no_red = cv2.countNonZero(mask)

    return int(no_red) > fire_border, cv2.bitwise_and(frame, hsv, mask=mask)


async def _detect_loop(stream, margin, lower, upper) -> None:
    signal_handler = SignalHandler()
    while signal_handler.KEEP_PROCESSING:
        frame = stream.read()

        if frame is None:
            break

        frame = cv2.resize(frame, (960, 540))

        fire, output = await _detect_fire(frame, margin, lower, upper)

        cv2.imshow("output", output)

        if fire:
            print("FIRE!")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
