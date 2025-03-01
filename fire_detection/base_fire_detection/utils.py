import cv2
import numpy as np


def _data_preparation(
    lower: np.ndarray,
    upper: np.ndarray,
):
    return lower, upper


async def _detect_fire(
    frame: np.ndarray,
    fire_border: int,
    border_lower: np.ndarray,
    border_upper: np.ndarray,
) -> bool:
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


def _cuda_data_preparation(
    lower: np.ndarray,
    upper: np.ndarray,
):
    gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (21, 21), 0)
    return lower, upper, gaussian_filter


async def _cuda_detect_fire(
    frame: np.ndarray, fire_border: int, border_lower: np.ndarray, border_upper: np.ndarray, gaussian_filter
) -> bool:
    """
    Detecting fire in gaven frame.

    :param frame: received frame
    :param fire_border: minimal number of pixels assumed as fire, needed to flag frame as containing fire
    :param border_lower: lower border of HSV values for fire color
    :param border_upper: upper border of HSV values for fire color
    :return: is frame containing fire
    """

    blur_gpu = gaussian_filter.apply(frame)
    hsv = cv2.cuda.cvtColor(blur_gpu, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv.download(), border_lower, border_upper)

    no_red = cv2.countNonZero(mask)

    return int(no_red) > fire_border
