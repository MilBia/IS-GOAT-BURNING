from typing import Protocol

import cv2
import numpy as np

from config import settings


class FireDetector(Protocol):
    def detect(self, frame: np.ndarray) -> bool: ...


class CPUFireDetector:
    def __init__(self, margin: int, lower: np.ndarray, upper: np.ndarray):
        self.margin = margin
        self.lower = lower
        self.upper = upper

    def detect(self, frame: np.ndarray) -> bool:
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        no_red = cv2.countNonZero(mask)
        return int(no_red) > self.margin


class CUDAFireDetector:
    def __init__(self, margin: int, lower: np.ndarray, upper: np.ndarray):
        self.margin = margin
        self.lower = lower
        self.upper = upper
        self.gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (21, 21), 0)

    def detect(self, frame: np.ndarray) -> bool:
        blur = self.gaussian_filter.apply(frame)
        hsv = cv2.cuda.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Split the GPU HSV image into its H, S, V channels
        h, s, v = cv2.cuda.split(hsv)

        # --- Process Hue Channel ---
        _, lower_h = cv2.cuda.threshold(h, self.lower[0], 255, cv2.THRESH_BINARY)
        _, upper_h = cv2.cuda.threshold(h, self.upper[0], 255, cv2.THRESH_BINARY_INV)
        in_range_h = cv2.cuda.bitwise_and(lower_h, upper_h)

        # --- Process Saturation Channel ---
        _, lower_s = cv2.cuda.threshold(s, self.lower[1], 255, cv2.THRESH_BINARY)
        _, upper_s = cv2.cuda.threshold(s, self.upper[1], 255, cv2.THRESH_BINARY_INV)
        in_range_s = cv2.cuda.bitwise_and(lower_s, upper_s)

        # --- Process Value Channel ---
        _, lower_v = cv2.cuda.threshold(v, self.lower[2], 255, cv2.THRESH_BINARY)
        _, upper_v = cv2.cuda.threshold(v, self.upper[2], 255, cv2.THRESH_BINARY_INV)
        in_range_v = cv2.cuda.bitwise_and(lower_v, upper_v)

        # Combine the individual channel masks
        final_mask_gpu = cv2.cuda.bitwise_and(in_range_h, in_range_s)
        final_mask_gpu = cv2.cuda.bitwise_and(final_mask_gpu, in_range_v)
        no_red = cv2.cuda.countNonZero(final_mask_gpu)

        return int(no_red) > self.margin


class OpenCLFireDetector:
    def __init__(self, margin: int, lower: np.ndarray, upper: np.ndarray):
        self.margin = margin
        self.lower = lower
        self.upper = upper

    def detect(self, frame: np.ndarray) -> bool:
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        no_red = cv2.countNonZero(mask)
        return int(no_red) > self.margin


def create_fire_detector(margin: int, lower: np.ndarray, upper: np.ndarray) -> FireDetector:
    if settings.open_cl:
        return OpenCLFireDetector(margin, lower, upper)
    if settings.cuda:
        return CUDAFireDetector(margin, lower, upper)
    return CPUFireDetector(margin, lower, upper)
