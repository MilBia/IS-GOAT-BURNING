from typing import Protocol

import cv2
import numpy as np

from config import settings


class FireDetector(Protocol):
    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]: ...


class CPUFireDetector:
    def __init__(self, margin: int, lower: np.ndarray, upper: np.ndarray):
        self.margin = margin
        self.lower = lower
        self.upper = upper

    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        no_red = cv2.countNonZero(mask)
        return int(no_red) > self.margin, cv2.bitwise_and(frame, hsv, mask=mask)


class CUDAFireDetector:
    def __init__(self, margin: int, lower: np.ndarray, upper: np.ndarray):
        self.margin = margin
        self.lower = lower
        self.upper = upper
        self.gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (21, 21), 0)

    def _create_channel_mask(
        self, channel: cv2.cuda_GpuMat, size, dtype, lower_bound: int, upper_bound: int
    ) -> cv2.cuda_GpuMat:
        """Creates a mask for a single channel based on lower and upper bounds."""
        lower_channel_gpu = cv2.cuda_GpuMat(size, dtype)
        lower_channel_gpu.setTo(int(lower_bound))
        upper_channel_gpu = cv2.cuda_GpuMat(size, dtype)
        upper_channel_gpu.setTo(int(upper_bound))

        lower_channel_mask = cv2.cuda.compare(channel, lower_channel_gpu, cv2.CMP_GE)
        upper_channel_mask = cv2.cuda.compare(channel, upper_channel_gpu, cv2.CMP_LE)
        return cv2.cuda.bitwise_and(lower_channel_mask, upper_channel_mask)

    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        blur = self.gaussian_filter.apply(frame)
        hsv = cv2.cuda.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Split the GPU HSV image into its H, S, V channels
        h, s, v = cv2.cuda.split(hsv)

        # --- Process Hue Channel ---
        size = h.size()
        dtype = h.type()
        in_range_h = self._create_channel_mask(h, size, dtype, int(self.lower[0]), int(self.upper[0]))

        # --- Process Saturation Channel ---
        in_range_s = self._create_channel_mask(s, size, dtype, int(self.lower[1]), int(self.upper[1]))

        # --- Process Value Channel ---
        in_range_v = self._create_channel_mask(v, size, dtype, int(self.lower[2]), int(self.upper[2]))

        # Combine the individual channel masks
        final_mask_gpu = cv2.cuda.bitwise_and(in_range_h, in_range_s)
        final_mask_gpu = cv2.cuda.bitwise_and(final_mask_gpu, in_range_v)

        no_red = cv2.cuda.countNonZero(final_mask_gpu)

        result_gpu = cv2.cuda_GpuMat(frame.size(), frame.type())
        result_gpu.setTo((0, 0, 0))
        frame.copyTo(result_gpu, final_mask_gpu)

        return int(no_red) > self.margin, result_gpu.download()


class OpenCLFireDetector(CPUFireDetector):
    """
    Uses the CPU-based fire detection implementation.
    OpenCV functions used within are transparently accelerated by OpenCL when available
    and when input frames are of type cv2.UMat.
    """

    pass


def create_fire_detector(margin: int, lower: np.ndarray, upper: np.ndarray) -> FireDetector:
    if settings.open_cl:
        return OpenCLFireDetector(margin, lower, upper)
    if settings.cuda:
        return CUDAFireDetector(margin, lower, upper)
    return CPUFireDetector(margin, lower, upper)
