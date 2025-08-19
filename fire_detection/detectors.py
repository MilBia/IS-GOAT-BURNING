from typing import Protocol

import cv2
import numpy as np

from config import settings


class FireDetector(Protocol):
    def detect(self, frame: np.ndarray | cv2.UMat | cv2.cuda.GpuMat) -> tuple[bool, np.ndarray]: ...


class CPUFireDetector:
    def __init__(self, margin: int, lower: np.ndarray, upper: np.ndarray):
        self.margin = margin
        self.lower = lower
        self.upper = upper

    def _detect_logic(self, frame: np.ndarray | cv2.UMat) -> np.ndarray | cv2.UMat:
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        no_red = cv2.countNonZero(mask)
        return int(no_red) > self.margin, cv2.bitwise_and(frame, frame, mask=mask)

    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        return self._detect_logic(frame)


class CUDAFireDetector:
    def __init__(self, margin: int, lower: np.ndarray, upper: np.ndarray):
        self.margin = margin
        self.lower = lower
        self.upper = upper
        self.base_bound_set: bool = False
        self.lower_channel: list[cv2.cuda.GpuMat] = []
        self.upper_channel: list[cv2.cuda.GpuMat] = []
        self.gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (21, 21), 0)
        self._last_frame_size = None

    def _create_lower_upper_masks(self, channel: cv2.cuda.GpuMat) -> None:
        size = channel.size()
        dtype = channel.type()
        for i in range(3):
            lower_channel_gpu = cv2.cuda.GpuMat(size, dtype)
            lower_channel_gpu.setTo(int(self.lower[i]))
            self.lower_channel.append(lower_channel_gpu)
            upper_channel_gpu = cv2.cuda.GpuMat(size, dtype)
            upper_channel_gpu.setTo(int(self.upper[i]))
            self.upper_channel.append(upper_channel_gpu)

    def _create_channel_mask(
        self, channel: cv2.cuda.GpuMat, lower_channel_gpu: cv2.cuda.GpuMat, upper_channel_gpu: cv2.cuda.GpuMat
    ) -> cv2.cuda.GpuMat:
        """Creates a mask for a single channel based on lower and upper bounds."""

        lower_channel_mask = cv2.cuda.compare(channel, lower_channel_gpu, cv2.CMP_GE)
        upper_channel_mask = cv2.cuda.compare(channel, upper_channel_gpu, cv2.CMP_LE)
        return cv2.cuda.bitwise_and(lower_channel_mask, upper_channel_mask)

    def detect(self, frame: cv2.cuda.GpuMat) -> tuple[bool, np.ndarray]:
        blur = self.gaussian_filter.apply(frame)
        hsv = cv2.cuda.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Split the GPU HSV image into its H, S, V channels
        h, s, v = cv2.cuda.split(hsv)
        if not self.base_bound_set or self._last_frame_size != h.size():
            self.lower_channel.clear()
            self.upper_channel.clear()
            self._create_lower_upper_masks(h)
            self.base_bound_set = True
            self._last_frame_size = h.size()

        # --- Process Hue Channel ---
        in_range_h = self._create_channel_mask(h, self.lower_channel[0], self.upper_channel[0])

        # --- Process Saturation Channel ---
        in_range_s = self._create_channel_mask(s, self.lower_channel[1], self.upper_channel[1])

        # --- Process Value Channel ---
        in_range_v = self._create_channel_mask(v, self.lower_channel[2], self.upper_channel[2])

        # Combine the individual channel masks
        final_mask_gpu = cv2.cuda.bitwise_and(in_range_h, in_range_s)
        final_mask_gpu = cv2.cuda.bitwise_and(final_mask_gpu, in_range_v)

        no_red = cv2.cuda.countNonZero(final_mask_gpu)

        result_gpu = cv2.cuda.GpuMat(frame.size(), frame.type())
        result_gpu.setTo((0, 0, 0))
        frame.copyTo(result_gpu, final_mask_gpu)

        return int(no_red) > self.margin, result_gpu.download()


class OpenCLFireDetector(CPUFireDetector):
    """
    Uses the CPU-based fire detection implementation.
    OpenCV functions used within are transparently accelerated by OpenCL when available
    and when input frames are of type cv2.UMat.
    """

    def detect(self, frame: cv2.UMat) -> tuple[bool, np.ndarray]:
        fire, annotated_frame = self._detect_logic(frame)
        return fire, annotated_frame.get()


def create_fire_detector(margin: int, lower: np.ndarray, upper: np.ndarray) -> FireDetector:
    if settings.open_cl:
        return OpenCLFireDetector(margin, lower, upper)
    if settings.cuda:
        return CUDAFireDetector(margin, lower, upper)
    return CPUFireDetector(margin, lower, upper)
