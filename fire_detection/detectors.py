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

    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        blur = self.gaussian_filter.apply(frame)
        hsv = cv2.cuda.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Split the GPU HSV image into its H, S, V channels
        h, s, v = cv2.cuda.split(hsv)

        # --- Process Hue Channel ---
        size = h.size()
        dtype = h.type()
        lower_h_gpu = cv2.cuda_GpuMat(size, dtype)
        lower_h_gpu.setTo(int(self.lower[0]))
        upper_h_gpu = cv2.cuda_GpuMat(size, dtype)
        upper_h_gpu.setTo(int(self.upper[0]))

        lower_h_mask = cv2.cuda.compare(h, lower_h_gpu, cv2.CMP_GE)
        upper_h_mask = cv2.cuda.compare(h, upper_h_gpu, cv2.CMP_LE)
        in_range_h = cv2.cuda.bitwise_and(lower_h_mask, upper_h_mask)

        # --- Process Saturation Channel ---
        lower_s_gpu = cv2.cuda_GpuMat(size, dtype)
        lower_s_gpu.setTo(int(self.lower[1]))
        upper_s_gpu = cv2.cuda_GpuMat(size, dtype)
        upper_s_gpu.setTo(int(self.upper[1]))

        lower_s_mask = cv2.cuda.compare(s, lower_s_gpu, cv2.CMP_GE)
        upper_s_mask = cv2.cuda.compare(s, upper_s_gpu, cv2.CMP_LE)
        in_range_s = cv2.cuda.bitwise_and(lower_s_mask, upper_s_mask)

        # --- Process Value Channel ---
        lower_v_gpu = cv2.cuda_GpuMat(size, dtype)
        lower_v_gpu.setTo(int(self.lower[2]))
        upper_v_gpu = cv2.cuda_GpuMat(size, dtype)
        upper_v_gpu.setTo(int(self.upper[2]))

        lower_v_mask = cv2.cuda.compare(v, lower_v_gpu, cv2.CMP_GE)
        upper_v_mask = cv2.cuda.compare(v, upper_v_gpu, cv2.CMP_LE)
        in_range_v = cv2.cuda.bitwise_and(lower_v_mask, upper_v_mask)

        # Combine the individual channel masks
        final_mask_gpu = cv2.cuda.bitwise_and(in_range_h, in_range_s)
        final_mask_gpu = cv2.cuda.bitwise_and(final_mask_gpu, in_range_v)

        no_red = cv2.cuda.countNonZero(final_mask_gpu)

        result_gpu = cv2.cuda_GpuMat(frame.size(), frame.type())
        result_gpu.setTo((0, 0, 0))
        frame.copyTo(result_gpu, final_mask_gpu)

        return int(no_red) > self.margin, result_gpu.download()


class OpenCLFireDetector(CPUFireDetector):
    pass


def create_fire_detector(margin: int, lower: np.ndarray, upper: np.ndarray) -> FireDetector:
    if settings.open_cl:
        return OpenCLFireDetector(margin, lower, upper)
    if settings.cuda:
        return CUDAFireDetector(margin, lower, upper)
    return CPUFireDetector(margin, lower, upper)
