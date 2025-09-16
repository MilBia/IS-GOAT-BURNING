"""Provides different implementations for the fire detection algorithm.

This module defines a protocol for fire detection and offers several concrete
implementations:
- CPUFireDetector: A standard, pure-Python implementation using OpenCV.
- CUDAFireDetector: A high-performance version accelerated with NVIDIA CUDA.
- OpenCLFireDetector: A version that can be accelerated by OpenCL-compatible
  devices.

A factory function `create_fire_detector` is provided to select the appropriate
detector based on the application's configuration.
"""

from typing import Protocol

import cv2
import numpy as np

from is_goat_burning.config import settings

# --- Constants ---
GAUSSIAN_BLUR_KERNEL_SIZE = (21, 21)


class FireDetector(Protocol):
    """A protocol defining the interface for all fire detector classes."""

    def detect(self, frame: np.ndarray | cv2.UMat | cv2.cuda.GpuMat) -> tuple[bool, np.ndarray]:
        """Analyzes a video frame for the presence of fire.

        Args:
            frame: The input video frame in a format compatible with the
                   specific implementation (e.g., np.ndarray for CPU).

        Returns:
            A tuple containing:
            - A boolean indicating if fire was detected.
            - The annotated frame with the fire mask applied.
        """
        ...


class CPUFireDetector:
    """Detects fire using standard CPU-based OpenCV functions."""

    def __init__(self, margin: int, lower: np.ndarray, upper: np.ndarray) -> None:
        """Initializes the CPU fire detector.

        Args:
            margin: The minimum number of non-zero pixels in the color mask
                    to trigger a positive fire detection.
            lower: The lower bound of the HSV color range for fire.
            upper: The upper bound of the HSV color range for fire.
        """
        self.margin = margin
        self.lower = lower
        self.upper = upper

    def _detect_logic(self, frame: np.ndarray | cv2.UMat) -> tuple[bool, np.ndarray | cv2.UMat]:
        """Core fire detection logic shared by CPU and OpenCL detectors.

        Args:
            frame: The input frame (np.ndarray or cv2.UMat).

        Returns:
            A tuple containing the fire detection result and the masked frame.
        """
        blur = cv2.GaussianBlur(frame, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        no_red = cv2.countNonZero(mask)
        return no_red > self.margin, cv2.bitwise_and(frame, frame, mask=mask)

    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Analyzes a standard NumPy ndarray frame for fire."""
        return self._detect_logic(frame)


class CUDAFireDetector:
    """Detects fire using CUDA-accelerated OpenCV functions for NVIDIA GPUs."""

    def __init__(self, margin: int, lower: np.ndarray, upper: np.ndarray) -> None:
        """Initializes the CUDA fire detector.

        Args:
            margin: The minimum number of non-zero pixels in the color mask
                    to trigger a positive fire detection.
            lower: The lower bound of the HSV color range for fire.
            upper: The upper bound of the HSV color range for fire.
        """
        self.margin = margin
        self.lower = lower
        self.upper = upper
        self.base_bound_set: bool = False
        self.lower_channel: list[cv2.cuda.GpuMat] = []
        self.upper_channel: list[cv2.cuda.GpuMat] = []
        self.gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        self._last_frame_size: tuple[int, int] | None = None

    def _create_lower_upper_masks(self, channel: cv2.cuda.GpuMat) -> None:
        """Pre-allocates GpuMat masks for the HSV color range bounds.

        This is done to avoid re-creating these masks on every frame.

        Args:
            channel: A sample GpuMat channel (e.g., Hue) used to get the
                     correct size and type for the new masks.
        """
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
        """Creates a mask for a single channel based on its bounds.

        Args:
            channel: The single-channel GpuMat to be masked (e.g., H, S, or V).
            lower_channel_gpu: A GpuMat of the same size as `channel`, filled
                               with the lower bound value.
            upper_channel_gpu: A GpuMat of the same size as `channel`, filled
                               with the upper bound value.

        Returns:
            A single-channel mask where pixels within the bounds are 255 and
            others are 0.
        """
        lower_channel_mask = cv2.cuda.compare(channel, lower_channel_gpu, cv2.CMP_GE)
        upper_channel_mask = cv2.cuda.compare(channel, upper_channel_gpu, cv2.CMP_LE)
        return cv2.cuda.bitwise_and(lower_channel_mask, upper_channel_mask)

    def detect(self, frame: cv2.cuda.GpuMat) -> tuple[bool, np.ndarray]:
        """Analyzes a cv2.cuda.GpuMat frame for fire using CUDA operations."""
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

        return no_red > self.margin, result_gpu.download()


class OpenCLFireDetector(CPUFireDetector):
    """Detects fire using OpenCL-accelerated OpenCV functions.

    This class inherits the CPU logic but operates on `cv2.UMat` frames,
    allowing OpenCV to dispatch operations to an OpenCL-compatible device
    (like an integrated or discrete GPU) if available.
    """

    def detect(self, frame: cv2.UMat) -> tuple[bool, np.ndarray]:
        """Analyzes a cv2.UMat frame for fire.

        The result is downloaded to a standard NumPy array before being returned.
        """
        fire, annotated_frame = self._detect_logic(frame)
        return fire, annotated_frame.get()


def create_fire_detector(
    margin: int, lower: np.ndarray, upper: np.ndarray, use_open_cl: bool | None = None, use_cuda: bool | None = None
) -> FireDetector:
    """Factory function to create the appropriate fire detector.

    Selects the detector implementation based on the provided boolean flags or
    falls back to the global application settings. The order of preference is
    CUDA, then OpenCL, then CPU.

    Args:
        margin: The fire detection threshold (pixel count).
        lower: The lower HSV color bound.
        upper: The upper HSV color bound.
        use_open_cl: Explicitly request the OpenCL detector. If None, uses
                     the global `settings.open_cl`.
        use_cuda: Explicitly request the CUDA detector. If None, uses
                  the global `settings.cuda`.

    Returns:
        An instance of a class that conforms to the FireDetector protocol.
    """
    should_use_open_cl = settings.open_cl if use_open_cl is None else use_open_cl
    should_use_cuda = settings.cuda if use_cuda is None else use_cuda

    if should_use_cuda:
        return CUDAFireDetector(margin, lower, upper)
    if should_use_open_cl:
        return OpenCLFireDetector(margin, lower, upper)
    return CPUFireDetector(margin, lower, upper)
