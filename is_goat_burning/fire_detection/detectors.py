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

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from typing import Protocol

import cv2
import numpy as np

if TYPE_CHECKING:
    from is_goat_burning.fire_detection.gemini_detector import GeminiFireDetector

from is_goat_burning.config import settings
from is_goat_burning.logger import get_logger

# --- Constants ---
GAUSSIAN_BLUR_KERNEL_SIZE = (21, 21)
GAUSSIAN_BLUR_SIGMA_X = 0


class FireDetector(Protocol):
    """A protocol defining the interface for all fire detector classes."""

    async def detect(self, frame: np.ndarray | cv2.UMat | cv2.cuda.GpuMat) -> tuple[bool, np.ndarray]:
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
    """Detects fire using standard CPU-based OpenCV functions with motion verification.

    This detector uses a two-step approach:
    1. HSV color-based fire detection (orange/red colors)
    2. Temporal motion verification via frame differencing

    Static objects (e.g., walls lit by sunset) are filtered out after the first frame
    because they don't exhibit the dynamic flickering characteristic of real fire.
    """

    def __init__(
        self,
        margin: int,
        lower: np.ndarray,
        upper: np.ndarray,
        motion_threshold: int = 25,
    ) -> None:
        """Initializes the CPU fire detector.

        Args:
            margin: The minimum number of non-zero pixels in the color mask
                    to trigger a positive fire detection.
            lower: The lower bound of the HSV color range for fire.
            upper: The upper bound of the HSV color range for fire.
            motion_threshold: The pixel intensity change threshold for motion
                              detection (0-255). Higher = less sensitive.
        """
        self.margin = margin
        self.lower = lower
        self.upper = upper
        self.motion_threshold = motion_threshold
        self._previous_frame: np.ndarray | cv2.UMat | None = None
        self._logger = get_logger(self.__class__.__name__)

    def _detect_logic(self, frame: np.ndarray | cv2.UMat) -> tuple[bool, np.ndarray | cv2.UMat]:
        """Core fire detection logic with temporal motion verification.

        The algorithm:
        1. Apply Gaussian blur and convert to HSV for color masking.
        2. Create a color mask for fire-like colors.
        3. Create a motion mask by comparing with the previous frame.
        4. Combine masks: fire = color match AND motion detected.
        5. Store current grayscale frame for next iteration.

        On the first frame, motion is assumed valid to establish a baseline.

        Args:
            frame: The input frame (np.ndarray or cv2.UMat).

        Returns:
            A tuple containing the fire detection result and the masked frame.
        """
        # Step 1: Blur and convert to HSV for color detection
        blur = cv2.GaussianBlur(frame, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.lower, self.upper)

        # Step 2: Convert current frame to grayscale for motion detection
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 3: Create motion mask (or assume motion on first frame)
        is_umat = isinstance(frame, cv2.UMat)

        if self._previous_frame is None:
            # First frame: assume all motion is valid to establish baseline
            self._logger.debug("Motion detection initialized (first frame)")
            if is_umat:
                # For UMat, create the motion mask on-device using the same shape as color_mask
                motion_mask = cv2.UMat(np.full(color_mask.get().shape, 255, dtype=np.uint8))
            else:
                motion_mask = np.ones_like(current_gray, dtype=np.uint8) * 255
        else:
            # Compute absolute difference between frames (works for both UMat and np.ndarray)
            frame_diff = cv2.absdiff(current_gray, self._previous_frame)
            # Threshold to create binary motion mask
            _, motion_mask = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)

        # Step 4: Store current grayscale for next frame comparison, keeping it on-device for UMat
        if is_umat:
            self._previous_frame = current_gray
        else:
            self._previous_frame = current_gray.copy()

        # Step 5: Combine color and motion masks
        final_mask = cv2.bitwise_and(color_mask, motion_mask)

        # Count pixels matching both color AND motion criteria
        no_red = cv2.countNonZero(final_mask)
        is_fire = no_red > self.margin
        self._logger.debug("Detection result: pixels=%d, margin=%d, fire=%s", no_red, self.margin, is_fire)
        return is_fire, cv2.bitwise_and(frame, frame, mask=final_mask)

    async def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Analyzes a standard NumPy ndarray frame for fire."""
        return await asyncio.to_thread(self._detect_logic, frame)


class CUDAFireDetector:
    """Detects fire using CUDA-accelerated OpenCV functions with motion verification.

    This detector uses a two-step approach:
    1. HSV color-based fire detection using CUDA operations
    2. Temporal motion verification via frame differencing on GPU

    Static objects are filtered out after the first frame because they don't
    exhibit the dynamic flickering characteristic of real fire.
    """

    def __init__(
        self,
        margin: int,
        lower: np.ndarray,
        upper: np.ndarray,
        motion_threshold: int = 25,
    ) -> None:
        """Initializes the CUDA fire detector.

        Args:
            margin: The minimum number of non-zero pixels in the color mask
                    to trigger a positive fire detection.
            lower: The lower bound of the HSV color range for fire.
            upper: The upper bound of the HSV color range for fire.
            motion_threshold: The pixel intensity change threshold for motion
                              detection (0-255). Higher = less sensitive.
        """
        self.margin = margin
        self.lower = lower
        self.upper = upper
        self.motion_threshold = motion_threshold
        self.base_bound_set: bool = False
        self.lower_channel: list[cv2.cuda.GpuMat] = []
        self.upper_channel: list[cv2.cuda.GpuMat] = []
        self.gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X
        )
        self._last_frame_size: tuple[int, int] | None = None
        self._previous_frame_gpu: cv2.cuda.GpuMat | None = None
        self._logger = get_logger(self.__class__.__name__)

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

    async def detect(self, frame: cv2.cuda.GpuMat) -> tuple[bool, np.ndarray]:
        """Analyzes a cv2.cuda.GpuMat frame for fire using CUDA operations."""
        return await asyncio.to_thread(self._detect_logic, frame)

    def _detect_logic(self, frame: cv2.cuda.GpuMat) -> tuple[bool, np.ndarray]:
        """Synchronous core logic for CUDA detection with motion verification."""
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

        # Combine the individual channel masks for color detection
        color_mask_gpu = cv2.cuda.bitwise_and(in_range_h, in_range_s)
        color_mask_gpu = cv2.cuda.bitwise_and(color_mask_gpu, in_range_v)

        # --- Motion Detection ---
        current_gray_gpu = cv2.cuda.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._previous_frame_gpu is None:
            # First frame: assume all motion is valid to establish baseline
            motion_mask_gpu = cv2.cuda.GpuMat(current_gray_gpu.size(), cv2.CV_8UC1)
            motion_mask_gpu.setTo(255)
        else:
            # Compute absolute difference between frames
            frame_diff_gpu = cv2.cuda.absdiff(current_gray_gpu, self._previous_frame_gpu)
            # Threshold to create binary motion mask
            _, motion_mask_gpu = cv2.cuda.threshold(frame_diff_gpu, self.motion_threshold, 255, cv2.THRESH_BINARY)

        # Store current grayscale for next frame comparison
        self._previous_frame_gpu = current_gray_gpu.clone()

        # Combine color and motion masks
        final_mask_gpu = cv2.cuda.bitwise_and(color_mask_gpu, motion_mask_gpu)

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

    async def detect(self, frame: cv2.UMat) -> tuple[bool, np.ndarray]:
        """Analyzes a cv2.UMat frame for fire.

        The result is downloaded to a standard NumPy array before being returned.
        """
        # Note: OpenCL operations in OpenCV are generally asynchronous on the device,
        # but the Python calls can still block. We offload to executor for consistency.
        # However, passing UMat to another thread might be tricky depending on context.
        # For now, we assume it's safe or that the overhead is acceptable.
        # Actually, UMat context is thread-local in some cases.
        # To be safe and simple, we'll wrap the logic.

        def _run() -> tuple[bool, np.ndarray]:
            fire, annotated_frame = self._detect_logic(frame)
            return fire, annotated_frame.get()

        return await asyncio.to_thread(_run)


class HybridFireDetector:
    """Two-stage fire detector: CV gatekeeper + Gemini verifier.

    This detector uses a lightweight local CV detector (e.g., CPUFireDetector)
    as a fast first pass. If the local detector finds potential fire, it then
    uses the GeminiFireDetector for high-accuracy verification. This approach
    minimizes API costs while reducing false positives.
    """

    def __init__(
        self,
        local_detector: CPUFireDetector | CUDAFireDetector | OpenCLFireDetector,
        gemini_detector: GeminiFireDetector,
    ) -> None:
        """Initializes the HybridFireDetector.

        Args:
            local_detector: A local CV-based detector for the fast first pass.
            gemini_detector: The Gemini API detector for verification.
        """
        self.local_detector = local_detector
        self.gemini_detector = gemini_detector
        self._logger = get_logger("HybridFireDetector")

    async def detect(self, frame: np.ndarray | cv2.UMat | cv2.cuda.GpuMat) -> tuple[bool, np.ndarray]:
        """Analyzes a frame for fire using two-stage detection.

        Stage 1 (Gatekeeper): Uses the local CV detector for a fast check.
        If no fire is detected, returns immediately without calling the API.

        Stage 2 (Verifier): If local detection is positive, calls the Gemini
        API to verify the detection and filter false positives.

        Args:
            frame: The input video frame (BGR numpy array, UMat, or GpuMat).

        Returns:
            A tuple containing:
            - A boolean indicating if fire was confirmed by both detectors.
            - The annotated frame from the local detector.
        """
        # Stage 1: Fast local check (gatekeeper)
        local_result, annotated_frame = await self.local_detector.detect(frame)

        if not local_result:
            # No fire detected locally - return immediately without API call
            return False, annotated_frame

        # Stage 2: Gemini verification
        # Convert frame to numpy array for Gemini detector (it requires np.ndarray)
        numpy_frame: np.ndarray
        if isinstance(frame, cv2.cuda.GpuMat):
            numpy_frame = frame.download()
        elif isinstance(frame, cv2.UMat):
            numpy_frame = frame.get()
        else:
            numpy_frame = frame

        gemini_result, _ = await self.gemini_detector.detect(numpy_frame)

        if not gemini_result:
            self._logger.info("CV detector found fire but Gemini rejected it - false positive filtered.")

        return gemini_result, annotated_frame


def _create_local_detector(
    margin: int,
    lower: np.ndarray,
    upper: np.ndarray,
    use_open_cl: bool | None = None,
    use_cuda: bool | None = None,
    motion_threshold: int | None = None,
) -> CPUFireDetector | CUDAFireDetector | OpenCLFireDetector:
    """Creates a local CV-based fire detector based on hardware settings.

    Args:
        margin: The fire detection threshold (pixel count).
        lower: The lower HSV color bound.
        upper: The upper HSV color bound.
        use_open_cl: Explicitly request the OpenCL detector. If None, uses
                     the global `settings.open_cl`.
        use_cuda: Explicitly request the CUDA detector. If None, uses
                  the global `settings.cuda`.
        motion_threshold: The pixel intensity change threshold for motion
                          detection. If None, uses `settings.motion_detection_threshold`.

    Returns:
        A CPU, CUDA, or OpenCL fire detector instance.
    """
    should_use_open_cl = settings.open_cl if use_open_cl is None else use_open_cl
    should_use_cuda = settings.cuda if use_cuda is None else use_cuda
    threshold = settings.motion_detection_threshold if motion_threshold is None else motion_threshold

    if should_use_cuda:
        return CUDAFireDetector(margin, lower, upper, threshold)
    if should_use_open_cl:
        return OpenCLFireDetector(margin, lower, upper, threshold)
    return CPUFireDetector(margin, lower, upper, threshold)


def create_fire_detector(
    margin: int,
    lower: np.ndarray,
    upper: np.ndarray,
    use_open_cl: bool | None = None,
    use_cuda: bool | None = None,
    strategy: str | None = None,
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
        strategy: The detection strategy to use ("classic", "gemini", or "hybrid").
                  If None, uses `settings.detection_strategy`.

    Returns:
        An instance of a class that conforms to the FireDetector protocol.
    """
    selected_strategy = strategy if strategy is not None else settings.detection_strategy

    if selected_strategy in ("gemini", "hybrid"):
        from is_goat_burning.fire_detection.gemini_detector import GeminiFireDetector

    if selected_strategy == "gemini":
        return GeminiFireDetector()

    # For both "classic" and "hybrid", we need a local detector.
    local_detector = _create_local_detector(margin, lower, upper, use_open_cl, use_cuda)

    if selected_strategy == "hybrid":
        gemini_detector = GeminiFireDetector()
        return HybridFireDetector(local_detector, gemini_detector)
    # "classic" strategy
    return local_detector
