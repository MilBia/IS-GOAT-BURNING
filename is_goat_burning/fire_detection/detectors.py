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

import asyncio
from collections import deque
from pathlib import Path
import threading
import time
from time import perf_counter
from typing import TYPE_CHECKING
from typing import Protocol

import cv2
import numpy as np

from is_goat_burning.config import settings
from is_goat_burning.fire_detection.escalation_log import EscalationLog
from is_goat_burning.fire_detection.escalation_log import get_escalation_log
from is_goat_burning.fire_detection.verifier_health import VerifierError
from is_goat_burning.fire_detection.verifier_health import VerifierStatus
from is_goat_burning.fire_detection.verifier_health import publish_verifier_status
from is_goat_burning.logger import get_logger

# --- Constants ---
GAUSSIAN_BLUR_KERNEL_SIZE = (21, 21)
GAUSSIAN_BLUR_SIGMA_X = 0
DEFAULT_MOTION_THRESHOLD = 25
MAX_PIXEL_VALUE = 255
# Area-relative mode: ignore tiny warm regions and tiny amounts of motion (fractions of the frame area).
MIN_WARM_FRACTION = 0.002
MIN_MOVING_FRACTION = 0.0005

if TYPE_CHECKING:
    from is_goat_burning.config import EdgeSettings
    from is_goat_burning.fire_detection.edge_detector import EdgeInferenceDetector

_factory_logger = get_logger("create_fire_detector")
# Name under which edge verifier availability is published (matches HybridFireDetector's type(verifier).__name__).
EDGE_VERIFIER_NAME = "EdgeInferenceDetector"
# True while the last hybrid_edge build fell back to the local detector (a model load failure was published).
_edge_load_failed = False

# Successfully built edge verifiers, kept across factory calls (i.e. stream reconnects) so the ONNX model is not
# re-read, re-warmed-up and its CUDA session not recreated each time. Only successes are stored.
_edge_verifier_cache: dict[tuple, EdgeInferenceDetector] = {}
_edge_verifier_cache_lock = threading.Lock()


def is_fire_by_pixels(
    moving_warm: int,
    warm: int,
    margin: int,
    relative_threshold: float | None = None,
    frame_area: int = 0,
) -> bool:
    """Applies the decision rule to the pixel counts of one frame.

    Two modes are supported:

    * ``relative_threshold is None`` (default): fire if the warm AND moving pixels exceed ``margin``
      (a fixed share of the whole frame).
    * ``relative_threshold`` set: fire if the warm region is at least ``MIN_WARM_FRACTION`` of the frame,
      the moving part of it is at least ``MIN_MOVING_FRACTION`` of the frame, and the moving pixels exceed
      ``relative_threshold`` times the warm pixels. This scales with the size of the flame instead of
      the frame, so small fires are not out of reach.

    Args:
        moving_warm: Pixels that are both fire-coloured and moving.
        warm: Pixels that are fire-coloured.
        margin: Pixel count for the frame-fraction mode.
        relative_threshold: Share of warm pixels that must be moving; None selects the frame-fraction mode.
        frame_area: Number of pixels in the frame (needed for the area-relative mode).

    Returns:
        True if the counts indicate fire.
    """
    if relative_threshold is None:
        return moving_warm > margin
    return (
        warm >= MIN_WARM_FRACTION * frame_area
        and moving_warm > MIN_MOVING_FRACTION * frame_area
        and moving_warm > relative_threshold * warm
    )


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


class FireVerifier(Protocol):
    """A protocol for second-stage verifiers used by `HybridFireDetector`.

    Verifiers always receive a BGR `np.ndarray` (never a GPU/UMat frame), which is
    satisfied by `GeminiFireDetector`, `DecisionFireDetector` and offline edge models.
    """

    async def verify(self, frame: np.ndarray) -> bool:
        """Verifies whether a frame contains fire.

        Args:
            frame: The input video frame as a BGR numpy array.

        Returns:
            True if fire was confirmed, False if the verifier rejected it.

        Raises:
            VerifierError: If the verifier could not produce a verdict (API error, timeout, failed inference).
                This is distinct from "no fire" and is handled by the ``VERIFIER_FAILURE_POLICY``.
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

    _logger = get_logger("CPUFireDetector")

    def __init__(
        self,
        margin: int,
        lower: np.ndarray,
        upper: np.ndarray,
        motion_threshold: int = DEFAULT_MOTION_THRESHOLD,
        relative_threshold: float | None = None,
    ) -> None:
        """Initializes the CPU fire detector.

        Args:
            margin: The minimum number of non-zero pixels in the color mask
                    to trigger a positive fire detection.
            lower: The lower bound of the HSV color range for fire.
            upper: The upper bound of the HSV color range for fire.
            motion_threshold: The pixel intensity change threshold for motion
                              detection (0-255). Higher = less sensitive.
            relative_threshold: If set, use the area-relative rule (share of warm
                                pixels that must be moving) instead of ``margin``.
        """
        self.margin = margin
        self.lower = lower
        self.upper = upper
        self.motion_threshold = motion_threshold
        self.relative_threshold = relative_threshold
        self._previous_frame: np.ndarray | cv2.UMat | None = None
        #: When False the masked "annotated" frame is not built and the input frame is returned instead
        #: (saves ~7 ms per check on a Raspberry Pi 4). Only the debug window (`VIDEO_OUTPUT`) needs the real one.
        self.annotate: bool = True

    def _detect_logic(self, frame: np.ndarray | cv2.UMat) -> tuple[bool, np.ndarray | cv2.UMat]:
        """Core fire detection logic with temporal motion verification.

        The algorithm:
        1. Apply Gaussian blur and convert to HSV for color masking.
        2. Create a color mask for fire-like colors.
        3. Create a motion mask by comparing with the previous frame.
        4. Combine masks: fire = color match AND motion detected.
        5. Store current grayscale frame for next iteration.

        On the first frame (or after a reset) there is nothing to compare with, so no motion is reported and the
        frame cannot trigger a detection; it only establishes the baseline.

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

        # Step 3: Create motion mask (empty on the first frame: no baseline, no evidence of motion)
        is_umat = isinstance(frame, cv2.UMat)

        frame_diff = None
        if self._previous_frame is not None:
            if is_umat:
                # OpenCL path: UMat doesn't expose shape/size directly in Python without .get()
                # so we rely on try-except to catch resolution mismatches.
                try:
                    frame_diff = cv2.absdiff(current_gray, self._previous_frame)
                except cv2.error:
                    self._logger.warning("Resolution change detected or frame mismatch (OpenCL), resetting motion baseline")
                    self._previous_frame = None
            else:
                # CPU path: Explicit resolution check for NumPy arrays
                if self._previous_frame.shape != current_gray.shape:
                    self._logger.warning("Resolution change detected (CPU), resetting motion baseline")
                    self._previous_frame = None
                else:
                    frame_diff = cv2.absdiff(current_gray, self._previous_frame)

        if self._previous_frame is None:
            # First frame (or reset): no baseline, so no motion. A colour-only match must not raise an alarm.
            self._logger.debug("Motion detection initialized (first frame or reset)")
            if is_umat:
                # For UMat, create the (empty) motion mask directly on-device.
                motion_mask = cv2.compare(color_mask, color_mask, cv2.CMP_NE)
            else:
                motion_mask = np.zeros_like(current_gray, dtype=np.uint8)
        else:
            # Threshold to create binary motion mask using the pre-computed frame_diff
            _, motion_mask = cv2.threshold(frame_diff, self.motion_threshold, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)

        # Step 4: Store current grayscale for next frame comparison, keeping it on-device for UMat
        if is_umat:
            self._previous_frame = current_gray
        else:
            self._previous_frame = current_gray.copy()

        # Step 5: Combine color and motion masks
        final_mask = cv2.bitwise_and(color_mask, motion_mask)

        # Count pixels matching both color AND motion criteria
        no_red = cv2.countNonZero(final_mask)
        warm = 0
        frame_area = 0
        if self.relative_threshold is not None:
            warm = cv2.countNonZero(color_mask)
            # Per frame, so a mid-stream resolution change cannot leave a stale area behind.
            gray_shape = current_gray.get().shape if is_umat else current_gray.shape
            frame_area = gray_shape[0] * gray_shape[1]
        is_fire = is_fire_by_pixels(no_red, warm, self.margin, self.relative_threshold, frame_area)
        self._logger.debug("Detection result: pixels=%d, warm=%d, margin=%d, fire=%s", no_red, warm, self.margin, is_fire)
        if not self.annotate:
            return is_fire, frame
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

    _logger = get_logger("CUDAFireDetector")

    def __init__(
        self,
        margin: int,
        lower: np.ndarray,
        upper: np.ndarray,
        motion_threshold: int = DEFAULT_MOTION_THRESHOLD,
        relative_threshold: float | None = None,
    ) -> None:
        """Initializes the CUDA fire detector.

        Args:
            margin: The minimum number of non-zero pixels in the color mask
                    to trigger a positive fire detection.
            lower: The lower bound of the HSV color range for fire.
            upper: The upper bound of the HSV color range for fire.
            motion_threshold: The pixel intensity change threshold for motion
                              detection (0-255). Higher = less sensitive.
            relative_threshold: If set, use the area-relative rule (share of warm
                                pixels that must be moving) instead of ``margin``.
        """
        self.margin = margin
        self.lower = lower
        self.upper = upper
        self.motion_threshold = motion_threshold
        self.relative_threshold = relative_threshold
        self.base_bound_set: bool = False
        self.lower_channel: list[cv2.cuda.GpuMat] = []
        self.upper_channel: list[cv2.cuda.GpuMat] = []
        self.gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X
        )
        self._last_frame_size: tuple[int, int] | None = None
        self._previous_frame_gpu: cv2.cuda.GpuMat | None = None

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

        # Check for resolution change
        if self._previous_frame_gpu is not None and self._previous_frame_gpu.size() != current_gray_gpu.size():
            self._logger.warning("Resolution change detected (CUDA), resetting motion baseline")
            self._previous_frame_gpu = None

        if self._previous_frame_gpu is None:
            # First frame: no baseline, so no motion
            motion_mask_gpu = cv2.cuda.GpuMat(current_gray_gpu.size(), cv2.CV_8UC1)
            motion_mask_gpu.setTo(0)
        else:
            try:
                # Compute absolute difference between frames
                frame_diff_gpu = cv2.cuda.absdiff(current_gray_gpu, self._previous_frame_gpu)
                # Threshold to create binary motion mask
                _, motion_mask_gpu = cv2.cuda.threshold(
                    frame_diff_gpu, self.motion_threshold, MAX_PIXEL_VALUE, cv2.THRESH_BINARY
                )
            except cv2.error:
                self._logger.warning("Frame mismatch during CUDA processing, resetting motion baseline.")
                self._previous_frame_gpu = None
                # Fallback to treating it as a first frame
                motion_mask_gpu = cv2.cuda.GpuMat(current_gray_gpu.size(), cv2.CV_8UC1)
                motion_mask_gpu.setTo(0)

        # Store current grayscale for next frame comparison
        # GpuMat is reference counted, so simple assignment is sufficient (no need for clone)
        self._previous_frame_gpu = current_gray_gpu

        # Combine color and motion masks
        final_mask_gpu = cv2.cuda.bitwise_and(color_mask_gpu, motion_mask_gpu)

        no_red = cv2.cuda.countNonZero(final_mask_gpu)
        warm = 0
        frame_area = 0
        if self.relative_threshold is not None:
            warm = cv2.cuda.countNonZero(color_mask_gpu)
            width, height = frame.size()
            frame_area = width * height

        result_gpu = cv2.cuda.GpuMat(frame.size(), frame.type())
        result_gpu.setTo((0, 0, 0))
        frame.copyTo(result_gpu, final_mask_gpu)

        is_fire = is_fire_by_pixels(no_red, warm, self.margin, self.relative_threshold, frame_area)
        return is_fire, result_gpu.download()


class OpenCLFireDetector(CPUFireDetector):
    """Detects fire using OpenCL-accelerated OpenCV functions.

    This class inherits the CPU logic but operates on `cv2.UMat` frames,
    allowing OpenCV to dispatch operations to an OpenCL-compatible device
    (like an integrated or discrete GPU) if available.
    """

    _logger = get_logger("OpenCLFireDetector")

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


def _to_numpy(frame: np.ndarray | cv2.UMat | cv2.cuda.GpuMat) -> np.ndarray:
    """Returns the frame as a host numpy array (downloads GPU/UMat frames)."""
    if isinstance(frame, cv2.cuda.GpuMat):
        return frame.download()
    if isinstance(frame, cv2.UMat):
        return frame.get()
    return frame


class HybridFireDetector:
    """Two-stage fire detector: CV gatekeeper + pluggable verifier.

    This detector uses a lightweight local CV detector (e.g., CPUFireDetector)
    as a fast first pass. If the local detector finds potential fire, it then
    uses a `FireVerifier` (e.g. Gemini, Decisions API or an edge model) for
    high-accuracy verification. This approach minimizes verifier cost while
    reducing false positives.

    A verifier error (``VerifierError``: the verifier did not answer) is never treated as "no fire": it is not cached
    for the cooldown, the verifier is asked again on the next positive check, the first ``errors_before_policy - 1``
    errors are reported as inconclusive (``last_result_inconclusive``; ``StreamFireDetector`` keeps its state), and
    once ``errors_before_policy`` errors occurred in a row (or the hourly call cap is reached)
    the ``failure_policy`` decides: ``alert_unverified`` reports the gate's fire as an unverified detection
    (``last_result_unverified`` is True), ``suppress`` reports no fire. While unavailable, the verifier is probed at
    most once per ``cooldown_seconds`` and the policy result is reported in between. Entering and leaving that degraded state
    is logged (CRITICAL / INFO) and published through ``verifier_health.publish_verifier_status``.

    A positive verdict is reused for the whole cooldown. A negative verdict is not reused (the verifier is asked again
    on every CV-positive check) while a positive verdict is at most ``cooldown_seconds + extinguish_debounce_seconds``
    old (#188), nor for ``extinguish_debounce_seconds`` after the first negative verdict that follows a reported fire
    (a verified or reused positive, or an ``alert_unverified`` policy result; #191), so a single false negative cannot
    end an ongoing fire. Otherwise (e.g. a persistent CV false positive) a negative verdict is reused for the cooldown.
    """

    def __init__(
        self,
        local_detector: CPUFireDetector | CUDAFireDetector | OpenCLFireDetector,
        verifier: FireVerifier,
        cooldown_seconds: float | None = None,
        max_calls_per_hour: int | None = None,
        failure_policy: str | None = None,
        errors_before_policy: int | None = None,
        escalation_log: EscalationLog | None = None,
        extinguish_debounce_seconds: float | None = None,
    ) -> None:
        """Initializes the HybridFireDetector.

        Args:
            local_detector: A local CV-based detector for the fast first pass.
            verifier: The second-stage verifier (any `FireVerifier`) used to confirm detections.
            cooldown_seconds: Minimum time between verifier calls. While it runs, the last verdict is reused (so the
                debounce is not reset by a gap). Defaults to ``settings.verifier_cooldown_seconds``; 0 disables it.
                Only real verdicts are reused; an error is never cached, and a negative verdict shortly after a
                positive one, or within ``extinguish_debounce_seconds`` of the first negative after a reported fire,
                is not reused either (one false negative must not end an ongoing fire; see the class docstring).
            max_calls_per_hour: Hard cap of verifier calls in a rolling hour; once reached the verifier is not called
                and the verifier counts as unavailable (``failure_policy`` applies). Defaults to
                ``settings.verifier_max_calls_per_hour``; 0 means unlimited.
            failure_policy: ``alert_unverified`` or ``suppress``; what to report while the verifier is unavailable.
                Defaults to ``settings.verifier_failure_policy``.
            errors_before_policy: Consecutive verifier errors after which ``failure_policy`` applies (before that an
                error reports "no fire" and the verifier is re-asked on the next check). Defaults to
                ``settings.verifier_errors_before_policy``.
            escalation_log: Recorder of every frame sent to the verifier with its verdict. Defaults to the shared log
                from ``settings.escalation_log`` (None while it is disabled).
            extinguish_debounce_seconds: The fire-extinguished debounce of ``StreamFireDetector``. A negative verdict
                is not reused while a positive verdict is at most ``cooldown_seconds + extinguish_debounce_seconds``
                old, nor for this long after the first negative verdict following a reported fire (see the class
                docstring). Defaults to ``settings.fire_extinguished_debounce_seconds``.
        """
        self.escalation_log = escalation_log if escalation_log is not None else get_escalation_log()
        self.extinguish_debounce_seconds = (
            settings.fire_extinguished_debounce_seconds if extinguish_debounce_seconds is None else extinguish_debounce_seconds
        )
        self.local_detector = local_detector
        self.verifier = verifier
        self.cooldown_seconds = settings.verifier_cooldown_seconds if cooldown_seconds is None else cooldown_seconds
        self.max_calls_per_hour = settings.verifier_max_calls_per_hour if max_calls_per_hour is None else max_calls_per_hour
        self.failure_policy = settings.verifier_failure_policy if failure_policy is None else failure_policy
        self.errors_before_policy = (
            settings.verifier_errors_before_policy if errors_before_policy is None else errors_before_policy
        )
        #: True when the last ``detect()`` reported fire only because the verifier is unavailable.
        self.last_result_unverified = False
        #: True when the last ``detect()`` got a verifier error below ``errors_before_policy``: the result is
        #: neither fire nor no-fire, and ``StreamFireDetector`` keeps its debounce state for that frame.
        self.last_result_inconclusive = False
        self._last_call_time: float | None = None
        self._last_verdict = False
        self._last_positive_time: float | None = None
        #: The result last reported for a CV-positive check (inconclusive results excluded), see ``_record_verdict``.
        self._reported_fire = False
        #: Until when a negative verdict is not reused (#191); None when no hold is active.
        self._negative_hold_until: float | None = None
        self._last_attempt_time: float | None = None
        self._call_times: deque[float] = deque()
        self._limit_warned = False
        self._consecutive_errors = 0
        self._unavailable = False
        self._logger = get_logger("HybridFireDetector")
        self._logger.info(f"Initialized hybrid detector with verifier {type(verifier).__name__}")

    def _cap_reached(self, now: float) -> bool:
        """Applies the rolling hourly cap; records the call when it is allowed."""
        if not self.max_calls_per_hour:
            return False
        while self._call_times and now - self._call_times[0] >= 3600:
            self._call_times.popleft()
        if len(self._call_times) >= self.max_calls_per_hour:
            if not self._limit_warned:
                self._limit_warned = True
                self._mark_unavailable(f"call limit {self.max_calls_per_hour}/h reached")
            return True
        self._limit_warned = False
        self._call_times.append(now)
        return False

    def _mark_unavailable(self, reason: str) -> None:
        """Logs CRITICAL and notifies listeners once when the verifier becomes unavailable."""
        if self._unavailable:
            return
        self._unavailable = True
        outcome = (
            "fires seen by the local detector are reported as UNVERIFIED"
            if self.failure_policy == "alert_unverified"
            else "fires seen by the local detector are SUPPRESSED (no alert)"
        )
        self._logger.critical(
            "Verifier %s unavailable (%s); policy %s: %s until it recovers.",
            type(self.verifier).__name__,
            reason,
            self.failure_policy,
            outcome,
        )
        publish_verifier_status(
            VerifierStatus(available=False, verifier=type(self.verifier).__name__, reason=reason, policy=self.failure_policy)
        )

    def _mark_available(self) -> None:
        """Resets the error state after a real verdict and reports the verifier as available.

        The INFO log is per detector instance; the published status is de-duplicated by ``verifier_health``, so a
        recovery seen by a detector rebuilt after a reconnect still reaches the listeners exactly once.
        """
        self._consecutive_errors = 0
        if self._unavailable:
            self._unavailable = False
            self._logger.info("Verifier %s recovered; fires are verified again.", type(self.verifier).__name__)
        publish_verifier_status(
            VerifierStatus(
                available=True, verifier=type(self.verifier).__name__, reason="recovered", policy=self.failure_policy
            )
        )

    def _record_escalation(
        self, frame: np.ndarray, verdict: str, latency_ms: float, error_type: str | None = None, policy: str | None = None
    ) -> None:
        """Hands one verifier call to the escalation log (no-op when disabled; never raises)."""
        if self.escalation_log is None:
            return
        probability = getattr(self.verifier, "last_probability", None)
        self.escalation_log.record(
            frame,
            type(self.verifier).__name__,
            verdict,
            probability if isinstance(probability, int | float) and verdict != "error" else None,
            latency_ms,
            error_type,
            policy,
        )

    def _handle_verifier_error(self, exc: VerifierError, frame: np.ndarray, latency_ms: float) -> bool:
        """Counts a verifier error, records it and returns the result to report (False when inconclusive)."""
        self._consecutive_errors += 1
        inconclusive = self._consecutive_errors < self.errors_before_policy and not self._unavailable
        self._record_escalation(
            frame,
            "error",
            latency_ms,
            type(exc.__cause__ or exc).__name__,
            "inconclusive" if inconclusive else self.failure_policy,
        )
        if inconclusive:
            self._logger.warning(
                "Verifier error %d/%d (%s); result inconclusive, asking again on the next check.",
                self._consecutive_errors,
                self.errors_before_policy,
                exc,
            )
            self.last_result_inconclusive = True
            return False
        self._mark_unavailable(f"{self._consecutive_errors} consecutive errors, last: {exc}")
        return self._policy_result()

    def _can_reuse_verdict(self, now: float) -> bool:
        """Whether the last verdict is reused instead of calling the verifier (inside the cooldown)."""
        if not self.cooldown_seconds or self._last_call_time is None or now - self._last_call_time >= self.cooldown_seconds:
            return False
        return self._last_verdict or not (self._fire_recently_verified(now) or self._negative_hold_active(now))

    def _probe_throttled(self, now: float) -> bool:
        """Whether an unavailable verifier was already probed within the cooldown (the policy result is reported).

        Without this every CV-positive check would call a failing verifier (up to one call per check, each possibly
        blocking the detect loop for the verifier timeout). Errors below ``errors_before_policy`` are not throttled.
        """
        return (
            self._unavailable
            and bool(self.cooldown_seconds)
            and self._last_attempt_time is not None
            and now - self._last_attempt_time < self.cooldown_seconds
        )

    def _fire_recently_verified(self, now: float) -> bool:
        """Whether a positive verdict is recent enough that a negative one must not be reused (#188).

        A positive verdict is reused for the whole cooldown, so the next real call, which may be a false negative,
        comes at least ``cooldown_seconds`` after it. Reusing that negative for another cooldown would end the fire
        after the (shorter) extinguish debounce. Until ``cooldown_seconds + extinguish_debounce_seconds`` after the
        last positive verdict the verifier is therefore asked again on every CV-positive check instead; a persistent
        CV false positive without any positive verdict keeps the full cooldown.
        """
        if self._last_positive_time is None:
            return False
        return now < self._last_positive_time + self.cooldown_seconds + self.extinguish_debounce_seconds

    def _negative_hold_active(self, now: float) -> bool:
        """Whether a negative verdict must not be reused because it follows a reported fire too closely (#191).

        The #188 window is anchored at the last positive verdict, so a first real call that comes late (after a verifier
        outage reported as fire under ``alert_unverified``, or after a gap in CV-positive checks) can still return a
        false negative that would be reused for a whole cooldown and end the fire. The first negative verdict after a
        reported fire therefore starts a hold of ``extinguish_debounce_seconds`` during which the verifier is asked on
        every CV-positive check. Later negatives inside the hold do not extend it; a positive verdict clears it.
        """
        return self._negative_hold_until is not None and now < self._negative_hold_until

    def _record_verdict(self, verified: bool, now: float) -> None:
        """Stores a real verdict for reuse and updates the #188 anchor and the #191 negative hold."""
        self._last_verdict = verified
        if verified:
            self._last_positive_time = now
            self._negative_hold_until = None
        elif self._reported_fire:
            self._negative_hold_until = now + self.extinguish_debounce_seconds  # first negative after a reported fire

    def _report(self, result: bool) -> bool:
        """Remembers the result reported for a CV-positive check (unless inconclusive) and returns it."""
        if not self.last_result_inconclusive:
            self._reported_fire = result
        return result

    def _policy_result(self) -> bool:
        """Returns the result reported while the verifier is unavailable, per ``failure_policy``."""
        unverified = self.failure_policy == "alert_unverified"
        self.last_result_unverified = unverified
        return unverified

    async def detect(self, frame: np.ndarray | cv2.UMat | cv2.cuda.GpuMat) -> tuple[bool, np.ndarray]:
        """Analyzes a frame for fire using two-stage detection.

        Stage 1 (Gatekeeper): Uses the local CV detector for a fast check.
        If no fire is detected, returns immediately without calling the verifier.

        Stage 2 (Verifier): If local detection is positive, calls the
        verifier to confirm the detection and filter false positives. If the
        verifier is unavailable, the failure policy decides (see class docstring).

        Args:
            frame: The input video frame (BGR numpy array, UMat, or GpuMat).

        Returns:
            A tuple containing:
            - A boolean indicating if fire was confirmed by both detectors (or, under
              ``alert_unverified``, seen by the local detector while the verifier is unavailable).
            - The annotated frame from the local detector.
        """
        self.last_result_unverified = False
        self.last_result_inconclusive = False
        # Stage 1: Fast local check (gatekeeper)
        local_result, annotated_frame = await self.local_detector.detect(frame)

        if not local_result:
            # No fire detected locally - return immediately without verifier call
            return False, annotated_frame

        # Stage 2: verifier confirmation
        numpy_frame = _to_numpy(frame)  # verifiers require np.ndarray

        now = time.monotonic()
        if self._can_reuse_verdict(now):
            # reuse the last verdict instead of calling the API again
            return self._report(self._last_verdict), annotated_frame
        if self._probe_throttled(now):
            return self._report(self._policy_result()), annotated_frame  # unavailable: probe at most once per cooldown
        if self._cap_reached(now):
            if self.escalation_log is not None:
                self.escalation_log.record_capped(type(self.verifier).__name__, self.failure_policy)
            return self._report(self._policy_result()), annotated_frame
        previous_call_time = self._last_call_time
        self._last_call_time = now
        self._last_attempt_time = now

        started = perf_counter()
        try:
            verified = await self.verifier.verify(numpy_frame)
        except VerifierError as exc:
            self._last_call_time = previous_call_time  # an error is not a verdict: no cooldown, ask again next check
            result = self._handle_verifier_error(exc, numpy_frame, (perf_counter() - started) * 1000.0)
            return self._report(result), annotated_frame
        except BaseException:
            self._last_call_time = previous_call_time  # a failed call must not start the cooldown
            raise
        self._record_escalation(numpy_frame, "fire" if verified else "no_fire", (perf_counter() - started) * 1000.0)
        self._record_verdict(verified, now)
        self._mark_available()

        if not verified:
            self._logger.info(
                f"CV detector found fire but verifier {type(self.verifier).__name__} rejected it - false positive filtered."
            )

        return self._report(verified), annotated_frame


class UnverifiedLocalDetector:
    """The local CV detector used when the ``hybrid_edge`` verifier model cannot be loaded (#171).

    Every result is unverified, so ``VERIFIER_FAILURE_POLICY`` applies to each check: ``alert_unverified`` reports
    the local result (``last_result_unverified`` is True for a fire), ``suppress`` reports no fire (the local detector
    still runs, so its motion baseline and the annotated frame stay current).
    """

    def __init__(
        self, local_detector: CPUFireDetector | CUDAFireDetector | OpenCLFireDetector, failure_policy: str | None = None
    ) -> None:
        """Initializes the wrapper.

        Args:
            local_detector: The local CV detector to delegate to.
            failure_policy: ``alert_unverified`` or ``suppress``. Defaults to ``settings.verifier_failure_policy``.
        """
        self.local_detector = local_detector
        self.failure_policy = settings.verifier_failure_policy if failure_policy is None else failure_policy
        self.last_result_unverified = False

    async def detect(self, frame: np.ndarray | cv2.UMat | cv2.cuda.GpuMat) -> tuple[bool, np.ndarray]:
        """Runs the local detector and applies the failure policy to its result.

        Args:
            frame: The input video frame.

        Returns:
            A tuple of the reported fire state and the local detector's annotated frame.
        """
        fire, annotated_frame = await self.local_detector.detect(frame)
        reported = fire and self.failure_policy == "alert_unverified"
        self.last_result_unverified = reported
        return reported, annotated_frame


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

    relative_threshold: float | None = None
    if settings.fire_detection_mode == "area_relative":
        relative_threshold = settings.area_relative_threshold

    if should_use_cuda:
        return CUDAFireDetector(margin, lower, upper, threshold, relative_threshold)
    if should_use_open_cl:
        return OpenCLFireDetector(margin, lower, upper, threshold, relative_threshold)
    return CPUFireDetector(margin, lower, upper, threshold, relative_threshold)


def clear_edge_verifier_cache() -> None:
    """Drops every cached ``EdgeInferenceDetector`` so the next ``hybrid_edge`` factory call rebuilds it.

    Intended for tests (to prevent state leaking between them) and for callers that need to force a model reload.
    Also forgets an earlier model load failure (see ``create_fire_detector``).
    """
    global _edge_load_failed  # noqa: PLW0603
    with _edge_verifier_cache_lock:
        _edge_verifier_cache.clear()
        _edge_load_failed = False


def _edge_cache_key(edge_settings: EdgeSettings) -> tuple:
    """Builds the cache key from everything that determines the edge verifier.

    The model file's mtime and size are part of the key, so a replaced model file is picked up on the next call.

    Args:
        edge_settings: The edge configuration.

    Returns:
        A hashable key.
    """
    model_path = edge_settings.model_path
    resolved: str | None = None
    mtime_ns: int | None = None
    size: int | None = None
    if model_path:
        path = Path(model_path)
        try:
            resolved = str(path.resolve())
            stat = path.stat()
            mtime_ns, size = stat.st_mtime_ns, stat.st_size
        except OSError:
            resolved = resolved or str(model_path)
    return (
        resolved,
        mtime_ns,
        size,
        str(edge_settings.accelerator),
        edge_settings.input_size,
        edge_settings.confidence_threshold,
    )


def _get_edge_verifier(edge_settings: EdgeSettings) -> EdgeInferenceDetector:
    """Returns a cached ``EdgeInferenceDetector`` for these settings, building (and caching) it on a miss.

    A cached verifier that reports ``needs_rebuild`` (inference errors) is replaced. A failed build raises and is
    NOT cached, so the next call retries.

    Args:
        edge_settings: The edge configuration.

    Returns:
        The (possibly shared) verifier.
    """
    from is_goat_burning.fire_detection.edge_detector import EdgeInferenceDetector

    key = _edge_cache_key(edge_settings)
    with _edge_verifier_cache_lock:
        verifier = _edge_verifier_cache.get(key)
        if verifier is not None and verifier.needs_rebuild:
            _factory_logger.warning("Rebuilding the edge verifier after inference errors.")
            verifier = None
        if verifier is None:
            # Drop our reference first so the old session can be freed as soon as no detector uses it.
            _edge_verifier_cache.clear()
            verifier = EdgeInferenceDetector(edge_settings)
            _edge_verifier_cache[key] = verifier
        else:
            _factory_logger.info("Reusing the cached edge verifier (model unchanged).")
        return verifier


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
        strategy: The detection strategy to use ("classic", "gemini",
                  "hybrid_cloud", "hybrid_edge", "gemini_decision",
                  "hybrid_decision", or the legacy alias "hybrid"). If None,
                  uses `settings.detection_strategy`.

    Returns:
        An instance of a class that conforms to the FireDetector protocol. For
        ``hybrid_edge`` this is a ``HybridFireDetector``, or, if the edge model
        cannot be loaded at startup, an ``UnverifiedLocalDetector`` (see Note).

    Note:
        A missing or unset ``EDGE__MODEL_PATH`` is rejected earlier, at settings
        validation (fail fast on misconfiguration). The factory itself covers
        model LOAD failures only: if building the edge verifier raises (corrupt
        or invalid model, input-size mismatch, unavailable accelerator, missing
        onnxruntime, unexpected errors), it logs CRITICAL and falls back to the
        local detector (classic rule; honours CUDA/OpenCL settings) wrapped in
        ``UnverifiedLocalDetector``, which applies ``VERIFIER_FAILURE_POLICY``:
        fires are reported marked unverified (``alert_unverified``) or not at
        all (``suppress``), instead of startup failing.

        The built edge verifier is cached across calls (keyed by resolved model path, file mtime and size,
        accelerator, input size and confidence threshold), so a stream reconnect does not reload the model; only
        successes are cached, a failed build is retried (and re-logged) on the next call. The local detector is
        always created fresh (its motion baseline must not survive a reconnect).
    """
    global _edge_load_failed  # noqa: PLW0603 - module-level load-failure state, like the verifier cache
    selected_strategy = strategy if strategy is not None else settings.detection_strategy

    # "hybrid" is the deprecated alias for "hybrid_cloud" (see Settings migration).
    cloud_hybrid_strategies = ("hybrid_cloud", "hybrid")

    if selected_strategy == "hybrid_edge":
        from is_goat_burning.fire_detection.edge_detector import EdgeModelError

        edge_local_detector = _create_local_detector(margin, lower, upper, use_open_cl, use_cuda)
        try:
            edge_detector = HybridFireDetector(edge_local_detector, _get_edge_verifier(settings.edge))
        except Exception as exc:  # noqa: BLE001 - any load failure must degrade to the local detector, never abort startup
            fallback = UnverifiedLocalDetector(edge_local_detector)
            outcome = (
                "fires seen by it will NOT be reported (VERIFIER_FAILURE_POLICY=suppress)"
                if fallback.failure_policy == "suppress"
                else "fires will still be reported, but marked UNVERIFIED (no edge-model verification)"
            )
            _factory_logger.critical(
                "hybrid_edge verification is unavailable (%s). Falling back to the local detector (classic rule): %s.",
                exc,
                outcome,
                exc_info=not isinstance(exc, EdgeModelError),
            )
            # The fallback applies VERIFIER_FAILURE_POLICY to every check (see UnverifiedLocalDetector).
            _edge_load_failed = True
            publish_verifier_status(
                VerifierStatus(
                    available=False,
                    verifier=EDGE_VERIFIER_NAME,
                    reason=f"model load failed: {type(exc).__name__}",
                    policy=fallback.failure_policy,
                )
            )
            return fallback
        if _edge_load_failed:
            # Only a load failure is cleared here; runtime unavailability (errors, call cap) recovers on a real verdict.
            _edge_load_failed = False
            publish_verifier_status(
                VerifierStatus(
                    available=True, verifier=EDGE_VERIFIER_NAME, reason="model loaded", policy=edge_detector.failure_policy
                )
            )
        return edge_detector

    if selected_strategy in ("gemini_decision", "hybrid_decision"):
        from is_goat_burning.fire_detection.decision_detector import DecisionFireDetector

        if selected_strategy == "gemini_decision":
            return DecisionFireDetector()
        return HybridFireDetector(
            _create_local_detector(margin, lower, upper, use_open_cl, use_cuda),
            DecisionFireDetector(),
        )

    if selected_strategy == "gemini" or selected_strategy in cloud_hybrid_strategies:
        from is_goat_burning.fire_detection.gemini_detector import GeminiFireDetector

    if selected_strategy == "gemini":
        return GeminiFireDetector()

    # For both "classic" and the cloud hybrid strategies, we need a local detector.
    local_detector = _create_local_detector(margin, lower, upper, use_open_cl, use_cuda)

    if selected_strategy in cloud_hybrid_strategies:
        return HybridFireDetector(local_detector, GeminiFireDetector())
    # "classic" strategy
    return local_detector
