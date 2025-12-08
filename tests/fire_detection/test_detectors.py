"""Unit tests for the fire detector implementations in `detectors.py`.

This module tests the CPU, CUDA, and OpenCL fire detection algorithms, as well
as the `create_fire_detector` factory function. Tests for CUDA and OpenCL are
skipped if the necessary hardware or libraries are not available.
"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from is_goat_burning.fire_detection.detectors import CPUFireDetector
from is_goat_burning.fire_detection.detectors import CUDAFireDetector
from is_goat_burning.fire_detection.detectors import OpenCLFireDetector
from is_goat_burning.fire_detection.detectors import create_fire_detector

# --- Test Setup ---

LOWER_HSV = np.array([18, 50, 50], dtype="uint8")
UPPER_HSV = np.array([35, 255, 255], dtype="uint8")
FIRE_COLOR_HSV = (25, 200, 200)
NON_FIRE_COLOR_HSV = (100, 200, 200)


def create_test_image(color_hsv: tuple[int, int, int], size: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Creates a solid-color BGR image from an HSV color tuple."""
    h, w = size
    hsv_image = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_image[:] = color_hsv
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


# --- Test Cases ---


@pytest.mark.asyncio
async def test_cpu_fire_detector() -> None:
    """Tests the CPUFireDetector with both fire and non-fire images."""
    detector = CPUFireDetector(margin=100, lower=LOWER_HSV, upper=UPPER_HSV)

    fire_image = create_test_image(FIRE_COLOR_HSV)
    is_fire, _ = await detector.detect(fire_image)
    assert is_fire is True, "CPU detector failed to detect fire"

    no_fire_image = create_test_image(NON_FIRE_COLOR_HSV)
    is_fire, _ = await detector.detect(no_fire_image)
    assert is_fire is False, "CPU detector incorrectly detected fire"


@pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0, reason="No CUDA-enabled GPU found")
@pytest.mark.asyncio
async def test_cuda_fire_detector() -> None:
    """Tests the CUDAFireDetector with both fire and non-fire images."""
    detector = CUDAFireDetector(margin=100, lower=LOWER_HSV, upper=UPPER_HSV)
    gpu_frame = cv2.cuda.GpuMat()

    fire_image = create_test_image(FIRE_COLOR_HSV)
    gpu_frame.upload(fire_image)
    is_fire, _ = await detector.detect(gpu_frame)
    assert is_fire is True, "CUDA detector failed to detect fire"

    no_fire_image = create_test_image(NON_FIRE_COLOR_HSV)
    gpu_frame.upload(no_fire_image)
    is_fire, _ = await detector.detect(gpu_frame)
    assert is_fire is False, "CUDA detector incorrectly detected fire"


@pytest.mark.skipif(not cv2.ocl.haveOpenCL(), reason="OpenCL is not available/enabled in this OpenCV build")
@pytest.mark.asyncio
async def test_opencl_fire_detector() -> None:
    """Tests the OpenCLFireDetector with both fire and non-fire images."""
    detector = OpenCLFireDetector(margin=100, lower=LOWER_HSV, upper=UPPER_HSV)

    fire_image = create_test_image(FIRE_COLOR_HSV)
    umat_frame = cv2.UMat(fire_image)
    is_fire, _ = await detector.detect(umat_frame)
    assert is_fire is True, "OpenCL detector failed to detect fire"

    no_fire_image = create_test_image(NON_FIRE_COLOR_HSV)
    umat_frame = cv2.UMat(no_fire_image)
    is_fire, _ = await detector.detect(umat_frame)
    assert is_fire is False, "OpenCL detector incorrectly detected fire"


@pytest.fixture
def mock_settings_classic():
    """Mocks settings to ensure classic strategy is used."""
    with patch("is_goat_burning.fire_detection.detectors.settings") as mock:
        mock.detection_strategy = "classic"
        yield mock


def test_create_fire_detector_factory_returns_cpu_by_default(mock_settings_classic) -> None:  # noqa: ARG001
    """Verifies the factory returns a CPUFireDetector when no flags are set."""
    detector = create_fire_detector(100, LOWER_HSV, UPPER_HSV, use_open_cl=False, use_cuda=False)
    assert isinstance(detector, CPUFireDetector)


@pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0, reason="No CUDA-enabled GPU found")
def test_create_fire_detector_factory_returns_cuda_when_requested(mock_settings_classic) -> None:  # noqa: ARG001
    """Verifies the factory returns a CUDAFireDetector when requested."""
    detector = create_fire_detector(100, LOWER_HSV, UPPER_HSV, use_open_cl=False, use_cuda=True)
    assert isinstance(detector, CUDAFireDetector)


@pytest.mark.skipif(not cv2.ocl.haveOpenCL(), reason="OpenCL is not available/enabled in this OpenCV build")
def test_create_fire_detector_factory_returns_opencl_when_requested(mock_settings_classic) -> None:  # noqa: ARG001
    """Verifies the factory returns an OpenCLFireDetector when requested."""
    detector = create_fire_detector(100, LOWER_HSV, UPPER_HSV, use_open_cl=True, use_cuda=False)
    assert isinstance(detector, OpenCLFireDetector)


# --- Motion Verification Tests ---

# Constants for motion tests
MOTION_THRESHOLD = 25
PIXEL_INTENSITY_CHANGE = 50  # Must be > MOTION_THRESHOLD to be detected as motion
TEST_MARGIN = 100  # Fire detection threshold for tests
MIN_PIXEL_VALUE = 0
MAX_PIXEL_VALUE = 255


def create_varied_fire_image(
    color_hsv: tuple[int, int, int],
    size: tuple[int, int] = (100, 100),
    value_offset: int = 0,
) -> np.ndarray:
    """Creates a fire-colored BGR image with optional value offset for motion simulation."""
    h, w = size
    hsv_image = np.zeros((h, w, 3), dtype=np.uint8)
    h_val, s_val, v_val = color_hsv
    # Apply offset to value channel, clamping to valid range
    v_val_adjusted = max(MIN_PIXEL_VALUE, min(MAX_PIXEL_VALUE, v_val + value_offset))
    hsv_image[:] = (h_val, s_val, v_val_adjusted)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


@pytest.mark.asyncio
async def test_cpu_detector_filters_static_false_positive() -> None:
    """Tests that static orange frames are filtered after the first frame.

    This simulates a static false positive like an orange wall lit by sunset.
    Frame 1: Fire detected (first frame, motion assumed valid).
    Frame 2+: No fire detected (identical frames = no motion).
    """
    detector = CPUFireDetector(margin=TEST_MARGIN, lower=LOWER_HSV, upper=UPPER_HSV, motion_threshold=MOTION_THRESHOLD)

    static_fire_image = create_test_image(FIRE_COLOR_HSV)

    # Frame 1: First frame, should detect fire (motion assumed)
    is_fire, _ = await detector.detect(static_fire_image)
    assert is_fire is True, "Frame 1: Should detect fire on first frame"

    # Frame 2: Identical frame, should NOT detect fire (no motion)
    is_fire, _ = await detector.detect(static_fire_image)
    assert is_fire is False, "Frame 2: Should filter static false positive"

    # Frame 3: Still identical, should NOT detect fire
    is_fire, _ = await detector.detect(static_fire_image)
    assert is_fire is False, "Frame 3: Should continue filtering static false positive"


@pytest.mark.asyncio
async def test_cpu_detector_detects_dynamic_fire() -> None:
    """Tests that dynamic fire-like frames with motion are detected.

    Real fire flickers and changes intensity rapidly. This simulates
    that by varying the brightness (V channel) between frames.
    """
    detector = CPUFireDetector(margin=TEST_MARGIN, lower=LOWER_HSV, upper=UPPER_HSV, motion_threshold=MOTION_THRESHOLD)

    # Create a sequence of frames with varying intensity (simulating fire flicker)
    frame1 = create_varied_fire_image(FIRE_COLOR_HSV, value_offset=0)
    frame2 = create_varied_fire_image(FIRE_COLOR_HSV, value_offset=PIXEL_INTENSITY_CHANGE)
    frame3 = create_varied_fire_image(FIRE_COLOR_HSV, value_offset=-PIXEL_INTENSITY_CHANGE)

    # Frame 1: First frame, should detect fire (motion assumed)
    is_fire, _ = await detector.detect(frame1)
    assert is_fire is True, "Frame 1: Should detect fire on first frame"

    # Frame 2: Different intensity, should detect fire (motion detected)
    is_fire, _ = await detector.detect(frame2)
    assert is_fire is True, "Frame 2: Should detect dynamic fire"

    # Frame 3: Different intensity again, should detect fire
    is_fire, _ = await detector.detect(frame3)
    assert is_fire is True, "Frame 3: Should continue detecting dynamic fire"


@pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0, reason="No CUDA-enabled GPU found")
@pytest.mark.asyncio
async def test_cuda_detector_filters_static_false_positive() -> None:
    """Tests that static orange frames are filtered by CUDA detector after frame 1."""
    detector = CUDAFireDetector(margin=TEST_MARGIN, lower=LOWER_HSV, upper=UPPER_HSV, motion_threshold=MOTION_THRESHOLD)
    gpu_frame = cv2.cuda.GpuMat()

    static_fire_image = create_test_image(FIRE_COLOR_HSV)
    gpu_frame.upload(static_fire_image)

    # Frame 1: First frame, should detect fire
    is_fire, _ = await detector.detect(gpu_frame)
    assert is_fire is True, "Frame 1: Should detect fire on first frame"

    # Frame 2: Identical frame, should NOT detect fire
    is_fire, _ = await detector.detect(gpu_frame)
    assert is_fire is False, "Frame 2: Should filter static false positive"


@pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0, reason="No CUDA-enabled GPU found")
@pytest.mark.asyncio
async def test_cuda_detector_detects_dynamic_fire() -> None:
    """Tests that dynamic fire-like frames with motion are detected by CUDA detector."""
    detector = CUDAFireDetector(margin=TEST_MARGIN, lower=LOWER_HSV, upper=UPPER_HSV, motion_threshold=MOTION_THRESHOLD)
    gpu_frame = cv2.cuda.GpuMat()

    frame1 = create_varied_fire_image(FIRE_COLOR_HSV, value_offset=0)
    frame2 = create_varied_fire_image(FIRE_COLOR_HSV, value_offset=PIXEL_INTENSITY_CHANGE)

    gpu_frame.upload(frame1)
    is_fire, _ = await detector.detect(gpu_frame)
    assert is_fire is True, "Frame 1: Should detect fire on first frame"

    gpu_frame.upload(frame2)
    is_fire, _ = await detector.detect(gpu_frame)
    assert is_fire is True, "Frame 2: Should detect dynamic fire"


@pytest.mark.skipif(not cv2.ocl.haveOpenCL(), reason="OpenCL is not available/enabled in this OpenCV build")
@pytest.mark.asyncio
async def test_opencl_detector_filters_static_false_positive() -> None:
    """Tests that static orange frames are filtered by OpenCL detector after frame 1."""
    detector = OpenCLFireDetector(margin=TEST_MARGIN, lower=LOWER_HSV, upper=UPPER_HSV, motion_threshold=MOTION_THRESHOLD)

    static_fire_image = create_test_image(FIRE_COLOR_HSV)

    # Frame 1: First frame, should detect fire
    umat_frame = cv2.UMat(static_fire_image)
    is_fire, _ = await detector.detect(umat_frame)
    assert is_fire is True, "Frame 1: Should detect fire on first frame"

    # Frame 2: Identical frame, should NOT detect fire
    umat_frame = cv2.UMat(static_fire_image)
    is_fire, _ = await detector.detect(umat_frame)
    assert is_fire is False, "Frame 2: Should filter static false positive"


@pytest.mark.skipif(not cv2.ocl.haveOpenCL(), reason="OpenCL is not available/enabled in this OpenCV build")
@pytest.mark.asyncio
async def test_opencl_detector_detects_dynamic_fire() -> None:
    """Tests that dynamic fire-like frames with motion are detected by OpenCL detector."""
    detector = OpenCLFireDetector(margin=TEST_MARGIN, lower=LOWER_HSV, upper=UPPER_HSV, motion_threshold=MOTION_THRESHOLD)

    frame1 = create_varied_fire_image(FIRE_COLOR_HSV, value_offset=0)
    frame2 = create_varied_fire_image(FIRE_COLOR_HSV, value_offset=PIXEL_INTENSITY_CHANGE)

    umat_frame = cv2.UMat(frame1)
    is_fire, _ = await detector.detect(umat_frame)
    assert is_fire is True, "Frame 1: Should detect fire on first frame"

    umat_frame = cv2.UMat(frame2)
    is_fire, _ = await detector.detect(umat_frame)
    assert is_fire is True, "Frame 2: Should detect dynamic fire"
