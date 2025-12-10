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


# --- Helper Functions for Parametrized Tests ---


def to_cpu(img: np.ndarray) -> np.ndarray:
    """Returns the image as a NumPy array (CPU)."""
    return img


def to_cuda(img: np.ndarray) -> cv2.cuda.GpuMat:
    """Returns the image as a GpuMat (CUDA)."""
    gpu_frame = cv2.cuda.GpuMat()
    gpu_frame.upload(img)
    return gpu_frame


def to_opencl(img: np.ndarray) -> cv2.UMat:
    """Returns the image as a UMat (OpenCL)."""
    return cv2.UMat(img)


# --- Parametrized Test Cases ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detector_class, to_device_func",
    [
        (CPUFireDetector, to_cpu),
        pytest.param(
            CUDAFireDetector,
            to_cuda,
            marks=pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0, reason="No CUDA-enabled GPU found"),
        ),
        pytest.param(
            OpenCLFireDetector,
            to_opencl,
            marks=pytest.mark.skipif(not cv2.ocl.haveOpenCL(), reason="OpenCL is not available/enabled"),
        ),
    ],
)
async def test_detector_filters_static_false_positive(detector_class, to_device_func) -> None:
    """Tests that static orange frames are filtered after the first frame for all detectors."""
    detector = detector_class(margin=TEST_MARGIN, lower=LOWER_HSV, upper=UPPER_HSV, motion_threshold=MOTION_THRESHOLD)

    static_fire_image = create_test_image(FIRE_COLOR_HSV)
    frame = to_device_func(static_fire_image)

    # Frame 1: First frame, should detect fire (motion assumed)
    # Note: For stateful detectors, we might need to re-upload/create frame if it's modified in place,
    # but here detect() shouldn't modify the input frame structure significantly for next calls
    # except for UMat/GpuMat where we might need fresh inputs if they were consumed/modified?
    # Actually, our detectors don't modify input frame content destructively for detection logic.
    # But to be safe and simulate real stream where new frames come in:

    is_fire, _ = await detector.detect(frame)
    assert is_fire is True, "Frame 1: Should detect fire on first frame"

    # Frame 2: Identical frame, should NOT detect fire (no motion)
    # Re-create frame to ensure distinct object if needed, though content is same
    frame = to_device_func(static_fire_image)
    is_fire, _ = await detector.detect(frame)
    assert is_fire is False, "Frame 2: Should filter static false positive"

    # Frame 3: Still identical, should NOT detect fire
    frame = to_device_func(static_fire_image)
    is_fire, _ = await detector.detect(frame)
    assert is_fire is False, "Frame 3: Should continue filtering static false positive"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detector_class, to_device_func",
    [
        (CPUFireDetector, to_cpu),
        pytest.param(
            CUDAFireDetector,
            to_cuda,
            marks=pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0, reason="No CUDA-enabled GPU found"),
        ),
        pytest.param(
            OpenCLFireDetector,
            to_opencl,
            marks=pytest.mark.skipif(not cv2.ocl.haveOpenCL(), reason="OpenCL is not available/enabled"),
        ),
    ],
)
async def test_detector_detects_dynamic_fire(detector_class, to_device_func) -> None:
    """Tests that dynamic fire-like frames with motion are detected for all detectors."""
    detector = detector_class(margin=TEST_MARGIN, lower=LOWER_HSV, upper=UPPER_HSV, motion_threshold=MOTION_THRESHOLD)

    # Create a sequence of frames with varying intensity (simulating fire flicker)
    frame1_np = create_varied_fire_image(FIRE_COLOR_HSV, value_offset=0)
    frame2_np = create_varied_fire_image(FIRE_COLOR_HSV, value_offset=PIXEL_INTENSITY_CHANGE)
    frame3_np = create_varied_fire_image(FIRE_COLOR_HSV, value_offset=-PIXEL_INTENSITY_CHANGE)

    # Frame 1: First frame, should detect fire (motion assumed)
    frame1 = to_device_func(frame1_np)
    is_fire, _ = await detector.detect(frame1)
    assert is_fire is True, "Frame 1: Should detect fire on first frame"

    # Frame 2: Different intensity, should detect fire (motion detected)
    frame2 = to_device_func(frame2_np)
    is_fire, _ = await detector.detect(frame2)
    assert is_fire is True, "Frame 2: Should detect dynamic fire"

    # Frame 3: Different intensity again, should detect fire
    frame3 = to_device_func(frame3_np)
    is_fire, _ = await detector.detect(frame3)
    assert is_fire is True, "Frame 3: Should continue detecting dynamic fire"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detector_class, to_device_func",
    [
        (CPUFireDetector, to_cpu),
        pytest.param(
            CUDAFireDetector,
            to_cuda,
            marks=pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0, reason="No CUDA-enabled GPU found"),
        ),
        pytest.param(
            OpenCLFireDetector,
            to_opencl,
            marks=pytest.mark.skipif(not cv2.ocl.haveOpenCL(), reason="OpenCL is not available/enabled"),
        ),
    ],
)
async def test_detector_handles_resolution_change(detector_class, to_device_func) -> None:
    """Tests that detectors handle resolution changes gracefully by resetting motion baseline."""
    detector = detector_class(margin=TEST_MARGIN, lower=LOWER_HSV, upper=UPPER_HSV, motion_threshold=MOTION_THRESHOLD)

    # Frame 1: Size 100x100
    frame1_np = create_varied_fire_image(FIRE_COLOR_HSV, size=(100, 100))
    frame1 = to_device_func(frame1_np)
    is_fire, _ = await detector.detect(frame1)
    assert is_fire is True, "Frame 1 (100x100): Should detect fire on first frame"

    # Frame 2: Size 100x100 (Identical) -> Should be filtered
    frame2 = to_device_func(frame1_np)
    is_fire, _ = await detector.detect(frame2)
    assert is_fire is False, "Frame 2 (100x100): Should filter static false positive"

    # Frame 3: Size 150x150 (Resolution Change) -> Should reset and detect fire (treated as first frame)
    frame3_np = create_varied_fire_image(FIRE_COLOR_HSV, size=(150, 150))
    frame3 = to_device_func(frame3_np)
    is_fire, _ = await detector.detect(frame3)
    assert is_fire is True, "Frame 3 (150x150): Should detect fire after resolution change (reset)"

    # Frame 4: Size 150x150 (Identical to Frame 3) -> Should be filtered
    frame4 = to_device_func(frame3_np)
    is_fire, _ = await detector.detect(frame4)
    assert is_fire is False, "Frame 4 (150x150): Should filter static false positive after reset"
