import cv2
import numpy as np
import pytest

from is_goat_burning.config import settings
from is_goat_burning.fire_detection.detectors import CPUFireDetector
from is_goat_burning.fire_detection.detectors import CUDAFireDetector
from is_goat_burning.fire_detection.detectors import OpenCLFireDetector
from is_goat_burning.fire_detection.detectors import create_fire_detector

# --- Test Setup ---

# Define a standard color range for fire used across tests
LOWER_HSV = np.array([18, 50, 50], dtype="uint8")
UPPER_HSV = np.array([35, 255, 255], dtype="uint8")
FIRE_COLOR_HSV = (25, 200, 200)  # A color within the fire range
NON_FIRE_COLOR_HSV = (100, 200, 200)  # A color outside the fire range (blueish)


def create_test_image(color_hsv: tuple[int, int, int], size: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Creates a BGR image of a solid color, converted from HSV."""
    h, w = size
    hsv_image = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_image[:] = color_hsv
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


# --- Test Cases ---


def test_cpu_fire_detector():
    """Tests the CPUFireDetector with fire and no-fire images."""
    detector = CPUFireDetector(margin=100, lower=LOWER_HSV, upper=UPPER_HSV)

    # Test with an image that should trigger fire detection
    fire_image = create_test_image(FIRE_COLOR_HSV)
    is_fire, _ = detector.detect(fire_image)
    assert is_fire is True, "CPU detector failed to detect fire"

    # Test with an image that should NOT trigger fire detection
    no_fire_image = create_test_image(NON_FIRE_COLOR_HSV)
    is_fire, _ = detector.detect(no_fire_image)
    assert is_fire is False, "CPU detector incorrectly detected fire"


@pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0, reason="No CUDA-enabled GPU found")
def test_cuda_fire_detector():
    """Tests the CUDAFireDetector. Skipped if no CUDA GPU is available."""
    detector = CUDAFireDetector(margin=100, lower=LOWER_HSV, upper=UPPER_HSV)
    gpu_frame = cv2.cuda.GpuMat()

    # Test with fire image
    fire_image = create_test_image(FIRE_COLOR_HSV)
    gpu_frame.upload(fire_image)
    is_fire, _ = detector.detect(gpu_frame)
    assert is_fire is True, "CUDA detector failed to detect fire"

    # Test with no-fire image
    no_fire_image = create_test_image(NON_FIRE_COLOR_HSV)
    gpu_frame.upload(no_fire_image)
    is_fire, _ = detector.detect(gpu_frame)
    assert is_fire is False, "CUDA detector incorrectly detected fire"


@pytest.mark.skipif(not cv2.ocl.haveOpenCL(), reason="OpenCL is not available/enabled in this OpenCV build")
def test_opencl_fire_detector():
    """Tests the OpenCLFireDetector. Skipped if OpenCL is not available."""
    detector = OpenCLFireDetector(margin=100, lower=LOWER_HSV, upper=UPPER_HSV)

    # Test with fire image, uploaded as a UMat for OpenCL processing
    fire_image = create_test_image(FIRE_COLOR_HSV)
    umat_frame = cv2.UMat(fire_image)
    is_fire, _ = detector.detect(umat_frame)
    assert is_fire is True, "OpenCL detector failed to detect fire"

    # Test with no-fire image
    no_fire_image = create_test_image(NON_FIRE_COLOR_HSV)
    umat_frame = cv2.UMat(no_fire_image)
    is_fire, _ = detector.detect(umat_frame)
    assert is_fire is False, "OpenCL detector incorrectly detected fire"


def test_create_fire_detector_factory(monkeypatch):
    """Tests the create_fire_detector factory function."""
    # Test CPU case
    monkeypatch.setattr(settings, "open_cl", False)
    monkeypatch.setattr(settings, "cuda", False)
    detector = create_fire_detector(100, LOWER_HSV, UPPER_HSV)
    assert isinstance(detector, CPUFireDetector)

    # Test CUDA case
    monkeypatch.setattr(settings, "open_cl", False)
    monkeypatch.setattr(settings, "cuda", True)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        detector = create_fire_detector(100, LOWER_HSV, UPPER_HSV)
        assert isinstance(detector, CUDAFireDetector)

    # Test OpenCL case
    monkeypatch.setattr(settings, "open_cl", True)
    monkeypatch.setattr(settings, "cuda", False)
    if cv2.ocl.haveOpenCL():
        detector = create_fire_detector(100, LOWER_HSV, UPPER_HSV)
        assert isinstance(detector, OpenCLFireDetector)
