"""Tests for the area-relative decision rule (#108)."""

from unittest.mock import patch

import cv2
import numpy as np
from pydantic import ValidationError
import pytest

from is_goat_burning.config import Settings
from is_goat_burning.fire_detection.detectors import CPUFireDetector
from is_goat_burning.fire_detection.detectors import OpenCLFireDetector
from is_goat_burning.fire_detection.detectors import _create_local_detector
from is_goat_burning.fire_detection.detectors import is_fire_by_pixels

HEIGHT, WIDTH = 240, 320
AREA = HEIGHT * WIDTH
LOWER = np.array([18, 50, 50], dtype="uint8")
UPPER = np.array([35, 255, 255], dtype="uint8")
FLAME_BGR = (0, 200, 255)


def _frame(x: int, size: int, colour: tuple[int, int, int] = FLAME_BGR) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), 20, dtype=np.uint8)
    frame[100 : 100 + size, x : x + size] = colour
    return frame


def test_frame_fraction_mode_is_unchanged() -> None:
    assert is_fire_by_pixels(101, 5000, margin=100) is True
    assert is_fire_by_pixels(100, 5000, margin=100) is False


def test_area_relative_detects_small_mostly_moving_flame() -> None:
    warm = int(0.01 * AREA)  # 1 % of the frame, far below a 10 % frame threshold
    assert is_fire_by_pixels(int(0.6 * warm), warm, margin=int(0.1 * AREA), relative_threshold=0.23, frame_area=AREA)


@pytest.mark.parametrize(
    ("moving_warm", "warm"),
    [
        (int(0.05 * 0.3 * AREA), int(0.3 * AREA)),  # large warm region (sunset wall), little of it moving
        (10, 20),  # warm region below 0.2 % of the frame
        (int(0.0004 * AREA), int(0.001 * AREA)),  # nothing worth calling motion
    ],
)
def test_area_relative_rejects(moving_warm: int, warm: int) -> None:
    assert not is_fire_by_pixels(moving_warm, warm, margin=int(0.1 * AREA), relative_threshold=0.23, frame_area=AREA)


def _detector(relative: float | None) -> CPUFireDetector:
    return CPUFireDetector(
        margin=int(0.1 * AREA), lower=LOWER, upper=UPPER, motion_threshold=25, relative_threshold=relative, frame_area=AREA
    )


@pytest.mark.asyncio
async def test_detector_small_flickering_flame_only_in_area_relative_mode() -> None:
    frames = [_frame(40, 24), _frame(80, 24)]  # 24x24 block (0.75 % of the frame) jumping sideways

    classic = _detector(None)
    relative = _detector(0.23)
    classic_results = [(await classic.detect(f))[0] for f in frames]
    relative_results = [(await relative.detect(f))[0] for f in frames]

    assert classic_results == [False, False]
    assert relative_results == [False, True]  # first frame never alarms, second has motion


@pytest.mark.asyncio
async def test_detector_static_warm_scene_is_rejected_in_area_relative_mode() -> None:
    detector = _detector(0.23)
    frame = _frame(40, 90)  # big static warm block
    assert [(await detector.detect(frame))[0] for _ in range(3)] == [False, False, False]


def test_factory_passes_mode_and_requires_frame_area() -> None:
    with patch("is_goat_burning.fire_detection.detectors.settings") as mock_settings:
        mock_settings.fire_detection_mode = "area_relative"
        mock_settings.area_relative_threshold = 0.3
        mock_settings.motion_detection_threshold = 25
        mock_settings.open_cl = False
        mock_settings.cuda = False

        detector = _create_local_detector(10, LOWER, UPPER, frame_area=AREA)
        assert detector.relative_threshold == 0.3
        assert detector.frame_area == AREA

        with pytest.raises(ValueError, match="frame area"):
            _create_local_detector(10, LOWER, UPPER)


def test_factory_default_mode_is_frame_fraction() -> None:
    with patch("is_goat_burning.fire_detection.detectors.settings") as mock_settings:
        mock_settings.fire_detection_mode = "frame_fraction"
        mock_settings.motion_detection_threshold = 25
        mock_settings.open_cl = False
        mock_settings.cuda = False
        assert _create_local_detector(10, LOWER, UPPER).relative_threshold is None


def test_settings_validate_mode_and_threshold() -> None:
    assert Settings(SOURCE="x", _env_file=None).fire_detection_mode == "frame_fraction"
    with pytest.raises(ValidationError):
        Settings(SOURCE="x", FIRE_DETECTION_MODE="nope", _env_file=None)
    with pytest.raises(ValidationError):
        Settings(SOURCE="x", AREA_RELATIVE_THRESHOLD=0, _env_file=None)


def test_constants_match_evaluation_harness() -> None:
    """The shipped rule must stay identical to the one evaluated in ``tools.fire_rule_eval``."""
    from is_goat_burning.fire_detection import detectors
    from tools.fire_rule_eval import rules

    assert detectors.MIN_WARM_FRACTION == rules.MIN_WARM_FRACTION
    assert detectors.MIN_MOVING_FRACTION == rules.MIN_MOVING_FRACTION


@pytest.mark.skipif(not cv2.ocl.haveOpenCL(), reason="OpenCL is not available/enabled")
@pytest.mark.asyncio
async def test_opencl_detector_area_relative() -> None:
    detector = OpenCLFireDetector(
        margin=int(0.1 * AREA), lower=LOWER, upper=UPPER, motion_threshold=25, relative_threshold=0.23, frame_area=AREA
    )
    results = [(await detector.detect(cv2.UMat(f)))[0] for f in (_frame(40, 24), _frame(80, 24), _frame(80, 24))]
    assert results == [False, True, False]
