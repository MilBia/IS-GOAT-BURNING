"""Per-check feature extraction that mirrors the production colour + motion pipeline."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from is_goat_burning.fire_detection.detector_core import DEFAULT_LOWER_HSV_FIRE
from is_goat_burning.fire_detection.detector_core import DEFAULT_UPPER_HSV_FIRE
from is_goat_burning.fire_detection.detectors import GAUSSIAN_BLUR_KERNEL_SIZE
from is_goat_burning.fire_detection.detectors import GAUSSIAN_BLUR_SIGMA_X
from is_goat_burning.fire_detection.detectors import MAX_PIXEL_VALUE

MOTION_THRESHOLDS = (15, 25, 40)
TILE_GRID = (8, 4)  # (columns, rows)
AVERAGE_WINDOW_CHECKS = 3
CC_SCALE = 4
WHITE_LOWER = np.array([0, 0, 200], dtype=np.uint8)
WHITE_UPPER = np.array([179, 49, 255], dtype=np.uint8)
WHITE_NEIGHBOURHOOD = 25

FEATURE_NAMES = (
    "t",
    "area",
    "first",
    "warm",
    "wm15",
    "wm25",
    "wm40",
    "tile_warm",
    "tile_wm25",
    "cc_max",
    "cc_close_max",
    "wm_avg25",
    "white_wm25",
)


@dataclass(frozen=True)
class ExtractOptions:
    """Options controlling which part of a video is analysed.

    Attributes:
        check_interval_s: Seconds between analysed frames (production default is ``1 / checks_per_second``).
        start_s: Offset in seconds to start from.
        max_checks: Optional cap on the number of checks.
        scale: Resize factor applied to each analysed frame (1.0 keeps production parity).
    """

    check_interval_s: float = 1.0
    start_s: float = 0.0
    max_checks: int | None = None
    scale: float = 1.0


def iter_check_frames(path: Path, options: ExtractOptions) -> Iterator[tuple[float, np.ndarray]]:
    """Yields ``(time_s, frame)`` pairs sampled at the check interval.

    Args:
        path: Video file.
        options: Sampling options.

    Yields:
        Time in seconds and the BGR frame at that time.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(round(options.check_interval_s * fps), 1)
        index = round(options.start_s * fps)
        if index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        produced = 0
        while not count or index < count:
            if options.max_checks is not None and produced >= options.max_checks:
                return
            ok, frame = cap.read()
            if not ok:
                return
            if options.scale != 1.0:
                frame = cv2.resize(frame, None, fx=options.scale, fy=options.scale, interpolation=cv2.INTER_AREA)
            yield index / fps, frame
            produced += 1
            for _ in range(step - 1):
                if not cap.grab():
                    return
            index += step
    finally:
        cap.release()


def _largest_component(mask: np.ndarray, close: bool) -> int:
    """Returns the area (in full-resolution pixels) of the largest connected component.

    The mask is evaluated at reduced resolution for speed.
    """
    small = cv2.resize(mask, None, fx=1 / CC_SCALE, fy=1 / CC_SCALE, interpolation=cv2.INTER_AREA)
    small = (small > 127).astype(np.uint8)
    if close:
        small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(small, connectivity=8)
    if n <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max()) * CC_SCALE * CC_SCALE


def _tile_fractions(mask: np.ndarray) -> np.ndarray:
    """Returns per-tile fraction of set pixels, flattened row-major."""
    frac = cv2.resize((mask > 0).astype(np.float32), TILE_GRID, interpolation=cv2.INTER_AREA)
    return frac.reshape(-1)


def extract_features(frames: Iterator[tuple[float, np.ndarray]]) -> dict[str, np.ndarray]:
    """Computes per-check features from a stream of sampled frames.

    The first check of a stream has no previous frame; production used to treat every pixel as moving
    (colour only; fixed since, see ``skip_first``). That old behaviour is reproduced and flagged through the ``first`` feature.

    Args:
        frames: Iterator of ``(time_s, frame)`` pairs.

    Returns:
        A mapping of feature name to a NumPy array with one row per check.
    """
    rows: dict[str, list] = {name: [] for name in FEATURE_NAMES}
    previous_gray: np.ndarray | None = None
    history: deque[np.ndarray] = deque(maxlen=AVERAGE_WINDOW_CHECKS)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (WHITE_NEIGHBOURHOOD, WHITE_NEIGHBOURHOOD))
    tiles = TILE_GRID[0] * TILE_GRID[1]

    for t, frame in frames:
        blur = cv2.GaussianBlur(frame, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        color = cv2.inRange(hsv, DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        area = frame.shape[0] * frame.shape[1]
        warm = cv2.countNonZero(color)
        first = previous_gray is None

        rows["t"].append(t)
        rows["area"].append(area)
        rows["first"].append(int(first))
        rows["warm"].append(warm)

        if first:
            for name in ("wm15", "wm25", "wm40", "wm_avg25"):
                rows[name].append(warm)
            rows["white_wm25"].append(0)
            rows["tile_warm"].append(_tile_fractions(color))
            rows["tile_wm25"].append(_tile_fractions(color))
            largest = _largest_component(color, close=False)
            rows["cc_max"].append(largest)
            rows["cc_close_max"].append(_largest_component(color, close=True))
        else:
            diff = cv2.absdiff(gray, previous_gray)
            masks = {}
            for threshold in MOTION_THRESHOLDS:
                _, motion = cv2.threshold(diff, threshold, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)
                masks[threshold] = cv2.bitwise_and(color, motion)
                rows[f"wm{threshold}"].append(cv2.countNonZero(masks[threshold]))
            joint = masks[25]
            rows["tile_warm"].append(_tile_fractions(color))
            rows["tile_wm25"].append(_tile_fractions(joint))
            rows["cc_max"].append(_largest_component(joint, close=False))
            rows["cc_close_max"].append(_largest_component(joint, close=True))

            average = np.mean(np.stack(history), axis=0).astype(np.float32)
            avg_motion = (np.abs(gray.astype(np.float32) - average) > 25).astype(np.uint8) * MAX_PIXEL_VALUE
            rows["wm_avg25"].append(cv2.countNonZero(cv2.bitwise_and(color, avg_motion)))

            white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
            near_warm = cv2.dilate(color, dilate_kernel)
            _, motion25 = cv2.threshold(diff, 25, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)
            rows["white_wm25"].append(cv2.countNonZero(cv2.bitwise_and(cv2.bitwise_and(white, near_warm), motion25)))

        previous_gray = gray
        history.append(gray)

    out: dict[str, np.ndarray] = {}
    for name, values in rows.items():
        if name.startswith("tile_"):
            out[name] = np.array(values, dtype=np.float32).reshape(-1, tiles)
        elif name == "t":
            out[name] = np.array(values, dtype=np.float64)
        else:
            out[name] = np.array(values, dtype=np.int64)
    return out
