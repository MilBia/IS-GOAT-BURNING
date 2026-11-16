"""Per-check cost of the #134 candidate features, measured on real frames (decode excluded)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from is_goat_burning.fire_detection.detector_core import DEFAULT_LOWER_HSV_FIRE
from is_goat_burning.fire_detection.detector_core import DEFAULT_UPPER_HSV_FIRE
from is_goat_burning.fire_detection.detectors import GAUSSIAN_BLUR_KERNEL_SIZE
from is_goat_burning.fire_detection.detectors import GAUSSIAN_BLUR_SIGMA_X
from tools.fire_rule_eval.features import FINE_GRID
from tools.fire_rule_eval.features import TILE_GRID
from tools.fire_rule_eval.features import ExtractOptions
from tools.fire_rule_eval.features import blob_table
from tools.fire_rule_eval.features import iter_check_frames
from tools.fire_rule_eval.tracking import BlobTracker
from tools.fire_rule_eval.tracking import LeakyIntegrator
from tools.fire_rule_eval.tracking import TrackGate
from tools.fire_rule_eval.tracking import box_gate

GATE = TrackGate(3, 3, anchored=True)
GATE_MIN_HITS = TrackGate(2, 2, anchored=True)
DECAY = 0.85


def _median_ms(step: Callable[[int], None], count: int, repeats: int) -> float:
    """Median over ``repeats`` passes of the mean per-check time of ``step(i)`` for ``i < count`` (ms)."""
    passes = []
    for _ in range(repeats):
        started = time.perf_counter()
        for i in range(count):
            step(i)
        passes.append((time.perf_counter() - started) / count * 1000)
    return statistics.median(passes)


def run(path: Path, checks: int, scale: float, repeats: int, threads: int) -> dict[str, float]:
    """Measures the baseline per-check pipeline and the incremental cost of P15, P8 and P15+P8.

    Args:
        path: Video file.
        checks: Number of 1 Hz checks to decode up front.
        scale: Resize factor applied to every frame before anything is timed (0.25 gives 480x270 from 1080p).
        repeats: Timing passes (the median is reported).
        threads: OpenCV thread count.

    Returns:
        Milliseconds per check by component, plus the frame size.
    """
    cv2.setNumThreads(threads)
    frames = [f for _, f in iter_check_frames(path, ExtractOptions(1.0, 0.0, checks, scale))]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    colors, joints = [], []
    for i, frame in enumerate(frames):
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X), cv2.COLOR_BGR2HSV)
        colors.append(cv2.inRange(hsv, DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE))
        if i:
            _, motion = cv2.threshold(cv2.absdiff(grays[i], grays[i - 1]), 25, 255, cv2.THRESH_BINARY)
            joints.append(cv2.bitwise_and(colors[i], motion))
    n = len(joints)

    def baseline(i: int) -> None:
        """Existing per-check work up to the warm-AND-moving mask (shared by every candidate)."""
        k = i + 1
        hsv = cv2.cvtColor(cv2.GaussianBlur(frames[k], GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X), cv2.COLOR_BGR2HSV)
        color = cv2.inRange(hsv, DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE)
        _, motion = cv2.threshold(
            cv2.absdiff(cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY), grays[i]), 25, 255, cv2.THRESH_BINARY
        )
        cv2.countNonZero(cv2.bitwise_and(color, motion))

    tracker = BlobTracker()

    def p15(i: int) -> None:
        tracker.update(blob_table(joints[i]))
        tracker.score(GATE)

    blobs = [blob_table(j) for j in joints]
    tracker2 = BlobTracker()

    def p15_tracker_only(i: int) -> None:
        tracker2.update(blobs[i])
        tracker2.score(GATE)

    fine = LeakyIntegrator(DECAY, (FINE_GRID[1], FINE_GRID[0]))
    coarse = LeakyIntegrator(DECAY, (TILE_GRID[1], TILE_GRID[0]))

    def p8(i: int) -> None:
        mask = (joints[i] > 0).astype(np.float32)
        fine.update(cv2.resize(mask, FINE_GRID, interpolation=cv2.INTER_AREA))
        coarse.update(cv2.resize(mask, TILE_GRID, interpolation=cv2.INTER_AREA))

    tracker3 = BlobTracker()
    gated = LeakyIntegrator(DECAY, (FINE_GRID[1], FINE_GRID[0]))

    def p15p8(i: int) -> None:
        tracker3.update(blob_table(joints[i]))
        tracker3.score(GATE)
        evidence = cv2.resize((joints[i] > 0).astype(np.float32), FINE_GRID, interpolation=cv2.INTER_AREA)
        gated.update(evidence * box_gate(tracker3.persistent_boxes(GATE_MIN_HITS), (FINE_GRID[1], FINE_GRID[0])))

    return {
        "height": float(frames[0].shape[0]),
        "width": float(frames[0].shape[1]),
        "baseline_mask_ms": _median_ms(baseline, n, repeats),
        "p15_ms": _median_ms(p15, n, repeats),
        "p15_tracker_only_ms": _median_ms(p15_tracker_only, n, repeats),
        "p8_ms": _median_ms(p8, n, repeats),
        "p15_p8_ms": _median_ms(p15p8, n, repeats),
    }
