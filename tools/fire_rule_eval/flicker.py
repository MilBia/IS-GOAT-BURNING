"""Temporal-flicker feature (#105): a windowed reference and an O(1)-per-pixel incremental version.

Fire flickers at roughly 1-15 Hz while static warm objects do not. Both implementations work on a 30 fps
stream of 480x270 frames and report, at one check per second, how many warm pixels flicker:

* ``windowed_flicker_mask``: per pixel over the last 30 frames, temporal std of grey >= ``std_min`` and at
  least ``crossings_min`` crossings of the temporal mean (the reference, too slow for production).
* ``IncrementalFlicker``: exponential moving mean, moving absolute deviation and a decayed sign-flip
  counter; no stored window.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from is_goat_burning.fire_detection.detector_core import DEFAULT_LOWER_HSV_FIRE
from is_goat_burning.fire_detection.detector_core import DEFAULT_UPPER_HSV_FIRE

WIDTH, HEIGHT = 480, 270
FPS = 30.0
WINDOW = 30
COLOUR_BLUR = (5, 5)
FLICKER_FEATURE_NAMES = ("t", "area", "warm", "flick_win", "flick_inc")


@dataclass(frozen=True)
class FlickerParams:
    """Thresholds shared by both implementations, plus the incremental filter constants.

    Attributes:
        std_min: Minimum temporal deviation of grey level for a pixel to count as flickering.
        crossings_min: Minimum mean crossings in the 30-frame window (windowed reference).
        mean_alpha: Weight of the newest frame in the moving mean (incremental).
        deviation_decay: Decay of the moving absolute deviation (incremental).
        flip_decay: Decay of the sign-flip counter (incremental).
        flips_min: Minimum decayed flip count (incremental).
    """

    std_min: float = 6.0
    crossings_min: int = 4
    mean_alpha: float = 1 / 16
    deviation_decay: float = 0.94
    flip_decay: float = 0.97
    flips_min: float = 2.0


def warm_mask(frame: np.ndarray) -> np.ndarray:
    """Returns the boolean fire-colour mask (same HSV range as production, light blur)."""
    blur = cv2.GaussianBlur(frame, COLOUR_BLUR, 0)
    return cv2.inRange(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV), DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE) > 0


def windowed_flicker_mask(window: np.ndarray, warm: np.ndarray, params: FlickerParams) -> np.ndarray:
    """Reference implementation.

    Args:
        window: Grey frames, shape ``(WINDOW, H, W)``.
        warm: Boolean fire-colour mask of the newest frame.
        params: Thresholds.

    Returns:
        Boolean mask of warm pixels that flicker.
    """
    stack = window.astype(np.float32)
    deviation = stack - stack.mean(axis=0)
    above = deviation > 0
    crossings = (above[1:] != above[:-1]).sum(axis=0)
    return (stack.std(axis=0) >= params.std_min) & (crossings >= params.crossings_min) & warm


class IncrementalFlicker:
    """Streaming flicker detector with constant memory and O(1) work per pixel and frame."""

    def __init__(self, shape: tuple[int, int], params: FlickerParams) -> None:
        """Allocates the per-pixel state.

        Args:
            shape: ``(height, width)`` of the grey frames.
            params: Thresholds and filter constants.
        """
        self._params = params
        self._mean: np.ndarray | None = None
        self._deviation = np.zeros(shape, np.float32)
        self._flips = np.zeros(shape, np.float32)
        self._sign = np.zeros(shape, np.bool_)

    def update(self, gray: np.ndarray, warm: np.ndarray) -> np.ndarray:
        """Feeds one frame.

        Args:
            gray: Grey frame.
            warm: Boolean fire-colour mask of the same frame.

        Returns:
            Boolean mask of warm pixels that currently flicker.
        """
        p = self._params
        gray_f = gray.astype(np.float32)
        if self._mean is None:
            self._mean = gray_f.copy()
        deviation = gray_f - self._mean
        cv2.accumulateWeighted(gray_f, self._mean, p.mean_alpha)
        sign = deviation > 0
        self._flips = self._flips * p.flip_decay + (sign != self._sign)
        self._sign = sign
        self._deviation = self._deviation * p.deviation_decay + np.abs(deviation) * (1 - p.deviation_decay)
        return (self._deviation >= p.std_min) & (self._flips >= p.flips_min) & warm


def iter_frames_30fps(path: Path) -> Iterator[np.ndarray]:
    """Yields 480x270 BGR frames resampled in time to 30 fps."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or not np.isfinite(fps) or fps <= 0:
            fps = FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_index, frame = -1, None
        for k in range(int(total / fps * FPS) if total > 0 else 10**9):
            index = int(round(k * fps / FPS))
            while last_index < index:
                ok, frame = cap.read()
                if not ok:
                    return
                last_index += 1
            yield cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    finally:
        cap.release()


def extract_flicker_features(
    frames: Iterator[np.ndarray], params: FlickerParams | None = None, windowed: bool = True
) -> dict[str, np.ndarray]:
    """Runs both implementations over a 30 fps stream and records one row per second (from t = 1 s).

    Args:
        frames: 30 fps BGR frames of equal size.
        params: Thresholds (defaults to ``FlickerParams()``).
        windowed: Also compute the (slow) reference; ``flick_win`` is -1 when disabled.

    Returns:
        Arrays named like ``FLICKER_FEATURE_NAMES``; ``warm``/``flick_*`` are pixel counts.
    """
    params = params or FlickerParams()
    ring: deque[np.ndarray] = deque(maxlen=WINDOW)
    rows: dict[str, list[float]] = {name: [] for name in FLICKER_FEATURE_NAMES}
    incremental: IncrementalFlicker | None = None
    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        warm = warm_mask(frame)
        if incremental is None:
            incremental = IncrementalFlicker(gray.shape, params)
        inc_mask = incremental.update(gray, warm)
        ring.append(gray)
        if index == 0 or index % WINDOW != 0:
            continue  # one row per second, aligned with the 1 Hz checks of the colour + motion features
        rows["t"].append(index / FPS)
        rows["area"].append(gray.size)
        rows["warm"].append(int(warm.sum()))
        rows["flick_inc"].append(int(inc_mask.sum()))
        win = int(windowed_flicker_mask(np.stack(ring), warm, params).sum()) if windowed else -1
        rows["flick_win"].append(win)
    return {name: np.asarray(values, dtype=np.float64) for name, values in rows.items()}
