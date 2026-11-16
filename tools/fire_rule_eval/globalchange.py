"""Whole-scene change features (#138: a global gate for ``area_relative``).

PRE-REGISTRATION (frozen in the commit that adds this file, BEFORE any feature was extracted and before any ``compare``).

Hypothesis: accelerated time-lapses, sunsets and panning / moving cameras change the whole scene, a real fire (and the
goat's small warm lights) changes a local part of it. A gate on a *global* change statistic should therefore silence
``area_relative`` on the former without touching the latter. The statistics are computed from the consecutive-check pair
(grey ``Delta`` = |gray_t - gray_{t-1}|, mean V of the blurred HSV frame, the raw warm mask) on the 480x270 working frame:

* ``gl_dv``    signed change of the mean V channel (0..255 levels) between checks, taken over the pixels OUTSIDE the warm
  mask (a fire's own brightness must not count as a global change: a 3 % flame at +150 levels would otherwise already move the
  frame mean by 4.5 levels); the gate uses ``|gl_dv|``.
* ``gl_out``   fraction of the frame with ``Delta > 25`` OUTSIDE the warm mask (the production motion threshold).
* ``gl_tiles`` fraction of the 16x9 tiles (30x30 px) in which >= 10 % of the pixels have ``Delta > 25`` ("changed" tiles).
* ``gl_coh``   uniformity of the brightness change: the larger of the shares of tiles whose mean grey changed by more than
  +3 / less than -3 levels (a sunset or an exposure jump moves almost all tiles in one direction; fire does not).
* ``gl_wg``    warm guard: the share of the changed tiles that contain warm pixels (>= 2 % of the tile). A local fire
  changes tiles that contain flames (high); a global change changes tiles without warm pixels (low). Used only to *relax*
  the tiles gate, so that a large fire that lights up much of the scene is not treated as a global change.
* ``gl_shift`` / ``gl_resp``  Hann-windowed phase correlation of the 120x68 grey frames: shift magnitude as a fraction of
  the width and the response (targets panning cameras; weak for forward-moving ones).

The warm mask used here is the RAW production mask: the post hoc mosaic exclusion of ``extract --exclude-mosaic`` does not
change any ``gl_*`` value, so the ROI and the full-frame runs differ only through the warm features of the baseline.
All features are 0 on the first check of a clip (no previous frame). Gate thresholds are fixed constants in
``rules.GATE_*`` (physical, not goat- or hard-negative derived) and are never tuned.
"""

from __future__ import annotations

import cv2
import numpy as np

MOTION_THRESHOLD = 25
TILE_GRID = (16, 9)  # (columns, rows): 30x30 px tiles at 480x270
TILE_CHANGED_FRACTION = 0.10  # share of moving pixels that makes a tile "changed"
TILE_BRIGHTNESS_STEP = 3.0  # grey levels: a tile counts as brightened / darkened beyond this mean change
TILE_WARM_FRACTION = 0.02  # share of warm pixels that makes a tile a "warm" tile
PHASE_SIZE = (120, 68)

GLOBAL_NAMES = ("gl_dv", "gl_out", "gl_tiles", "gl_coh", "gl_wg", "gl_shift", "gl_resp")


def zero_global_scores() -> dict[str, float]:
    """Feature values of a check that has no previous frame."""
    return dict.fromkeys(GLOBAL_NAMES, 0.0)


class GlobalChange:
    """Per-stream state (previous mean V, tile means and small grey frame) producing the ``gl_*`` features."""

    def __init__(self) -> None:
        self._previous_v: float | None = None
        self._previous_tiles: np.ndarray | None = None
        self._previous_small: np.ndarray | None = None
        self._window = cv2.createHanningWindow(PHASE_SIZE, cv2.CV_32F)

    def update(self, hsv: np.ndarray, gray: np.ndarray, diff: np.ndarray | None, color: np.ndarray) -> dict[str, float]:
        """Feeds one check and returns its features.

        Args:
            hsv: HSV image of the (blurred) frame; only the mean V channel outside the warm mask is used.
            gray: Grey frame of this check.
            diff: ``absdiff`` of this and the previous grey frame (``None`` on the first check of a stream).
            color: RAW warm mask (0 / 255) of this check.

        Returns:
            The ``gl_*`` features (all zero on the first check).
        """
        mean_v = float(cv2.mean(hsv, mask=cv2.bitwise_not(color))[2])
        tiles = cv2.resize(gray, TILE_GRID, interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.resize(gray, PHASE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._previous_v is None:
            self._previous_v, self._previous_tiles, self._previous_small = mean_v, tiles, small
            return zero_global_scores()

        motion = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        area = gray.shape[0] * gray.shape[1]
        outside = cv2.countNonZero(motion) - cv2.countNonZero(cv2.bitwise_and(motion, color))

        moved = cv2.resize(motion, TILE_GRID, interpolation=cv2.INTER_AREA) > TILE_CHANGED_FRACTION * 255
        warm_tiles = cv2.resize(color, TILE_GRID, interpolation=cv2.INTER_AREA) > TILE_WARM_FRACTION * 255
        changed = int(moved.sum())
        delta = tiles - self._previous_tiles
        n_tiles = tiles.size

        (dx, dy), response = cv2.phaseCorrelate(self._previous_small, small, self._window)
        out = {
            "gl_dv": mean_v - self._previous_v,
            "gl_out": outside / area,
            "gl_tiles": changed / n_tiles,
            "gl_coh": max(float((delta > TILE_BRIGHTNESS_STEP).sum()), float((delta < -TILE_BRIGHTNESS_STEP).sum())) / n_tiles,
            "gl_wg": float((moved & warm_tiles).sum()) / changed if changed else 1.0,
            "gl_shift": float(np.hypot(dx, dy)) / PHASE_SIZE[0],
            "gl_resp": float(response),
        }
        self._previous_v, self._previous_tiles, self._previous_small = mean_v, tiles, small
        return out
