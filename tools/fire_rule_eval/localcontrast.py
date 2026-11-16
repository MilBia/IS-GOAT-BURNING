"""Local-contrast small-target scores (#136: P9 CFAR, P10 top-hat / LCM, P1 spatial wavelet).

Everything here works on the **480x270 working frame** (extract with ``--scale 0.25``); window sizes are given in
pixels of that frame (CFAR / top-hat / LCM run on its 4x reduction for speed, so their constants are divided by 4) and
were derived from the target-size arithmetic below BEFORE any comparison was run
(pre-registration, see ``rules.LOCAL_CONTRAST_RULES`` and the commit message).

Target-size arithmetic at 480x270 (129 600 px): a 0.3 % fire is ~390 px (about 20x20), 1 % about 36x36, and the
anchored sunlit building that fixes the goat's false-alarm threshold (2025-12-27, ~1.4 % of the frame) about 43x43.
The hypothesis under test is **size selectivity**: windows that let a ~20 px target stand out while a ~43 px
anchored blob contaminates its own reference (CFAR), survives the structuring element (top-hat) or fills the
neighbouring cells (LCM).

"AND with the gate" is read at pixel level: every map is built from the production warm AND moving mask
(``joint``), so a score can only come from pixels the baseline gate already marks. Each variant yields one scalar
per check: the ``N``-th largest value of its contrast map (an order statistic, so a few stray pixels cannot trigger
it; ``N`` ~ the response area of a minimal target, frozen below). No division by the total warm area is used.
"""

from __future__ import annotations

import cv2
import numpy as np

FRAME_WIDTH = 480  # working resolution these windows were designed for
FRAME_HEIGHT = 270
BLUR_KERNEL = (5, 5)  # the production (21, 21) blur scaled by 480 / 1920
REDUCE = 4  # CFAR, top-hat and LCM run on the 4x reduced maps (120x67): the 480x270 windows below are divided by 4

# P9 CA-CFAR on the warm-AND-moving density (120x67): cell 3 (12 px), guard 7 (28 px >= a 20-25 px target),
# outer window 15 (60 px > the 43 px blob)
CFAR_CELL = 3
CFAR_GUARD = 7
CFAR_OUTER = 15
CFAR_SIGMA_FLOOR = 0.02  # density units: keeps the normalised score finite on an empty background
CFAR_RANK = 4  # reduced pixels, ~64 px = 0.05 % of the frame

# P10 top-hat of the heat map R - B inside warm AND moving (120x67): rectangular SE 7 (28 px: > 20, < 43)
TOPHAT_SE = 7
TOPHAT_RANK = 4

# P10 local contrast measure (120x67): cell 4 (16 px: a 20 px target covers ~one cell, a 43 px blob three), 8 neighbours
LCM_CELL = 4
LCM_RANK = 2

# P1 Haar level-1 detail energy of the unblurred R, G, B inside the region, at 240x135 (1 subband px = 2x2 px)
WAVELET_MIN_REGION = 16  # subband pixels (~0.05 % of the frame)
WAVELET_WINDOW = 9  # subband pixels (18 px)
WAVELET_MIN_WINDOW_PIXELS = 20
WAVELET_RANK = 16

SCORE_NAMES = (
    "lc_p9_cfar_z",
    "lc_p9_cfar_diff",
    "lc_p10_tophat",
    "lc_p10_lcm",
    "lc_p1_global",
    "lc_p1_local",
    "lc_ctl_peak",
)
LOCAL_CONTRAST_NAMES = SCORE_NAMES


def nth_largest(values: np.ndarray, rank: int) -> float:
    """Returns the ``rank``-th largest strictly positive value of ``values`` (0 when fewer than ``rank`` are positive)."""
    positive = values[values > 0]
    if positive.size < rank:
        return 0.0
    return float(np.partition(positive, positive.size - rank)[positive.size - rank])


def reduce_map(image: np.ndarray) -> np.ndarray:
    """Exact 4x ``INTER_AREA`` reduction (the last 2 rows of a 270-row map are dropped): 480x270 -> 120x67."""
    height = image.shape[0] // REDUCE * REDUCE
    return cv2.resize(image[:height], (image.shape[1] // REDUCE, height // REDUCE), interpolation=cv2.INTER_AREA)


def _ring_mean(source: np.ndarray, outer: int, guard: int) -> np.ndarray:
    """Mean of ``source`` over the square annulus between the guard window and the outer window (guard excluded)."""
    big = cv2.boxFilter(source, -1, (outer, outer)) * float(outer * outer)
    small = cv2.boxFilter(source, -1, (guard, guard)) * float(guard * guard)
    return (big - small) / float(outer * outer - guard * guard)


def cfar_maps(density: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CA-CFAR on the density of the warm-AND-moving mask.

    Args:
        density: ``float32`` density (0..1) of the warm-AND-moving mask on the reduced grid.

    Returns:
        ``(x, z, diff)``: the cell density ``x`` (mean over a 3x3 cell), the CFAR statistic
        ``(x - mu) / (sigma + floor)`` and the plain difference ``x - mu`` where ``mu`` / ``sigma`` are the mean and the
        standard deviation of the cell densities in the reference ring (guard 7, outer 15).
    """
    x = cv2.boxFilter(density, -1, (CFAR_CELL, CFAR_CELL))
    mu = _ring_mean(x, CFAR_OUTER, CFAR_GUARD)
    second = _ring_mean(x * x, CFAR_OUTER, CFAR_GUARD)
    sigma = np.sqrt(np.maximum(second - mu * mu, 0.0))
    diff = x - mu
    return x, diff / (sigma + np.float32(CFAR_SIGMA_FLOOR)), diff


def heat_map(blur: np.ndarray, joint: np.ndarray) -> np.ndarray:
    """``uint8`` heat map (R - B of the blurred frame, saturated) restricted to the warm-AND-moving pixels."""
    return cv2.bitwise_and(cv2.subtract(blur[..., 2], blur[..., 0]), joint)


def tophat_map(heat: np.ndarray) -> np.ndarray:
    """White top-hat (rectangular 7x7 SE) of the reduced heat map (``uint8``)."""
    return cv2.morphologyEx(heat, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_SE, TOPHAT_SE)))


def lcm_map(heat: np.ndarray) -> np.ndarray:
    """Local contrast: mean of a 4x4 cell minus the largest mean of its 8 neighbouring cells (>= 0, ``float32``)."""
    cells = cv2.blur(heat.astype(np.float32) * np.float32(1 / 255), (LCM_CELL, LCM_CELL))
    height, width = cells.shape
    step = LCM_CELL
    padded = cv2.copyMakeBorder(cells, step, step, step, step, cv2.BORDER_CONSTANT, value=0)
    best = np.zeros_like(cells)
    for dy in (-step, 0, step):
        for dx in (-step, 0, step):
            if dy or dx:
                np.maximum(best, padded[step + dy : step + dy + height, step + dx : step + dx + width], out=best)
    return np.maximum(cells - best, 0.0)


def haar_energy_explicit(frame: np.ndarray) -> np.ndarray:
    """Dense level-1 Haar detail energy ``LH^2 + HL^2 + HH^2`` averaged over the colour channels (reference for tests).

    Args:
        frame: ``uint8`` BGR frame.

    Returns:
        ``float32`` array ``(H // 2, W // 2)`` in squared grey levels.
    """
    height, width = frame.shape[0] // 2 * 2, frame.shape[1] // 2 * 2
    f = frame[:height, :width].astype(np.float32)
    a, b, c, d = f[0::2, 0::2], f[0::2, 1::2], f[1::2, 0::2], f[1::2, 1::2]
    lh = (a + b - c - d) * 0.25
    hl = (a - b + c - d) * 0.25
    hh = (a - b - c + d) * 0.25
    return (lh * lh + hl * hl + hh * hh).mean(axis=2)


def wavelet_scores(frame: np.ndarray, joint: np.ndarray) -> tuple[float, float]:
    """P1: ``(global, local)`` detail-energy scores (RMS grey levels) of the warm-AND-moving region.

    The detail energy of a 2x2 block (orthonormal-scaled Haar) equals its pixel variance, and is evaluated only where
    the region is (the region is sparse), on the unblurred frame (the production blur would erase the texture).
    ``global`` is the Toreyin-style regional statistic: RMS energy over all region pixels (0 when the region has fewer
    than 16 subband pixels). ``local`` is the 16th largest windowed (9x9 subband) region-mean RMS energy.
    """
    height, width = joint.shape[0] // 2 * 2, joint.shape[1] // 2 * 2
    region = cv2.resize(joint[:height, :width], (width // 2, height // 2), interpolation=cv2.INTER_AREA) >= 128
    ys, xs = np.nonzero(region)
    if ys.size < WAVELET_MIN_REGION:
        return 0.0, 0.0
    quad = np.stack([frame[2 * ys + dy, 2 * xs + dx] for dy in (0, 1) for dx in (0, 1)]).astype(np.float32)  # (4, n, 3)
    energy = ((quad * quad).mean(axis=0) - quad.mean(axis=0) ** 2).mean(axis=1)  # variance of each 2x2 block, mean of channels
    window = (WAVELET_WINDOW, WAVELET_WINDOW)
    sparse = np.zeros(region.shape, dtype=np.float32)
    sparse[ys, xs] = energy
    n = cv2.boxFilter(region.astype(np.float32), -1, window, normalize=False)
    total = cv2.boxFilter(sparse, -1, window, normalize=False)
    local = np.where(n >= WAVELET_MIN_WINDOW_PIXELS, np.sqrt(total / np.maximum(n, 1.0)), 0.0).astype(np.float32)
    return float(np.sqrt(energy.mean())), nth_largest(local, WAVELET_RANK)


def local_contrast_scores(frame: np.ndarray, blur: np.ndarray, joint: np.ndarray) -> dict[str, float]:
    """All #136 per-check scores.

    Args:
        frame: ``uint8`` BGR frame at the working resolution (480x270).
        blur: The same frame after the (scaled) production Gaussian blur.
        joint: ``uint8`` warm-AND-moving mask (0 / 255).

    Returns:
        A mapping of ``SCORE_NAMES`` to scalars.
    """
    density = reduce_map(joint).astype(np.float32) * np.float32(1 / 255)
    x, z, diff = cfar_maps(density)
    heat = reduce_map(heat_map(blur, joint))
    p1_global, p1_local = wavelet_scores(frame, joint)
    return {
        "lc_p9_cfar_z": nth_largest(z, CFAR_RANK),
        "lc_p9_cfar_diff": nth_largest(diff, CFAR_RANK),
        "lc_p10_tophat": nth_largest(tophat_map(heat), TOPHAT_RANK) / 255.0,
        "lc_p10_lcm": nth_largest(lcm_map(heat), LCM_RANK),
        "lc_p1_global": p1_global,
        "lc_p1_local": p1_local,
        "lc_ctl_peak": nth_largest(x, CFAR_RANK),
    }


def zero_scores() -> dict[str, float]:
    """Scores of a check that has no motion information (the first check of a stream)."""
    return dict.fromkeys(SCORE_NAMES, 0.0)
