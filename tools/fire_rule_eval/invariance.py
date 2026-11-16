"""Illumination-invariant motion masks (#135: P4 drift normalisation, P11 census, P12 two-EMA band).

The production motion mask is ``|gray_k - gray_(k-1)| > 25``. The candidates below replace it (or are ANDed with
it). They are pre-registered (names and parameters fixed before the first comparison, see ``VARIANTS``) and each
yields, per check, three scalars:

* ``wm_<v>``  = count(warm AND mask_v)                (the "replace" form),
* ``and_<v>`` = count(warm AND motion25 AND mask_v)   (the "AND" form),
* ``mv_<v>``  = count(mask_v)                         (how much of the frame the mask marks, a global-change gauge).

Only counts are stored, so a rule sweep is cheap. All masks work at the frame resolution that is passed in.
"""

from __future__ import annotations

import cv2
import numpy as np

from is_goat_burning.fire_detection.detector_core import DEFAULT_LOWER_HSV_FIRE
from is_goat_burning.fire_detection.detector_core import DEFAULT_UPPER_HSV_FIRE
from is_goat_burning.fire_detection.detectors import MAX_PIXEL_VALUE

MOTION_THRESHOLD = 25  # same as production
CENSUS_TOLERANCE = 4  # ternary census: a neighbour counts as brighter / darker only beyond +-4 grey levels
CENSUS_DISTANCES = (2, 4, 6)  # changed ternary bits (of 16) that make a pixel "moving"
EMA_FAST_TAU = 3.0  # checks (1 Hz: seconds)
EMA_SLOW_TAUS = (15.0, 120.0)
EMA_THRESHOLDS = (25, 15)
CHROMA_TAU = 10.0  # checks; reference of the per-channel medians used by P4c
CHROMA_GAIN_LIMITS = (0.7, 1.4)
GAIN_LIMITS = (0.5, 2.0)
DRIFT_SAMPLE_ROWS = 135  # the global drift is estimated on a ~135-row subsample (median)
DRIFT_TILES = (8, 4)  # (columns, rows) for the local-drift variant

P4_VARIANTS = ("p4s", "p4g", "p4t", "p4c")
P11_VARIANTS = tuple(f"p11_k{k}" for k in CENSUS_DISTANCES)
P12_VARIANTS = ("p12_a15", "p12_a120", "p12_p15", "p12_p120", "p12_a15_t15", "p12_a120_t15")
VARIANTS = (*P4_VARIANTS, *P11_VARIANTS, *P12_VARIANTS)

# 8 neighbours of the 3x3 census window (dy, dx)
_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

INVARIANCE_NAMES = (
    "mv25",
    "warm_p4c",
    *(f"{kind}_{v}" for v in VARIANTS for kind in ("wm", "and", "mv")),
)


def census_codes(gray: np.ndarray, tolerance: int = CENSUS_TOLERANCE) -> tuple[np.ndarray, np.ndarray]:
    """Ternary 3x3 census transform.

    Args:
        gray: ``uint8`` grey image.
        tolerance: Grey-level margin below which a neighbour counts as equal to the centre.

    Returns:
        Two ``uint8`` images: bit ``i`` of the first is set when neighbour ``i`` is brighter than ``centre + tolerance``,
        bit ``i`` of the second when it is darker than ``centre - tolerance``. A uniform brightness shift that
        keeps every neighbour difference unchanged therefore leaves both codes unchanged.
    """
    height, width = gray.shape
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    upper = cv2.add(gray, tolerance)
    lower = cv2.subtract(gray, tolerance)
    brighter = np.zeros_like(gray)
    darker = np.zeros_like(gray)
    for bit, (dy, dx) in enumerate(_OFFSETS):
        neighbour = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        cv2.bitwise_or(brighter, cv2.bitwise_and(cv2.compare(neighbour, upper, cv2.CMP_GT), 1 << bit), dst=brighter)
        cv2.bitwise_or(darker, cv2.bitwise_and(cv2.compare(neighbour, lower, cv2.CMP_LT), 1 << bit), dst=darker)
    return brighter, darker


def census_distance(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Hamming distance (0..16) between two census code pairs, per pixel."""
    return cv2.add(
        cv2.LUT(cv2.bitwise_xor(a[0], b[0]), _POPCOUNT),
        cv2.LUT(cv2.bitwise_xor(a[1], b[1]), _POPCOUNT),
    )


def _sample(image: np.ndarray) -> np.ndarray:
    step = max(image.shape[0] // DRIFT_SAMPLE_ROWS, 1)
    return image[::step, ::step]


def shift_compensated_motion(gray: np.ndarray, previous: np.ndarray, threshold: int = MOTION_THRESHOLD) -> np.ndarray:
    """P4 (additive): motion mask after removing the global brightness shift (median of the frame difference)."""
    shift = float(np.median(_sample(gray).astype(np.int16) - _sample(previous)))
    adjusted = cv2.addWeighted(previous, 1.0, previous, 0.0, shift)
    return cv2.threshold(cv2.absdiff(gray, adjusted), threshold, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)[1]


def gain_compensated_motion(gray: np.ndarray, previous: np.ndarray, threshold: int = MOTION_THRESHOLD) -> np.ndarray:
    """P4 (multiplicative): motion mask after removing the global gain (ratio of the frame medians)."""
    gain = float(np.median(_sample(gray))) / max(float(np.median(_sample(previous))), 1.0)
    adjusted = cv2.convertScaleAbs(previous, alpha=float(np.clip(gain, *GAIN_LIMITS)))
    return cv2.threshold(cv2.absdiff(gray, adjusted), threshold, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)[1]


def tile_compensated_motion(gray: np.ndarray, previous: np.ndarray, threshold: int = MOTION_THRESHOLD) -> np.ndarray:
    """P4 (local): as ``shift_compensated_motion`` with the shift estimated per 8x4 tile and interpolated."""
    columns, rows = DRIFT_TILES
    delta = _sample(gray).astype(np.float32) - _sample(previous)
    height, width = delta.shape
    delta = delta[: height // rows * rows, : width // columns * columns]
    blocks = delta.reshape(rows, delta.shape[0] // rows, columns, delta.shape[1] // columns)
    shifts = np.median(blocks, axis=(1, 3)).astype(np.float32)
    shift_map = cv2.resize(shifts, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    residual = np.abs(gray.astype(np.float32) - previous.astype(np.float32) - shift_map)
    return (residual > threshold).astype(np.uint8) * MAX_PIXEL_VALUE


def _ema_alpha(tau: float, n: int) -> float:
    """EMA weight of check ``n`` (0-based): at least the running-mean weight, so short clips are not dominated by check 0."""
    return max(1.0 - float(np.exp(-1.0 / tau)), 1.0 / (n + 1))


class InvarianceMasks:
    """Per-stream state of the P4 / P11 / P12 masks; feed every check in order with :meth:`update`."""

    def __init__(self) -> None:
        self.n = 0
        self._previous_gray: np.ndarray | None = None
        self._previous_census: tuple[np.ndarray, np.ndarray] | None = None
        self._fast: np.ndarray | None = None
        self._slow: dict[float, np.ndarray] = {}
        self._reference: np.ndarray | None = None

    def _normalised_colour(self, blur: np.ndarray) -> np.ndarray:
        """P4c: warm mask of the frame after gray-world gains towards a running reference of the channel medians."""
        medians = np.median(_sample(blur).reshape(-1, 3), axis=0)
        if self._reference is None:
            gains = np.ones(3)
        else:
            gains = np.clip(self._reference / np.maximum(medians, 1.0), *CHROMA_GAIN_LIMITS)
        alpha = _ema_alpha(CHROMA_TAU, self.n)
        self._reference = medians if self._reference is None else (1 - alpha) * self._reference + alpha * medians
        normalised = cv2.multiply(blur, (float(gains[0]), float(gains[1]), float(gains[2]), 0.0))
        hsv = cv2.cvtColor(normalised, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE)

    def _masks(self, gray: np.ndarray) -> dict[str, np.ndarray]:
        previous = self._previous_gray
        masks: dict[str, np.ndarray] = {}
        masks["p4s"] = shift_compensated_motion(gray, previous)
        masks["p4g"] = gain_compensated_motion(gray, previous)
        masks["p4t"] = tile_compensated_motion(gray, previous)
        masks["p4c"] = masks["p4s"]  # motion part of P4c; its colour mask is normalised separately
        census = census_codes(gray)
        distance = census_distance(census, self._previous_census)
        for k in CENSUS_DISTANCES:
            masks[f"p11_k{k}"] = cv2.threshold(distance, k - 1, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)[1]
        self._previous_census = census
        return masks

    def _ema_masks(self, gray: np.ndarray) -> dict[str, np.ndarray]:
        cv2.accumulateWeighted(gray, self._fast, _ema_alpha(EMA_FAST_TAU, self.n))
        for tau, slow in self._slow.items():
            cv2.accumulateWeighted(gray, slow, _ema_alpha(tau, self.n))
        masks: dict[str, np.ndarray] = {}
        for tau in EMA_SLOW_TAUS:
            band = self._fast - self._slow[tau]
            tag = int(tau)
            masks[f"p12_a{tag}"] = (np.abs(band) > MOTION_THRESHOLD).astype(np.uint8) * MAX_PIXEL_VALUE
            masks[f"p12_p{tag}"] = (band > MOTION_THRESHOLD).astype(np.uint8) * MAX_PIXEL_VALUE
            masks[f"p12_a{tag}_t15"] = (np.abs(band) > EMA_THRESHOLDS[1]).astype(np.uint8) * MAX_PIXEL_VALUE
        return masks

    def update(self, blur: np.ndarray, color: np.ndarray, gray: np.ndarray, motion25: np.ndarray | None) -> dict[str, int]:
        """Processes one check.

        Args:
            blur: Gaussian-blurred BGR frame (production pre-processing).
            color: Production warm mask (``inRange`` on ``blur``), ``uint8`` 0 / 255.
            gray: ``uint8`` grey frame.
            motion25: Production motion mask ``|gray - previous| > 25`` (``None`` on the first check of a stream).

        Returns:
            Scalar counts keyed by ``INVARIANCE_NAMES``. On the first check there is no motion information, so the
            ``wm_`` / ``and_`` counts equal the warm count (as for the production features) and ``mv_`` is zero.
        """
        warm = int(cv2.countNonZero(color))
        color_p4c = self._normalised_colour(blur)
        warm_p4c = int(cv2.countNonZero(color_p4c))
        out: dict[str, int] = {"mv25": 0, "warm_p4c": warm_p4c}
        if self._previous_gray is None:
            self._fast = gray.astype(np.float32)
            self._slow = {tau: gray.astype(np.float32) for tau in EMA_SLOW_TAUS}
            self._previous_census = census_codes(gray)
            for v in VARIANTS:
                base = warm_p4c if v == "p4c" else warm
                out.update({f"wm_{v}": base, f"and_{v}": base, f"mv_{v}": 0})
        else:
            masks = {**self._masks(gray), **self._ema_masks(gray)}
            out["mv25"] = int(cv2.countNonZero(motion25))
            colored_motion = cv2.bitwise_and(color, motion25)
            colored_motion_p4c = cv2.bitwise_and(color_p4c, motion25)
            for v in VARIANTS:
                mask = masks[v]
                if v == "p4c":
                    out["wm_p4c"] = int(cv2.countNonZero(cv2.bitwise_and(color_p4c, mask)))
                    out["and_p4c"] = int(cv2.countNonZero(cv2.bitwise_and(colored_motion_p4c, mask)))
                else:
                    out[f"wm_{v}"] = int(cv2.countNonZero(cv2.bitwise_and(color, mask)))
                    out[f"and_{v}"] = int(cv2.countNonZero(cv2.bitwise_and(colored_motion, mask)))
                out[f"mv_{v}"] = int(cv2.countNonZero(mask))
        self._previous_gray = gray
        self.n += 1
        return out
