"""Background-model foreground masks for the warm-AND-moving rule (#139: MOG2, KNN, running median vs frame differencing).

PRE-REGISTRATION (frozen in the commit that adds this file, BEFORE any background-model feature was extracted and before any
``compare``). Hypothesis: the production motion mask ``|gray_t - gray_(t-1)| > 25`` flags every pixel that changed between two
checks, so it fires on a flicker of the lamp as well as on a flame; a background model flags only what differs from the
*learned scene*, which should (a) suppress periodic / static-ish warm clutter of the target scene and (b) make a small flame
stand out, so that composites of 0.3 % and 1 % of the frame (invisible to the baseline) become detectable.

Models (all on the 8-bit grey frame, unblurred, ``detectShadows=False``, one state per clip, nothing else tuned):

====================  =====================================================================================================
name                  model
====================  =====================================================================================================
``mog2_h30``          MOG2, history 30, varThreshold 16 (OpenCV default), learning rate ``-1`` (auto, ``1 / min(2n, 30)``)
``mog2_h120f``        MOG2, history 120, varThreshold 16, auto for the first ``FIXED_AFTER`` = 5 checks, then fixed ``1 / 120``
``mog2v40_h30``       MOG2, history 30, varThreshold 40 (about 25 grey levels at the initial variance 15: the production
                      motion threshold), auto
``knn_h30``           KNN, history 30, dist2Threshold 400 (20 grey levels, OpenCV default), auto
``med5`` / ``med9``   median of the previous 5 / 9 checks (current check excluded, at least 3 samples), ``|gray - median| > 25``
====================  =====================================================================================================

Why two learning-rate schedules: a toy test (``diag139/absorb.py``, a static square appearing at check 6 / 15 / 40 on noise)
shows that with the auto rate a *static* foreground object turns into background after 1-3 checks (MOG2) / 4-9 checks (KNN)
whatever ``history`` is, because ``lr = 1 / min(2 n, history)`` is large for short streams; a fixed rate ``1 / history``
absorbs it after about 0.1 * history checks (13 checks for history 120). The 1 Hz check rate plus the 10-60 check clips make
this the dominant parameter, so ``mog2_h120f`` (slow absorption) is a pre-registered candidate next to the auto ones.
The sensitivity set (``SENSITIVITY_VARIANTS``: other histories and the fixed schedule of the other models) is reported in a
separately labelled table; it is NOT part of the candidate configurations and cannot "win".

Per check and model ``v`` the extractor stores (``extract --bg-only``): ``bgwm_v`` = count(warm AND fg), ``bgand_v`` =
count(warm AND fg AND motion25), ``bgmv_v`` = count(fg) (frame-wide foreground, a clutter gauge) and ``tile_bg_v`` = the 8x4
tile fractions of warm AND fg.

Rules (``rules.BG_CANDIDATE_RULES``), swept as the baselines are: ``bg_<v>`` = ``area_relative`` with the model's mask in
place of motion25 (share of the warm region that is foreground; same minimum warm / moving areas), ``bg_<v>_tiled`` = max over
the 8x4 tiles of the warm-AND-fg fraction, ``bg_<v>_abs`` = warm-AND-fg count > p * frame area (the production ``skip_first``
form), and ``bg_<v>_deb3`` = ``bg_<v>`` alarmed on 3 consecutive checks (the D=2 s baseline control is ``area_relative_deb3``).
Controls on the same features: ``area_relative``, ``area_relative_deb3``, ``tiled``, ``abs_wm25``.

Counted: 6 models x 3 forms + 6 debounced = 24 candidate configurations (+ 4 controls), judged on 9 real, 4 synth, 8 effective
hard and 40 goat clips (17 labelled clips for recall / hard). Treat any single "win" as a hypothesis, not a result.

Predicted ceiling (written before the comparison, from the baseline ``warm`` column of the composites only): the goat
background already holds about 10 % warm pixels, so ``area_relative`` (foreground share of the WHOLE warm region) cannot exceed
fraction / warm = 0.03 (0.3 %), 0.10 (1 %), 0.31 (3 %), 0.99 (10 %) even with a perfect mask; the baseline's continuous
threshold is t = 0.106 at 480x270. Hence ``bg_<v>`` (``area_relative`` form) is expected NOT to meet the 0.3 / 1 % criterion by
construction; only ``_tiled`` and ``_abs`` (absolute area) can. Disclosure of prior knowledge: before this registration the
foreground fractions of ``mog2_h30``, ``mog2v40_h30``, ``knn_h30`` and ``med5`` were measured on three goat clips (17.12,
22.12, 27.12; ``diag139/mosaic.py``): every model marks the privacy mosaic 2-15x more often than the rest of the frame, as the
frame differencing does (the mosaic is NOT removed by any model); the variants were not chosen from those numbers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from is_goat_burning.fire_detection.detector_core import DEFAULT_LOWER_HSV_FIRE
from is_goat_burning.fire_detection.detector_core import DEFAULT_UPPER_HSV_FIRE
from is_goat_burning.fire_detection.detectors import GAUSSIAN_BLUR_KERNEL_SIZE
from is_goat_burning.fire_detection.detectors import GAUSSIAN_BLUR_SIGMA_X
from is_goat_burning.fire_detection.detectors import MAX_PIXEL_VALUE

MOTION_THRESHOLD = 25  # production motion threshold; also the median model's threshold
FIXED_AFTER = 5  # checks of the auto learning rate before a fixed schedule takes over (equal to the protocol warm-up W)
MEDIAN_MIN_SAMPLES = 3
BG_TILE_GRID = (8, 4)  # (columns, rows), as ``features.TILE_GRID``


@dataclass(frozen=True)
class BgSpec:
    """One background model.

    Attributes:
        kind: ``mog2``, ``knn`` or ``median``.
        history: History length (``mog2`` / ``knn``) or number of previous checks (``median``).
        threshold: ``varThreshold`` (``mog2``) or ``dist2Threshold`` (``knn``); unused for ``median``.
        fixed: Whether the learning rate is fixed to ``1 / history`` after ``FIXED_AFTER`` checks (else always auto).
    """

    kind: str
    history: int
    threshold: float = 0.0
    fixed: bool = False


CANDIDATE_VARIANTS: dict[str, BgSpec] = {
    "mog2_h30": BgSpec("mog2", 30, 16.0),
    "mog2_h120f": BgSpec("mog2", 120, 16.0, fixed=True),
    "mog2v40_h30": BgSpec("mog2", 30, 40.0),
    "knn_h30": BgSpec("knn", 30, 400.0),
    "med5": BgSpec("median", 5),
    "med9": BgSpec("median", 9),
}
SENSITIVITY_VARIANTS: dict[str, BgSpec] = {
    "mog2_h10": BgSpec("mog2", 10, 16.0),
    "mog2_h120": BgSpec("mog2", 120, 16.0),
    "mog2_h30f": BgSpec("mog2", 30, 16.0, fixed=True),
    "mog2v40_h30f": BgSpec("mog2", 30, 40.0, fixed=True),
    "knn_h10": BgSpec("knn", 10, 400.0),
    "knn_h120": BgSpec("knn", 120, 400.0),
    "knn_h120f": BgSpec("knn", 120, 400.0, fixed=True),
}
ALL_VARIANTS = {**CANDIDATE_VARIANTS, **SENSITIVITY_VARIANTS}

BG_BASE_NAMES = ("t", "area", "first", "warm", "wm25", "mv25", "tile_wm25")


def bg_names(variants: tuple[str, ...]) -> tuple[str, ...]:
    """Feature names stored per check for the given models."""
    return tuple(f"{kind}_{v}" for v in variants for kind in ("bgwm", "bgand", "bgmv", "tile_bg"))


class _Model:
    """Foreground mask of one model; call :meth:`apply` once per check, in order."""

    def __init__(self, spec: BgSpec) -> None:
        self.spec = spec
        self.n = 0
        self._buffer: list[np.ndarray] = []
        if spec.kind == "mog2":
            self._sub = cv2.createBackgroundSubtractorMOG2(spec.history, spec.threshold, False)
        elif spec.kind == "knn":
            self._sub = cv2.createBackgroundSubtractorKNN(spec.history, spec.threshold, False)
        elif spec.kind != "median":
            raise ValueError(spec.kind)

    def apply(self, gray: np.ndarray) -> np.ndarray:
        """Returns the ``uint8`` foreground mask (0 / 255) of ``gray`` and updates the model."""
        spec = self.spec
        if spec.kind == "median":
            if len(self._buffer) < MEDIAN_MIN_SAMPLES:
                fg = np.zeros_like(gray)
            else:
                stack = np.stack(self._buffer)
                middle = len(self._buffer) // 2
                median = np.partition(stack, middle, axis=0)[middle]
                fg = cv2.threshold(cv2.absdiff(gray, median), MOTION_THRESHOLD, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)[1]
            self._buffer = [*self._buffer[-(spec.history - 1) :], gray]
        else:
            rate = 1.0 / spec.history if spec.fixed and self.n >= FIXED_AFTER else -1.0
            fg = self._sub.apply(gray, learningRate=rate)
        self.n += 1
        return fg


class BgModels:
    """Per-stream state of a set of background models (feed every check in order with :meth:`update`)."""

    def __init__(self, variants: tuple[str, ...] = tuple(CANDIDATE_VARIANTS)) -> None:
        self.variants = variants
        self._models = {name: _Model(ALL_VARIANTS[name]) for name in variants}

    def masks(self, gray: np.ndarray) -> dict[str, np.ndarray]:
        """Foreground masks of all models for this check (updates every model)."""
        return {name: model.apply(gray) for name, model in self._models.items()}

    def update(self, gray: np.ndarray, color: np.ndarray, motion25: np.ndarray | None) -> dict[str, float | np.ndarray]:
        """Processes one check.

        Args:
            gray: ``uint8`` grey frame.
            color: Production warm mask (0 / 255).
            motion25: Production motion mask of the check (``None`` on the first check of a stream).

        Returns:
            Counts and tile fractions keyed by ``bg_names(self.variants)``. ``bgand`` equals ``bgwm`` on a first check.
        """
        out: dict[str, float | np.ndarray] = {}
        for name, fg in self.masks(gray).items():
            joint = cv2.bitwise_and(color, fg)
            out[f"bgwm_{name}"] = int(cv2.countNonZero(joint))
            out[f"bgand_{name}"] = (
                out[f"bgwm_{name}"] if motion25 is None else int(cv2.countNonZero(cv2.bitwise_and(joint, motion25)))
            )
            out[f"bgmv_{name}"] = int(cv2.countNonZero(fg))
            out[f"tile_bg_{name}"] = cv2.resize(
                (joint > 0).astype(np.float32), BG_TILE_GRID, interpolation=cv2.INTER_AREA
            ).reshape(-1)
        return out


def extract_bg_features(  # noqa: C901
    frames: Iterator[tuple[float, np.ndarray]],
    blur_kernel: tuple[int, int] = GAUSSIAN_BLUR_KERNEL_SIZE,
    exclude_regions: tuple[tuple[float, float, float, float], ...] = (),
    variants: tuple[str, ...] = tuple(CANDIDATE_VARIANTS),
    models: BgModels | None = None,
    second_regions: tuple[tuple[float, float, float, float], ...] = (),
) -> dict[str, np.ndarray]:
    """Minimal per-check extraction: the baseline columns the rules need plus the background-model columns.

    ``warm``, ``wm25`` and ``tile_wm25`` are computed exactly as in ``features.extract_features`` (verified bit-identical).

    Args:
        frames: Iterator of ``(time_s, frame)`` pairs.
        blur_kernel: Gaussian blur of the colour mask (production (21, 21); (5, 5) at 480x270).
        exclude_regions: Fractional regions whose warm pixels are ignored (post hoc mosaic run).
        variants: Models to run (ignored when ``models`` is given).
        models: Optional model state carried over from a previous clip (the ``--carry-state`` sensitivity run).
        second_regions: #140: when given, the model counts are also stored for the warm mask with these regions excluded, under
            ``roi__<column>`` (needs ``models.update_dual``); the models run once, so the mosaic run costs no second pass.

    Returns:
        Feature name to array, one row per check.
    """
    from tools.fire_rule_eval.features import TILE_GRID
    from tools.fire_rule_eval.features import _region_mask
    from tools.fire_rule_eval.features import _tile_fractions

    state = models or BgModels(variants)
    rows: dict[str, list] = {name: [] for name in (*BG_BASE_NAMES, *bg_names(state.variants))}
    previous: np.ndarray | None = None
    for t, frame in frames:
        blur = cv2.GaussianBlur(frame, blur_kernel, GAUSSIAN_BLUR_SIGMA_X)
        color = cv2.inRange(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV), DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE)
        if exclude_regions:
            color = color.copy()
            color[_region_mask(frame.shape[:2], exclude_regions) > 0] = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        warm = cv2.countNonZero(color)
        first = previous is None
        motion = (
            None
            if first
            else cv2.threshold(cv2.absdiff(gray, previous), MOTION_THRESHOLD, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)[1]
        )
        joint = color if first else cv2.bitwise_and(color, motion)
        color_roi = None
        if second_regions:
            color_roi = color.copy()
            color_roi[_region_mask(frame.shape[:2], second_regions) > 0] = 0
            warm_roi = cv2.countNonZero(color_roi)
            joint_roi = color_roi if first else cv2.bitwise_and(color_roi, motion)
            rows.setdefault("roi__warm", []).append(warm_roi)
            rows.setdefault("roi__wm25", []).append(warm_roi if first else cv2.countNonZero(joint_roi))
            rows.setdefault("roi__tile_wm25", []).append(_tile_fractions(joint_roi))
        rows["t"].append(t)
        rows["area"].append(frame.shape[0] * frame.shape[1])
        rows["first"].append(int(first))
        rows["warm"].append(warm)
        rows["wm25"].append(warm if first else cv2.countNonZero(joint))
        rows["mv25"].append(0 if first else cv2.countNonZero(motion))
        rows["tile_wm25"].append(_tile_fractions(joint))
        if second_regions:
            plain, roi = state.update_dual(gray, color, color_roi, motion)
            for name, value in plain.items():
                rows.setdefault(name, []).append(value)
            for name, value in roi.items():
                rows.setdefault(f"roi__{name}", []).append(value)
        else:
            for name, value in state.update(gray, color, motion).items():
                rows.setdefault(name, []).append(value)
        previous = gray
    tiles = TILE_GRID[0] * TILE_GRID[1]
    out: dict[str, np.ndarray] = {}
    for name, values in rows.items():
        if name.startswith("tile_") or name.startswith("roi__tile_"):
            out[name] = np.array(values, dtype=np.float32).reshape(-1, tiles)
        elif name == "t":
            out[name] = np.array(values, dtype=np.float64)
        else:
            out[name] = np.array(values, dtype=np.int64)
    return out
