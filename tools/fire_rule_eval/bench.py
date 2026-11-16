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


def run_invariance(path: Path, checks: int, scale: float, repeats: int, threads: int) -> dict[str, float]:
    """Measures the per-check cost of the #135 motion masks on real frames (decode excluded).

    Every figure is the incremental time of one method over the production colour + ``absdiff`` mask: the mask
    itself plus the AND with the warm mask and the pixel count, with the method's own state (previous census,
    EMA accumulators) kept as in streaming use.

    Args:
        path: Video file.
        checks: Number of 1 Hz checks to decode up front.
        scale: Resize factor applied to every frame before anything is timed.
        repeats: Timing passes (the median is reported).
        threads: OpenCV thread count.

    Returns:
        Milliseconds per check by method, plus the frame size.
    """
    from tools.fire_rule_eval.invariance import EMA_FAST_TAU
    from tools.fire_rule_eval.invariance import EMA_SLOW_TAUS
    from tools.fire_rule_eval.invariance import InvarianceMasks
    from tools.fire_rule_eval.invariance import census_codes
    from tools.fire_rule_eval.invariance import census_distance
    from tools.fire_rule_eval.invariance import gain_compensated_motion
    from tools.fire_rule_eval.invariance import shift_compensated_motion
    from tools.fire_rule_eval.invariance import tile_compensated_motion

    cv2.setNumThreads(threads)
    frames = [f for _, f in iter_check_frames(path, ExtractOptions(1.0, 0.0, checks, scale))]
    blurs = [cv2.GaussianBlur(f, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X) for f in frames]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    colors = [cv2.inRange(cv2.cvtColor(b, cv2.COLOR_BGR2HSV), DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE) for b in blurs]
    motions = [None] + [
        cv2.threshold(cv2.absdiff(grays[i], grays[i - 1]), 25, 255, cv2.THRESH_BINARY)[1] for i in range(1, len(frames))
    ]
    n = len(frames) - 1

    def baseline(i: int) -> None:
        k = i + 1
        cv2.countNonZero(
            cv2.bitwise_and(colors[k], cv2.threshold(cv2.absdiff(grays[k], grays[i]), 25, 255, cv2.THRESH_BINARY)[1])
        )

    def stateless(function: Callable) -> Callable[[int], None]:
        def step(i: int) -> None:
            cv2.countNonZero(cv2.bitwise_and(colors[i + 1], function(grays[i + 1], grays[i])))

        return step

    previous_census = census_codes(grays[0])

    def p11(i: int) -> None:
        nonlocal previous_census
        census = census_codes(grays[i + 1])
        mask = cv2.threshold(census_distance(census, previous_census), 3, 255, cv2.THRESH_BINARY)[1]
        previous_census = census
        cv2.countNonZero(cv2.bitwise_and(colors[i + 1], mask))

    fast = grays[0].astype(np.float32)
    slow = grays[0].astype(np.float32)
    alpha_fast = 1 - float(np.exp(-1 / EMA_FAST_TAU))
    alpha_slow = 1 - float(np.exp(-1 / EMA_SLOW_TAUS[0]))

    def p12(i: int) -> None:
        cv2.accumulateWeighted(grays[i + 1], fast, alpha_fast)
        cv2.accumulateWeighted(grays[i + 1], slow, alpha_slow)
        mask = (cv2.absdiff(fast, slow) > 25).astype(np.uint8) * 255
        cv2.countNonZero(cv2.bitwise_and(colors[i + 1], mask))

    chroma = InvarianceMasks()

    def p4c(i: int) -> None:
        cv2.countNonZero(chroma._normalised_colour(blurs[i + 1]))  # noqa: SLF001 - timing the colour part only

    everything = InvarianceMasks()
    everything.update(blurs[0], colors[0], grays[0], None)

    def all_variants(i: int) -> None:
        everything.update(blurs[i + 1], colors[i + 1], grays[i + 1], motions[i + 1])

    return {
        "height": float(frames[0].shape[0]),
        "width": float(frames[0].shape[1]),
        "baseline_mask_ms": _median_ms(baseline, n, repeats),
        "p4s_ms": _median_ms(stateless(shift_compensated_motion), n, repeats),
        "p4g_ms": _median_ms(stateless(gain_compensated_motion), n, repeats),
        "p4t_ms": _median_ms(stateless(tile_compensated_motion), n, repeats),
        "p4c_colour_ms": _median_ms(p4c, n, repeats),
        "p11_ms": _median_ms(p11, n, repeats),
        "p12_ms": _median_ms(p12, n, repeats),
        "all_variants_ms": _median_ms(all_variants, n, repeats),
    }


def run_local_contrast(path: Path, checks: int, scale: float, repeats: int, threads: int) -> dict[str, float]:
    """Measures the per-check cost of the #136 local-contrast scores on real frames (decode excluded).

    Every figure is the stand-alone cost of one score given the frame, its blur and the warm-AND-moving mask (the
    production mask itself is reported separately as ``baseline_mask_ms``): it includes the reduction of the masks, the
    maps and the order statistic. ``dense_p1_ms`` is the wavelet score on a mask that is set everywhere (worst case of
    the region-only evaluation).

    Args:
        path: Video file.
        checks: Number of 1 Hz checks to decode up front.
        scale: Resize factor applied to every frame before anything is timed (0.25 gives the 480x270 working frame).
        repeats: Timing passes (the median is reported).
        threads: OpenCV thread count.

    Returns:
        Milliseconds per check by score, plus the frame size.
    """
    from tools.fire_rule_eval import localcontrast as lc

    cv2.setNumThreads(threads)
    kernel = (5, 5) if scale < 0.5 else GAUSSIAN_BLUR_KERNEL_SIZE
    frames = [f for _, f in iter_check_frames(path, ExtractOptions(1.0, 0.0, checks, scale))]
    blurs = [cv2.GaussianBlur(f, kernel, GAUSSIAN_BLUR_SIGMA_X) for f in frames]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    colors = [cv2.inRange(cv2.cvtColor(b, cv2.COLOR_BGR2HSV), DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE) for b in blurs]
    joints = [
        cv2.bitwise_and(colors[i], cv2.threshold(cv2.absdiff(grays[i], grays[i - 1]), 25, 255, cv2.THRESH_BINARY)[1])
        for i in range(1, len(frames))
    ]
    n = len(joints)

    def baseline(i: int) -> None:
        k = i + 1
        hsv = cv2.cvtColor(cv2.GaussianBlur(frames[k], kernel, GAUSSIAN_BLUR_SIGMA_X), cv2.COLOR_BGR2HSV)
        color = cv2.inRange(hsv, DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE)
        motion = cv2.threshold(cv2.absdiff(cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY), grays[i]), 25, 255, cv2.THRESH_BINARY)[
            1
        ]
        cv2.countNonZero(cv2.bitwise_and(color, motion))

    def cfar(i: int) -> None:
        density = lc.reduce_map(joints[i]).astype(np.float32) * np.float32(1 / 255)
        _, z, _ = lc.cfar_maps(density)
        lc.nth_largest(z, lc.CFAR_RANK)

    def tophat(i: int) -> None:
        heat = lc.reduce_map(lc.heat_map(blurs[i + 1], joints[i]))
        lc.nth_largest(lc.tophat_map(heat), lc.TOPHAT_RANK)

    def lcm(i: int) -> None:
        heat = lc.reduce_map(lc.heat_map(blurs[i + 1], joints[i]))
        lc.nth_largest(lc.lcm_map(heat), lc.LCM_RANK)

    def wavelet(i: int) -> None:
        lc.wavelet_scores(frames[i + 1], joints[i])

    dense = np.full_like(joints[0], 255)

    def dense_wavelet(i: int) -> None:
        lc.wavelet_scores(frames[i + 1], dense)

    def everything(i: int) -> None:
        lc.local_contrast_scores(frames[i + 1], blurs[i + 1], joints[i])

    return {
        "height": float(frames[0].shape[0]),
        "width": float(frames[0].shape[1]),
        "baseline_mask_ms": _median_ms(baseline, n, repeats),
        "p9_cfar_ms": _median_ms(cfar, n, repeats),
        "p10_tophat_ms": _median_ms(tophat, n, repeats),
        "p10_lcm_ms": _median_ms(lcm, n, repeats),
        "p1_wavelet_ms": _median_ms(wavelet, n, repeats),
        "dense_p1_ms": _median_ms(dense_wavelet, n, repeats),
        "all_ms": _median_ms(everything, n, repeats),
        "joint_share": float(np.mean([cv2.countNonZero(j) / j.size for j in joints])),
    }


def run_global(path: Path, checks: int, scale: float, repeats: int, threads: int) -> dict[str, float]:
    """Measures the per-check cost of the #138 global-change features on real frames (decode excluded).

    ``global_ms`` is the stand-alone cost of ``GlobalChange.update`` given the HSV frame, the grey frame, their
    ``absdiff`` and the warm mask (it includes its own motion threshold, which production already has: a conservative
    over-count); ``baseline_mask_ms`` is the production colour + motion mask on the same frames.

    Args:
        path: Video file.
        checks: Number of 1 Hz checks to decode up front.
        scale: Resize factor applied to every frame before anything is timed (0.25 gives the 480x270 working frame).
        repeats: Timing passes (the median is reported).
        threads: OpenCV thread count.

    Returns:
        Milliseconds per check, plus the frame size.
    """
    from tools.fire_rule_eval.globalchange import GlobalChange

    cv2.setNumThreads(threads)
    kernel = (5, 5) if scale < 0.5 else GAUSSIAN_BLUR_KERNEL_SIZE
    frames = [f for _, f in iter_check_frames(path, ExtractOptions(1.0, 0.0, checks, scale))]
    hsvs = [cv2.cvtColor(cv2.GaussianBlur(f, kernel, GAUSSIAN_BLUR_SIGMA_X), cv2.COLOR_BGR2HSV) for f in frames]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    colors = [cv2.inRange(h, DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE) for h in hsvs]
    diffs = [cv2.absdiff(grays[i + 1], grays[i]) for i in range(len(frames) - 1)]
    n = len(diffs)
    state = GlobalChange()
    state.update(hsvs[0], grays[0], None, colors[0])

    def baseline(i: int) -> None:
        k = i + 1
        hsv = cv2.cvtColor(cv2.GaussianBlur(frames[k], kernel, GAUSSIAN_BLUR_SIGMA_X), cv2.COLOR_BGR2HSV)
        color = cv2.inRange(hsv, DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE)
        motion = cv2.threshold(cv2.absdiff(cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY), grays[i]), 25, 255, cv2.THRESH_BINARY)[
            1
        ]
        cv2.countNonZero(cv2.bitwise_and(color, motion))

    def features(i: int) -> None:
        state.update(hsvs[i + 1], grays[i + 1], diffs[i], colors[i + 1])

    return {
        "height": float(frames[0].shape[0]),
        "width": float(frames[0].shape[1]),
        "baseline_mask_ms": _median_ms(baseline, n, repeats),
        "global_ms": _median_ms(features, n, repeats),
    }


def run_bg(path: Path, checks: int, scale: float, repeats: int, threads: int, lowrank: bool = False) -> dict[str, float]:  # noqa: C901
    """Measures the per-check cost of the #139 background-model masks on real frames (decode excluded).

    Each model is timed on its own: ``model.apply`` plus the work that turns the mask into the stored counts (warm AND fg,
    AND motion25, the foreground count and the 8x4 tile fractions). The models are fed the same frames cyclically for
    ``repeats`` passes after one warm-up pass, so the timings are steady-state. The frame differencing baseline is timed
    on the same frames (``baseline_mask_ms``); the figures are increments over it (the models do not need its absdiff).

    Args:
        path: Video file.
        checks: Number of 1 Hz checks to decode up front.
        scale: Resize factor applied to every frame before anything is timed.
        repeats: Timing passes (the median is reported).
        threads: OpenCV thread count.
        lowrank: Measure the #140 low-rank models (``LowRankModels``) instead of the #139 set; adds ``<name>_iters`` (mean RPCA
            iterations) and ``<name>_apply`` is then the mask only.

    Returns:
        Milliseconds per check by model (plus ``all_candidates``, the sum of the pre-registered candidates), and the frame size.
    """
    from tools.fire_rule_eval.bgmodel import ALL_VARIANTS
    from tools.fire_rule_eval.bgmodel import CANDIDATE_VARIANTS
    from tools.fire_rule_eval.bgmodel import BgModels

    if lowrank:
        from tools.fire_rule_eval.lowrank import ALL_VARIANTS
        from tools.fire_rule_eval.lowrank import CANDIDATE_VARIANTS
        from tools.fire_rule_eval.lowrank import LowRankModels as BgModels

    cv2.setNumThreads(threads)
    kernel = (5, 5) if scale < 0.5 else GAUSSIAN_BLUR_KERNEL_SIZE
    frames = [f for _, f in iter_check_frames(path, ExtractOptions(1.0, 0.0, checks, scale))]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    colors = [
        cv2.inRange(
            cv2.cvtColor(cv2.GaussianBlur(f, kernel, GAUSSIAN_BLUR_SIGMA_X), cv2.COLOR_BGR2HSV),
            DEFAULT_LOWER_HSV_FIRE,
            DEFAULT_UPPER_HSV_FIRE,
        )
        for f in frames
    ]
    motions = [None] + [
        cv2.threshold(cv2.absdiff(grays[i], grays[i - 1]), 25, 255, cv2.THRESH_BINARY)[1] for i in range(1, len(frames))
    ]
    n = len(frames)

    def baseline(i: int) -> None:
        k = i % n
        hsv = cv2.cvtColor(cv2.GaussianBlur(frames[k], kernel, GAUSSIAN_BLUR_SIGMA_X), cv2.COLOR_BGR2HSV)
        color = cv2.inRange(hsv, DEFAULT_LOWER_HSV_FIRE, DEFAULT_UPPER_HSV_FIRE)
        _, motion = cv2.threshold(
            cv2.absdiff(cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY), grays[k - 1]), 25, 255, cv2.THRESH_BINARY
        )
        cv2.countNonZero(cv2.bitwise_and(color, motion))

    out: dict[str, float] = {
        "height": float(frames[0].shape[0]),
        "width": float(frames[0].shape[1]),
        "baseline_mask_ms": _median_ms(baseline, n, repeats),
    }
    for name in ALL_VARIANTS:
        state = BgModels((name,))
        for i in range(n):  # warm-up pass: the model has seen the clip once
            state.update(grays[i], colors[i], motions[i])
        out[name] = _median_ms(lambda i, s=state: s.update(grays[i % n], colors[i % n], motions[i % n]), n, repeats)
        bare = BgModels((name,))
        for i in range(n):
            bare.masks(grays[i])
        out[f"{name}_apply"] = _median_ms(lambda i, s=bare: s.masks(grays[i % n]), n, repeats)
        if lowrank:
            out[f"{name}_iters"] = float(state._models[name].iterations_used)
    out["all_candidates"] = sum(out[name] for name in CANDIDATE_VARIANTS)
    return out
