"""Decision layer on the per-check gate score: debounce vs CUSUM / Bernoulli-SPRT vs a conformal threshold (#137).

PRE-REGISTRATION (frozen in the commit that introduces this module, before the first ``compare`` of any candidate).
Only the baseline controls (debounce of the unchanged gates) and the goat-only distribution of the gate score
and the (non-positive-derived) positive score series printed while designing were looked at before freezing.

Gate scores (``g``, one number per check, 0 on the first check of a clip and where a gate fails):
    * ``ar``: ``area_relative`` score (share of the warm region that moves), the baseline of #104-#136;
    * ``ff``: ``frame_fraction`` score (warm AND moving pixels / frame area), the production default mode.

Production debounce (``StreamFireDetector._handle_fire_detection``): the first alarmed check stamps ``start``; the alarm
fires when ``now - start >= FIRE_DETECTED_DEBOUNCE_SECONDS`` (D). At ``CHECKS_PER_SECOND = 1`` this needs ``D + 1``
consecutive alarmed checks (D=0: 1 check, D=3: 4 checks). The "debounce 3" controls of #134-#136 (3 consecutive checks)
are therefore Settings D=2. A non-alarmed check resets the timer; the fire state ends after
``FIRE_EXTINGUISHED_DEBOUNCE_SECONDS`` (E, default 5) of non-alarmed checks.

Frozen candidates (7 per feature set, all on ``ar``):
    * ``cusum_{q90,half,q99,max}``: ``S = max(0, S + g - k)``, alarm iff ``S > h``; ``k`` = the 90th / 99th percentile,
      half the maximum and the maximum of ``g`` over the counted goat checks of the same feature set (goat-only statistic);
      ``h`` is the swept parameter.
    * ``sprt_{q90,q99,q99p25}``: Bernoulli log-likelihood-ratio CUSUM on the hits ``g > gate`` (gate = the 90th / 99th goat
      percentile of ``g``); ``p0`` = the goat hit rate at the gate (clipped to >= 1/(2n)), ``p1`` = 0.5 (``q99p25``: 0.25)
      fixed a priori; ``S = max(0, S + llr)`` with ``llr = ln(p1/p0)`` on a hit and ``ln((1-p1)/(1-p0))`` on a miss; alarm
      iff ``S > h``; ``h`` swept. (A Page/CUSUM chain of Wald SPRTs.)
Controls: debounce D in {0, 1, 2, 3, 5, 10} s (K = D + 1 checks) for ``ar`` and ``ff``, each at its OWN best safe point.

Operating point of every rule: the largest score over the counted goat checks of the 40 per-clip series, the canonical chained
stream and ``N_PERM`` random orders of the chained clips (state carried over the joins), i.e. alarm iff ``score > t``:
zero goat alarms in all of them, zero margin (in-sample). Counted checks = after the warm-up window ``W = 5``.

Verdict per comparator D (criterion of #137), over the 13 positive clips (9 real + 4 synthetic), with the production state
machine (E = 5 s) replayed from a cold start at the first check with motion information: pass (i) iff clips detected >= the
comparator's, real check recall >= the comparator's and the median time to alarm over the clips detected by both <= the
comparator's; pass (ii) iff (clips detected or real recall higher, none lower) and the median latency (time to alarm + 1 s)
<= 2x the comparator's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import math
import re
import time

import numpy as np

from tools.fire_rule_eval.report import HARD_NEGATIVE
from tools.fire_rule_eval.report import Clip
from tools.fire_rule_eval.rules import Features
from tools.fire_rule_eval.rules import Rule
from tools.fire_rule_eval.rules import area_relative_score
from tools.fire_rule_eval.rules import debounced_score

DEBOUNCE_SECONDS = (0, 1, 2, 3, 5, 10)
EXTINGUISH_SECONDS = (0, 5, 10)
EXTINGUISH_DEFAULT = 5.0
CHECKS_PER_SECOND = 1.0
N_PERM = 200
PERM_SEED = 137
# production default of area_relative; README recommendation and production default of frame_fraction
FIXED_THRESHOLDS = {"ar": (0.23,), "ff": (0.05, 0.10)}
CUSUM_KS = ("q90", "half", "q99", "max")
SPRT_VARIANTS = {"q90": (0.90, 0.5), "q99": (0.99, 0.5), "q99p25": (0.99, 0.25)}  # name -> (goat quantile of the gate, p1)
REAL_GROUP = "commons_fire"
SYNTH_GROUP = "synth_goat"
CHAIN_KEYS = ("t", "first", "area", "warm", "wm25")
H_GRID = tuple(float(x) for x in np.geomspace(1e-3, 30.0, 260))
AR_GRID = tuple(float(x) for x in np.linspace(0.005, 0.95, 190))
FF_GRID = tuple(float(x) for x in np.geomspace(0.0002, 0.25, 120))


def debounce_checks(seconds: float, checks_per_second: float = CHECKS_PER_SECOND) -> int:
    """Consecutive alarmed checks the production debounce needs (``now - start >= D`` with ``start`` = the first alarm)."""
    return int(math.ceil(seconds * checks_per_second - 1e-9)) + 1


def replay_production(
    alarm: np.ndarray, on_s: float, off_s: float, step_s: float = 1.0
) -> tuple[np.ndarray, list[int], list[int]]:
    """Replays ``StreamFireDetector._handle_fire_detection`` on per-check raw alarms (checks ``step_s`` apart).

    Returns:
        ``(state, entries, exits)``: ``state[i]`` = ``fire_is_currently_detected`` after check ``i``; check indices of the
        transitions into / out of the fire state.
    """
    state = False
    start: float | None = None
    end: float | None = None
    out = np.zeros(len(alarm), dtype=bool)
    entries: list[int] = []
    exits: list[int] = []
    for i, fire in enumerate(alarm):
        now = i * step_s
        if fire:
            end = None
            if not state:
                if start is None:
                    start = now
                if now - start >= on_s:
                    state = True
                    start = None
                    entries.append(i)
        else:
            start = None
            if state:
                if end is None:
                    end = now
                if now - end >= off_s:
                    state = False
                    end = None
                    exits.append(i)
        out[i] = state
    return out, entries, exits


def cusum_series(g: np.ndarray, k: float) -> np.ndarray:
    """One-sided CUSUM ``S = max(0, S + g - k)`` of a score series."""
    out = np.zeros(len(g))
    s = 0.0
    for i, x in enumerate(g):
        s = max(0.0, s + float(x) - k)
        out[i] = s
    return out


def llr_series(hit: np.ndarray, p0: float, p1: float) -> np.ndarray:
    """Bernoulli log-likelihood-ratio CUSUM of a hit series (hit rate ``p0`` under no fire, ``p1`` under fire)."""
    up, down = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    out = np.zeros(len(hit))
    s = 0.0
    for i, h in enumerate(hit):
        s = max(0.0, s + (up if h else down))
        out[i] = s
    return out


def ff_score(f: Features) -> np.ndarray:
    """``frame_fraction`` score: warm AND moving pixels / frame area (0 on the first check)."""
    return np.where(f["first"] == 0, f["wm25"] / np.maximum(f["area"], 1), 0.0)


GATES: dict[str, Callable[[Features], np.ndarray]] = {"ar": area_relative_score(), "ff": ff_score}


@dataclass(frozen=True)
class Candidate:
    """A decision rule built on a gate score.

    Attributes:
        name: Identifier.
        family: ``debounce`` / ``cusum`` / ``sprt``.
        mode: Gate (``ar`` / ``ff``).
        description: One line.
        score: Per-check statistic; the rule alarms iff ``score > p``.
        gate: The underlying gate score function.
        raw: For debounce rules the raw per-check alarm ``gate > p`` (fed to the production state machine); None otherwise.
        on_s: Production on-debounce applied to the raw alarm (D for debounce rules, 0 for sequential tests).
        grid: Parameter sweep for ``compare``.
        params: Frozen parameters (for the report).
    """

    name: str
    family: str
    mode: str
    description: str
    score: Callable[[Features], np.ndarray]
    gate: Callable[[Features], np.ndarray]
    on_s: float = 0.0
    grid: tuple[float, ...] = H_GRID
    params: dict = field(default_factory=dict)

    @property
    def rule(self) -> Rule:
        """``Rule`` view (for ``compare`` / the continuous margin analysis)."""
        score = self.score
        return Rule(self.name, self.description, self.grid, lambda f, p: score(f) > p, score=score)

    def alarm(self, f: Features, p: float) -> np.ndarray:
        """Per-check alarm of the rule (includes the debounce / sequential test)."""
        return self.score(f) > p

    def raw(self, f: Features, p: float) -> np.ndarray:
        """Raw per-check alarm fed to the production state machine."""
        return (self.gate(f) > p) if self.family == "debounce" else (self.score(f) > p)


def goat_scores(gate: Callable[[Features], np.ndarray], clips: list[Clip]) -> np.ndarray:
    """Gate scores over the counted checks of the goat negatives (the only data candidate parameters are derived from)."""
    return np.concatenate([gate(c.features)[c.mask] for c in clips if c.label == "neg" and c.mask.any()])


def debounce_candidate(mode: str, seconds: int) -> Candidate:
    """The production debounce of ``seconds`` on the ``mode`` gate (K = D + 1 consecutive alarmed checks at 1 Hz)."""
    gate = GATES[mode]
    k = debounce_checks(seconds)
    return Candidate(
        f"{mode}_deb{seconds}s",
        "debounce",
        mode,
        f"{mode}: alarm after {k} consecutive alarmed checks (FIRE_DETECTED_DEBOUNCE_SECONDS={seconds})",
        debounced_score(gate, k),
        gate,
        float(seconds),
        AR_GRID if mode == "ar" else FF_GRID,
        {"D": seconds, "K": k},
    )


def build_candidates(clips: list[Clip]) -> list[Candidate]:
    """Controls (debounce) and the 7 frozen candidates; goat-derived parameters are computed from ``clips`` only."""
    cands: list[Candidate] = [debounce_candidate(mode, d) for mode in GATES for d in DEBOUNCE_SECONDS]
    gate = GATES["ar"]
    g = goat_scores(gate, clips)
    ks = {
        "q90": float(np.quantile(g, 0.90)),
        "half": 0.5 * float(g.max()),
        "q99": float(np.quantile(g, 0.99)),
        "max": float(g.max()),
    }
    for label in CUSUM_KS:
        k = ks[label]
        cands.append(
            Candidate(
                f"cusum_{label}",
                "cusum",
                "ar",
                f"CUSUM of the ar score, allowance k={k:.4f} (goat {label})",
                lambda f, k=k: cusum_series(gate(f), k),
                gate,
                params={"k": k},
            )
        )
    for label, (quantile, p1) in SPRT_VARIANTS.items():
        thr = float(np.quantile(g, quantile))
        p0 = max(float((g > thr).mean()), 1 / (2 * len(g)))
        cands.append(
            Candidate(
                f"sprt_{label}",
                "sprt",
                "ar",
                f"Bernoulli LLR-CUSUM on hits ar > {thr:.4f} (goat q{int(quantile * 100)}), p0={p0:.4f}, p1={p1}",
                lambda f, thr=thr, p0=p0, p1=p1: llr_series(gate(f) > thr, p0, p1),
                gate,
                params={"gate": thr, "p0": p0, "p1": p1},
            )
        )
    return cands


# ---- operating point ------------------------------------------------------------------------------------------------


def chain_stream(clips: list[Clip]) -> tuple[Features, np.ndarray]:
    """Concatenates clips into one stream (state carried over the joins; first rows mark them) and its counted mask."""
    feats = {k: np.concatenate([c.features[k] for c in clips]) for k in CHAIN_KEYS}
    return feats, np.concatenate([c.mask for c in clips])


def goat_chains(negatives: list[Clip]) -> list[tuple[Features, np.ndarray]]:
    """The canonical (name-ordered) chained goat stream followed by ``N_PERM`` random orders of the same clips."""
    ordered = sorted(negatives, key=lambda c: c.name)
    rng = np.random.default_rng(PERM_SEED)
    chains = [chain_stream(ordered)]
    chains.extend(chain_stream([ordered[i] for i in rng.permutation(len(ordered))]) for _ in range(N_PERM))
    return chains


@dataclass(frozen=True)
class Point:
    """Operating point of a candidate: ``t_clip`` (per-clip only), ``t_chain`` (canonical chain), ``t_safe`` (all)."""

    t_clip: float
    t_chain: float
    t_safe: float
    perm_alarm_orders_at_clip: int  # of N_PERM random orders alarming at the per-clip-only threshold


def operating_point(cand: Candidate, negatives: list[Clip], chains: list[tuple[Features, np.ndarray]]) -> Point:
    """Largest counted goat score per clip, in the canonical chain and in the permuted chains."""
    t_clip = max((float(cand.score(c.features)[c.mask].max()) for c in negatives if c.mask.any()), default=0.0)
    maxima = [float(cand.score(f)[m].max()) for f, m in chains]
    t_chain = maxima[0]
    orders = sum(1 for v in maxima[1:] if v > t_clip)
    return Point(t_clip, t_chain, max(t_clip, *maxima), orders)


# ---- operational evaluation of positives ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PositiveResult:
    """Cold-start production replay of one positive clip."""

    name: str
    group: str
    detected: bool
    ttfa: float | None  # seconds from the first check with motion information (real) / the fire onset (synthetic)
    available: float  # seconds from that start to the last check (the largest measurable time to alarm)
    recall: float  # share of checks from the start in the fire state
    entries: int
    exits: int
    protocol_recall: float  # run-length alarm over the warm-up-masked checks, state warmed from the first check (#134-#136)


def start_index(clip: Clip) -> int:
    """First check with motion information (real clips) / the first fire check (synthetic: ``fire_from_s``)."""
    t = clip.features["t"]
    if clip.group == SYNTH_GROUP:
        return int(np.flatnonzero(t >= clip.onset_s)[0])
    return int(np.flatnonzero(clip.features["first"] == 0)[0])


def positive_results(cand: Candidate, p: float, off_s: float, clips: list[Clip]) -> list[PositiveResult]:
    """Production replay (on-debounce of the rule, extinguish ``off_s``) of every positive clip."""
    out = []
    for clip in clips:
        if not clip.is_positive:
            continue
        f = clip.features
        i0 = start_index(clip)
        state, entries, exits = replay_production(cand.raw(f, p), cand.on_s, off_s)
        hit = np.flatnonzero(state[i0:])
        t = f["t"]
        step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
        out.append(
            PositiveResult(
                clip.name,
                clip.group,
                bool(len(hit)),
                float(hit[0]) * step if len(hit) else None,
                float(len(t) - 1 - i0) * step,
                float(state[i0:].mean()),
                len(entries),
                len(exits),
                float(cand.alarm(f, p)[clip.mask].mean()),
            )
        )
    return out


def is_real(r: PositiveResult) -> bool:
    """Whether the result belongs to a real (not synthetic) fire clip."""
    return r.group.startswith(REAL_GROUP)


def summarise(results: list[PositiveResult]) -> dict:
    """Pooled summary: clips detected (real / synthetic), mean check recall of real clips, median time to alarm."""
    real = [r for r in results if is_real(r)]
    synth = [r for r in results if not is_real(r)]
    delays = lambda rs: [r.ttfa for r in rs if r.ttfa is not None]  # noqa: E731
    med = lambda xs: float(np.median(xs)) if xs else float("nan")  # noqa: E731
    return {
        "real_det": sum(r.detected for r in real),
        "real_n": len(real),
        "synth_det": sum(r.detected for r in synth),
        "synth_n": len(synth),
        "real_recall": float(np.mean([r.recall for r in real])) if real else float("nan"),
        "real_protocol_recall": float(np.mean([r.protocol_recall for r in real])) if real else float("nan"),
        "synth_recall": float(np.mean([r.recall for r in synth])) if synth else float("nan"),
        "ttfa_real": med(delays(real)),
        "ttfa_synth": med(delays(synth)),
        "ttfa_all": med(delays(results)),
        "censored": sum(not r.detected for r in results),
    }


def paired_median_ttfa(a: list[PositiveResult], b: list[PositiveResult]) -> tuple[float, float, int]:
    """Median time to alarm of ``a`` and ``b`` over the clips both detected (``nan`` when none), and their number."""
    pairs = [(x.ttfa, y.ttfa) for x, y in zip(a, b, strict=True) if x.ttfa is not None and y.ttfa is not None]
    if not pairs:
        return float("nan"), float("nan"), 0
    return float(np.median([p[0] for p in pairs])), float(np.median([p[1] for p in pairs])), len(pairs)


def verdict(cand: list[PositiveResult], base: list[PositiveResult]) -> tuple[str, str]:
    """Pre-registered #137 verdict of a candidate against one debounce comparator: ``(label, detail)``."""
    c, b = summarise(cand), summarise(base)
    det_c, det_b = c["real_det"] + c["synth_det"], b["real_det"] + b["synth_det"]
    rec_c, rec_b = round(c["real_recall"], 2), round(b["real_recall"], 2)
    tc, tb, n = paired_median_ttfa(cand, base)
    detail = f"det {det_c}/{det_b}, recall {rec_c:.2f}/{rec_b:.2f}, ttfa {tc:.1f}/{tb:.1f} (n={n})"
    if n == 0:
        return ("n/a", detail)
    no_worse = det_c >= det_b and rec_c >= rec_b
    better = det_c > det_b or rec_c > rec_b
    if no_worse and tc <= tb:
        return ("PASS(i)" if not better else "PASS(i)+", detail)
    if no_worse and better and tc + 1 <= 2 * (tb + 1):
        return ("PASS(ii)", detail)
    return ("fail", detail)


# ---- conformal threshold (simplified P6) ----------------------------------------------------------------------------


def conformal_threshold(calibration: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold: the ``ceil((n+1)(1-alpha))``-th smallest calibration score (``inf`` when n is too small)."""
    n = len(calibration)
    rank = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if rank > n else float(np.sort(calibration)[rank - 1])


def goat_day(name: str) -> str:
    """Recording day of a goat clip."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else "?"


# ---- cost -----------------------------------------------------------------------------------------------------------


def bench_decision(checks: int = 200_000, repeats: int = 5) -> dict[str, float]:  # noqa: C901
    """Per-check cost (microseconds, pure Python, one thread) of the decision layers fed with a random score stream."""
    rng = np.random.default_rng(0)
    g = [float(x) for x in rng.random(checks) * 0.2]
    results: dict[str, float] = {}

    def timed(fn: Callable[[], None]) -> float:
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best / checks * 1e6

    def empty() -> None:
        for x in g:
            _ = x > 0.1

    def debounce() -> None:
        run = 0
        for x in g:
            run = run + 1 if x > 0.1 else 0
            _ = run >= 4

    def cusum() -> None:
        s = 0.0
        for x in g:
            s = max(0.0, s + x - 0.07)
            _ = s > 0.2

    def sprt() -> None:
        s = 0.0
        for x in g:
            s = max(0.0, s + (2.3 if x > 0.1 else -0.64))
            _ = s > 4.6

    results["loop + threshold"] = timed(empty)
    results["debounce (run counter)"] = timed(debounce)
    results["cusum"] = timed(cusum)
    results["sprt (LLR cusum)"] = timed(sprt)
    return results


# ---- post hoc additions (after the first run, see the report): glitch protection and hold --------------------------


def saturated_response(cand: Candidate, p: float, hits: int = 5, length: int = 80) -> tuple[int | None, int]:
    """Response of a rule to a saturated gate (score 1) after a quiet start.

    Returns:
        ``(n_min, hold)``: the number of consecutive saturated checks needed for the rule to alarm (None when it never
        does within ``length``), and the number of further checks the rule's own alarm stays on after ``hits`` saturated
        checks followed by zeros (0 for a plain run-length debounce).
    """
    n = length
    f = {
        "first": np.zeros(n, dtype=int),
        "area": np.full(n, 100.0),
        "warm": np.full(n, 50.0),
        "wm25": np.concatenate([np.full(hits, 50.0), np.zeros(n - hits)]),
        "t": np.arange(n, dtype=float),
    }
    f["first"][0] = 1  # the first check of a stream has no motion information
    f["wm25"][0] = 0.0
    on = cand.alarm(f, p)
    saturated = dict(f, wm25=np.where(np.arange(n) >= 1, 50.0, 0.0))
    first = np.flatnonzero(cand.alarm(saturated, p))
    n_min = int(first[0]) if len(first) else None
    last = int(np.flatnonzero(on)[-1]) if on.any() else -1
    return n_min, max(last - (hits - 1), 0)


__all__ = [
    "HARD_NEGATIVE",
    "Candidate",
    "PositiveResult",
    "Point",
    "build_candidates",
    "debounce_checks",
    "replay_production",
]
