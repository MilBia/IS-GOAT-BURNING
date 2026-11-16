"""Candidate decision rules evaluated on stored per-check features.

Every rule maps ``(features, parameter) -> boolean array`` (one entry per check). Rules other than
``deployed`` never alarm on the first check of a stream, where no motion information exists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tools.fire_rule_eval.bgmodel import CANDIDATE_VARIANTS
from tools.fire_rule_eval.bgmodel import SENSITIVITY_VARIANTS
from tools.fire_rule_eval.features import FINE_GRID
from tools.fire_rule_eval.features import TILE_GRID
from tools.fire_rule_eval.invariance import VARIANTS as INVARIANCE_VARIANTS
from tools.fire_rule_eval.lowrank import CANDIDATE_VARIANTS as LR_CANDIDATE_VARIANTS
from tools.fire_rule_eval.lowrank import CONTROL_VARIANTS as LR_CONTROL_VARIANTS
from tools.fire_rule_eval.lowrank import SENSITIVITY_VARIANTS as LR_SENSITIVITY_VARIANTS
from tools.fire_rule_eval.tracking import BlobTracker
from tools.fire_rule_eval.tracking import LeakyIntegrator
from tools.fire_rule_eval.tracking import TrackGate
from tools.fire_rule_eval.tracking import box_gate

Features = dict[str, np.ndarray]

MIN_WARM_FRACTION = 0.002
MIN_MOVING_FRACTION = 0.0005

FRACTION_GRID = tuple(float(x) for x in np.geomspace(0.0002, 0.25, 36))
RATIO_GRID = tuple(float(x) for x in np.linspace(0.02, 0.95, 32))
TILE_GRID_VALUES = tuple(float(x) for x in np.linspace(0.005, 0.9, 36))


@dataclass(frozen=True)
class Rule:
    """A candidate rule with a single swept parameter.

    Attributes:
        name: Identifier used in reports.
        description: One-line explanation.
        grid: Parameter values to sweep (ascending = more conservative).
        decide: Callable returning a boolean alarm array for a given parameter value.
        production_value: The production default of the swept parameter, if the rule is deployed.
        score: Optional continuous per-check score such that ``decide(f, p) == score(f) > p`` (zero where the rule's
            gates fail); enables the continuous margin analysis of the comparison protocol.
        chain: Whether chaining the negative clips into one stream is meaningful (False when the feature state is
            computed per clip at extraction time and therefore not carried across clip joins).
    """

    name: str
    description: str
    grid: tuple[float, ...]
    decide: Callable[[Features, float], np.ndarray]
    production_value: float | None = None
    score: Callable[[Features], np.ndarray] | None = None
    chain: bool = True


def _not_first(f: Features) -> np.ndarray:
    return f["first"] == 0


def _deployed(f: Features, p: float) -> np.ndarray:
    return f["wm25"] > p * f["area"]


def _skip_first(f: Features, p: float) -> np.ndarray:
    return _deployed(f, p) & _not_first(f)


def _threshold(key: str) -> Callable[[Features, float], np.ndarray]:
    def decide(f: Features, p: float) -> np.ndarray:
        return (f[key] > p * f["area"]) & _not_first(f)

    return decide


def _area_relative(f: Features, p: float) -> np.ndarray:
    warm_enough = f["warm"] >= MIN_WARM_FRACTION * f["area"]
    moving_enough = f["wm25"] > MIN_MOVING_FRACTION * f["area"]
    return warm_enough & moving_enough & (f["wm25"] > p * f["warm"]) & _not_first(f)


def area_relative_score(wm_key: str = "wm25", warm_key: str = "warm") -> Callable[[Features], np.ndarray]:
    """Continuous score of the ``area_relative`` family: share of the warm region that moves (0 where a gate fails)."""

    def score(f: Features) -> np.ndarray:
        warm_enough = f[warm_key] >= MIN_WARM_FRACTION * f["area"]
        moving_enough = f[wm_key] > MIN_MOVING_FRACTION * f["area"]
        share = f[wm_key] / np.maximum(f[warm_key], 1)
        return np.where(warm_enough & moving_enough & _not_first(f), share, 0.0)

    return score


def _area_relative_family(wm_key: str, warm_key: str = "warm") -> Callable[[Features, float], np.ndarray]:
    score = area_relative_score(wm_key, warm_key)

    def decide(f: Features, p: float) -> np.ndarray:
        return score(f) > p

    return decide


def _tiled(f: Features, p: float) -> np.ndarray:
    moving_enough = f["wm25"] > MIN_MOVING_FRACTION * f["area"]
    return (f["tile_wm25"].max(axis=1) > p) & moving_enough & _not_first(f)


def tiled_score(wm_key: str = "wm25", tile_key: str = "tile_wm25") -> Callable[[Features], np.ndarray]:
    """Continuous score of the ``tiled`` rule: the largest tile fraction of warm AND moving (0 where the moving gate fails)."""

    def score(f: Features) -> np.ndarray:
        moving_enough = f[wm_key] > MIN_MOVING_FRACTION * f["area"]
        return np.where(moving_enough & _not_first(f), f[tile_key].max(axis=1), 0.0)

    return score


def abs_score(key: str) -> Callable[[Features], np.ndarray]:
    """Continuous score of the production ``skip_first`` form: ``key`` count as a share of the frame (0 on a first check)."""

    def score(f: Features) -> np.ndarray:
        return np.where(_not_first(f), f[key] / np.maximum(f["area"], 1), 0.0)

    return score


def _white_core(f: Features, p: float) -> np.ndarray:
    return ((f["wm25"] + f["white_wm25"]) > p * f["area"]) & _not_first(f)


# ---- #134: candidate persistence (P15) and leaky evidence accumulation (P8) -------------------------------------
# Stateful rules: the whole clip is processed from its first check (the evaluation applies the warm-up mask
# afterwards) and the per-check score series is cached in the feature dict, so a parameter sweep is cheap.

TILE_FRACTION_GRID = tuple(float(x) for x in np.geomspace(0.0005, 0.9, 40))

P15_GATES = {
    "p15_persist3": TrackGate(3, 3),
    "p15_anchor3": TrackGate(3, 3, anchored=True),
    "p15_anchor3of5": TrackGate(3, 5, anchored=True),
    "p15_anchor5of8": TrackGate(5, 8, anchored=True),
    "p15_grow3": TrackGate(3, 3, anchored=True, min_growth=1.0),
    "p15_tall3": TrackGate(3, 3, anchored=True, min_elongation=1.0),
}
P8_DECAYS = (0.5, 0.7, 0.85)
P8_GRIDS = {"tile": TILE_GRID, "fine": FINE_GRID}
GATE_MIN_HITS = TrackGate(2, 2, anchored=True)


def _first_rows(f: Features) -> np.ndarray:
    return f["first"] == 1


def _p15_series(f: Features, key: str) -> np.ndarray:
    cache_key = f"_series_{key}"
    if cache_key not in f:
        gate = P15_GATES[key]
        tracker = BlobTracker()
        out = np.zeros(len(f["t"]))
        for i, blobs in enumerate(f["blobs"]):
            if f["first"][i]:
                tracker.reset()
                continue
            tracker.update(blobs)
            out[i] = tracker.score(gate)
        f[cache_key] = out
    return f[cache_key]


def _evidence(f: Features, grid_name: str) -> np.ndarray:
    columns, rows = P8_GRIDS[grid_name]
    source = f["tile_wm25"] if grid_name == "tile" else f["fine_wm25"]
    evidence = source.reshape(-1, rows, columns).astype(np.float64).copy()
    evidence[_first_rows(f)] = 0.0  # a first check has no motion information
    return evidence


def _p8_series(f: Features, grid_name: str, decay: float, gated: bool) -> np.ndarray:
    cache_key = f"_series_p8_{grid_name}_{decay}_{gated}"
    if cache_key not in f:
        columns, rows = P8_GRIDS[grid_name]
        evidence = _evidence(f, grid_name)
        integrator = LeakyIntegrator(decay, (rows, columns))
        tracker = BlobTracker()
        out = np.zeros(len(f["t"]))
        for i in range(len(out)):
            x = evidence[i]
            if gated:
                if f["first"][i]:
                    tracker.reset()
                else:
                    tracker.update(f["blobs"][i])
                x = x * box_gate(tracker.persistent_boxes(GATE_MIN_HITS), (rows, columns))
            out[i] = integrator.update(x)
        f[cache_key] = out
    return f[cache_key]


def _p15_rule(key: str) -> Callable[[Features, float], np.ndarray]:
    def decide(f: Features, p: float) -> np.ndarray:
        return (_p15_series(f, key) > p) & _not_first(f)

    return decide


def _p8_rule(grid_name: str, decay: float, gated: bool) -> Callable[[Features, float], np.ndarray]:
    def decide(f: Features, p: float) -> np.ndarray:
        return (_p8_series(f, grid_name, decay, gated) > p) & _not_first(f)

    return decide


def debounced_score(score: Callable[[Features], np.ndarray], checks: int) -> Callable[[Features], np.ndarray]:
    """Score of the debounced rule: the minimum over the last ``checks`` scores (``> p`` iff all of them are)."""

    def wrapped(f: Features) -> np.ndarray:
        values = score(f)
        out = values.copy()
        for shift in range(1, checks):
            lagged = np.concatenate([np.zeros(shift), values[:-shift]]) if shift < len(values) else np.zeros(len(values))
            out = np.minimum(out, lagged)
        return out

    return wrapped


def debounced(decide: Callable[[Features, float], np.ndarray], checks: int) -> Callable[[Features, float], np.ndarray]:
    """Wraps a rule so that it alarms only after ``checks`` consecutive alarmed checks (the production debounce)."""

    def wrapped(f: Features, p: float) -> np.ndarray:
        alarm = decide(f, p)
        run = np.zeros(len(alarm), dtype=int)
        for i, value in enumerate(alarm):
            run[i] = (run[i - 1] + 1 if i else 1) if value else 0
        return run >= checks

    return wrapped


TRACKING_RULES: tuple[Rule, ...] = (
    *(
        Rule(key, f"P15: persistent candidate area (gate {gate}) > p * frame area", FRACTION_GRID, _p15_rule(key))
        for key, gate in P15_GATES.items()
    ),
    *(
        Rule(
            f"p8_{grid}_{int(decay * 100)}",
            f"P8: leaky integrator (decay {decay}) of warm-AND-moving evidence on a {grid} grid, max tile > p",
            TILE_FRACTION_GRID,
            _p8_rule(grid, decay, False),
        )
        for grid in P8_GRIDS
        for decay in P8_DECAYS
    ),
    *(
        Rule(
            f"p15p8_{grid}_{int(decay * 100)}",
            f"P15+P8: leaky integrator (decay {decay}, {grid} grid) fed from tiles under persistent tracks, max > p",
            TILE_FRACTION_GRID,
            _p8_rule(grid, decay, True),
        )
        for grid in P8_GRIDS
        for decay in P8_DECAYS
    ),
)

# ---- #135: illumination-invariant motion masks (P4, P11, P12; see ``invariance.py``) --------------------------------
# Every variant has a "replace" rule (warm AND mask_v) and an "AND" rule (warm AND motion25 AND mask_v), both in the
# ``area_relative`` form (share of the warm region that moves). P4c also normalises the warm mask (``warm_p4c``).
# P12 state is computed per clip at extraction time, so the chained-goat stream is not meaningful for it.

INVARIANCE_RULES: tuple[Rule, ...] = tuple(
    Rule(
        f"{variant}{suffix}",
        f"#135 {variant}: share of the warm region that is {'AND-ed with motion25 and ' if suffix else ''}"
        f"flagged by the {variant} mask > p",
        RATIO_GRID,
        _area_relative_family(f"{kind}_{variant}", "warm_p4c" if variant == "p4c" else "warm"),
        score=area_relative_score(f"{kind}_{variant}", "warm_p4c" if variant == "p4c" else "warm"),
        chain=not variant.startswith("p12"),
    )
    for variant in INVARIANCE_VARIANTS
    for suffix, kind in (("", "wm"), ("_and", "and"))
)

# ---- #136: local-contrast small-target scores (P9 CFAR, P10 top-hat / LCM, P1 Haar; see ``localcontrast.py``) --------
# PRE-REGISTERED before the first comparison of any candidate (variants, windows and rank orders are fixed in
# ``localcontrast.py``; nothing is tuned). 6 variants x 2 forms + 1 control = 13 configurations (+ the 2 baseline controls
# ``area_relative`` / ``area_relative_deb3`` recomputed on the same 480x270 features = 15):
#   * ``<variant>_and``: gate AND score. The gate is the production warm-AND-moving pixel mask (the scores are built from
#     it) plus the ``area_relative`` minimum warm / moving areas; alarm iff score > p.
#   * ``<variant>_or``: ``area_relative`` at ``OR_PARTNER_THRESHOLD`` (the baseline's own best safe point at 480x270,
#     fixed from the baseline control run before any candidate was compared) OR (the ``_and`` rule), p swept under zero goat
#     alarms of the combination. Tests whether the small-target channel adds anything to the baseline.
#   * ``ctl_peak_and``: control without the local reference (N-th largest cell density), i.e. "is a local peak enough?".
# All scores are order statistics normalised by frame area / in absolute units (no division by the total warm area).

OR_PARTNER_THRESHOLD = (
    0.11  # grid best safe point of area_relative on the 480x270 features (baseline run, before any candidate)
)
LC_GRID = tuple(float(x) for x in np.geomspace(1e-4, 1e3, 220))
LC_VARIANTS = ("p9_cfar_z", "p9_cfar_diff", "p10_tophat", "p10_lcm", "p1_global", "p1_local")
LC_DESCRIPTIONS = {
    "p9_cfar_z": "CA-CFAR z-score of the warm-AND-moving density (guard 25, outer 61)",
    "p9_cfar_diff": "CA-CFAR difference (cell density minus ring mean, no noise normalisation)",
    "p10_tophat": "top-hat (31x31) of the heat map R-B inside warm-AND-moving",
    "p10_lcm": "local contrast measure (cell 15 vs the brightest of 8 neighbours) of the heat map",
    "p1_global": "Haar level-1 detail energy over the whole warm-AND-moving region (Toreyin)",
    "p1_local": "Haar level-1 detail energy, windowed (18 px) region mean, 16th largest",
}
OR_FLAG = 1e9  # score of a check where the OR partner alarms (any p below it alarms)


def lc_gate(f: Features) -> np.ndarray:
    """Minimum-area gate shared with ``area_relative`` (warm >= 0.2 %, moving > 0.05 %) and no first check."""
    return (f["warm"] >= MIN_WARM_FRACTION * f["area"]) & (f["wm25"] > MIN_MOVING_FRACTION * f["area"]) & _not_first(f)


def lc_and_score(key: str) -> Callable[[Features], np.ndarray]:
    """Score of the AND form: the stored local-contrast value where the gate holds, else 0."""

    def score(f: Features) -> np.ndarray:
        return np.where(lc_gate(f), f[f"lc_{key}"], 0.0)

    return score


def lc_or_score(key: str) -> Callable[[Features], np.ndarray]:
    """Score of the OR form: flagged (``OR_FLAG``) where the ``area_relative`` partner alarms, else the AND score."""
    and_score = lc_and_score(key)

    def score(f: Features) -> np.ndarray:
        return np.where(_area_relative(f, OR_PARTNER_THRESHOLD), OR_FLAG, and_score(f))

    return score


def _score_rule(score: Callable[[Features], np.ndarray]) -> Callable[[Features, float], np.ndarray]:
    def decide(f: Features, p: float) -> np.ndarray:
        return score(f) > p

    return decide


def _lc_rule(name: str, description: str, score: Callable[[Features], np.ndarray]) -> Rule:
    return Rule(name, description, LC_GRID, _score_rule(score), score=score)


LOCAL_CONTRAST_RULES: tuple[Rule, ...] = (
    *(_lc_rule(f"{key}_and", f"#136 AND: {LC_DESCRIPTIONS[key]} > p (gated)", lc_and_score(key)) for key in LC_VARIANTS),
    *(
        _lc_rule(
            f"{key}_or", f"#136 OR: area_relative({OR_PARTNER_THRESHOLD}) OR [{LC_DESCRIPTIONS[key]} > p]", lc_or_score(key)
        )
        for key in LC_VARIANTS
    ),
    _lc_rule(
        "ctl_peak_and",
        "#136 control: N-th largest 9x9 cell density of warm-AND-moving > p (no local reference)",
        lc_and_score("ctl_peak"),
    ),
)


# ---- #138: global-change gate on ``area_relative`` (features in ``globalchange.py``) ---------------------------------
# PRE-REGISTERED before any ``gl_*`` feature was extracted and before any ``compare``: 6 single gates x 2 fixed thresholds
# + 2 combined gates (strict / loose) = 14 gate configurations at D=0, + the 2 combined gates with the 3-checks debounce
# = 16 candidates (+ 2 controls ``area_relative`` / ``area_relative_deb3``). Thresholds are physical constants, not derived
# from any clip; nothing is tuned afterwards. The rule is ``area_relative`` whose score is zeroed where the gate says that
# the whole scene changed ("blocked"); the swept parameter ``p`` is the ``area_relative`` threshold, as for the baseline.
# Pre-stated expectation (so that a 3/8 reads as predicted, not as bad luck): the best a global gate can do is silence the
# two tunnels, the Ropeway and the two time-lapses that alarm today (Funchal, Norway); Ljubljana (local blinking lights)
# and Falling leaves (local motion) are not global changes and are expected to survive, i.e. the floor is about 2-3/8.

GATE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "dv": (3.0, 6.0),  # |change of the mean V| in grey levels
    "out": (0.25, 0.5),  # share of the frame that changes outside the warm mask
    "tiles": (0.4, 0.7),  # share of 16x9 tiles that changed
    "coh": (0.5, 0.75),  # share of tiles that brightened / darkened together
    "tileswg": (0.4, 0.7),  # as ``tiles`` but only blocking while < 50 % of the changed tiles contain warm pixels
    "shift": (0.02, 0.05),  # phase-correlation shift, fraction of the width
}
GATE_WARM_GUARD = 0.5
GATE_LEVELS = ("strict", "loose")  # the first / second threshold of every single gate, combined with OR
COMBINED_GATES = ("dv", "out", "tiles", "coh", "shift")  # ``tileswg`` is evaluated alone (it is the guarded ``tiles``)


def gate_blocked(f: Features, kind: str, threshold: float) -> np.ndarray:
    """Per check: whether the single gate ``kind`` says that the whole scene changed (the alarm is suppressed)."""
    if kind == "dv":
        return np.abs(f["gl_dv"]) >= threshold
    if kind == "out":
        return f["gl_out"] >= threshold
    if kind == "tiles":
        return f["gl_tiles"] >= threshold
    if kind == "coh":
        return f["gl_coh"] >= threshold
    if kind == "tileswg":
        return (f["gl_tiles"] >= threshold) & (f["gl_wg"] < GATE_WARM_GUARD)
    if kind == "shift":
        return f["gl_shift"] >= threshold
    raise KeyError(kind)


def combined_blocked(f: Features, level: str) -> np.ndarray:
    """Per check: any of the combined single gates (at the ``strict`` or ``loose`` thresholds) blocks."""
    index = GATE_LEVELS.index(level)
    blocked = np.zeros(len(f["t"]), dtype=bool)
    for kind in COMBINED_GATES:
        blocked |= gate_blocked(f, kind, GATE_THRESHOLDS[kind][index])
    return blocked


def gated_score(blocked: Callable[[Features], np.ndarray]) -> Callable[[Features], np.ndarray]:
    """``area_relative`` score that is zero wherever ``blocked`` holds."""
    base = area_relative_score()

    def score(f: Features) -> np.ndarray:
        return np.where(blocked(f), 0.0, base(f))

    return score


def _gate_rule(name: str, description: str, score: Callable[[Features], np.ndarray], checks: int = 1) -> Rule:
    if checks > 1:
        score = debounced_score(score, checks)
    return Rule(name, description, RATIO_GRID, _score_rule(score), score=score)


def _single_gate_rules() -> tuple[Rule, ...]:
    rules = []
    for kind, thresholds in GATE_THRESHOLDS.items():
        for level, threshold in zip(GATE_LEVELS, thresholds, strict=True):
            rules.append(
                _gate_rule(
                    f"gate_{kind}_{level}",
                    f"#138: area_relative, blocked where the {kind} global-change gate fires at {threshold}",
                    gated_score(lambda f, k=kind, t=threshold: gate_blocked(f, k, t)),
                )
            )
    return tuple(rules)


GATE_RULES: tuple[Rule, ...] = (
    *_single_gate_rules(),
    *(
        _gate_rule(
            f"gate_all_{level}{suffix}",
            f"#138: area_relative, blocked where any of {COMBINED_GATES} fires ({level} thresholds)"
            + (", alarmed on 3 consecutive checks" if checks > 1 else ""),
            gated_score(lambda f, lv=level: combined_blocked(f, lv)),
            checks,
        )
        for checks, suffix in ((1, ""), (3, "_deb3"))
        for level in GATE_LEVELS
    ),
)


# ---- #139: background-model foreground masks (MOG2 / KNN / running median; see ``bgmodel.py``) -----------------------
# PRE-REGISTERED (``bgmodel.py`` docstring) before any ``bg*`` feature was extracted and before any ``compare``:
# 6 candidate models x 3 forms (``area_relative`` share, ``tiled``, absolute area) + 6 debounced (D=2) = 24 candidates,
# + 4 controls on the same features. The sensitivity models (other histories / schedules) get the same three forms but
# are only reported in a separately labelled table. State is computed per clip at extraction time (``chain=False``).


def _bg_rules(variants: tuple[str, ...], debounce: bool) -> tuple[Rule, ...]:
    rules: list[Rule] = []
    for v in variants:
        wm = f"bgwm_{v}"
        ratio = area_relative_score(wm)
        rules.append(
            Rule(
                f"bg_{v}",
                f"#139 {v}: share of the warm region that is background-model foreground > p",
                RATIO_GRID,
                _score_rule(ratio),
                score=ratio,
                chain=False,
            )
        )
        tiled = tiled_score(wm, f"tile_bg_{v}")
        rules.append(
            Rule(
                f"bg_{v}_tiled",
                f"#139 {v}: max 8x4 tile fraction of warm AND foreground > p",
                TILE_GRID_VALUES,
                _score_rule(tiled),
                score=tiled,
                chain=False,
            )
        )
        absolute = abs_score(wm)
        rules.append(
            Rule(
                f"bg_{v}_abs",
                f"#139 {v}: warm AND foreground > p * frame area",
                FRACTION_GRID,
                _score_rule(absolute),
                score=absolute,
                chain=False,
            )
        )
        if debounce:
            deb = debounced_score(ratio, 3)
            rules.append(
                Rule(
                    f"bg_{v}_deb3",
                    f"#139 {v}: bg_{v} alarmed on 3 consecutive checks",
                    RATIO_GRID,
                    _score_rule(deb),
                    score=deb,
                    chain=False,
                )
            )
    return tuple(rules)


BG_CANDIDATE_RULES: tuple[Rule, ...] = _bg_rules(tuple(CANDIDATE_VARIANTS), True)
BG_SENSITIVITY_RULES: tuple[Rule, ...] = _bg_rules(tuple(SENSITIVITY_VARIANTS), False)
BG_CONTROL_RULES: tuple[Rule, ...] = (
    Rule(
        "abs_wm25",
        "#139 control: warm AND motion25 > p * frame area (production skip_first, with a score)",
        FRACTION_GRID,
        _score_rule(abs_score("wm25")),
        score=abs_score("wm25"),
    ),
)


# ---- #140: low-rank + sparse masks (eigenbackground / GoDec / RPCA; see ``lowrank.py``) --------------------------------
# PRE-REGISTERED (``lowrank.py`` docstring) before any ``lr_*`` feature was extracted and before any ``compare``: 8 candidate
# models x 3 forms (``area_relative`` share, ``tiled``, absolute area) + 8 debounced (D=2) = 32 candidate configurations;
# controls: the mean-of-window model (same forms + debounce), ``bg_med5`` / ``bg_med9`` of #139 (re-extracted in the same pass)
# and the baselines on the same features. Sensitivity models get the three forms but are reported separately.

LR_CANDIDATE_RULES: tuple[Rule, ...] = _bg_rules(tuple(LR_CANDIDATE_VARIANTS), True)
LR_CONTROL_RULES: tuple[Rule, ...] = _bg_rules(tuple(LR_CONTROL_VARIANTS), True)
LR_SENSITIVITY_RULES: tuple[Rule, ...] = _bg_rules(tuple(LR_SENSITIVITY_VARIANTS), False)


RULES: tuple[Rule, ...] = (
    Rule("deployed", "production: warm AND motion(25) > p * area; first check is colour only", FRACTION_GRID, _deployed, 0.1),
    Rule("skip_first", "deployed, but never alarm on the first check after (re)start", FRACTION_GRID, _skip_first, 0.1),
    Rule("motion15", "skip_first with motion threshold 15", FRACTION_GRID, _threshold("wm15")),
    Rule("motion40", "skip_first with motion threshold 40", FRACTION_GRID, _threshold("wm40")),
    Rule("avg_motion", "motion vs the mean of the last 3 checks, not the previous one", FRACTION_GRID, _threshold("wm_avg25")),
    Rule(
        "area_relative",
        "share of the warm region that moves > p (min warm / moving area)",
        RATIO_GRID,
        _area_relative,
        score=area_relative_score(),
    ),
    Rule(
        "tiled",
        "max over 8x4 tiles of the tile fraction that is warm AND moving > p",
        TILE_GRID_VALUES,
        _tiled,
        score=tiled_score(),
    ),
    Rule("component", "largest connected warm-AND-moving component > p * frame area", FRACTION_GRID, _threshold("cc_max")),
    Rule("component_closed", "as component, after a 5x5 morphological close", FRACTION_GRID, _threshold("cc_close_max")),
    Rule("white_core", "warm-AND-moving plus bright desaturated moving pixels near warm", FRACTION_GRID, _white_core),
)

MIN_FLICKER_FRACTION = 0.0005
AREA_RELATIVE_SAFE = 0.14  # most sensitive zero-alarm point of ``area_relative`` in the #104 evaluation
AREA_RELATIVE_PRODUCTION = 0.23  # shipped default of AREA_RELATIVE_THRESHOLD


def _flicker(key: str) -> Callable[[Features, float], np.ndarray]:
    def decide(f: Features, p: float) -> np.ndarray:
        return f[key] > p * f["flick_area"]  # flicker counts are at the flicker resolution, not at ``area``

    return decide


def _flicker_relative(f: Features, p: float) -> np.ndarray:
    warm_enough = f["fwarm"] >= MIN_WARM_FRACTION * f["flick_area"]
    return warm_enough & (f["flick_inc"] > MIN_FLICKER_FRACTION * f["flick_area"]) & (f["flick_inc"] > p * f["fwarm"])


def _area_or_flicker(f: Features, p: float) -> np.ndarray:
    return _area_relative(f, AREA_RELATIVE_PRODUCTION) | _flicker("flick_inc")(f, p)


def _area_or_flicker_relative(f: Features, p: float) -> np.ndarray:
    return _area_relative(f, AREA_RELATIVE_PRODUCTION) | _flicker_relative(f, p)


def _area_and_flicker(f: Features, p: float) -> np.ndarray:
    return _area_relative(f, AREA_RELATIVE_SAFE) & _flicker("flick_inc")(f, p)


# Rules that need the columns added by ``load_clips(..., flicker_dir=...)`` (see ``flicker.py``).
FLICKER_RULES: tuple[Rule, ...] = (
    Rule("flicker_win", "windowed reference: flickering warm pixels > p * frame area", FRACTION_GRID, _flicker("flick_win")),
    Rule("flicker_inc", "incremental: flickering warm pixels > p * frame area", FRACTION_GRID, _flicker("flick_inc")),
    Rule("flicker_rel", "incremental: share of the warm region that flickers > p", RATIO_GRID, _flicker_relative),
    Rule("area_or_flicker", "area_relative(0.23) OR incremental flicker > p * area", FRACTION_GRID, _area_or_flicker),
    Rule(
        "area_or_flicker_rel",
        "area_relative(0.23) OR share of the warm region that flickers > p",
        RATIO_GRID,
        _area_or_flicker_relative,
    ),
    Rule("area_and_flicker", "area_relative(0.14) AND incremental flicker > p * area", FRACTION_GRID, _area_and_flicker),
)

CONTROL_RULES: tuple[Rule, ...] = (
    Rule(
        "area_relative_deb3",
        "control: area_relative alarmed on 3 consecutive checks",
        RATIO_GRID,
        debounced(_area_relative, 3),
        score=debounced_score(area_relative_score(), 3),
    ),
    Rule("skip_first_deb3", "control: skip_first alarmed on 3 consecutive checks", FRACTION_GRID, debounced(_skip_first, 3)),
)

RULES_BY_NAME = {
    rule.name: rule
    for rule in (
        *RULES,
        *FLICKER_RULES,
        *TRACKING_RULES,
        *INVARIANCE_RULES,
        *LOCAL_CONTRAST_RULES,
        *GATE_RULES,
        *BG_CANDIDATE_RULES,
        *BG_SENSITIVITY_RULES,
        *BG_CONTROL_RULES,
        *LR_CANDIDATE_RULES,
        *LR_CONTROL_RULES,
        *LR_SENSITIVITY_RULES,
        *CONTROL_RULES,
    )
}
