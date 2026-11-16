"""Candidate decision rules evaluated on stored per-check features.

Every rule maps ``(features, parameter) -> boolean array`` (one entry per check). Rules other than
``deployed`` never alarm on the first check of a stream, where no motion information exists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tools.fire_rule_eval.features import FINE_GRID
from tools.fire_rule_eval.features import TILE_GRID
from tools.fire_rule_eval.invariance import VARIANTS as INVARIANCE_VARIANTS
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
    Rule("tiled", "max over 8x4 tiles of the tile fraction that is warm AND moving > p", TILE_GRID_VALUES, _tiled),
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

RULES_BY_NAME = {rule.name: rule for rule in (*RULES, *FLICKER_RULES, *TRACKING_RULES, *INVARIANCE_RULES, *CONTROL_RULES)}
