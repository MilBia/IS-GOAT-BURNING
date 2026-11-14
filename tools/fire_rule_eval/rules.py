"""Candidate decision rules evaluated on stored per-check features.

Every rule maps ``(features, parameter) -> boolean array`` (one entry per check). Rules other than
``deployed`` never alarm on the first check of a stream, where no motion information exists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

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
    """

    name: str
    description: str
    grid: tuple[float, ...]
    decide: Callable[[Features, float], np.ndarray]
    production_value: float | None = None


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


def _tiled(f: Features, p: float) -> np.ndarray:
    moving_enough = f["wm25"] > MIN_MOVING_FRACTION * f["area"]
    return (f["tile_wm25"].max(axis=1) > p) & moving_enough & _not_first(f)


def _white_core(f: Features, p: float) -> np.ndarray:
    return ((f["wm25"] + f["white_wm25"]) > p * f["area"]) & _not_first(f)


RULES: tuple[Rule, ...] = (
    Rule("deployed", "production: warm AND motion(25) > p * area; first check is colour only", FRACTION_GRID, _deployed, 0.1),
    Rule("skip_first", "deployed, but never alarm on the first check after (re)start", FRACTION_GRID, _skip_first, 0.1),
    Rule("motion15", "skip_first with motion threshold 15", FRACTION_GRID, _threshold("wm15")),
    Rule("motion40", "skip_first with motion threshold 40", FRACTION_GRID, _threshold("wm40")),
    Rule("avg_motion", "motion vs the mean of the last 3 checks, not the previous one", FRACTION_GRID, _threshold("wm_avg25")),
    Rule("area_relative", "share of the warm region that moves > p (min warm / moving area)", RATIO_GRID, _area_relative),
    Rule("tiled", "max over 8x4 tiles of the tile fraction that is warm AND moving > p", TILE_GRID_VALUES, _tiled),
    Rule("component", "largest connected warm-AND-moving component > p * frame area", FRACTION_GRID, _threshold("cc_max")),
    Rule("component_closed", "as component, after a 5x5 morphological close", FRACTION_GRID, _threshold("cc_close_max")),
    Rule("white_core", "warm-AND-moving plus bright desaturated moving pixels near warm", FRACTION_GRID, _white_core),
)

RULES_BY_NAME = {rule.name: rule for rule in RULES}
