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

RULES_BY_NAME = {rule.name: rule for rule in (*RULES, *FLICKER_RULES)}
