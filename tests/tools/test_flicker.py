"""Tests for the temporal-flicker feature of the evaluation harness (#105)."""

import numpy as np

from tools.fire_rule_eval.flicker import HEIGHT
from tools.fire_rule_eval.flicker import WIDTH
from tools.fire_rule_eval.flicker import FlickerParams
from tools.fire_rule_eval.flicker import extract_flicker_features
from tools.fire_rule_eval.report import _merge_flicker
from tools.fire_rule_eval.rules import FLICKER_RULES

SECONDS = 6
FLICKER_AREA = (slice(100, 140), slice(100, 160))  # 40x60 px
STATIC_AREA = (slice(100, 140), slice(300, 360))


def _frames(flicker: bool) -> list[np.ndarray]:
    """Dark scene with one orange block that flickers (every other frame) and one static orange block."""
    frames = []
    for i in range(SECONDS * 30):
        frame = np.full((HEIGHT, WIDTH, 3), 20, dtype=np.uint8)
        frame[STATIC_AREA] = (0, 190, 255)
        frame[FLICKER_AREA] = (0, 240, 255) if (i % 2 == 0 or not flicker) else (0, 130, 200)
        frames.append(frame)
    return frames


def test_flickering_block_is_found_by_both_implementations_and_static_is_not() -> None:
    flicker = extract_flicker_features(iter(_frames(True)))
    static = extract_flicker_features(iter(_frames(False)))

    block = 40 * 60
    assert flicker["t"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]  # rows at 1 s, aligned with 1 Hz checks
    for key in ("flick_win", "flick_inc"):
        assert flicker[key][-1] >= 0.6 * block, key
        assert static[key].max() == 0, key


def test_incremental_tracks_windowed_reference() -> None:
    feats = extract_flicker_features(iter(_frames(True)))
    assert abs(feats["flick_inc"][-1] - feats["flick_win"][-1]) < 0.3 * feats["flick_win"][-1]


def test_windowed_can_be_disabled() -> None:
    feats = extract_flicker_features(iter(_frames(True)), FlickerParams(), windowed=False)
    assert (feats["flick_win"] == -1).all()


def test_merge_aligns_by_check_index_and_drops_settling_checks() -> None:
    n = 8
    colour = {
        "t": np.arange(n) * 1.001,  # NTSC-like timestamps must not break alignment
        "area": np.full(n, 100.0),
        "first": np.array([1] + [0] * (n - 1)),
    }
    flicker = {
        "t": np.arange(1, n, dtype=float),
        "warm": np.arange(1, n) * 10.0,
        "flick_win": np.arange(1, n) * 1.0,
        "flick_inc": np.arange(1, n) * 2.0,
    }
    merged = _merge_flicker(colour, flicker)
    assert merged is not None
    assert merged["flick_inc"].tolist() == [6.0, 8.0, 10.0, 12.0, 14.0]  # checks 3..7
    assert merged["first"].tolist() == [0] * 5
    assert _merge_flicker({"t": np.arange(2.0), "area": np.ones(2), "first": np.zeros(2)}, flicker) is None


def test_flicker_rules_use_only_flicker_columns() -> None:
    feats = {
        "area": np.array([1000.0, 1000.0]),
        "fwarm": np.array([100.0, 100.0]),
        "flick_win": np.array([0.0, 50.0]),
        "flick_inc": np.array([0.0, 50.0]),
        "warm": np.array([100.0, 100.0]),
        "wm25": np.array([0.0, 0.0]),
        "first": np.array([0, 0]),
    }
    for rule in FLICKER_RULES:
        if rule.name.startswith(("flicker_win", "flicker_inc", "flicker_rel")):
            assert rule.decide(feats, rule.grid[len(rule.grid) // 4]).tolist() == [False, True], rule.name
