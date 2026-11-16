"""Tests for the #139 background-model masks, extraction columns and rules."""

import cv2
import numpy as np

from tools.fire_rule_eval.bgmodel import ALL_VARIANTS
from tools.fire_rule_eval.bgmodel import CANDIDATE_VARIANTS
from tools.fire_rule_eval.bgmodel import BgModels
from tools.fire_rule_eval.bgmodel import bg_names
from tools.fire_rule_eval.bgmodel import extract_bg_features
from tools.fire_rule_eval.features import extract_features
from tools.fire_rule_eval.rules import BG_CANDIDATE_RULES
from tools.fire_rule_eval.rules import RULES_BY_NAME
from tools.fire_rule_eval.rules import tiled_score

HEIGHT, WIDTH = 270, 480
SQUARE = (slice(100, 140), slice(200, 260))


def _frame(rng: np.random.Generator, square: bool) -> np.ndarray:
    gray = rng.normal(100, 2.0, (HEIGHT, WIDTH)).clip(0, 255).astype(np.uint8)
    if square:
        gray[SQUARE] = 240
    return gray


def _share(models: BgModels, rng: np.random.Generator, checks: int, appears: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {v: [] for v in models.variants}
    for i in range(checks):
        for name, fg in models.masks(_frame(rng, i >= appears)).items():
            out[name].append(float((fg[SQUARE] > 0).mean()))
    return out


def test_new_static_object_is_foreground_then_absorbed() -> None:
    rng = np.random.default_rng(0)
    shares = _share(BgModels(tuple(CANDIDATE_VARIANTS)), rng, 40, 15)
    for name, series in shares.items():
        assert series[15] > 0.9, name  # the check on which the square appears
        assert series[-1] < 0.1, name  # absorbed (or, for the median, no longer different from the buffer) later on


def test_fixed_schedule_absorbs_slower_than_auto() -> None:
    def checks_until_absorbed(name: str) -> int:
        series = _share(BgModels((name,)), np.random.default_rng(3), 60, 10)[name][10:]
        return next(i for i, share in enumerate(series) if share < 0.5)

    assert checks_until_absorbed("mog2_h120f") > checks_until_absorbed("mog2_h30") + 3


def test_median_excludes_the_current_check() -> None:
    models = BgModels(("med5",))
    rng = np.random.default_rng(1)
    for _ in range(4):
        models.masks(_frame(rng, False))
    assert models.masks(_frame(rng, True))["med5"][SQUARE].mean() == 255  # the new square is compared to a clean median


def test_first_checks_of_median_are_empty() -> None:
    models = BgModels(("med9",))
    rng = np.random.default_rng(2)
    assert all(not models.masks(_frame(rng, True))["med9"].any() for _ in range(3))


def test_extraction_matches_baseline_columns() -> None:
    rng = np.random.default_rng(4)
    frames = []
    for i in range(8):
        frame = cv2.cvtColor(_frame(rng, False), cv2.COLOR_GRAY2BGR)
        if i >= 5:
            frame[60:90, 300:340] = (0, 190, 255)  # a warm patch that appears at check 5
        frames.append((float(i), frame))
    base = extract_features(iter(frames), blur_kernel=(5, 5))
    bg = extract_bg_features(iter(frames), blur_kernel=(5, 5), variants=("mog2_h30", "med5"))
    for key in ("t", "area", "first", "warm", "wm25", "tile_wm25"):
        assert np.array_equal(base[key], bg[key]), key
    assert set(bg_names(("mog2_h30", "med5"))) <= set(bg)
    assert bg["wm25"].max() > 0 and bg["bgwm_mog2_h30"].max() > 0
    assert np.all(bg["bgand_mog2_h30"] <= bg["bgwm_mog2_h30"])
    assert bg["tile_bg_med5"].shape == (8, 32)


def test_carried_state_continues_the_model() -> None:
    rng = np.random.default_rng(5)
    models = BgModels(("mog2_h30",))
    first = [(float(i), cv2.cvtColor(_frame(rng, False), cv2.COLOR_GRAY2BGR)) for i in range(10)]
    extract_bg_features(iter(first), (5, 5), models=models)
    assert models._models["mog2_h30"].n == 10


def test_rules_registered_and_scored() -> None:
    assert len(BG_CANDIDATE_RULES) == 6 * 4
    for rule in BG_CANDIDATE_RULES:
        assert rule.name in RULES_BY_NAME and rule.score is not None and not rule.chain
    assert {r.name for r in BG_CANDIDATE_RULES} >= {
        f"bg_{v}_{form}" for v in CANDIDATE_VARIANTS for form in ("tiled", "abs", "deb3")
    }
    assert all(v in ALL_VARIANTS for v in CANDIDATE_VARIANTS)


def test_score_matches_decide() -> None:
    n = 6
    f = {
        "first": np.array([1, 0, 0, 0, 0, 0]),
        "area": np.full(n, 1000.0),
        "warm": np.full(n, 100.0),
        "wm25": np.array([0, 5, 30, 60, 0, 20.0]),
        "bgwm_med5": np.array([0, 0, 30, 60, 0, 20.0]),
        "tile_bg_med5": np.tile(np.linspace(0, 0.5, 32), (n, 1)),
        "t": np.arange(n, dtype=float),
    }
    for name in ("bg_med5", "bg_med5_tiled", "bg_med5_abs", "bg_med5_deb3", "tiled", "abs_wm25"):
        f.setdefault("tile_wm25", f["tile_bg_med5"])
        rule = RULES_BY_NAME[name]
        for p in rule.grid[:: max(len(rule.grid) // 6, 1)]:
            assert np.array_equal(rule.decide(f, p), rule.score(f) > p), (name, p)
    assert tiled_score()(f)[0] == 0.0  # first check never scores
