"""Tests for the #138 global-change features and the gated ``area_relative`` rules."""

import cv2
import numpy as np

from tools.fire_rule_eval.features import FEATURE_NAMES
from tools.fire_rule_eval.features import extract_features
from tools.fire_rule_eval.globalchange import GLOBAL_NAMES
from tools.fire_rule_eval.globalchange import GlobalChange
from tools.fire_rule_eval.rules import GATE_RULES
from tools.fire_rule_eval.rules import GATE_THRESHOLDS
from tools.fire_rule_eval.rules import RULES_BY_NAME
from tools.fire_rule_eval.rules import gate_blocked

HEIGHT, WIDTH = 270, 480


def _base(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(80, 120, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    return cv2.GaussianBlur(frame, (5, 5), 0)


def _step(state: GlobalChange, previous: np.ndarray, current: np.ndarray, color: np.ndarray | None = None) -> dict[str, float]:
    """Feeds ``previous`` and then ``current`` and returns the features of ``current``."""
    color = np.zeros((HEIGHT, WIDTH), np.uint8) if color is None else color
    zero = np.zeros_like(color)
    for frame, diff, mask in ((previous, None, zero), (current, None, color)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame is current:
            diff = cv2.absdiff(gray, cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY))
        out = state.update(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), gray, diff, mask)
    return out


def test_first_check_is_zero() -> None:
    frame = _base()
    out = GlobalChange().update(
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        None,
        np.zeros((HEIGHT, WIDTH), np.uint8),
    )
    assert out == dict.fromkeys(GLOBAL_NAMES, 0.0)


def test_uniform_brightness_shift_trips_every_gate() -> None:
    before = _base()
    after = np.clip(before.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    out = _step(GlobalChange(), before, after)
    assert out["gl_dv"] > 30
    assert out["gl_out"] > 0.95 and out["gl_tiles"] > 0.95 and out["gl_coh"] > 0.95
    for kind, (strict, loose) in GATE_THRESHOLDS.items():
        if kind in {"shift"}:
            continue  # a uniform shift is not a translation
        feats = {key: np.array([value]) for key, value in out.items()}
        assert gate_blocked(feats, kind, strict)[0] and gate_blocked(feats, kind, loose)[0], kind


def test_local_warm_blob_trips_no_gate() -> None:
    before = _base()
    after = before.copy()
    after[100:160, 200:260] = (30, 90, 250)  # a flame-coloured 60x60 patch
    color = np.zeros((HEIGHT, WIDTH), np.uint8)
    color[100:160, 200:260] = 255
    out = _step(GlobalChange(), before, after, color)
    assert out["gl_tiles"] < 0.1 and out["gl_out"] < 0.01 and out["gl_wg"] == 1.0
    feats = {key: np.array([value]) for key, value in out.items()}
    for kind, (strict, _) in GATE_THRESHOLDS.items():
        assert not gate_blocked(feats, kind, strict)[0], kind


def test_warm_guard_relaxes_the_tiles_gate() -> None:
    feats = {"gl_tiles": np.array([0.8, 0.8]), "gl_wg": np.array([0.9, 0.1])}
    assert gate_blocked(feats, "tileswg", 0.7).tolist() == [False, True]


def test_pan_gives_a_phase_correlation_shift() -> None:
    before = _base()
    after = np.roll(before, 30, axis=1)
    assert _step(GlobalChange(), before, after)["gl_shift"] > 0.05


def test_extract_adds_global_columns_and_ignores_the_mosaic_exclusion() -> None:
    frames = [(float(i), _base(i % 2)) for i in range(4)]
    plain = extract_features(iter(frames), (5, 5))
    masked = extract_features(iter(frames), (5, 5), ((0.0, 0.0, 1.0, 1.0),))
    assert all(name in FEATURE_NAMES for name in GLOBAL_NAMES)
    for name in GLOBAL_NAMES:
        assert plain[name].shape == (4,) and plain[name][0] == 0.0
        assert np.array_equal(plain[name], masked[name])


def test_gated_rule_is_zero_where_blocked_and_equals_the_baseline_elsewhere() -> None:
    n = 3
    feats = {
        "t": np.arange(n, dtype=float),
        "area": np.full(n, 1000),
        "first": np.zeros(n, dtype=int),
        "warm": np.full(n, 100),
        "wm25": np.full(n, 50),
        "gl_dv": np.array([0.0, 10.0, 0.0]),
        "gl_out": np.zeros(n),
        "gl_tiles": np.zeros(n),
        "gl_coh": np.zeros(n),
        "gl_wg": np.ones(n),
        "gl_shift": np.zeros(n),
    }
    base = RULES_BY_NAME["area_relative"].score(feats)
    gated = RULES_BY_NAME["gate_dv_strict"].score(feats)
    assert gated.tolist() == [base[0], 0.0, base[2]]
    for rule in GATE_RULES:
        assert (rule.decide(feats, 0.1) == (rule.score(feats) > 0.1)).all()
