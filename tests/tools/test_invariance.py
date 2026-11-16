"""Tests for the #135 illumination-invariant motion masks (P4, P11, P12)."""

import cv2
import numpy as np

from tools.fire_rule_eval.features import FEATURE_NAMES
from tools.fire_rule_eval.features import extract_features
from tools.fire_rule_eval.invariance import INVARIANCE_NAMES
from tools.fire_rule_eval.invariance import VARIANTS
from tools.fire_rule_eval.invariance import InvarianceMasks
from tools.fire_rule_eval.invariance import census_codes
from tools.fire_rule_eval.invariance import census_distance
from tools.fire_rule_eval.invariance import gain_compensated_motion
from tools.fire_rule_eval.invariance import shift_compensated_motion
from tools.fire_rule_eval.invariance import tile_compensated_motion
from tools.fire_rule_eval.rules import RULES_BY_NAME


def _scene(seed: int = 0, size: tuple[int, int] = (135, 240)) -> np.ndarray:
    """A textured grey scene whose values stay away from saturation (60..180)."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 120, size, dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 1.5) + 60


def test_census_codes_ignore_a_uniform_brightness_shift() -> None:
    gray = _scene()
    brighter = cv2.add(gray, 30)
    assert census_distance(census_codes(gray), census_codes(brighter)).max() == 0


def test_census_distance_sees_a_local_change() -> None:
    gray = _scene()
    changed = gray.copy()
    changed[40:80, 60:120] = 255 - changed[40:80, 60:120]
    distance = census_distance(census_codes(gray), census_codes(changed))
    assert distance[50:70, 70:110].max() > 4
    assert distance[:30].max() == 0


def test_global_drift_compensation_removes_a_uniform_shift_but_keeps_a_local_change() -> None:
    gray = _scene()
    shifted = cv2.add(gray, 40)
    for compensate in (shift_compensated_motion, tile_compensated_motion):
        assert cv2.countNonZero(compensate(shifted, gray)) == 0
    assert cv2.countNonZero(cv2.threshold(cv2.absdiff(shifted, gray), 25, 255, cv2.THRESH_BINARY)[1]) == gray.size
    changed = shifted.copy()
    changed[40:60, 60:100] = 255
    assert cv2.countNonZero(shift_compensated_motion(changed, gray)) >= 20 * 40 * 0.9


def test_gain_compensation_removes_a_uniform_gain() -> None:
    gray = _scene()
    scaled = cv2.convertScaleAbs(gray, alpha=1.3)
    assert cv2.countNonZero(gain_compensated_motion(scaled, gray)) == 0


def test_two_ema_band_is_empty_on_a_static_scene_and_sees_a_persistent_change() -> None:
    frame = np.zeros((135, 240, 3), dtype=np.uint8)
    frame[:] = (40, 80, 120)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color = np.zeros(gray.shape, dtype=np.uint8)
    color[:] = 255
    masks = InvarianceMasks()
    for _ in range(5):
        out = masks.update(frame, color, gray, np.zeros_like(gray) if masks.n else None)
    assert all(out[f"mv_p12_{kind}"] == 0 for kind in ("a15", "a120", "p15", "p120"))
    bright = frame.copy()
    bright[20:60, 20:80] = 255
    bright_gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
    for _ in range(6):
        out = masks.update(bright, color, bright_gray, np.zeros_like(gray))
    assert out["mv_p12_p15"] > 0


def test_extract_features_returns_every_invariance_column() -> None:
    rng = np.random.default_rng(1)
    frames = [(float(i), rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)) for i in range(4)]
    features = extract_features(iter(frames))
    assert set(INVARIANCE_NAMES) <= set(FEATURE_NAMES) <= set(features)
    assert len(features["wm_p4c"]) == 4
    assert features["first"][0] == 1
    assert features["wm_p11_k4"][0] == features["warm"][0]  # the first check is colour only, like the production features
    for name in VARIANTS:
        assert (features[f"and_{name}"] <= features["wm25"]).all() or name == "p4c"


def test_score_form_matches_the_decide_form_of_the_baseline_and_the_debounce_control() -> None:
    rng = np.random.default_rng(3)
    n = 60
    area = np.full(n, 100_000)
    warm = rng.integers(0, 20_000, n)
    features = {
        "area": area,
        "warm": warm,
        "wm25": (warm * rng.random(n)).astype(np.int64),
        "first": np.zeros(n, dtype=int),
        "t": np.arange(n, dtype=float),
    }
    features["first"][0] = 1
    for name in ("area_relative", "area_relative_deb3"):
        rule = RULES_BY_NAME[name]
        for p in (0.05, 0.2, 0.5):
            assert (rule.score(features) > p).tolist() == rule.decide(features, p).tolist()


def test_invariance_rules_are_registered_in_replace_and_and_forms() -> None:
    for variant in VARIANTS:
        assert RULES_BY_NAME[variant].score is not None
        assert RULES_BY_NAME[f"{variant}_and"].score is not None
    assert not RULES_BY_NAME["p12_a15"].chain
    assert RULES_BY_NAME["p11_k4"].chain
