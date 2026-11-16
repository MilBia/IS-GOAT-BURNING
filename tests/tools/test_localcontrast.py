"""Tests for the #136 local-contrast small-target scores (P9 CFAR, P10 top-hat / LCM, P1 Haar)."""

import cv2
import numpy as np

from tools.fire_rule_eval.features import FEATURE_NAMES
from tools.fire_rule_eval.features import extract_features
from tools.fire_rule_eval.localcontrast import SCORE_NAMES
from tools.fire_rule_eval.localcontrast import haar_energy_explicit
from tools.fire_rule_eval.localcontrast import local_contrast_scores
from tools.fire_rule_eval.localcontrast import nth_largest
from tools.fire_rule_eval.localcontrast import wavelet_scores
from tools.fire_rule_eval.localcontrast import zero_scores
from tools.fire_rule_eval.rules import LC_VARIANTS
from tools.fire_rule_eval.rules import OR_FLAG
from tools.fire_rule_eval.rules import RULES_BY_NAME
from tools.fire_rule_eval.rules import lc_gate

HEIGHT, WIDTH = 270, 480
SELECTIVE = ("lc_p9_cfar_z", "lc_p10_tophat", "lc_p10_lcm")


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(100, 130, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[..., 2] += 80  # reddish: a positive heat map R - B
    return frame


def _mask(side: int, y: int = 100, x: int = 200) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[y : y + side, x : x + side] = 255
    return mask


def _scores(mask: np.ndarray, frame: np.ndarray | None = None) -> dict[str, float]:
    frame = _frame() if frame is None else frame
    return local_contrast_scores(frame, cv2.GaussianBlur(frame, (5, 5), 0), mask)


def test_nth_largest_needs_enough_positive_values() -> None:
    values = np.array([0.0, 3.0, 1.0, 2.0, -1.0], dtype=np.float32)
    assert nth_largest(values, 2) == 2.0
    assert nth_largest(values, 4) == 0.0


def test_empty_mask_gives_zero_everywhere() -> None:
    assert _scores(np.zeros((HEIGHT, WIDTH), dtype=np.uint8)) == zero_scores()


def test_a_small_target_stands_out_and_a_large_anchored_blob_is_suppressed() -> None:
    small, blob = _scores(_mask(20)), _scores(_mask(60))
    for name in SELECTIVE:
        assert small[name] > 3 * blob[name], name
    assert small["lc_ctl_peak"] == blob["lc_ctl_peak"] == 1.0  # a plain local peak cannot tell them apart


def test_scores_are_unchanged_by_a_uniform_brightness_shift() -> None:
    frame = _frame()
    shifted = cv2.add(frame, 20)
    mask = _mask(20)
    first, second = _scores(mask, frame), _scores(mask, shifted)
    for name in ("lc_p9_cfar_z", "lc_p9_cfar_diff", "lc_p10_tophat", "lc_p10_lcm", "lc_ctl_peak"):
        assert first[name] == second[name], name
    assert abs(first["lc_p1_global"] - second["lc_p1_global"]) < 1e-3


def test_sparse_haar_energy_equals_the_explicit_haar_transform() -> None:
    frame = np.random.default_rng(3).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    mask = _mask(40)
    energy = haar_energy_explicit(frame)
    region = cv2.resize(mask, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA) >= 128
    expected = float(np.sqrt(energy[region].mean()))
    assert abs(wavelet_scores(frame, mask)[0] - expected) < 1e-2 * expected


def test_wavelet_score_separates_texture_from_a_uniform_region() -> None:
    flat = np.full((HEIGHT, WIDTH, 3), 150, dtype=np.uint8)
    textured = flat.copy()
    textured[100:140, 200:240] = np.random.default_rng(1).integers(100, 200, (40, 40, 3), dtype=np.uint8)
    mask = _mask(40)
    assert wavelet_scores(flat, mask)[0] == 0.0
    assert wavelet_scores(textured, mask)[0] > 10.0


def test_tiny_regions_give_no_wavelet_score() -> None:
    assert wavelet_scores(_frame(), _mask(5)) == (0.0, 0.0)


def test_extract_features_has_every_local_contrast_column() -> None:
    rng = np.random.default_rng(0)
    frames = [(float(i), rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)) for i in range(3)]
    features = extract_features(iter(frames), (5, 5))
    assert set(SCORE_NAMES) <= set(FEATURE_NAMES) <= set(features)
    assert all(features[name][0] == 0 for name in SCORE_NAMES)  # first check: no motion information


def test_rules_are_registered_and_the_or_form_flags_the_partner() -> None:
    for key in LC_VARIANTS:
        assert f"{key}_and" in RULES_BY_NAME and f"{key}_or" in RULES_BY_NAME
    assert "ctl_peak_and" in RULES_BY_NAME
    f = {
        "area": np.array([1000.0, 1000.0, 1000.0]),
        "warm": np.array([500.0, 500.0, 500.0]),
        "wm25": np.array([480.0, 0.3, 480.0]),
        "first": np.array([0, 0, 1]),
        "lc_p9_cfar_z": np.array([2.0, 9.0, 9.0]),
    }
    assert lc_gate(f).tolist() == [True, False, False]  # moving < 0.05 % of the area is gated; check 2 is the first
    and_rule, or_rule = RULES_BY_NAME["p9_cfar_z_and"], RULES_BY_NAME["p9_cfar_z_or"]
    assert and_rule.decide(f, 1.0).tolist() == [True, False, False]
    assert or_rule.score(f)[0] == OR_FLAG  # 480 of 500 warm pixels move: the area_relative partner fires
    assert or_rule.decide(f, 1e6).tolist() == [True, False, False]
