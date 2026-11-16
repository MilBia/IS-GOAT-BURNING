"""Tests for the #140 low-rank + sparse masks."""

import numpy as np

from tools.fire_rule_eval.lowrank import ALL_VARIANTS
from tools.fire_rule_eval.lowrank import CANDIDATE_VARIANTS
from tools.fire_rule_eval.lowrank import LowRankModels
from tools.fire_rule_eval.lowrank import eig_background
from tools.fire_rule_eval.lowrank import eig_background_naive
from tools.fire_rule_eval.lowrank import godec_low_rank
from tools.fire_rule_eval.lowrank import rpca_low_rank

HEIGHT, WIDTH = 60, 80
SQUARE = (slice(20, 30), slice(30, 45))


def _frame(rng: np.random.Generator, square: bool = False, gain: float = 1.0) -> np.ndarray:
    gray = rng.normal(100, 2.0, (HEIGHT, WIDTH)) * gain
    if square:
        gray[SQUARE] = 240
    return gray.clip(0, 255).astype(np.uint8)


def _run(names: tuple[str, ...], frames: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
    models = LowRankModels(names)
    out: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for frame in frames:
        for name, fg in models.masks(frame).items():
            out[name].append(fg)
    return out


def test_eig_fast_path_equals_naive_svd() -> None:
    rng = np.random.default_rng(0)
    base = rng.uniform(50, 200, 400)
    for rank in (0, 1, 2):
        window = np.stack([base * g + rng.normal(0, 3, 400) for g in (0.8, 0.9, 1.0, 1.1, 1.2, 0.95, 1.05)])
        current = base * 1.15 + rng.normal(0, 3, 400)
        shift = window[0]
        w, c = (window - shift).astype(np.float32), (current - shift).astype(np.float32)
        fast = eig_background(w, (w @ w.T).astype(np.float64), w @ c, rank) + shift
        naive = eig_background_naive(window, current, rank)
        assert np.allclose(fast, naive, atol=0.05), rank


def test_eig_gram_is_maintained_incrementally() -> None:
    rng = np.random.default_rng(1)
    names = ("lr_eig2_n9",)
    models = LowRankModels(names)
    for i in range(20):  # the ring buffer wraps after 9 frames
        models.masks(_frame(rng, square=i % 7 == 0))
    window = models._models["lr_eig2_n9"].window
    stack = window.view()
    assert np.allclose(window.gram, stack @ stack.T, rtol=1e-4, atol=1.0)


def test_empty_until_min_samples_then_new_square_is_foreground() -> None:
    rng = np.random.default_rng(2)
    frames = [_frame(rng) for _ in range(8)] + [_frame(rng, square=True)]
    for name, masks in _run(tuple(CANDIDATE_VARIANTS), frames).items():
        assert all(m.max() == 0 for m in masks[:3]), name
        if name == "lr_godec2_n9":  # rank 2 on 10 columns: the second component absorbs the new object at once (see below)
            continue
        assert (masks[-1][SQUARE] > 0).mean() > 0.9, name
        assert (masks[-1] > 0).mean() < 0.2, name


def test_godec_rank_two_absorbs_a_new_object_that_rank_one_reports() -> None:
    rng = np.random.default_rng(2)
    frames = [_frame(rng) for _ in range(8)] + [_frame(rng, square=True)]
    masks = _run(("lr_godec1_n9", "lr_godec2_n9"), frames)
    assert (masks["lr_godec1_n9"][-1][SQUARE] > 0).mean() > 0.9
    assert (masks["lr_godec2_n9"][-1][SQUARE] > 0).mean() < 0.1


def test_static_square_is_absorbed_after_a_window() -> None:
    rng = np.random.default_rng(3)
    frames = [_frame(rng) for _ in range(6)] + [_frame(rng, square=True) for _ in range(30)]
    for name, masks in _run(("lr_eig1_n5", "lr_godec1_n5", "lr_rpca_n5", "lr_mean_n9"), frames).items():
        assert (masks[-1][SQUARE] > 0).mean() < 0.1, name


def test_rank_one_absorbs_a_global_gain_change() -> None:
    rng = np.random.default_rng(4)
    gains = [1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12]
    frames = [_frame(rng, gain=g) for g in gains]
    shares = {n: (m[-1] > 0).mean() for n, m in _run(("lr_eig1_n9", "lr_godec1_n9", "lr_mean_n9"), frames).items()}
    assert shares["lr_eig1_n9"] <= shares["lr_mean_n9"]
    assert shares["lr_godec1_n9"] < 0.05


def test_godec_and_rpca_recover_rank_one_plus_sparse() -> None:
    rng = np.random.default_rng(5)
    background = rng.uniform(60, 180, 500)
    data = np.outer(np.linspace(0.9, 1.1, 8), background)
    sparse = np.zeros_like(data)
    sparse[7, 100:120] = 80.0
    observed = data + sparse
    low = godec_low_rank(observed.astype(np.float32), 1, 5, rng.standard_normal((500, 1)).astype(np.float32))
    assert np.abs(low[-1] - data[-1]).mean() < 3.0
    low_rpca, iterations = rpca_low_rank(observed)
    assert np.abs(low_rpca[-1] - data[-1]).mean() < 3.0
    assert 1 <= iterations <= 50


def test_rpca_iteration_cap_and_log() -> None:
    rng = np.random.default_rng(6)
    models = LowRankModels(("lr_rpca_n9_c5",))
    for _ in range(12):
        models.masks(_frame(rng))
    assert 1 <= models._models["lr_rpca_n9_c5"].iterations_used <= 5


def test_update_columns() -> None:
    rng = np.random.default_rng(7)
    models = LowRankModels(("lr_eig1_n5", "lr_rpca_n5"))
    color = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    for _ in range(6):
        columns = models.update(_frame(rng), color, np.zeros_like(color))
    for name in ("lr_eig1_n5", "lr_rpca_n5"):
        assert {f"bgwm_{name}", f"bgand_{name}", f"bgmv_{name}", f"tile_bg_{name}", f"lriter_{name}"} <= set(columns)
    assert columns["tile_bg_lr_eig1_n5"].shape == (32,)


def test_variant_registry() -> None:
    assert len(CANDIDATE_VARIANTS) == 8
    assert set(CANDIDATE_VARIANTS) <= set(ALL_VARIANTS)
