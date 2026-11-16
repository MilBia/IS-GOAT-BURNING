"""Tests of the #137 decision layer (production debounce replay, CUSUM / LLR statistics, conformal threshold)."""

import numpy as np
import pytest

from tools.fire_rule_eval.decision import GATES
from tools.fire_rule_eval.decision import PositiveResult
from tools.fire_rule_eval.decision import bench_decision
from tools.fire_rule_eval.decision import conformal_threshold
from tools.fire_rule_eval.decision import cusum_series
from tools.fire_rule_eval.decision import debounce_candidate
from tools.fire_rule_eval.decision import debounce_checks
from tools.fire_rule_eval.decision import llr_series
from tools.fire_rule_eval.decision import replay_production
from tools.fire_rule_eval.decision import verdict
from tools.fire_rule_eval.rules import debounced


@pytest.mark.parametrize(("seconds", "checks"), [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (10, 11)])
def test_debounce_checks_matches_production_timer(seconds: int, checks: int) -> None:
    alarm = np.ones(30, dtype=bool)
    _, entries, _ = replay_production(alarm, float(seconds), 0.0)
    assert entries == [checks - 1]
    assert debounce_checks(seconds) == checks


def test_replay_equals_run_length_debounce_without_extinguish_delay() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        alarm = rng.random(60) < 0.6
        for d in (0, 1, 2, 3):
            state, _, _ = replay_production(alarm, float(d), 0.0)
            expected = debounced(lambda f, _p: f["a"], d + 1)({"a": alarm}, 0.0)
            assert np.array_equal(state, expected)


def test_replay_extinguish_delay_keeps_the_state_through_short_gaps() -> None:
    alarm = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    state, entries, exits = replay_production(alarm, 0.0, 3.0)
    assert entries == [0, 12]
    assert exits == [9]  # 3 s after the first miss of the long gap (index 6)
    assert state[:8].all()


def test_cusum_accumulates_and_floors_at_zero() -> None:
    s = cusum_series(np.array([0.3, 0.3, 0.0, 0.0, 0.0, 0.3]), 0.1)
    assert s == pytest.approx([0.2, 0.4, 0.3, 0.2, 0.1, 0.3])
    assert cusum_series(np.zeros(3), 0.1).tolist() == [0.0, 0.0, 0.0]


def test_llr_series_tolerates_a_dropout_a_run_length_debounce_does_not() -> None:
    hit = np.array([1, 1, 0, 1, 1], dtype=bool)
    s = llr_series(hit, 0.05, 0.5)
    assert s[1] == pytest.approx(2 * np.log(10))
    assert s[2] < s[1]
    assert s[4] > s[1]  # the dropout costs less than a hit gives
    assert not debounced(lambda f, _p: f["a"], 3)({"a": hit}, 0.0).any()


def test_debounce_candidate_score_is_min_of_last_k() -> None:
    f = {
        "first": np.array([1, 0, 0, 0, 0]),
        "area": np.full(5, 100.0),
        "warm": np.full(5, 10.0),
        "wm25": np.array([0, 5.0, 3.0, 6.0, 6.0]),
    }
    cand = debounce_candidate("ar", 1)
    assert cand.score(f).tolist() == pytest.approx([0.0, 0.0, 0.3, 0.3, 0.6])
    assert GATES["ar"](f).tolist() == pytest.approx([0.0, 0.5, 0.3, 0.6, 0.6])


def test_conformal_threshold_rank_and_infinite_case() -> None:
    cal = np.arange(1.0, 100.0)  # n = 99
    assert conformal_threshold(cal, 0.1) == 90.0  # ceil(100 * 0.9) = 90
    assert conformal_threshold(cal, 0.005) == float("inf")


def _res(name: str, group: str, ttfa: float | None, recall: float) -> PositiveResult:
    return PositiveResult(name, group, ttfa is not None, ttfa, 20.0, recall, 1, 0, recall)


def test_verdict_pass_i_and_ii_and_fail() -> None:
    base = [_res("a", "commons_fire__x", 3.0, 0.5), _res("b", "commons_fire__y", 3.0, 0.5)]
    same_faster = [_res("a", "commons_fire__x", 2.0, 0.5), _res("b", "commons_fire__y", 2.0, 0.5)]
    assert verdict(same_faster, base)[0].startswith("PASS(i)")
    better_slower = [_res("a", "commons_fire__x", 6.0, 0.7), _res("b", "commons_fire__y", 6.0, 0.7)]
    assert verdict(better_slower, base)[0] == "PASS(ii)"
    much_slower = [_res("a", "commons_fire__x", 9.0, 0.7), _res("b", "commons_fire__y", 9.0, 0.7)]
    assert verdict(much_slower, base)[0] == "fail"
    worse = [_res("a", "commons_fire__x", 1.0, 0.3), _res("b", "commons_fire__y", 1.0, 0.3)]
    assert verdict(worse, base)[0] == "fail"


def test_bench_decision_runs() -> None:
    result = bench_decision(checks=2000, repeats=1)
    assert set(result) >= {"cusum", "sprt (LLR cusum)", "debounce (run counter)"}
