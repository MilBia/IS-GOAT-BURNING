"""Tests for the #134 candidate features (blob table, tracker, leaky integrator, stateful rules)."""

import cv2
import numpy as np

from tools.fire_rule_eval.features import BLOB_COLUMNS
from tools.fire_rule_eval.features import BLOB_SLOTS
from tools.fire_rule_eval.features import blob_table
from tools.fire_rule_eval.rules import RULES_BY_NAME
from tools.fire_rule_eval.rules import debounced
from tools.fire_rule_eval.tracking import BlobTracker
from tools.fire_rule_eval.tracking import LeakyIntegrator
from tools.fire_rule_eval.tracking import TrackGate
from tools.fire_rule_eval.tracking import box_gate


def _blobs(*boxes: tuple[float, float, float, float]) -> np.ndarray:
    table = np.zeros((BLOB_SLOTS, BLOB_COLUMNS), dtype=np.float32)
    for i, (x, y, w, h) in enumerate(boxes):
        table[i] = (x, y, w, h, w * h)
    return table


def test_blob_table_reports_the_largest_components_in_frame_fractions() -> None:
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    cv2.rectangle(mask, (960, 540), (1151, 755), 255, -1)  # 192 x 216 px
    cv2.rectangle(mask, (100, 100), (139, 139), 255, -1)
    table = blob_table(mask)
    assert table.shape == (BLOB_SLOTS, BLOB_COLUMNS)
    assert table[0, 4] > table[1, 4] > 0 == table[2, 4]
    assert abs(table[0, 0] - 0.5) < 0.01
    assert abs(table[0, 4] - (192 * 216) / (1920 * 1080)) < 0.002


def test_tracker_counts_a_candidate_only_after_it_persisted() -> None:
    tracker = BlobTracker()
    gate = TrackGate(3, 3)
    scores = [tracker.update(_blobs((0.4, 0.5, 0.1, 0.2))) and tracker.score(gate) for _ in range(4)]
    assert scores[:2] == [0.0, 0.0]
    assert scores[2] > 0 and scores[3] > 0


def test_tracker_ignores_a_candidate_that_jumps_around() -> None:
    tracker = BlobTracker()
    gate = TrackGate(2, 2)
    for x in (0.05, 0.4, 0.75, 0.2):
        tracker.update(_blobs((x, 0.5, 0.05, 0.05)))
        assert tracker.score(gate) == 0.0


def test_anchor_gate_rejects_a_drifting_candidate() -> None:
    tracker = BlobTracker()
    gate = TrackGate(3, 3, anchored=True)
    for x in (0.10, 0.13, 0.16, 0.19, 0.22):
        tracker.update(_blobs((x, 0.5, 0.1, 0.1)))
    assert tracker.score(gate) == 0.0
    assert tracker.score(TrackGate(3, 3)) > 0.0


def test_leaky_integrator_never_exceeds_the_largest_sustained_input() -> None:
    integrator = LeakyIntegrator(0.85, (4, 8))
    evidence = np.zeros((4, 8))
    evidence[1, 2] = 0.3
    peak = max(integrator.update(evidence) for _ in range(100))
    assert 0.29 < peak <= 0.3
    burst = LeakyIntegrator(0.85, (4, 8))
    assert burst.update(np.ones((4, 8))) < 0.2


def test_box_gate_marks_the_overlapped_tiles() -> None:
    gate = box_gate([np.array([0.0, 0.0, 0.26, 0.26])], (4, 8))
    assert gate[0, 0] and gate[0, 2] and gate[1, 0]
    assert not gate[0, 3] and not gate[3, 7]


def test_debounce_needs_consecutive_checks() -> None:
    f = {"x": np.array([1, 1, 0, 1, 1, 1, 0], dtype=bool)}
    out = debounced(lambda feats, _p: feats["x"], 3)(f, 0.0)
    assert out.tolist() == [False, False, False, False, False, True, False]


def test_stateful_rule_runs_over_the_whole_clip_and_caches_its_series() -> None:
    n = 8
    f = {
        "t": np.arange(n, dtype=float),
        "first": np.array([1] + [0] * (n - 1)),
        "blobs": np.stack([_blobs((0.4, 0.5, 0.1, 0.1)) for _ in range(n)]),
        "tile_wm25": np.zeros((n, 32), dtype=np.float32),
        "fine_wm25": np.zeros((n, 576), dtype=np.float32),
    }
    f["blobs"][0] = 0
    alarm = RULES_BY_NAME["p15_anchor3"].decide(f, 0.001)
    assert alarm.tolist() == [False, False, False, True, True, True, True, True]
    assert "_series_p15_anchor3" in f
