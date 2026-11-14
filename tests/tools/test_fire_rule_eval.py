"""Tests for the fire-rule evaluation harness (issue #104)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from is_goat_burning.fire_detection.detector_core import DEFAULT_LOWER_HSV_FIRE
from is_goat_burning.fire_detection.detector_core import DEFAULT_UPPER_HSV_FIRE
from is_goat_burning.fire_detection.detectors import CPUFireDetector
from tools.fire_rule_eval.__main__ import main
from tools.fire_rule_eval.features import ExtractOptions
from tools.fire_rule_eval.features import extract_features
from tools.fire_rule_eval.features import iter_check_frames
from tools.fire_rule_eval.report import Clip
from tools.fire_rule_eval.report import best_safe_point
from tools.fire_rule_eval.report import evaluate
from tools.fire_rule_eval.report import load_clips
from tools.fire_rule_eval.report import render_markdown
from tools.fire_rule_eval.report import sweep
from tools.fire_rule_eval.rules import RULES
from tools.fire_rule_eval.rules import RULES_BY_NAME

HEIGHT, WIDTH = 240, 320
FIRE_BGR = (0, 200, 255)


def _frame(x: int, width: int = 80, colour: tuple[int, int, int] = FIRE_BGR) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), 20, dtype=np.uint8)
    frame[60:180, x : x + width] = colour
    return frame


def _detector(margin: int) -> CPUFireDetector:
    return CPUFireDetector(margin=margin, lower=DEFAULT_LOWER_HSV_FIRE, upper=DEFAULT_UPPER_HSV_FIRE, motion_threshold=25)


def test_features_match_production_detector() -> None:
    """Extracted features must agree with ``CPUFireDetector`` pixel-for-pixel.

    The first check is excluded: production no longer alarms on it (``skip_first`` rule), while the
    ``deployed`` rule keeps the earlier colour-only behaviour so old results stay reproducible.
    """
    frames = [_frame(20), _frame(60), _frame(100)]
    feats = extract_features(iter(enumerate(frames)))

    assert bool(_detector(0)._detect_logic(frames[0])[0]) is False  # no baseline, no alarm

    for index, frame in enumerate(frames):
        if index == 0:
            continue
        count = feats["wm25"][index]
        assert count > 0
        for margin, expected in ((count - 1, True), (count, False)):
            detector = _detector(margin)
            for earlier in frames[:index]:
                detector._detect_logic(earlier)
            fire, _ = detector._detect_logic(frame)
            assert bool(fire) is expected, (index, margin)


def test_first_check_is_colour_only_and_flagged() -> None:
    feats = extract_features(iter(enumerate([_frame(20), _frame(20)])))
    assert feats["first"].tolist() == [1, 0]
    assert feats["wm25"][0] == feats["warm"][0] > 0
    assert feats["wm25"][1] == 0  # static scene: no motion on the second check


def test_deployed_rule_alarms_on_first_check_but_skip_first_does_not() -> None:
    feats = extract_features(iter(enumerate([_frame(20), _frame(20)])))
    fraction = 0.1  # the fire block covers ~25 % of the frame
    assert RULES_BY_NAME["deployed"].decide(feats, fraction).tolist() == [True, False]
    assert RULES_BY_NAME["skip_first"].decide(feats, fraction).tolist() == [False, False]


def test_moving_fire_is_detected_by_every_non_deployed_rule() -> None:
    frames = [_frame(20 + 40 * i, width=60) for i in range(4)]
    feats = extract_features(iter(enumerate(frames)))
    for name in ("skip_first", "motion15", "motion40", "avg_motion", "area_relative", "tiled", "component", "component_closed"):
        rule = RULES_BY_NAME[name]
        alarms = rule.decide(feats, rule.grid[0])
        assert not alarms[0], name
        assert alarms[1:].any(), name


def test_static_scene_never_alarms_after_first_check() -> None:
    feats = extract_features(iter(enumerate([_frame(20)] * 5)))
    for rule in RULES:
        if rule.name == "deployed":
            continue
        assert not rule.decide(feats, rule.grid[0]).any(), rule.name


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 5.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_iter_check_frames_samples_at_interval(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_video(video, [_frame(20 + i) for i in range(20)], fps=5.0)  # 4 s
    times = [t for t, _ in iter_check_frames(video, ExtractOptions(check_interval_s=1.0))]
    assert times == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_best_safe_point_prefers_zero_false_alarms() -> None:
    static = extract_features(iter(enumerate([_frame(20)] * 6)))
    moving = extract_features(iter(enumerate([_frame(10 + 30 * i, width=60) for i in range(6)])))
    clips = [Clip("neg", "neg", "g", static), Clip("pos", "pos", "g", {k: v[1:] for k, v in moving.items()})]
    rule = RULES_BY_NAME["skip_first"]
    best = best_safe_point(sweep(rule, clips))
    assert best is not None
    assert best.neg_alarm_checks == 0
    assert best.pos_recall > 0
    assert evaluate(rule, rule.grid[0], clips).pos_clips == 1


def test_cli_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    neg_dir, pos_dir = tmp_path / "neg", tmp_path / "pos"
    neg_dir.mkdir()
    pos_dir.mkdir()
    _write_video(neg_dir / "static.mp4", [_frame(20)] * 20)
    _write_video(pos_dir / "moving.mp4", [_frame(10 + 6 * i, width=60) for i in range(20)])
    manifest, feats = tmp_path / "manifest.json", tmp_path / "feats"

    assert main(["scan", "--dir", str(neg_dir), "--label", "neg", "--group", "static", "--manifest", str(manifest)]) == 0
    assert main(["scan", "--dir", str(pos_dir), "--label", "pos", "--group", "moving", "--manifest", str(manifest)]) == 0
    assert len(json.loads(manifest.read_text())["clips"]) == 2
    assert main(["extract", "--manifest", str(manifest), "--out", str(feats)]) == 0
    assert main(["report", "--manifest", str(manifest), "--features", str(feats)]) == 0

    output = capsys.readouterr().out
    assert "Best operating point per rule" in output
    assert "Production default" in output
    clips = load_clips(manifest, feats)
    assert {c.label for c in clips} == {"neg", "pos"}
    assert "static__static" in render_markdown(clips)


def test_synth_composite_adds_flame_area(tmp_path: Path) -> None:
    from tools.fire_rule_eval.synth import SynthOptions
    from tools.fire_rule_eval.synth import composite_video

    background, fire, out = tmp_path / "bg.mp4", tmp_path / "fire.mp4", tmp_path / "out.mp4"
    _write_video(background, [np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)] * 15, fps=5.0)
    flame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    flame[40:200, 100:220] = FIRE_BGR
    _write_video(fire, [flame] * 15, fps=5.0)

    written = composite_video(background, fire, out, SynthOptions(fraction=0.05, seconds=3.0, anchor_x=0.5, anchor_y=0.9))
    assert written == 15
    feats = extract_features(iter_check_frames(out, ExtractOptions(check_interval_s=1.0)))
    share = feats["warm"] / feats["area"]
    assert 0.02 < float(share.mean()) < 0.1  # close to the requested 5 % (lossy codec, soft edges)
