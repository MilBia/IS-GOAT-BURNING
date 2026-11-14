"""Evaluation of rules against labelled clips (recall and false-positive tables)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from tools.fire_rule_eval.rules import RULES
from tools.fire_rule_eval.rules import Features
from tools.fire_rule_eval.rules import Rule

POSITIVE = "pos"
HARD_NEGATIVE = "neg_hard"
NEGATIVE_LABELS = ("neg", HARD_NEGATIVE)


@dataclass(frozen=True)
class Clip:
    """A labelled clip with its stored features."""

    name: str
    label: str
    group: str
    features: Features

    @property
    def is_positive(self) -> bool:
        """Whether the clip contains fire."""
        return self.label == POSITIVE


def load_clips(manifest_path: Path, features_dir: Path) -> list[Clip]:
    """Loads clips listed in a manifest that have stored features.

    Positive clips are restricted to the ``fire_from_s`` / ``fire_to_s`` window when given and always
    drop their first check (no motion information), so recall reflects the steady state.

    Args:
        manifest_path: JSON manifest with a ``clips`` list.
        features_dir: Directory holding ``<name>.npz`` files from ``extract``.

    Returns:
        Loaded clips (clips without features are skipped).
    """
    manifest = json.loads(manifest_path.read_text())
    clips: list[Clip] = []
    for entry in manifest["clips"]:
        npz = features_dir / f"{entry['name']}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as data:
            feats = {key: data[key] for key in data.files}
        keep = np.ones(len(feats["t"]), dtype=bool)
        if entry["label"] == POSITIVE:
            keep &= feats["first"] == 0
            keep &= feats["t"] >= entry.get("fire_from_s", 0.0)
            keep &= feats["t"] <= entry.get("fire_to_s", float("inf"))
            feats = {key: value[keep] for key, value in feats.items()}
        if len(feats["t"]):
            clips.append(Clip(entry["name"], entry["label"], entry.get("group", entry["label"]), feats))
    return clips


@dataclass(frozen=True)
class Point:
    """Result of one rule at one parameter value."""

    param: float
    neg_alarm_checks: int
    neg_first_alarms: int
    hard_clips_alarmed: int
    hard_clips: int
    neg_checks: int
    neg_clips_alarmed: int
    neg_clips: int
    pos_recall: float
    pos_clips_detected: int
    pos_clips: int


def evaluate(rule: Rule, param: float, clips: list[Clip]) -> Point:
    """Evaluates one rule at one parameter value over all clips."""
    neg_alarms = neg_first = neg_checks = neg_clips_alarmed = neg_clips = 0
    hard_alarmed = hard_clips = 0
    recalls: list[float] = []
    detected = pos_clips = 0
    for clip in clips:
        alarm = rule.decide(clip.features, param)
        if clip.is_positive:
            pos_clips += 1
            recalls.append(float(alarm.mean()))
            detected += int(alarm.any())
        elif clip.label == HARD_NEGATIVE:
            hard_clips += 1
            hard_alarmed += int((alarm & (clip.features["first"] == 0)).any())
        else:
            neg_clips += 1
            first = clip.features["first"] == 1
            steady = alarm & ~first
            neg_alarms += int(steady.sum())
            neg_first += int((alarm & first).sum())
            neg_checks += int((~first).sum())
            neg_clips_alarmed += int(steady.any())
    recall = float(np.mean(recalls)) if recalls else float("nan")
    return Point(
        param,
        neg_alarms,
        neg_first,
        hard_alarmed,
        hard_clips,
        neg_checks,
        neg_clips_alarmed,
        neg_clips,
        recall,
        detected,
        pos_clips,
    )


def sweep(rule: Rule, clips: list[Clip]) -> list[Point]:
    """Evaluates a rule over its whole parameter grid."""
    return [evaluate(rule, p, clips) for p in rule.grid]


def best_safe_point(points: list[Point]) -> Point | None:
    """Returns the point with the highest recall among those with zero negative alarms.

    Ties are broken towards the most conservative parameter that attains the same recall, which leaves
    the largest margin to the false-positive region.
    """
    safe = [p for p in points if p.neg_alarm_checks == 0]
    if not safe:
        return None
    best = max(p.pos_recall for p in safe)
    candidates = [p for p in safe if p.pos_recall >= best - 1e-12]
    return candidates[len(candidates) // 2]


def margin_to_first_false_alarm(point: Point, points: list[Point]) -> float | None:
    """Ratio between the parameter that first alarms on negatives and the chosen parameter.

    For rules where a larger parameter is more conservative, values above 1 mean headroom is on the
    sensitive side.
    """
    alarming = [p.param for p in points if p.neg_alarm_checks > 0]
    if not alarming:
        return None
    return point.param / max(alarming)


HEADROOM = 2.0


def headroom_recall(points: list[Point], factor: float) -> str:
    """Recall at the first grid value at least ``factor`` times the largest value that alarms on negatives."""
    alarming = [p.param for p in points if p.neg_alarm_checks > 0]
    if not alarming:
        return "n/a"
    target = factor * max(alarming)
    robust = [p for p in points if p.param >= target and p.neg_alarm_checks == 0]
    if not robust:
        return "out of grid"
    return f"{robust[0].pos_recall:.2f} (p={robust[0].param:.4g})"


def render_markdown(clips: list[Clip]) -> str:
    """Builds the full Markdown report."""
    lines: list[str] = []
    pos = [c for c in clips if c.is_positive]
    neg = [c for c in clips if not c.is_positive]
    lines.append(f"Clips: {len(pos)} positive, {len(neg)} negative; checks: {sum(len(c.features['t']) for c in clips)}\n")

    lines.append("### Best operating point per rule (zero steady-state alarms on target-scene negatives)\n")
    lines.append(
        "Negatives (`neg`) = target scene. Steady-state alarms exclude the first check after a (re)start, reported separately. "
        "`neg_hard` clips are warm/flickering non-fire scenes and are only counted, not used to pick the point.\n"
    )
    lines.append(
        "| Rule | Param | Recall | Clips detected | Neg alarms | First-check alarms | Margin | Recall @2x | Hard neg |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    details: dict[str, tuple[Rule, Point]] = {}
    for rule in RULES:
        points = sweep(rule, clips)
        best = best_safe_point(points)
        if best is None:
            lines.append(f"| {rule.name} | - | no safe point | - | - | - | - | - | - |")
            continue
        details[rule.name] = (rule, best)
        margin = margin_to_first_false_alarm(best, points)
        margin_text = "no false alarm in grid" if margin is None else f"x{margin:.2f}"
        lines.append(
            f"| {rule.name} | {best.param:.4g} | {best.pos_recall:.2f} | {best.pos_clips_detected}/{best.pos_clips} | "
            f"{best.neg_alarm_checks}/{best.neg_checks} | {best.neg_first_alarms}/{best.neg_clips} | {margin_text} | "
            f"{headroom_recall(points, HEADROOM)} | {best.hard_clips_alarmed}/{best.hard_clips} |"
        )

    production = next(r for r in RULES if r.name == "deployed")
    prod = evaluate(production, production.production_value or 0.1, clips)
    lines.append(
        f"\n### Production default (`deployed`, p={production.production_value})\n\n"
        f"Positive recall {prod.pos_recall:.2f} ({prod.pos_clips_detected}/{prod.pos_clips} clips detected); "
        f"steady-state negative alarms {prod.neg_alarm_checks}/{prod.neg_checks} checks "
        f"in {prod.neg_clips_alarmed}/{prod.neg_clips} clips; "
        f"first-check (colour only) alarms {prod.neg_first_alarms}/{prod.neg_clips} clips; "
        f"hard negatives alarmed {prod.hard_clips_alarmed}/{prod.hard_clips} clips.\n"
    )

    lines.append("### Per-clip detection (checks alarmed / checks) at each rule's best safe point\n")
    names = list(details)
    lines.append("| Clip | Label | Group | " + " | ".join(names) + " |")
    lines.append("|---|---|---|" + "---|" * len(names))
    for clip in clips:
        cells = []
        for name in names:
            rule, point = details[name]
            alarm = rule.decide(clip.features, point.param)
            cells.append(f"{int(alarm.sum())}/{len(alarm)}")
        lines.append(f"| {clip.name} | {clip.label} | {clip.group} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
