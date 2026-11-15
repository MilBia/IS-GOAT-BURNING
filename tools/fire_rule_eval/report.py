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


FLICKER_SETTLE_S = 3.0  # the incremental filters need a few seconds to settle


def _merge_flicker(feats: Features, flicker: Features) -> Features | None:
    """Adds flicker columns to the colour + motion features.

    Check ``k`` of the colour + motion features (frame ``k * fps``) matches flicker row ``k - 1`` (flicker
    rows start at 1 s). Matching by index rather than by time keeps NTSC clips (29.97 fps) aligned.
    """
    if not len(feats["t"]) or not len(flicker["t"]):
        return None  # an undecodable clip stores an empty file
    spacing = np.diff(feats["t"])
    if abs(float(feats["t"][0])) > 0.05 or (len(spacing) and np.abs(spacing - 1.0).max() > 0.05):
        raise ValueError("flicker merging needs colour + motion features extracted with --interval 1 --start 0")
    count = min(len(feats["t"]) - 1, len(flicker["t"]))
    if count < 1:
        return None
    checks = np.arange(1, count + 1)
    wanted = checks >= FLICKER_SETTLE_S
    if not wanted.any():
        return None
    checks = checks[wanted]
    rows = {key: value[checks] for key, value in feats.items()}
    for key in ("flick_win", "flick_inc"):
        rows[key] = flicker[key][checks - 1]
    rows["fwarm"] = flicker["warm"][checks - 1]
    # The flicker counts are at 480x270 while ``area`` is at the extraction resolution: keep both.
    rows["flick_area"] = flicker["area"][checks - 1]
    return rows


def load_clips(manifest_path: Path, features_dir: Path, flicker_dir: Path | None = None) -> list[Clip]:
    """Loads clips listed in a manifest that have stored features.

    Positive clips are restricted to the ``fire_from_s`` / ``fire_to_s`` window when given and always
    drop their first check (no motion information), so recall reflects the steady state.

    Args:
        manifest_path: JSON manifest with a ``clips`` list.
        features_dir: Directory holding ``<name>.npz`` files from ``extract``.
        flicker_dir: Optional directory with ``<name>.npz`` files from ``flicker``; when given, only clips
            present there are loaded, the first seconds are dropped and flicker columns are added.

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
        if flicker_dir is not None:
            flicker_npz = flicker_dir / f"{entry['name']}.npz"
            if not flicker_npz.exists():
                continue
            with np.load(flicker_npz) as data:
                merged = _merge_flicker(feats, {key: data[key] for key in data.files})
            if merged is None:
                continue
            feats = merged
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


AGREEMENT_THRESHOLDS = (0.0005, 0.001, 0.002, 0.005, 0.01)


def render_flicker_agreement(clips: list[Clip]) -> str:
    """Compares the incremental flicker feature with the windowed reference over all checks."""
    win = np.concatenate([c.features["flick_win"] / c.features["flick_area"] for c in clips])
    inc = np.concatenate([c.features["flick_inc"] / c.features["flick_area"] for c in clips])
    both = (win > 0) & (inc > 0)
    pearson = float(np.corrcoef(win, inc)[0, 1]) if win.std() > 0 and inc.std() > 0 else float("nan")
    ranks = np.corrcoef(np.argsort(np.argsort(win)), np.argsort(np.argsort(inc)))[0, 1]
    ratio = float(np.median(inc[both] / win[both])) if both.any() else float("nan")
    lines = [
        "### Incremental vs windowed flicker feature\n",
        f"Checks: {len(win)}; Pearson r = {pearson:.3f}; rank correlation = {float(ranks):.3f}; "
        f"median ratio incremental/windowed over checks where both are non-zero = {ratio:.2f}\n",
        "| Threshold (share of frame) | Both | Windowed only | Incremental only |",
        "|---|---|---|---|",
    ]
    for threshold in AGREEMENT_THRESHOLDS:
        a, b = win > threshold, inc > threshold
        lines.append(f"| {threshold:g} | {int((a & b).sum())} | {int((a & ~b).sum())} | {int((b & ~a).sum())} |")
    return "\n".join(lines) + "\n"


def _usable_rules(clips: list[Clip], rules: tuple[Rule, ...] | list[Rule]) -> list[Rule]:
    """Drops ``flicker_win`` when the windowed reference was not computed (stored as -1)."""
    no_reference = any((c.features.get("flick_win", np.zeros(1)) < 0).any() for c in clips)
    return [rule for rule in rules if not (no_reference and rule.name == "flicker_win")]


def render_markdown(clips: list[Clip], rules: tuple[Rule, ...] | list[Rule] = RULES) -> str:
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
    for rule in _usable_rules(clips, rules):
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

    if all("flick_win" in c.features and (c.features["flick_win"] >= 0).all() for c in clips):
        lines.append("")
        lines.append(render_flicker_agreement(clips))

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
