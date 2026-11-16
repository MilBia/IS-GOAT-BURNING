"""Common comparison protocol of the #132 sub-issues ("best safe point", warm-up, time to first alarm).

Every rule is swept over its grid. The operating point is the **lowest** parameter from which on all grid
values give zero steady-state alarms on the target-scene negatives (``neg``). Everything else (real fire,
synthetic fire per fraction, hard negatives, time to the first alarm, the goat clips chained into one stream)
is reported at that same point.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from tools.fire_rule_eval.report import HARD_NEGATIVE
from tools.fire_rule_eval.report import Clip
from tools.fire_rule_eval.rules import Features
from tools.fire_rule_eval.rules import Rule

SYNTH_GROUP = "synth_goat"
FIXED_CAMERA_EXCLUDED = ("tunnel", "Ropeway")  # moving-camera hard negatives (excluded from the #104 set)


@dataclass(frozen=True)
class Outcome:
    """Result of one rule at its best safe point."""

    rule: str
    param: float | None
    margin: float | None
    goat_alarms: int
    goat_checks: int
    chain_alarms: int
    real_detected: int
    real_clips: int
    real_recall: float
    synth: dict[str, tuple[float, int]]  # fraction -> (check recall, detected)
    ttfa_real: float | None
    ttfa_synth: float | None
    hard_alarmed: int
    hard_clips: int
    hard_fixed_alarmed: int
    hard_fixed_clips: int


def chain(clips: list[Clip]) -> Clip | None:
    """Concatenates negative clips (in the given order) into one stream; ``first`` rows mark the joins."""
    if not clips:
        return None
    keys = [k for k in clips[0].features if not k.startswith("_") and all(k in c.features for c in clips)]
    feats: Features = {k: np.concatenate([c.features[k] for c in clips]) for k in keys}
    valid = np.concatenate([c.mask for c in clips])
    return Clip("chain", "neg", "chain", feats, valid)


def _first_alarm_delay(clip: Clip, alarm: np.ndarray) -> float | None:
    hits = np.flatnonzero(alarm & clip.mask)
    if not len(hits):
        return None
    start = max(clip.onset_s, float(clip.features["t"][np.flatnonzero(clip.mask)[0]]))
    return float(clip.features["t"][hits[0]] - start)


def lowest_safe_param(rule: Rule, negatives: list[Clip]) -> tuple[float | None, float | None, list[int]]:
    """Returns ``(parameter, margin, goat alarm counts per grid value)``."""
    counts = []
    for p in rule.grid:
        counts.append(sum(int((rule.decide(c.features, p) & c.mask).sum()) for c in negatives))
    unsafe = [i for i, n in enumerate(counts) if n > 0]
    if not unsafe:
        return rule.grid[0], None, counts
    last = max(unsafe)
    if last + 1 >= len(rule.grid):
        return None, None, counts
    return rule.grid[last + 1], rule.grid[last + 1] / rule.grid[last], counts


def evaluate_rule(rule: Rule, clips: list[Clip]) -> Outcome:
    """Evaluates ``rule`` at its best safe point over ``clips``."""
    negatives = [c for c in clips if c.label == "neg"]
    param, margin, _ = lowest_safe_param(rule, negatives)
    if param is None:
        return Outcome(rule.name, None, None, -1, 0, -1, 0, 0, 0.0, {}, None, None, 0, 0, 0, 0)
    goat_alarms = sum(int((rule.decide(c.features, param) & c.mask).sum()) for c in negatives)
    goat_checks = sum(int(c.mask.sum()) for c in negatives)
    stream = chain(sorted(negatives, key=lambda c: c.name))
    chain_alarms = int((rule.decide(stream.features, param) & stream.mask).sum()) if stream else 0

    real_recalls, real_detected, real_delays = [], 0, []
    synth: dict[str, tuple[float, int]] = {}
    synth_delays = []
    real_clips = 0
    for clip in (c for c in clips if c.is_positive):
        alarm = rule.decide(clip.features, param)
        recall = float(alarm[clip.mask].mean())
        delay = _first_alarm_delay(clip, alarm)
        if clip.group == SYNTH_GROUP:
            fraction = re.search(r"(\d\.\d+)$", clip.name).group(1)
            synth[fraction] = (recall, int(delay is not None))
            if delay is not None:
                synth_delays.append(delay)
        elif clip.group.startswith("commons_fire"):
            real_clips += 1
            real_recalls.append(recall)
            real_detected += int(delay is not None)
            if delay is not None:
                real_delays.append(delay)
    hard = [c for c in clips if c.label == HARD_NEGATIVE]
    hard_hits = [bool((rule.decide(c.features, param) & c.mask).any()) for c in hard]
    fixed = [h for c, h in zip(hard, hard_hits, strict=True) if not any(k in c.name for k in FIXED_CAMERA_EXCLUDED)]
    return Outcome(
        rule.name,
        param,
        margin,
        goat_alarms,
        goat_checks,
        chain_alarms,
        real_detected,
        real_clips,
        float(np.mean(real_recalls)) if real_recalls else float("nan"),
        synth,
        float(np.median(real_delays)) if real_delays else None,
        float(np.median(synth_delays)) if synth_delays else None,
        sum(hard_hits),
        len(hard),
        sum(fixed),
        len(fixed),
    )


def render(outcomes: list[Outcome], warmup: int, configs: int) -> str:
    """Markdown table of the outcomes."""
    fractions = sorted({f for o in outcomes for f in o.synth})
    head = (
        "| Rule | Param | Margin | Goat alarms | Chained goat | Real clips | Real recall | "
        + " | ".join(f"Synth {float(f) * 100:g}% (recall, hit)" for f in fractions)
        + " | TTFA real / synth (s) | Hard (all) | Hard (fixed cam) |"
    )
    lines = [f"Warm-up: first {warmup} checks of every clip do not count. Rules/configurations swept: {configs}.\n", head]
    lines.append("|" + "---|" * (head.count("|") - 1))
    for o in outcomes:
        if o.param is None:
            lines.append(f"| {o.rule} | no safe point | | | | | | " + "| " * len(fractions) + "| | | |")
            continue
        margin = "n/a" if o.margin is None else f"x{o.margin:.2f}"
        synth = " | ".join(f"{o.synth[f][0]:.2f}, {o.synth[f][1]}" if f in o.synth else "-" for f in fractions)
        ttfa_real = "-" if o.ttfa_real is None else f"{o.ttfa_real:.0f}"
        ttfa_synth = "-" if o.ttfa_synth is None else f"{o.ttfa_synth:.0f}"
        lines.append(
            f"| {o.rule} | {o.param:.4g} | {margin} | {o.goat_alarms}/{o.goat_checks} | {o.chain_alarms} | "
            f"{o.real_detected}/{o.real_clips} | {o.real_recall:.2f} | {synth} | {ttfa_real} / {ttfa_synth} | "
            f"{o.hard_alarmed}/{o.hard_clips} | {o.hard_fixed_alarmed}/{o.hard_fixed_clips} |"
        )
    return "\n".join(lines)
