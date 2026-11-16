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
TIMELAPSE_TOKENS = ("Sunrise", "Sunset", "sunset")  # accelerated day/night time-lapses (the #135 "stress test" subset)


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
    hard_timelapse_alarmed: int = 0
    hard_timelapse_clips: int = 0
    chain_meaningful: bool = True


def is_timelapse(name: str) -> bool:
    """Whether a clip name belongs to the accelerated time-lapse hard negatives."""
    return any(token in name for token in TIMELAPSE_TOKENS)


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
    timelapse = [h for c, h in zip(hard, hard_hits, strict=True) if is_timelapse(c.name)]
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
        sum(timelapse),
        len(timelapse),
        rule.chain,
    )


def render(outcomes: list[Outcome], warmup: int, configs: int) -> str:
    """Markdown table of the outcomes."""
    fractions = sorted({f for o in outcomes for f in o.synth})
    head = (
        "| Rule | Param | Margin | Goat alarms | Chained goat | Real clips | Real recall | "
        + " | ".join(f"Synth {float(f) * 100:g}% (recall, hit)" for f in fractions)
        + " | TTFA real / synth (s) | Hard (all) | Hard (fixed cam) | Hard (time-lapse) |"
    )
    lines = [f"Warm-up: first {warmup} checks of every clip do not count. Rules/configurations swept: {configs}.\n", head]
    lines.append("|" + "---|" * (head.count("|") - 1))
    for o in outcomes:
        if o.param is None:
            lines.append(f"| {o.rule} | no safe point | | | | | | " + "| " * len(fractions) + "| | | | |")
            continue
        margin = "n/a" if o.margin is None else f"x{o.margin:.2f}"
        synth = " | ".join(f"{o.synth[f][0]:.2f}, {o.synth[f][1]}" if f in o.synth else "-" for f in fractions)
        ttfa_real = "-" if o.ttfa_real is None else f"{o.ttfa_real:.0f}"
        ttfa_synth = "-" if o.ttfa_synth is None else f"{o.ttfa_synth:.0f}"
        chained = o.chain_alarms if o.chain_meaningful else "n/a"
        lines.append(
            f"| {o.rule} | {o.param:.4g} | {margin} | {o.goat_alarms}/{o.goat_checks} | {chained} | "
            f"{o.real_detected}/{o.real_clips} | {o.real_recall:.2f} | {synth} | {ttfa_real} / {ttfa_synth} | "
            f"{o.hard_alarmed}/{o.hard_clips} | {o.hard_fixed_alarmed}/{o.hard_fixed_clips} | "
            f"{o.hard_timelapse_alarmed}/{o.hard_timelapse_clips} |"
        )
    return "\n".join(lines)


def _scores(rule: Rule, clip: Clip) -> np.ndarray:
    """Per-check scores of the checks that count (empty when the clip has none)."""
    return rule.score(clip.features)[clip.mask]


def continuous_row(rule: Rule, clips: list[Clip], grid_param: float | None) -> str | None:
    """One row of the continuous-margin table: the rule evaluated at exactly the maximum goat score.

    The threshold ``t`` is the largest score of any counted goat check, so the goat negatives never alarm
    (``score > t`` is false everywhere); the grid of the main table only coarsens this point.
    """
    if rule.score is None:
        return None
    negatives = [c for c in clips if c.label == "neg"]
    t = max((float(_scores(rule, c).max()) for c in negatives if c.mask.any()), default=0.0)
    real = [c for c in clips if c.is_positive and c.group.startswith("commons_fire")]
    real_max = [float(_scores(rule, c).max()) for c in real]
    real_recall = float(np.mean([(_scores(rule, c) > t).mean() for c in real])) if real else float("nan")
    detected = sum(m > t for m in real_max)
    synth = []
    for c in sorted((c for c in clips if c.is_positive and c.group == SYNTH_GROUP), key=lambda c: c.name):
        values = _scores(rule, c)
        synth.append(f"{(values > t).mean():.2f}, {int(bool((values > t).any()))}")
    hard = [c for c in clips if c.label == HARD_NEGATIVE and c.mask.any()]
    hard_max = {c.name: float(_scores(rule, c).max()) for c in hard}
    hits = {name: value > t for name, value in hard_max.items()}
    fixed = sum(h for c in hard for h in [hits[c.name]] if not any(k in c.name for k in FIXED_CAMERA_EXCLUDED))
    lapse = sum(hits[c.name] for c in hard if is_timelapse(c.name))
    ratio = lambda x: "inf" if t == 0 else f"{x / t:.2f}"  # noqa: E731
    grid = "-" if grid_param is None or t == 0 else f"x{grid_param / t:.2f}"
    return (
        f"| {rule.name} | {t:.4f} | {grid} | {detected}/{len(real)} | {real_recall:.2f} | {' | '.join(synth)} | "
        f"{sum(hits.values())}/{len(hard)} | {fixed} | {lapse} | {ratio(max(hard_max.values(), default=0.0))} | "
        f"{ratio(min(real_max, default=0.0))} |"
    )


def render_continuous(rules: list[Rule], clips: list[Clip], grid_params: dict[str, float | None]) -> str:
    """Markdown table of the continuous-margin analysis (see :func:`continuous_row`)."""
    fractions = sorted({re.search(r"(\d\.\d+)$", c.name).group(1) for c in clips if c.is_positive and c.group == SYNTH_GROUP})
    head = (
        "| Rule | max goat score t | grid p / t | Real clips (score > t) | Real recall | "
        + " | ".join(f"Synth {float(f) * 100:g}% (recall, hit)" for f in fractions)
        + " | Hard (all) | Hard (fixed cam) | Hard (time-lapse) | max hard score / t | weakest real clip / t |"
    )
    lines = [
        "Continuous margin: threshold = largest goat score (alarm iff score > t), so goat alarms are 0 by construction.\n",
        head,
    ]
    lines.append("|" + "---|" * (head.count("|") - 1))
    lines.extend(row for rule in rules if (row := continuous_row(rule, clips, grid_params.get(rule.name))) is not None)
    return "\n".join(lines)


def render_hard_detail(rules: list[Rule], clips: list[Clip], grid_params: dict[str, float | None]) -> str:
    """Per hard-negative clip: alarmed checks at the best safe point (and max score relative to the max goat score)."""
    hard = sorted((c for c in clips if c.label == HARD_NEGATIVE and c.mask.any()), key=lambda c: c.name)
    negatives = [c for c in clips if c.label == "neg"]
    head = "| Hard clip (kind) | " + " | ".join(r.name for r in rules) + " |"
    lines = [
        "Hard negatives at each rule's best safe point: alarmed checks / counted checks (max score / max goat score).\n",
        head,
    ]
    lines.append("|" + "---|" * (head.count("|") - 1))
    goat_max = {
        r.name: max((float(_scores(r, c).max()) for c in negatives if c.mask.any()), default=0.0) if r.score else 0.0
        for r in rules
    }
    for clip in hard:
        kind = (
            "time-lapse"
            if is_timelapse(clip.name)
            else "moving camera"
            if any(k in clip.name for k in FIXED_CAMERA_EXCLUDED)
            else "fixed"
        )
        cells = []
        for r in rules:
            p = grid_params.get(r.name)
            if p is None:
                cells.append("-")
                continue
            alarms = int((r.decide(clip.features, p) & clip.mask).sum())
            extra = f" ({_scores(r, clip).max() / goat_max[r.name]:.2f})" if r.score and goat_max[r.name] > 0 else ""
            cells.append(f"{alarms}/{int(clip.mask.sum())}{extra}")
        lines.append(f"| {clip.name.split('__', 1)[-1][:40]} ({kind}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_per_day(rules: list[Rule], clips: list[Clip]) -> str:
    """Maximum score per goat recording day, next to the maximum score of every synthetic composite.

    Shows which day sets the continuous threshold ``t`` (the largest goat score) and how far each composite is above or
    below it (ratio to ``t``), i.e. whether a rule separates the small fire from the goat's worst day.
    """
    scored = [r for r in rules if r.score is not None]
    negatives = [c for c in clips if c.label == "neg" and c.mask.any()]
    days = sorted(
        {re.search(r"(\d{4}-\d{2}-\d{2})", c.name).group(1) for c in negatives if re.search(r"\d{4}-\d{2}-\d{2}", c.name)}
    )
    synth = sorted((c for c in clips if c.is_positive and c.group == SYNTH_GROUP and c.mask.any()), key=lambda c: c.name)
    head = "| | " + " | ".join(r.name for r in scored) + " |"
    lines = ["Per-day maximum goat score and synthetic maxima (ratio to the largest goat score t in brackets).\n", head]
    lines.append("|" + "---|" * (head.count("|") - 1))
    goat_t = {r.name: max(float(_scores(r, c).max()) for c in negatives) for r in scored}
    for day in days:
        day_clips = [c for c in negatives if day in c.name]
        cells = [f"{max(float(_scores(r, c).max()) for c in day_clips):.4g}" for r in scored]
        lines.append(f"| goat {day} | " + " | ".join(cells) + " |")
    lines.append("| **t (max goat)** | " + " | ".join(f"{goat_t[r.name]:.4g}" for r in scored) + " |")
    for clip in synth:
        cells = []
        for r in scored:
            values = _scores(r, clip)
            ratio = float(values.max()) / goat_t[r.name] if goat_t[r.name] > 0 else float("inf")
            cells.append(f"{float(values.max()):.4g} (x{ratio:.2f}; {(values > goat_t[r.name]).mean():.2f})")
        lines.append(f"| {clip.name.split('__', 1)[-1]} max (x t; share of checks > t) | " + " | ".join(cells) + " |")
    return "\n".join(lines)
