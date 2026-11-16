"""Report of the #139 / #140 background-model experiments: what the models mark on the goat, on the baseline alarms, on fire."""

from __future__ import annotations

import numpy as np

from tools.fire_rule_eval.bgmodel import ALL_VARIANTS
from tools.fire_rule_eval.bgmodel import CANDIDATE_VARIANTS
from tools.fire_rule_eval.lowrank import ALL_VARIANTS as LR_ALL_VARIANTS
from tools.fire_rule_eval.protocol import FIXED_CAMERA_EXCLUDED as FIXED_CAMERA_EXCLUDED_KEYS
from tools.fire_rule_eval.protocol import SYNTH_GROUP
from tools.fire_rule_eval.protocol import is_timelapse
from tools.fire_rule_eval.report import HARD_NEGATIVE
from tools.fire_rule_eval.report import Clip
from tools.fire_rule_eval.rules import area_relative_score


def _variants(clips: list[Clip]) -> list[str]:
    present = [v for v in (*LR_ALL_VARIANTS, *ALL_VARIANTS) if all(f"bgwm_{v}" in c.features for c in clips)]
    lowrank = [v for v in LR_ALL_VARIANTS if v in present]
    known = (*lowrank, *CANDIDATE_VARIANTS)
    return lowrank + [v for v in CANDIDATE_VARIANTS if v in present] + [v for v in present if v not in known]


def _kind(name: str) -> str:
    if is_timelapse(name):
        return "time-lapse"
    return "moving camera" if any(k in name for k in FIXED_CAMERA_EXCLUDED_KEYS) else "fixed"


def goat_table(clips: list[Clip], variants: list[str]) -> str:
    """Frame-wide foreground share on the counted goat checks: motion25 versus every model (clutter gauge)."""
    goat = [c for c in clips if c.label == "neg"]
    head = "| Mask | mean fg share of the frame | p99 | max |\n|---|---|---|---|"
    rows = []
    series = {"motion25": np.concatenate([(c.features["mv25"] / c.features["area"])[c.mask] for c in goat])}
    for v in variants:
        series[v] = np.concatenate([(c.features[f"bgmv_{v}"] / c.features["area"])[c.mask] for c in goat])
    for name, x in series.items():
        rows.append(f"| {name} | {x.mean():.4f} | {np.percentile(x, 99):.4f} | {x.max():.4f} |")
    return (
        "Frame-wide foreground share on the counted goat checks (clutter gauge; includes the privacy mosaic).\n\n"
        + "\n".join([head, *rows])
    )


def retention_table(clips: list[Clip], variants: list[str]) -> str:
    """Per clip: of the pixels the BASELINE counts as warm-AND-moving, how many are foreground in each model.

    For hard negatives the checks are those where the baseline ``area_relative`` score exceeds the largest goat score (the
    baseline's alarmed checks at the continuous operating point); for positives all counted checks. Cells: median (min)
    of ``bgand / wm25`` over those checks; ``-`` when the clip has no such check.
    """
    score = area_relative_score()
    goat_t = max(float(score(c.features)[c.mask].max()) for c in clips if c.label == "neg" and c.mask.any())
    head = f"| Clip (kind) | baseline-alarmed checks (t={goat_t:.4f}) | " + " | ".join(variants) + " |"
    lines = [
        "Retention of the baseline's alarming pixels: median (min) over the checks of bgand / wm25 "
        "(share of warm-AND-motion25 pixels that are also model foreground).\n",
        head,
        "|" + "---|" * (head.count("|") - 1),
    ]
    groups = (
        ("hard negatives", [c for c in clips if c.label == HARD_NEGATIVE]),
        ("real fire", [c for c in clips if c.is_positive and c.group != SYNTH_GROUP]),
        ("synth", [c for c in clips if c.is_positive and c.group == SYNTH_GROUP]),
    )
    for title, members in groups:
        lines.append(f"| **{title}** |" + " |" * (len(variants) + 1))
        for c in sorted(members, key=lambda c: c.name):
            s = score(c.features)
            sel = c.mask & (s > goat_t) if c.label == HARD_NEGATIVE else c.mask
            if not sel.any():
                lines.append(f"| {c.name.split('__', 1)[-1][:38]} ({_kind(c.name)}) | 0 |" + " - |" * len(variants))
                continue
            wm = np.maximum(c.features["wm25"][sel], 1)
            cells = []
            for v in variants:
                r = c.features[f"bgand_{v}"][sel] / wm
                cells.append(f"{np.median(r):.2f} ({r.min():.2f})")
            lines.append(
                f"| {c.name.split('__', 1)[-1][:38]} ({_kind(c.name)}) | {int(sel.sum())} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)


def run(clips: list[Clip], label: str) -> str:
    """Builds the whole report."""
    variants = _variants(clips)
    head = f"# #139 / #140 background-model report: {label}"
    return f"{head}\n\n{goat_table(clips, variants)}\n\n{retention_table(clips, variants)}\n"
