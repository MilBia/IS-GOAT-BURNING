"""#138 report of the global-change gates: per hard clip, least-global alarmed checks, real-fire cost, scene-cut audit.

All rules are evaluated at their continuous operating point ``t`` = the largest score of any counted goat check
(alarm iff score > t), as in ``compare --continuous``. The scene-cut audit (``--cuts``) is POST HOC and uses the
independent histogram detector of ``cuts.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.fire_rule_eval.protocol import FIXED_CAMERA_EXCLUDED
from tools.fire_rule_eval.protocol import SYNTH_GROUP
from tools.fire_rule_eval.protocol import is_timelapse
from tools.fire_rule_eval.report import HARD_NEGATIVE
from tools.fire_rule_eval.report import Clip
from tools.fire_rule_eval.rules import GATE_RULES
from tools.fire_rule_eval.rules import GATE_THRESHOLDS
from tools.fire_rule_eval.rules import RULES_BY_NAME
from tools.fire_rule_eval.rules import Rule

GATE_FEATURES = (("|dv|", "gl_dv"), ("out", "gl_out"), ("tiles", "gl_tiles"), ("coh", "gl_coh"), ("shift", "gl_shift"))


def _short(name: str) -> str:
    return name.split("__", 1)[-1][:34]


def _kind(name: str) -> str:
    if is_timelapse(name):
        return "time-lapse"
    return "moving camera" if any(k in name for k in FIXED_CAMERA_EXCLUDED) else "fixed"


def threshold(rule: Rule, goat: list[Clip]) -> float:
    """Largest score of a counted goat check (the continuous operating point)."""
    return max((float(rule.score(c.features)[c.mask].max()) for c in goat if c.mask.any()), default=0.0)


def _value(f: dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.abs(f[key]) if key == "gl_dv" else f[key]


def load_cuts(path: Path | None) -> dict[str, set[int]]:
    """Check indices labelled as scene cuts by ``cuts.py`` (empty when no file is given)."""
    if path is None:
        return {}
    return {name: set(entry["cuts"]) for name, entry in json.loads(path.read_text()).items()}


def run(clips: list[Clip], title: str, cuts: dict[str, set[int]]) -> str:  # noqa: C901
    """Markdown report for the gate rules over ``clips``."""
    goat = [c for c in clips if c.label == "neg" and c.mask.any()]
    hard = sorted((c for c in clips if c.label == HARD_NEGATIVE and c.mask.any()), key=lambda c: c.name)
    real = sorted((c for c in clips if c.is_positive and c.group.startswith("commons_fire")), key=lambda c: c.name)
    synth = sorted((c for c in clips if c.group == SYNTH_GROUP), key=lambda c: c.name)
    rules = [RULES_BY_NAME["area_relative"], RULES_BY_NAME["area_relative_deb3"], *GATE_RULES]
    t = {r.name: threshold(r, goat) for r in rules}
    out = [
        f"# #138 global-change gates: {title}\n",
        "Operating point: continuous, t = max counted goat score of each rule (W=5).\n",
    ]

    # 1. summary per rule
    out.append("## 1. Summary per rule (continuous margin)\n")
    out.append(
        "| Rule | t | Real clips | Real recall | Synth hits (0.3/1/3/10 %) | Hard | Fixed | Time-lapse | Moving camera | Goat checks blocked |"  # noqa: E501
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    hits: dict[str, dict[str, bool]] = {}
    for r in rules:
        h = {c.name: bool((r.score(c.features)[c.mask] > t[r.name]).any()) for c in hard}
        hits[r.name] = h
        recalls = [float((r.score(c.features)[c.mask] > t[r.name]).mean()) for c in real]
        det = sum(bool((r.score(c.features)[c.mask] > t[r.name]).any()) for c in real)
        sy = "/".join(str(int(bool((r.score(c.features)[c.mask] > t[r.name]).any()))) for c in synth)
        fixed = sum(v for c in hard for v in [h[c.name]] if _kind(c.name) == "fixed")
        lapse = sum(v for c in hard for v in [h[c.name]] if _kind(c.name) == "time-lapse")
        moving = sum(v for c in hard for v in [h[c.name]] if _kind(c.name) == "moving camera")
        blocked_goat = "-"
        if r.name.startswith("gate_"):
            base_s = RULES_BY_NAME["area_relative"].score
            blocked_goat = str(sum(int(((base_s(c.features) > 0) & (r.score(c.features) == 0) & c.mask).sum()) for c in goat))
        out.append(
            f"| {r.name} | {t[r.name]:.4f} | {det}/{len(real)} | {np.mean(recalls):.3f} | {sy} | {sum(h.values())}/{len(hard)} | "  # noqa: E501
            f"{fixed}/{sum(_kind(c.name) == 'fixed' for c in hard)} | {lapse}/{sum(_kind(c.name) == 'time-lapse' for c in hard)} | "  # noqa: E501
            f"{moving}/{sum(_kind(c.name) == 'moving camera' for c in hard)} | {blocked_goat} |"
        )

    # 2. per hard clip
    out.append("\n## 2. Per hard negative: alarmed checks / counted checks at each rule's own t\n")
    cols = [r for r in rules if not r.name.endswith("_deb3") or r.name.startswith("gate_all")]
    out.append("| Hard clip (kind) | " + " | ".join(r.name.replace("gate_", "g_") for r in cols) + " |")
    out.append("|---|" + "---|" * len(cols))
    for c in hard:
        cells = [f"{int(((r.score(c.features) > t[r.name]) & c.mask).sum())}/{int(c.mask.sum())}" for r in cols]
        out.append(f"| {_short(c.name)} ({_kind(c.name)}) | " + " | ".join(cells) + " |")

    # 3. least-global alarmed check of each hard clip (baseline alarms)
    base = RULES_BY_NAME["area_relative"]
    out.append("\n## 3. Hard clips: least global baseline-alarmed check (min of each gate feature over the baseline alarms)\n")
    thr = {k: GATE_THRESHOLDS[k] for k in ("dv", "out", "tiles", "coh", "shift")}
    out.append(
        "Gate thresholds (strict / loose): "
        + "; ".join(f"{label} {thr[k][0]:g}/{thr[k][1]:g}" for (label, _), k in zip(GATE_FEATURES, thr, strict=True))
        + ". A gate removes the clip only if its feature is >= threshold on EVERY alarmed check, i.e. min >= threshold.\n"
    )
    out.append(
        "| Hard clip (kind) | baseline alarms | " + " | ".join(f"min {label}" for label, _ in GATE_FEATURES) + " | max wg |"
    )
    out.append("|---|---|" + "---|" * (len(GATE_FEATURES) + 1))
    for c in hard:
        alarm = (base.score(c.features) > t["area_relative"]) & c.mask
        if not alarm.any():
            out.append(f"| {_short(c.name)} ({_kind(c.name)}) | 0 | " + " | ".join("-" for _ in GATE_FEATURES) + " | - |")
            continue
        mins = [f"{_value(c.features, key)[alarm].min():.3f}" for _, key in GATE_FEATURES]
        out.append(
            f"| {_short(c.name)} ({_kind(c.name)}) | {int(alarm.sum())} | "
            + " | ".join(mins)
            + f" | {c.features['gl_wg'][alarm].max():.2f} |"
        )

    # 4. goat range of the features
    out.append("\n## 4. Goat range of the gl_* features (counted checks; target-scene noise level)\n")
    out.append("| Feature | median | p99 | max |")
    out.append("|---|---|---|---|")
    for label, key in (*GATE_FEATURES, ("wg", "gl_wg"), ("resp", "gl_resp")):
        values = np.concatenate([_value(c.features, key)[c.mask] for c in goat])
        out.append(f"| {label} | {np.median(values):.4f} | {np.quantile(values, 0.99):.4f} | {values.max():.4f} |")

    # 5. real fires: baseline-alarmed checks blocked
    out.append("\n## 5. Real fires: baseline alarms (alarmed checks / counted) and how many each gate removes\n")
    gates = [r for r in GATE_RULES if not r.name.endswith("_deb3")]
    out.append(
        "| Real clip | counted | cut checks | baseline alarms | "
        + " | ".join(r.name.replace("gate_", "g_") for r in gates)
        + " |"
    )
    out.append("|---|---|---|---|" + "---|" * len(gates))
    for c in real:
        alarm = (base.score(c.features) > t["area_relative"]) & c.mask
        cut_idx = cuts.get(c.name, set())
        cut_mask = np.isin(np.arange(len(c.mask)), list(cut_idx)) & c.mask
        cells = []
        for r in gates:
            blocked = alarm & (r.score(c.features) <= t[r.name])
            cells.append(str(int(blocked.sum())))
        out.append(
            f"| {_short(c.name)} | {int(c.mask.sum())} | {int(cut_mask.sum())} | {int(alarm.sum())} | "
            + " | ".join(cells)
            + " |"
        )

    # 6. post hoc: recall without scene cuts
    if cuts:
        out.append("\n## 6. POST HOC: real recall with and without the scene-cut checks (independent histogram detector)\n")
        out.append(
            "Cut checks are removed from numerator and denominator (counted checks only). The cut list comes from `cuts.py`.\n"
        )
        out.append(
            "| Rule | recall (all checks) | recall (no cut checks) | change vs baseline, no cuts [pp] | clips detected (no cuts) |"  # noqa: E501
        )
        out.append("|---|---|---|---|---|")
        base_nc = None
        for r in rules:
            all_r, nc_r, det = [], [], 0
            for c in real:
                cut_mask = np.isin(np.arange(len(c.mask)), list(cuts.get(c.name, set())))
                alarm = r.score(c.features) > t[r.name]
                all_r.append(float(alarm[c.mask].mean()))
                keep = c.mask & ~cut_mask
                if keep.any():
                    nc_r.append(float(alarm[keep].mean()))
                    det += int(alarm[keep].any())
            nc = float(np.mean(nc_r))
            base_nc = nc if base_nc is None else base_nc
            out.append(f"| {r.name} | {np.mean(all_r):.3f} | {nc:.3f} | {100 * (nc - base_nc):+.1f} | {det}/{len(nc_r)} |")
        counts = sum(len(v) for v in cuts.values())
        out.append(f"\nCut checks labelled in the 9 real clips: {counts}.")
    out.append(oracle_section(goat, hard, real, cuts, t["area_relative"]))
    # 7. gate also versus the hard clips excluding cuts is not applicable (hard clips are one continuous shot)
    out.append(
        "\nTime-lapse = stress test (accelerated day/night); real time is NOT measured (no goat stream until December 2026, #109)."  # noqa: E501
    )
    return "\n".join(out)


def oracle_section(goat: list[Clip], hard: list[Clip], real: list[Clip], cuts: dict[str, set[int]], t: float) -> str:
    """POST HOC oracle sweep: is ANY threshold of a single gl_* feature good enough? (tuned on the evaluation clips).

    For every feature the threshold is swept over all values the feature takes on the baseline-alarmed checks of the hard and
    real clips; the baseline alarm (score > t) is blocked where the feature is >= the threshold. Reports the best hard-negative
    count under a recall drop of at most 5 pp, and the best hard count regardless of recall, each with the recall it costs.
    This is an upper bound on what a gate of that form can do on these 8 + 9 clips (the threshold is chosen with the answer
    in view), not a result.
    """
    base = RULES_BY_NAME["area_relative"]
    scored = {c.name: (base.score(c.features) > t) & c.mask for c in (*hard, *real)}
    base_recall = float(np.mean([scored[c.name][c.mask].mean() for c in real]))

    def recall(c: Clip, blocked: np.ndarray, drop_cuts: bool) -> float:
        keep = c.mask.copy()
        if drop_cuts:
            keep &= ~np.isin(np.arange(len(c.mask)), list(cuts.get(c.name, set())))
        return float((scored[c.name] & ~blocked)[keep].mean()) if keep.any() else float("nan")

    lines = [
        "\n## 7. POST HOC oracle: best single-feature threshold, tuned on the evaluation clips (upper bound, NOT a result)\n",
        f"Baseline recall {base_recall:.3f} at t={t:.4f}; recall must stay >= {base_recall - 0.05:.3f}. "
        "A hard clip is silenced "
        "when the gate blocks every one of its baseline alarms. Hard negatives that alarm in the baseline: "
        f"{sum(bool(scored[c.name].any()) for c in hard)}/{len(hard)}.\n",
        "| Feature | best hard with recall drop <= 5 pp (threshold; recall; recall w/o cuts) "
        "| best hard at any recall (threshold; recall; recall w/o cuts) |",
        "|---|---|---|",
    ]
    for label, key in GATE_FEATURES:
        alarmed = np.concatenate([_value(c.features, key)[scored[c.name]] for c in (*hard, *real)])
        candidates = sorted({float(x) for x in alarmed})
        rows = []
        for threshold in candidates:
            blocked = {c.name: _value(c.features, key) >= threshold for c in (*hard, *real)}
            n_hard = sum(bool((scored[c.name] & ~blocked[c.name]).any()) for c in hard)
            r_all = float(np.mean([recall(c, blocked[c.name], False) for c in real]))
            r_nc = float(np.nanmean([recall(c, blocked[c.name], True) for c in real]))
            rows.append((n_hard, -r_all, threshold, r_all, r_nc))
        ok = [r for r in rows if r[3] >= base_recall - 0.05]
        best_ok = min(ok) if ok else None
        best_any = min(rows)
        fmt = lambda r: "-" if r is None else f"{r[0]}/{len(hard)} (>= {r[2]:.4g}; {r[3]:.3f}; {r[4]:.3f})"  # noqa: E731
        lines.append(f"| {label} | {fmt(best_ok)} | {fmt(best_any)} |")
    lines.append("\nMedian feature value on the baseline-alarmed checks (hard clips by kind vs real fires):\n")
    lines.append(
        "| Feature | goat (all counted) | time-lapse | moving camera | fixed hard | real fires | real fires w/o cut checks |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for label, key in (*GATE_FEATURES, ("wg", "gl_wg")):

        def med(group: list[Clip], drop_cuts: bool = False, key: str = key) -> str:
            values = []
            for c in group:
                keep = scored[c.name].copy()
                if drop_cuts:
                    keep &= ~np.isin(np.arange(len(c.mask)), list(cuts.get(c.name, set())))
                values.append(_value(c.features, key)[keep])
            values = np.concatenate(values) if values else np.array([])
            return f"{np.median(values):.3f}" if len(values) else "-"

        goat_values = np.concatenate([_value(c.features, key)[c.mask] for c in goat])
        lines.append(
            f"| {label} | {np.median(goat_values):.3f} | {med([c for c in hard if _kind(c.name) == 'time-lapse'])} | "
            f"{med([c for c in hard if _kind(c.name) == 'moving camera'])} | "
            f"{med([c for c in hard if _kind(c.name) == 'fixed'])} | "
            f"{med(real)} | {med(real, True)} |"
        )
    return "\n".join(lines)
