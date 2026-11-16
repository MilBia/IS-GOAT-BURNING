"""POST HOC analysis of #140 (after the pre-registered ``compare``): real-clip recall split fixed / moving camera.

The split is an objective proxy fixed before looking at any ``lr_*`` result: a real-fire clip is "moving camera" when the
median share of the frame that changes between consecutive checks (``mv25 / area`` of the baseline motion mask over the counted
checks) exceeds ``MOVING_SHARE`` = 0.15 (hand-held / panning footage: Bonfire on a farm, Durham, Greenland, Veel schade); the
other five real clips are called "fixed". Threshold t = the largest goat score (as ``compare --continuous``); recall = mean over
the clips of the share of counted checks with score > t.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.fire_rule_eval.protocol import SYNTH_GROUP
from tools.fire_rule_eval.report import load_clips
from tools.fire_rule_eval.rules import RULES_BY_NAME

MOVING_SHARE = 0.15


def split_table(manifest: Path, features: Path, warmup: int, names: list[str]) -> str:
    """Markdown table: goat threshold, recall on fixed / moving real clips, all real clips."""
    clips = load_clips(manifest, features, None, warmup)
    goat = [c for c in clips if c.label == "neg"]
    real = [c for c in clips if c.is_positive and c.group != SYNTH_GROUP]
    moving = {c.name for c in real if np.median((c.features["mv25"] / c.features["area"])[c.mask]) > MOVING_SHARE}
    lines = [
        f"Moving-camera real clips (median motion share > {MOVING_SHARE}): "
        + ", ".join(sorted(n.split("__", 1)[-1][:30] for n in moving)),
        "",
        "| Rule | t (max goat) | recall all (9) | recall fixed (5) | recall moving (4) | detected fixed | detected moving |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in names:
        rule = RULES_BY_NAME[name]
        if rule.score is None:
            continue
        t = max(float(rule.score(c.features)[c.mask].max()) for c in goat if c.mask.any())
        per = {c.name: float((rule.score(c.features)[c.mask] > t).mean()) for c in real}
        fixed = [v for n, v in per.items() if n not in moving]
        mov = [v for n, v in per.items() if n in moving]
        lines.append(
            f"| {name} | {t:.4f} | {np.mean(list(per.values())):.2f} | {np.mean(fixed):.2f} | {np.mean(mov):.2f} | "
            f"{sum(v > 0 for v in fixed)}/{len(fixed)} | {sum(v > 0 for v in mov)}/{len(mov)} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rules", nargs="+", required=True)
    args = parser.parse_args()
    print(split_table(Path(args.manifest), Path(args.features), args.warmup, args.rules))


if __name__ == "__main__":
    main()
