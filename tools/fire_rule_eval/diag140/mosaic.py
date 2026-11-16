# ruff: noqa: B007, B023, E401, E501, F401, I001, UP031
"""Exploratory pre-check of #140: does the sparse component of a low-rank model remove the privacy mosaic of the goat clips?

Decision rule written BEFORE the script was run (commit of this file precedes the pre-registration): a model "removes the
mosaic" iff, on each of the three goat clips (17.12, 22.12, 27.12, as in ``diag139/mosaic.py``), the foreground share inside the
mosaic region is at most 1.5x the share outside it AND at most one third of the share the frame differencing (``diff25``)
puts inside the mosaic. The controls ``diff25`` / ``med5`` / ``mog2_h30`` are printed next to the candidates (#139: 2-15x).
"""

import sys, glob, numpy as np, cv2
from pathlib import Path
from tools.fire_rule_eval.features import ExtractOptions, iter_check_frames, MOSAIC_REGIONS, _region_mask
from tools.fire_rule_eval.bgmodel import BgModels
from tools.fire_rule_eval.lowrank import ALL_VARIANTS, LowRankModels

clips = sorted(
    glob.glob("/exp/data/goat/event_2025-12-27*/goat-cam_2025-12-27_13-01-11.mp4")
    + glob.glob("/exp/data/goat/event_2025-12-22*/*.mp4")[:1]
    + glob.glob("/exp/data/goat/event_2025-12-17*/*.mp4")[:1]
)
LR = tuple(ALL_VARIANTS)
for c in clips:
    lr = LowRankModels(LR)
    ctrl = BgModels(("med5", "mog2_h30"))
    acc = {k: [] for k in ("diff25", *ctrl.variants, *LR)}
    prev = None
    mos = None
    for i, (t, f) in enumerate(iter_check_frames(Path(c), ExtractOptions(1.0, 0.0, 40, 0.25))):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if mos is None:
            mos = _region_mask(g.shape, MOSAIC_REGIONS) > 0
        masks = {}
        if prev is not None:
            masks["diff25"] = cv2.absdiff(g, prev) > 25
        masks.update({k: v > 127 for k, v in ctrl.masks(g).items()})
        masks.update({k: v > 127 for k, v in lr.masks(g).items()})
        prev = g
        if i >= 5:
            for k, m in masks.items():
                acc[k].append((m.mean(), m[mos].mean(), m[~mos].mean()))
    print(Path(c).parent.name, Path(c).name, len(acc["diff25"]), "checks", flush=True)
    base = np.array(acc["diff25"])[:, 1].mean()
    for k, v in acc.items():
        a = np.array(v)
        if len(a):
            ratio = a[:, 1].mean() / max(a[:, 2].mean(), 1e-9)
            removes = ratio <= 1.5 and a[:, 1].mean() <= base / 3
            print(
                f"  {k:16s} all {a[:, 0].mean():.4f} (max {a[:, 0].max():.4f})  mosaic {a[:, 1].mean():.4f}  outside {a[:, 2].mean():.4f}  ratio {ratio:5.1f}  removes={removes}",
                flush=True,
            )
