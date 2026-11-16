# ruff: noqa: B007, B023, E401, E501, F401, I001, UP031
"""Exploratory diagnostic of #139 (run before the pre-registration; see bgmodel.py)."""

import sys, glob, numpy as np, cv2
from pathlib import Path
from tools.fire_rule_eval.features import ExtractOptions, iter_check_frames, MOSAIC_REGIONS, _region_mask

clips = sorted(
    glob.glob("/exp/data/goat/event_2025-12-27*/goat-cam_2025-12-27_13-01-11.mp4")
    + glob.glob("/exp/data/goat/event_2025-12-22*/*.mp4")[:1]
    + glob.glob("/exp/data/goat/event_2025-12-17*/*.mp4")[:1]
)


def variants():
    return {
        "diff25": None,
        "mog2_h30": cv2.createBackgroundSubtractorMOG2(30, 16, False),
        "mog2_h30_v40": cv2.createBackgroundSubtractorMOG2(30, 40, False),
        "knn_h30": cv2.createBackgroundSubtractorKNN(30, 400.0, False),
    }


for c in clips:
    vs = variants()
    acc = {k: [] for k in vs}
    acc["med5"] = []
    prev = None
    buf = []
    mos = None
    for i, (t, f) in enumerate(iter_check_frames(Path(c), ExtractOptions(1.0, 0.0, 40, 0.25))):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if mos is None:
            mos = _region_mask(g.shape, MOSAIC_REGIONS) > 0
        masks = {}
        for k, v in vs.items():
            if v is None:
                if prev is not None:
                    masks[k] = cv2.absdiff(g, prev) > 25
            else:
                masks[k] = v.apply(g) > 127
        if len(buf) >= 3:
            masks["med5"] = np.abs(g.astype(np.int16) - np.median(np.stack(buf), axis=0)) > 25
        prev = g
        buf = (buf + [g])[-5:]
        if i >= 5:
            for k, m in masks.items():
                acc[k].append((m.mean(), m[mos].mean(), m[~mos].mean()))
    print(Path(c).parent.name, Path(c).name, len(acc["diff25"]), "checks")
    for k, v in acc.items():
        a = np.array(v)
        if len(a):
            print(
                f"  {k:14s} all {a[:, 0].mean():.4f} (max {a[:, 0].max():.4f})  mosaic {a[:, 1].mean():.4f}  outside {a[:, 2].mean():.4f}  ratio mosaic/outside {a[:, 1].mean() / max(a[:, 2].mean(), 1e-9):.1f}"
            )
