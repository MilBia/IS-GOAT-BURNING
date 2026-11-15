"""Evaluate a trained classifier on the held-out splits (issue #73).

Threshold is chosen on the validation split (highest threshold that keeps recall >= target), then applied to test.
Reports D-Fire recall / FPR and the Gavle false-positive count. Gavle frames are sampled every 2 s, so
FP/hour = false_positives / (frames * 2 s / 3600).

Example:
    python -m tools.edge_model.evaluate --manifest manifest.csv --run runs/mnv3s_224 --smoke-policy exclude
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.edge_model.train import load_manifest

SECONDS_PER_GAVLE_FRAME = 2.0


def pick_threshold(probs: np.ndarray, y: np.ndarray, recall_target: float) -> float:
    """Return the highest threshold whose recall on ``y`` is at least ``recall_target``."""
    for t in np.linspace(0.99, 0.01, 99):
        if ((probs >= t) & (y == 1)).sum() / max(1, (y == 1).sum()) >= recall_target:
            return float(t)
    return 0.01


def report(rows: list[dict], probs: np.ndarray, thr: float, name: str) -> None:
    """Print metrics for one split."""
    y = np.array([r["y"] for r in rows])
    pred = probs >= thr
    src = np.array([r["source"] for r in rows])
    fire = y == 1
    print(f"\n## {name} (threshold {thr:.2f})")
    print(f"- D-Fire recall: {(pred & fire & (src == 'dfire')).sum() / max(1, (fire & (src == 'dfire')).sum()):.3f}")
    neg_d = (~fire) & (src == "dfire")
    print(f"- D-Fire FPR: {(pred & neg_d).sum() / max(1, neg_d.sum()):.3f}")
    g = src == "gavle"
    if g.any():
        fp = int((pred & g).sum())
        hours = g.sum() * SECONDS_PER_GAVLE_FRAME / 3600
        print(f"- Gavle FP: {fp}/{int(g.sum())} frames ({fp / hours:.1f} per hour, max P(fire)={probs[g].max():.3f})")


def main() -> None:
    """Pick threshold on val and report val + test."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--smoke-policy", default="exclude")
    ap.add_argument("--recall-target", type=float, default=0.95)
    args = ap.parse_args()
    rows = load_manifest(args.manifest, args.smoke_policy)
    by = {s: [r for r in rows if r["split"] == s] for s in ("val", "test")}
    val_p = np.load(args.run / "val_probs.npy")
    dfire_val = np.array([(r["source"] == "dfire") for r in by["val"]])
    yv = np.array([r["y"] for r in by["val"]])
    thr = pick_threshold(val_p[dfire_val], yv[dfire_val], args.recall_target)
    report(by["val"], val_p, thr, "val")
    report(by["test"], np.load(args.run / "test_probs.npy"), thr, "test")


if __name__ == "__main__":
    main()
