"""Evaluate any pretrained ONNX fire classifier on the manifest splits (issue #73).

Preprocessing: squash-resize to the model input size (bilinear), RGB, ImageNet mean/std. A single output is treated
as a logit (sigmoid); two outputs as ``softmax[1]``. Threshold is picked on ``val`` D-Fire rows for the recall target and
applied to ``test``; Gavle false positives are reported with the 2 s per frame sampling caveat.

Example:
    python -m tools.edge_model.eval_onnx --manifest manifest.csv --model model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from tools.edge_model.evaluate import pick_threshold
from tools.edge_model.evaluate import report
from tools.edge_model.train import MEAN
from tools.edge_model.train import STD
from tools.edge_model.train import load_manifest


def score(sess: ort.InferenceSession, rows: list[dict], bs: int = 1) -> np.ndarray:
    """Return P(fire) for every row."""
    inp = sess.get_inputs()[0]
    size = int(inp.shape[2])
    mean = np.array(MEAN, np.float32).reshape(1, 3, 1, 1)
    std = np.array(STD, np.float32).reshape(1, 3, 1, 1)
    out = []
    for i in range(0, len(rows), bs):
        batch = []
        for r in rows[i : i + bs]:
            img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
            batch.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR))
        x = (np.stack(batch).astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - mean) / std
        y = sess.run(None, {inp.name: x})[0]
        if y.shape[-1] == 1:
            out.append(1 / (1 + np.exp(-y[:, 0])))
        else:
            e = np.exp(y - y.max(1, keepdims=True))
            out.append((e / e.sum(1, keepdims=True))[:, 1])
    return np.concatenate(out)


def main() -> None:
    """Score val and test, pick the threshold on val and print the report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--smoke-policy", default="exclude")
    ap.add_argument("--recall-target", type=float, default=0.95)
    args = ap.parse_args()
    rows = load_manifest(args.manifest, args.smoke_policy)
    sess = ort.InferenceSession(str(args.model))
    by = {s: [r for r in rows if r["split"] == s] for s in ("val", "test")}
    probs = {s: score(sess, by[s]) for s in by}
    dfire = np.array([r["source"] == "dfire" for r in by["val"]])
    yv = np.array([r["y"] for r in by["val"]])
    thr = pick_threshold(probs["val"][dfire], yv[dfire], args.recall_target)
    for s in ("val", "test"):
        report(by[s], probs[s], thr, s)


if __name__ == "__main__":
    main()
