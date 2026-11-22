"""ONNX vs PyTorch parity checks for the zero-shot head (#151).

1. Graph parity: the same project-preprocessed blob through the PyTorch head, ONNX Runtime CPU and cv2.dnn (new and
   classic engine); max abs difference of P(fire).
2. Pipeline difference (informational): project ``preprocess`` + ONNX vs the Hugging Face zero-shot pipeline (PIL
   resize, SigLIP processor, ``padding="max_length"``) reduced the same way. Only for single-prompt sets, where the
   pipeline's per-label sigmoid scores can be inverted to logits. Catches normalisation / tokenisation mistakes.

Example:
    python -m tools.zero_shot_eval.parity --model google/siglip2-base-patch16-224 --prompt-set rich --onnx m.onnx \
        --manifest ~/data/llm-verifier-eval/manifest_main.json --n 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from transformers import pipeline

from tools.zero_shot_eval.evaluate import fire_probability
from tools.zero_shot_eval.evaluate import preprocess
from tools.zero_shot_eval.export import build
from tools.zero_shot_eval.prompts import FIRE_CLASSES
from tools.zero_shot_eval.prompts import PROMPT_SETS


def _lse(v: np.ndarray) -> float:
    m = v.max()
    return float(m + np.log(np.exp(v - m).sum()))


def main() -> None:
    """Print parity numbers for one (model, prompt set, ONNX file)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-set", required=True, choices=sorted(PROMPT_SETS))
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8, help="images: the first n/2 positives and n/2 negatives")
    args = ap.parse_args()
    rows = json.loads(args.manifest.read_text())
    pick = [r for r in rows if r["label"] == "yes"][: args.n // 2] + [r for r in rows if r["label"] == "no"][: args.n // 2]
    head, size, _ = build(args.model, args.prompt_set)
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    nets = {"cv2-new": cv2.dnn.readNetFromONNX(str(args.onnx))}
    if hasattr(cv2.dnn, "ENGINE_CLASSIC"):
        nets["cv2-classic"] = cv2.dnn.readNetFromONNX(str(args.onnx), cv2.dnn.ENGINE_CLASSIC)
    prompt_set = PROMPT_SETS[args.prompt_set]
    single = all(len(p) == 1 for p in prompt_set.values())
    zs = pipeline("zero-shot-image-classification", model=args.model) if single else None
    labels = [p[0] for p in prompt_set.values()]
    is_fire = np.array([n in FIRE_CLASSES for n in prompt_set])
    diffs: dict[str, list[float]] = {k: [] for k in ("ort", *nets, "hf-pipeline")}
    for r in pick:
        frame = cv2.imread(str(args.manifest.parent / r["path"]))
        blob = preprocess(frame, size)
        with torch.no_grad():
            p_torch = fire_probability(head(torch.from_numpy(blob)).numpy())
        p_ort = fire_probability(sess.run(None, {"input": blob})[0])
        diffs["ort"].append(abs(p_ort - p_torch))
        for k, net in nets.items():
            net.setInput(blob)
            diffs[k].append(abs(fire_probability(net.forward()) - p_torch))
        if zs is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = {d["label"]: d["score"] for d in zs(img, candidate_labels=labels, hypothesis_template="{}")}
            sig = np.clip(np.array([res[lab] for lab in labels], dtype=np.float64), 1e-12, 1 - 1e-12)
            logit = np.log(sig / (1 - sig))  # = scale * cos + bias; bias cancels in the 2-way softmax
            p_hf = fire_probability(np.array([_lse(logit[~is_fire]), _lse(logit[is_fire])]))
            diffs["hf-pipeline"].append(abs(p_ort - p_hf))
        print(f"{r['cat']}: P(fire) torch {p_torch:.4f} ort {p_ort:.4f}")
    for k, v in diffs.items():
        if v:
            print(f"max |dP(fire)| {k} vs {'onnx-ort' if k == 'hf-pipeline' else 'torch'}: {max(v):.2e} (n={len(v)})")


if __name__ == "__main__":
    main()
