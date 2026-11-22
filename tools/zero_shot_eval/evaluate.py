"""Score the eval manifest with zero-shot ONNX models and print a Markdown report (#151).

Deployable path only: the copied ``edge_detector.preprocess`` -> ONNX Runtime CPU -> ``classifier_fire_probability``.
Threshold rule of umbrella #149 ("best safe point"): the lowest threshold with 0 false alarms on all negatives, i.e. just
above the highest negative score, rounded UP to 4 decimals (safe to copy into ``EDGE__CONFIDENCE_THRESHOLD``).
Categories starting with ``synth`` are excluded from headline recall, ``goat*`` form the goat FPR,
``dfire_fire`` / ``hard_warm_dfire`` are reported separately as well.

Per-image scores (with image paths) go to ``--out``; keep it outside the repository.

Example:
    python -m tools.zero_shot_eval.evaluate --manifest ~/data/llm-verifier-eval/manifest_main.json \
        --out ~/data/zero-shot-151/scores m1.onnx m2.onnx > report.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
DFIRE_CATS = ("dfire_fire", "hard_warm_dfire")
SWEEP = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def ceil4(x: float) -> float:
    """Round up to 4 decimals, so a printed threshold used with ``>=`` never admits the negative it was derived from."""
    return math.ceil(x * 1e4) / 1e4


def preprocess(frame: np.ndarray, input_size: int) -> np.ndarray:
    """Copy of ``is_goat_burning.fire_detection.edge_detector.preprocess`` (importing it needs the app config)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    return np.ascontiguousarray((blob - _MEAN) / _STD, dtype=np.float32)


def fire_probability(output: np.ndarray) -> float:
    """Copy of ``classifier_fire_probability`` for two logits (softmax index 1)."""
    logits = np.asarray(output, dtype=np.float64).reshape(-1)
    exp = np.exp(logits - logits.max())
    return float(exp[1] / exp.sum())


def wilson(k: int, n: int) -> tuple[float, float]:
    """95 % Wilson interval for k/n (same as ``tools.llm_verifier_eval``)."""
    if n == 0:
        return (0.0, 1.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def fmt(k: int, n: int) -> str:
    """Format k/n with its Wilson interval."""
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}% ({100 * lo:.0f}-{100 * hi:.0f})" if n else "-"


def score_model(model: Path, rows: list[dict], root: Path, threads: int) -> list[float]:
    """Return P(fire) per manifest row."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = int(inp.shape[2])
    out = []
    for r in rows:
        frame = cv2.imread(str(root / r["path"]))
        if frame is None:
            raise FileNotFoundError(r["path"])
        out.append(fire_probability(sess.run(None, {inp.name: preprocess(frame, size)})[0]))
    return out


def report(name: str, rows: list[dict], scores: list[float]) -> None:
    """Print the Markdown report for one model."""
    s = np.array(scores)
    cat = np.array([r["cat"] for r in rows])
    pos = np.array([r["label"] == "yes" for r in rows])
    synth = np.char.startswith(cat.astype(str), "synth")
    dfire = np.isin(cat, DFIRE_CATS)
    neg = ~pos
    head = pos & ~synth
    safe = ceil4(float(np.nextafter(s[neg].max(), 1.0)))

    def rec(mask: np.ndarray, t: float) -> str:
        return fmt(int((s[mask] >= t).sum()), int(mask.sum()))

    print(f"\n### {name}\n")
    print(
        f"best safe point: threshold {safe:.4f} (max negative score {s[neg].max():.6f}, from `{cat[neg][s[neg].argmax()]}`)\n"
    )
    print(f"- recall non-synth fire (headline): {rec(head, safe)}")
    safe_nd = ceil4(float(np.nextafter(s[neg & ~dfire].max(), 1.0)))
    print(f"- recall non-synth fire without D-Fire, same threshold: {rec(head & ~dfire, safe)}")
    print(
        f"- leak-free (both D-Fire categories dropped, threshold {safe_nd:.4f} from {int((neg & ~dfire).sum())} negatives): "
        f"recall {rec(head & ~dfire, safe_nd)}"
    )
    print(f"- FPR all negatives: {rec(neg, safe)}; goat: {rec(neg & np.char.startswith(cat.astype(str), 'goat'), safe)}")
    order = np.sort(s[neg])[::-1]
    for k in (1, 3):
        t = ceil4(float(np.nextafter(order[k], 1.0)))
        print(f"- for reference, allowing {k}/{int(neg.sum())} false alarms (thr {t:.4f}): recall {rec(head, t)}")
    print("\n| category | truth | n | median P(fire) | max P(fire) | alarms at safe point |\n|---|---|---:|---:|---:|---|")
    for c in sorted(set(cat)):
        m = cat == c
        truth = "fire" if pos[m][0] else "no fire"
        print(f"| {c} | {truth} | {int(m.sum())} | {np.median(s[m]):.3f} | {s[m].max():.3f} | {rec(m, safe)} |")
    print("\n| thr | recall non-synth | recall w/o D-Fire | FPR all neg | FPR goat | FPR non-goat neg |")
    print("|---|---|---|---|---|---|")
    goat = neg & np.char.startswith(cat.astype(str), "goat")
    for t in SWEEP:
        cols = [rec(m, t) for m in (head, head & ~dfire, neg, goat, neg & ~goat)]
        print(f"| {t} | " + " | ".join(cols) + " |")


def main() -> None:
    """Score every model and print the report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", type=Path, nargs="+")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="directory for per-image scores (outside the repo)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    rows = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["cat"]] += 1
    print(f"manifest: {len(rows)} frames, categories {dict(sorted(counts.items()))}")
    for model in args.models:
        scores = score_model(model, rows, args.manifest.parent, args.threads)
        with (args.out / f"{model.stem}.jsonl").open("w") as f:
            for r, sc in zip(rows, scores, strict=True):
                f.write(json.dumps({**r, "p_fire": sc}) + "\n")
        report(model.stem, rows, scores)


if __name__ == "__main__":
    main()
