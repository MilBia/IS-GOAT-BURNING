"""Accuracy and parity of OpenVINO IR variants vs the FP32 ONNX reference on the eval manifest (#152).

The reference is the original ONNX on ONNX Runtime CPU (the #151 path); each IR variant runs on OpenVINO CPU with
``INFERENCE_PRECISION_HINT=f32`` by default (what the N100 CPU does); ``--device GPU --precision default`` scores the
iGPU path (f16 compute) on a host with an Intel GPU. Note the defaults differ on purpose: this script defaults to
``--precision f32`` (accuracy of the exact model), ``bench.py`` to ``default`` (what the device does out of the box: f32
on the N100 CPU, f16 on the iGPU); pass ``--precision default`` here to score the bench's GPU configuration.
Threshold rule of umbrella #149 as in ``tools/zero_shot_eval/evaluate.py`` (#151): best safe point = lowest threshold
with 0 false alarms, rounded up to 4 decimals; headline recall excludes ``synth*``; leak-free drops both D-Fire
categories and recomputes the threshold.

Per variant the Markdown table shows the own safe threshold and recall there, recall and false alarms at the
reference threshold, the hit difference vs the reference and max |dP(fire)| over all frames. Per-image scores (with image
paths) go to ``--out``; keep it outside the repository.

Example:
    python -m tools.openvino_eval.evaluate ~/data/zero-shot-151/onnx/p16_224_minimal.onnx \
        --ir-dir ~/data/openvino-152/ir/p16_224_minimal --manifest ~/data/llm-verifier-eval/manifest_main.json \
        --out ~/data/openvino-152/scores
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from tools.openvino_eval.convert import VARIANTS
from tools.openvino_eval.convert import preprocess

DFIRE_CATS = ("dfire_fire", "hard_warm_dfire")


def ceil4(x: float) -> float:
    """Round up to 4 decimals, so a printed threshold used with ``>=`` never admits the negative it was derived from."""
    return math.ceil(x * 1e4) / 1e4


def wilson(k: int, n: int) -> tuple[float, float]:
    """95 % Wilson interval for k/n."""
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


def p_fire(output: np.ndarray) -> float:
    """Softmax index 1 of the two logits ``[no_fire, fire]`` (``classifier_fire_probability``)."""
    v = np.asarray(output, dtype=np.float64).reshape(-1)
    e = np.exp(v - v.max())
    return float(e[1] / e.sum())


def score(infer, size: int, rows: list[dict], root: Path) -> np.ndarray:
    """P(fire) per manifest row."""
    out = []
    for r in rows:
        frame = cv2.imread(str(root / r["path"]))
        if frame is None:
            raise FileNotFoundError(r["path"])
        out.append(p_fire(infer(preprocess(frame, size))))
    return np.array(out)


def ort_infer(model: Path, threads: int):
    """ONNX Runtime CPU session -> (infer, input size)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    return (lambda b: sess.run(None, {inp.name: b})[0]), int(inp.shape[2])


def ov_infer(model: Path, threads: int, device: str, precision: str):
    """Compile an IR with OpenVINO -> (infer, input size); CPU always gets ``threads``, ``precision`` may be ``default``."""
    import openvino as ov

    config: dict = {"PERFORMANCE_HINT": "LATENCY"}
    if device == "CPU":
        config["INFERENCE_NUM_THREADS"] = threads
    if precision != "default":
        config["INFERENCE_PRECISION_HINT"] = precision
    compiled = ov.Core().compile_model(str(model), device, config)
    req = compiled.create_infer_request()
    size = int(compiled.input(0).get_partial_shape()[2].get_length())
    return (lambda b: req.infer({0: b})[compiled.output(0)]), size


def safe_point(s: np.ndarray, neg: np.ndarray) -> float:
    """Lowest 4-decimal threshold above every negative score."""
    return ceil4(float(np.nextafter(s[neg].max(), 1.0)))


def main() -> None:
    """Score the reference and every IR variant and print the Markdown tables."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path, help="FP32 ONNX reference")
    ap.add_argument("--ir-dir", type=Path, required=True, help="directory with <variant>.xml from convert.py")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="directory for per-image scores (outside the repo)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="CPU", help="OpenVINO device for the IR variants (CPU or GPU)")
    ap.add_argument("--precision", choices=("default", "f32", "f16"), default="f32", help="INFERENCE_PRECISION_HINT")
    args = ap.parse_args()
    rows = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    cat = np.array([r["cat"] for r in rows]).astype(str)
    pos = np.array([r["label"] == "yes" for r in rows])
    neg = ~pos
    head = pos & ~np.char.startswith(cat, "synth")
    nodf = ~np.isin(cat, DFIRE_CATS)

    runs = {"onnx-ort": ort_infer(args.onnx, args.threads)}
    for v in VARIANTS:
        xml = args.ir_dir / f"{v}.xml"
        if xml.exists():
            runs[f"ir-{v}"] = ov_infer(xml, args.threads, args.device, args.precision)
    scores = {name: score(infer, size, rows, root) for name, (infer, size) in runs.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, s in scores.items():
        with (args.out / f"{args.onnx.stem}__{name}__{args.device}-{args.precision}.jsonl").open("w") as f:
            for r, p in zip(rows, s, strict=True):
                f.write(json.dumps({**r, "p_fire": float(p)}) + "\n")

    ref = scores["onnx-ort"]
    ref_thr = safe_point(ref, neg)
    ref_hits = int((ref[head] >= ref_thr).sum())
    n_head, n_neg = int(head.sum()), int(neg.sum())
    print(f"\n### {args.onnx.stem}, IR on {args.device} precision {args.precision}")
    print(f"\nreference threshold {ref_thr:.4f}, {len(rows)} frames\n")
    print(
        "| variant | own safe thr | recall non-synth @own | leak-free (own thr w/o D-Fire) | recall @ref thr | FP @ref thr "
        "| d hits vs ref @own | max abs dP(fire) | max abs dP fire frames |"
    )
    print("|---|---:|---|---|---|---:|---:|---:|---:|")
    for name, s in scores.items():
        thr = safe_point(s, neg)
        thr_nd = safe_point(s, neg & nodf)
        hits = int((s[head] >= thr).sum())
        leak_free = fmt(int((s[head & nodf] >= thr_nd).sum()), int((head & nodf).sum()))
        print(
            f"| {name} | {thr:.4f} | {fmt(hits, n_head)} | {leak_free} "
            f"| {fmt(int((s[head] >= ref_thr).sum()), n_head)} | {int((s[neg] >= ref_thr).sum())}/{n_neg} "
            f"| {hits - ref_hits:+d} | {np.abs(s - ref).max():.2e} | {np.abs(s - ref)[pos].max():.2e} |"
        )


if __name__ == "__main__":
    main()
