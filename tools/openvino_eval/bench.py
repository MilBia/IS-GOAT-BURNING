"""Self-contained latency / memory benchmark of a classifier (IR or ONNX) on OpenVINO CPU or GPU (#152).

Standalone (only numpy, opencv-python-headless and openvino): copy this file next to the models and a folder of a few
JPEGs. Preprocessing is the project's ``edge_detector.preprocess`` (included). One model/device per process, so the
peak RSS (``ru_maxrss``) belongs to that run only. Memory the GPU driver maps outside the process heap may not be
counted, so GPU RSS can be understated. Measures load+compile time, then median / p95 / max latency of preprocess +
inference over ``--runs`` calls cycling through the images, and prints a JSON summary without paths or host names,
including the full-precision P(fire) per image for CPU-vs-GPU parity (compare two outputs with ``--compare``).

The GPU plugin computes in f16 by default whatever the IR weight precision; ``--precision f32`` forces f32. Unlike
``evaluate.py`` (default ``f32``), this bench defaults to ``--precision default``, i.e. the device's own choice (f32 on
the N100 CPU, f16 on the iGPU), because that is what a deployment would run without extra configuration.

``--compare`` checks that both outputs have the same input size, image count and (if present) image file names in the
same order; outputs written before names were stored only trigger a warning.

Example (in the container from ``Dockerfile.bench``):
    python bench.py --model ov/p16_224_minimal/fp16.xml --images jpgs --device GPU --runs 50
    python bench.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import cv2
import numpy as np

_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    """BGR -> RGB, squash-resize, 0-1, ImageNet mean/std, NCHW float32 (as in ``edge_detector.preprocess``)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    return np.ascontiguousarray((blob - _MEAN) / _STD, dtype=np.float32)


def p_fire(out: np.ndarray) -> float:
    """Softmax index 1 of the two logits ``[no_fire, fire]``."""
    v = np.asarray(out, dtype=np.float64).reshape(-1)
    e = np.exp(v - v.max())
    return float(e[1] / e.sum())


def compare(a: Path, b: Path) -> None:
    """Print max |dP(fire)| between two bench JSON outputs after checking they cover the same images."""
    da, db = (json.loads(p.read_text()) for p in (a, b))
    for key in ("input_size", "images"):
        if da[key] != db[key]:
            raise SystemExit(f"{key} differs: {da[key]} vs {db[key]}")
    if "image_names" in da and "image_names" in db:
        if da["image_names"] != db["image_names"]:
            raise SystemExit(f"image names differ: {da['image_names']} vs {db['image_names']}")
    else:
        print("warning: image_names missing in at least one output (old format); order not verified", file=sys.stderr)
    pa, pb = np.array(da["p_fire"]), np.array(db["p_fire"])
    print(json.dumps({"max_abs_dp_fire": float(np.abs(pa - pb).max()), "images": len(pa)}))


def main() -> None:
    """Run the benchmark and print the JSON summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--images", type=Path, help="folder with a few .jpg files")
    ap.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    ap.add_argument("--precision", choices=("default", "f32", "f16"), default="default", help="INFERENCE_PRECISION_HINT")
    ap.add_argument("--threads", type=int, default=4, help="CPU inference threads (and cv2 threads for preprocess)")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--compare", type=Path, nargs=2, metavar="JSON")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if not args.model or not args.images:
        ap.error("--model and --images are required")
    import openvino as ov

    cv2.setNumThreads(args.threads)
    loaded = [(p.name, cv2.imread(str(p))) for p in sorted(args.images.glob("*.jp*g"))]
    names = [n for n, f in loaded if f is not None]
    frames = [f for _, f in loaded if f is not None]
    if not frames:
        raise SystemExit("no readable JPEGs in --images")
    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    core = ov.Core()
    config: dict = {"PERFORMANCE_HINT": "LATENCY"}
    if args.device == "CPU":
        config |= {"INFERENCE_NUM_THREADS": args.threads, "NUM_STREAMS": 1}
    if args.precision != "default":
        config["INFERENCE_PRECISION_HINT"] = args.precision
    t0 = time.perf_counter()
    compiled = core.compile_model(str(args.model), args.device, config)
    load_s = time.perf_counter() - t0
    req = compiled.create_infer_request()
    out_port = compiled.output(0)
    size = int(compiled.input(0).get_partial_shape()[2].get_length())

    def infer(blob: np.ndarray) -> np.ndarray:
        return req.infer({0: blob})[out_port]

    probs = [p_fire(infer(preprocess(f, size))) for f in frames]
    for i in range(args.warmup):
        infer(preprocess(frames[i % len(frames)], size))
    times = []
    for i in range(args.runs):
        start = time.perf_counter()
        infer(preprocess(frames[i % len(frames)], size))
        times.append((time.perf_counter() - start) * 1000)
    weights = args.model.with_suffix(".bin") if args.model.suffix == ".xml" else args.model
    print(
        json.dumps(
            {
                "model_file": f"{args.model.parent.name}/{args.model.name}",
                "weights_mb": round(weights.stat().st_size / 1e6, 1),
                "device": args.device,
                "device_name": core.get_property(args.device, "FULL_DEVICE_NAME"),
                "precision_hint": str(compiled.get_property("INFERENCE_PRECISION_HINT")),
                "runtime": f"openvino {ov.__version__}",
                "threads": args.threads,
                "input_size": size,
                "images": len(frames),
                "image_names": names,
                "runs": args.runs,
                "load_s": round(load_s, 2),
                "median_ms": round(float(np.median(times)), 1),
                "p95_ms": round(float(np.percentile(times, 95)), 1),
                "max_ms": round(float(np.max(times)), 1),
                "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "rss_before_load_mib": round(rss_before_kb / 1024, 1),
                "p_fire": probs,
                "cpu": platform.processor() or platform.machine(),
                "cores": os.cpu_count(),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
