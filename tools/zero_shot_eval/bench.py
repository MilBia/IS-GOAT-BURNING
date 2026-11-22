"""Self-contained latency / memory benchmark of a classifier ONNX on one backend (#151).

Standalone (only numpy, opencv-python-headless and, for ``ort``, onnxruntime): copy this file next to the ONNX and a
folder of a few JPEGs to the target host. Preprocessing is the project's ``edge_detector.preprocess`` (included), the
input size is read from the model. One backend per process, so the peak RSS (``ru_maxrss``) belongs to that backend
only. Measures model load time, then median / p95 latency of preprocess + inference over ``--runs`` calls cycling
through the images. Prints a JSON summary without paths or host names. Does NOT reproduce concurrent stream decoding.

``cores`` is ``os.cpu_count()`` of the host (a container ``--cpus`` quota is not reflected).

Example (bare metal, or in the container from ``Dockerfile.bench``):
    python bench.py --model model.onnx --images jpgs/ --backend ort --threads 4 --runs 50
    python bench.py --model model.onnx --images jpgs/ --backend cv2 --threads 4 --runs 50

    docker build -f Dockerfile.bench -t zs-bench .          # in tools/zero_shot_eval
    docker run --rm --cpus 4 -v "$PWD/work:/work:ro" zs-bench --model /work/model.onnx --images /work/jpgs --backend ort
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
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
    """Softmax index 1 for two logits, sigmoid for one."""
    v = np.asarray(out, dtype=np.float64).reshape(-1)
    if v.size == 1:
        return float(1 / (1 + np.exp(-v[0])))
    e = np.exp(v - v.max())
    return float(e[1] / e.sum())


def main() -> None:
    """Run the benchmark and print the JSON summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="folder with a few .jpg files")
    ap.add_argument("--backend", choices=("ort", "cv2", "cv2-classic"), required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--size", type=int, default=0, help="cv2 only: model input side (default: probe 224/256/384/512)")
    args = ap.parse_args()
    frames = [cv2.imread(str(p)) for p in sorted(args.images.glob("*.jp*g"))]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise SystemExit("no readable JPEGs in --images")
    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    if args.backend == "ort":
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = args.threads
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(args.model), opts, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        size = int(inp.shape[2])

        def infer(blob: np.ndarray) -> np.ndarray:
            return sess.run(None, {inp.name: blob})[0]

        version = f"onnxruntime {ort.__version__}"
    else:
        cv2.setNumThreads(args.threads)
        if args.backend == "cv2-classic":
            net = cv2.dnn.readNetFromONNX(str(args.model), cv2.dnn.ENGINE_CLASSIC)
        else:
            net = cv2.dnn.readNetFromONNX(str(args.model))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        size = args.size

        def infer(blob: np.ndarray) -> np.ndarray:
            net.setInput(blob)
            return net.forward()

        version = f"opencv {cv2.__version__}"
    load_s = time.perf_counter() - t0
    if not size:
        size = _probe_size(infer)
    blobs = [preprocess(f, size) for f in frames]
    probs = [p_fire(infer(b)) for b in blobs]
    for i in range(args.warmup):
        infer(preprocess(frames[i % len(frames)], size))
    times = []
    for i in range(args.runs):
        start = time.perf_counter()
        infer(preprocess(frames[i % len(frames)], size))
        times.append((time.perf_counter() - start) * 1000)
    print(
        json.dumps(
            {
                "model_file": args.model.name,
                "model_mb": round(args.model.stat().st_size / 1e6, 1),
                "backend": args.backend,
                "runtime": version,
                "threads": args.threads,
                "input_size": size,
                "images": len(frames),
                "runs": args.runs,
                "load_s": round(load_s, 2),
                "median_ms": round(float(np.median(times)), 1),
                "p95_ms": round(float(np.percentile(times, 95)), 1),
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "rss_before_load_mb": round(rss_before_kb / 1024, 1),
                "p_fire": [round(p, 4) for p in probs],
                "cpu": platform.processor() or platform.machine(),
                "cores": os.cpu_count(),
            },
            indent=1,
        )
    )


def _probe_size(infer: object) -> int:
    """Find the square input size cv2.dnn accepts (it does not expose the declared input shape)."""
    for s in (224, 256, 384, 512):
        try:
            infer(np.zeros((1, 3, s, s), np.float32))
            return s
        except cv2.error:
            continue
    raise SystemExit("could not determine the model input size")


if __name__ == "__main__":
    main()
