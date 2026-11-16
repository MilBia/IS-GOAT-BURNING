"""Correctness + speed harness for fused-kernel prototypes of the classic detector's per-check hot path (#133).

The reference is the exact OpenCV call sequence of ``CPUFireDetector._detect_logic`` *after* the Gaussian blur
(blur stays in OpenCV in every prototype; it is the largest single stage and is a separable convolution that
OpenCV already vectorises):

    hsv        = cvtColor(blurred, BGR2HSV); warm_mask = inRange(hsv, lower, upper)
    gray       = cvtColor(frame, BGR2GRAY);  motion    = threshold(absdiff(gray, prev_gray), thr, 255, BINARY)
    moving_warm = countNonZero(warm_mask & motion);  warm = countNonZero(warm_mask);  prev_gray[:] = gray

A candidate is any importable callable ``fused(blurred, frame, prev_gray, lower, upper, thr) -> (moving_warm, warm)``
that must update ``prev_gray`` in place with the current frame's gray image. Inputs are C-contiguous uint8 arrays:
``blurred``/``frame`` (H, W, 3) BGR, ``prev_gray`` (H, W), ``lower``/``upper`` (3,), ``thr`` int.
Gray must match OpenCV 5's BGR2GRAY bit-exactly: (b*3735 + g*19235 + r*9798 + 16384) >> 15,
HSV must match OpenCV's 8-bit BGR2HSV (H in 0..179).

    python -m tools.native_bench.proto_bench clip.mp4 --module my_ext --func fused
    python -m tools.native_bench.proto_bench clip.mp4            # reference only (shows the OpenCV baseline)
"""

import argparse
import importlib
import json
import statistics
import time

import cv2
import numpy as np

LOWER = np.array([18, 50, 50], dtype="uint8")
UPPER = np.array([35, 255, 255], dtype="uint8")
THR = 25


def reference(blurred, frame, prev_gray, lower, upper, thr):
    """The OpenCV call sequence of the detector (post-blur part)."""
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    warm_mask = cv2.inRange(hsv, lower, upper)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, motion = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    moving = cv2.bitwise_and(warm_mask, motion)
    prev_gray[:] = gray
    return cv2.countNonZero(moving), cv2.countNonZero(warm_mask)


def load_pairs(path: str, count: int = 12, step: int = 15) -> list[tuple[np.ndarray, np.ndarray]]:
    """(frame_t, frame_t+step) pairs from the clip plus random/saturated synthetic pairs for edge cases."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    frames = []
    for i in range(count * step + 1):
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
    pairs = list(zip(frames[:-1], frames[1:], strict=True))
    rng = np.random.default_rng(0)
    h, w = frames[0].shape[:2]
    for _ in range(3):  # uniformly random pixels: exercises every HSV branch (tie cases v==r==g etc.)
        pairs.append((rng.integers(0, 256, (h, w, 3), dtype=np.uint8), rng.integers(0, 256, (h, w, 3), dtype=np.uint8)))
    base = rng.integers(0, 4, (h, w, 3), dtype=np.uint8) * 85  # values in {0,85,170,255}: many equal-channel ties
    pairs.append((base, np.roll(base, 1, axis=0)))
    return pairs


def check(fn, pairs) -> dict:
    bad = 0
    worst = 0
    for a, b in pairs:
        blurred = cv2.GaussianBlur(b, (21, 21), 0)
        p_ref = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        p_new = p_ref.copy()
        r = reference(blurred, b, p_ref, LOWER, UPPER, THR)
        c = fn(blurred, b, p_new, LOWER, UPPER, THR)
        gray_ok = bool(np.array_equal(p_ref, p_new))
        c = (int(c[0]), int(c[1]))
        worst = max(worst, abs(c[0] - r[0]), abs(c[1] - r[1]))
        if c != r or not gray_ok:
            bad += 1
    return {"pairs": len(pairs), "mismatching_pairs": bad, "max_abs_count_error": worst}


def timeit(fn, pairs, n: int) -> float:
    a, b = pairs[3]
    blurred = cv2.GaussianBlur(b, (21, 21), 0)
    prev = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    fn(blurred, b, prev, LOWER, UPPER, THR)
    ts = []
    for _ in range(n):
        prev[:] = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        fn(blurred, b, prev, LOWER, UPPER, THR)
        ts.append(time.perf_counter() - t0)
    return round(1000 * statistics.median(ts), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--module")
    ap.add_argument("--func", default="fused")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    pairs = load_pairs(args.clip)
    out: dict = {"label": args.label or args.module or "reference", "res": "x".join(map(str, pairs[0][0].shape[1::-1]))}
    cv2.setNumThreads(1)  # single-thread comparison = CPU cost
    out["opencv_1thread_ms"] = timeit(reference, pairs, args.n)
    cv2.setNumThreads(0)
    cv2.setNumThreads(cv2.getNumberOfCPUs())
    out["opencv_allthreads_ms"] = timeit(reference, pairs, args.n)
    if args.module:
        fn = getattr(importlib.import_module(args.module), args.func)
        out["correctness"] = check(fn, pairs)
        out["candidate_ms"] = timeit(fn, pairs, args.n)
        out["speedup_vs_opencv_1thread"] = round(out["opencv_1thread_ms"] / out["candidate_ms"], 2)
        out["speedup_vs_opencv_allthreads"] = round(out["opencv_allthreads_ms"] / out["candidate_ms"], 2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
