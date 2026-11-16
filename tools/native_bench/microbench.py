"""Per-stage cost of the hot paths on real frames of a clip (#133). Prints one JSON object.

Every number is the median of ``--n`` repetitions in ms (wall time of the call, so OpenCV's own thread pool is
included; ``--threads 1`` pins OpenCV to one thread to get the CPU cost). The detector stages replicate
``CPUFireDetector._detect_logic`` call by call, so the sum can be compared with the real method (the difference is
the Python/interpreter overhead, i.e. the only part a native rewrite of the *glue* could remove).
"""

import argparse
import asyncio
import json
import statistics
import time

import cv2
import numpy as np

LOWER = np.array([18, 50, 50], dtype="uint8")
UPPER = np.array([35, 255, 255], dtype="uint8")


def med(fn, n: int) -> float:
    fn()  # warm-up
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return round(1000 * statistics.median(ts), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="resize frames first (e.g. 0.25) to see the resolution effect")
    args = ap.parse_args()
    if args.threads:
        cv2.setNumThreads(args.threads)

    cap = cv2.VideoCapture(args.clip, cv2.CAP_FFMPEG)
    frames = []
    t0 = time.perf_counter()
    for _ in range(60):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    decode_ms = round(1000 * (time.perf_counter() - t0) / len(frames), 3)
    if args.scale != 1.0:
        frames = [cv2.resize(f, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA) for f in frames]
    a, b = frames[10], frames[40]  # two frames ~1 s apart at 30 fps
    h, w = a.shape[:2]
    out: dict = {"clip": args.clip, "res": f"{w}x{h}", "threads": cv2.getNumThreads(), "decode_ms_per_frame": decode_ms}

    # --- JPEG compression of one frame (the MemoryBufferStrategy cost, paid on EVERY stream frame) ---
    jpeg = {}
    for q in (95, 90, 80, 70):
        params = [cv2.IMWRITE_JPEG_QUALITY, q]
        _, enc = cv2.imencode(".jpg", a, params)
        jpeg[f"q{q}"] = {"ms": med(lambda p=params: cv2.imencode(".jpg", a, p), args.n), "kb": round(len(enc) / 1024, 1)}
    out["jpeg_default_is_q95"] = jpeg
    out["frame_copy_ms"] = med(lambda: a.copy(), args.n)

    # --- Detector stages (the exact call sequence of CPUFireDetector._detect_logic) ---
    blur = cv2.GaussianBlur(a, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    color = cv2.inRange(hsv, LOWER, UPPER)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ga, gb)
    _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    final = cv2.bitwise_and(color, motion)
    stages = {
        "gaussian_blur_21": med(lambda: cv2.GaussianBlur(a, (21, 21), 0), args.n),
        "cvt_bgr2hsv": med(lambda: cv2.cvtColor(blur, cv2.COLOR_BGR2HSV), args.n),
        "in_range": med(lambda: cv2.inRange(hsv, LOWER, UPPER), args.n),
        "cvt_bgr2gray": med(lambda: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), args.n),
        "absdiff": med(lambda: cv2.absdiff(ga, gb), args.n),
        "threshold": med(lambda: cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY), args.n),
        "gray_copy": med(lambda: ga.copy(), args.n),
        "bitwise_and_masks": med(lambda: cv2.bitwise_and(color, motion), args.n),
        "count_nonzero": med(lambda: cv2.countNonZero(final), args.n),
        "annotated_bitwise_and_frame": med(lambda: cv2.bitwise_and(a, a, mask=final), args.n),
    }
    out["detector_stages_ms"] = stages
    out["detector_stage_sum_ms"] = round(sum(stages.values()), 3)

    from is_goat_burning.fire_detection.detectors import CPUFireDetector

    det = CPUFireDetector(margin=int(w * h * 0.1), lower=LOWER, upper=UPPER)
    det._detect_logic(a)
    seq = [a, b]
    state = {"i": 0}

    def real() -> None:
        det._detect_logic(seq[state["i"] % 2])
        state["i"] += 1

    out["detector_real_ms"] = med(real, args.n)

    # --- Cost of the asyncio hop (to_thread) per check, i.e. the pure-Python orchestration overhead ---
    async def hop(n: int) -> float:
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            await asyncio.to_thread(lambda: None)
            ts.append(time.perf_counter() - t)
        return round(1000 * statistics.median(ts), 4)

    out["to_thread_noop_ms"] = asyncio.run(hop(200))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
