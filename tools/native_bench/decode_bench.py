"""CPU cost of decoding a clip: ``read()`` on every frame (what ``VideoStreamer.frames`` does) vs ``grab()`` on
every frame and ``retrieve()`` only on the frames that are analysed (#133). Prints one JSON line per clip.

CPU time is process-wide (all FFmpeg worker threads), per frame, so the numbers compare directly across machines.
"""

import argparse
import json
import resource
import time

import cv2


def cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def run(path: str, mode: str, every: int, max_frames: int) -> dict:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    for _ in range(30):  # warm-up: open, thread pool, first GOP
        cap.read()
    n = 0
    c0, t0 = cpu(), time.perf_counter()
    while n < max_frames:
        if mode == "read":
            ok, _ = cap.read()
        else:
            ok = cap.grab()
            if ok and n % every == 0:
                cap.retrieve()
        if not ok:
            break
        n += 1
    c1, t1 = cpu(), time.perf_counter()
    cap.release()
    return {
        "clip": path.rsplit("/", 1)[-1],
        "res": f"{w}x{h}",
        "fps": fps,
        "mode": mode,
        "frames": n,
        "cpu_ms_per_frame": round(1000 * (c1 - c0) / n, 3),
        "wall_ms_per_frame": round(1000 * (t1 - t0) / n, 3),
        "cpu_cores_at_native_fps": round((c1 - c0) / n * fps, 3),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument(
        "--every", type=int, default=30, help="retrieve every Nth frame in grab mode (checks_per_second=1 at 30 fps)"
    )
    a = ap.parse_args()
    for p in a.clips:
        for m in ("read", "grab"):
            print(json.dumps(run(p, m, a.every, a.frames)), flush=True)
