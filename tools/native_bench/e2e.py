"""Real-time end-to-end profile of the production pipeline on a local clip (#133).

Runs the real ``StreamFireDetector`` / ``VideoStreamer`` / ``AsyncVideoChunkSaver`` / detector code, but feeds it
from a local file through a paced ``cv2.VideoCapture`` wrapper (one frame per ``1/fps`` s, looping the clip), so the
CPU, RAM and event-loop lag numbers are what the live stream would cost. Prints one JSON line.

Configuration comes from the normal environment (``CHECKS_PER_SECOND``, ``VIDEO__*``, ``DETECTION_STRATEGY``...),
so the same script measures every mode:

    SOURCE=mock://x CHECKS_PER_SECOND=1 python -m tools.native_bench.e2e clip.mp4 --seconds 90
    ... VIDEO__SAVE_VIDEO_CHUNKS=true VIDEO__VIDEO_OUTPUT_DIRECTORY=/tmp/v VIDEO__BUFFER_MODE=memory ...
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import resource
import time

import cv2


class PacedCapture:
    """Wraps ``cv2.VideoCapture``: paces ``read()`` to real time, loops the clip, ends after ``seconds``."""

    def __init__(self, path: str, seconds: float, fps_override: float | None = None) -> None:
        self.cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        self.fps = fps_override or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.seconds = seconds
        self.start = 0.0
        self.n = 0
        self.decode_s = 0.0
        self.late_s = 0.0

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return self.cap.get(prop)

    def isOpened(self) -> bool:  # noqa: N802
        return self.cap.isOpened()

    def release(self) -> None:
        self.cap.release()

    def read(self):
        if self.start == 0.0:
            self.start = time.monotonic()
        if time.monotonic() - self.start >= self.seconds:
            return False, None
        due = self.start + self.n / self.fps
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self.late_s += -delay
        t0 = time.perf_counter()
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        self.decode_s += time.perf_counter() - t0
        self.n += 1
        return ok, frame


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


def thread_cpu() -> dict[str, float]:
    """CPU seconds per thread name (/proc/self/task), to attribute decode vs python vs worker threads."""
    out: dict[str, float] = {}
    tick = os.sysconf("SC_CLK_TCK")
    for tid in os.listdir("/proc/self/task"):
        try:
            stat = Path(f"/proc/self/task/{tid}/stat").read_text()
            comm = stat[stat.index("(") + 1 : stat.rindex(")")]
            fields = stat[stat.rindex(")") + 2 :].split()
            out[comm] = out.get(comm, 0.0) + (int(fields[11]) + int(fields[12])) / tick
        except OSError, ValueError:
            continue
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--fire-at", type=float, default=None, help="signal a fire event after N seconds")
    ap.add_argument("--fps", type=float, default=None, help="override the clip fps (pacing)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    from is_goat_burning.config import settings
    from is_goat_burning.fire_detection.detector_core import DEFAULT_LOWER_HSV_FIRE
    from is_goat_burning.fire_detection.detector_core import DEFAULT_UPPER_HSV_FIRE
    from is_goat_burning.fire_detection.detector_core import StreamFireDetector
    from is_goat_burning.fire_detection.detectors import create_fire_detector
    from is_goat_burning.stream import VideoStreamer

    async def noop() -> None:
        return None

    cap = PacedCapture(args.clip, args.seconds, args.fps)
    det = StreamFireDetector(noop, False, settings.checks_per_second)
    det.stream = VideoStreamer(url=args.clip, cap=cap)  # type: ignore[arg-type]
    w, h = det.stream.frame_shape
    det.fire_detector = create_fire_detector(
        margin=int(w * h * settings.fire_detection_threshold),
        lower=DEFAULT_LOWER_HSV_FIRE,
        upper=DEFAULT_UPPER_HSV_FIRE,
        strategy=settings.detection_strategy,
    )

    lag_max = 0.0
    lag_sum = 0.0
    lag_n = 0
    rss_series: list[float] = []

    async def ticker() -> None:
        nonlocal lag_max, lag_sum, lag_n
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(0.01)
            lag = time.perf_counter() - t0 - 0.01
            lag_max = max(lag_max, lag)
            lag_sum += lag
            lag_n += 1

    async def sampler() -> None:
        while True:
            await asyncio.sleep(5)
            rss_series.append(round(rss_mb(), 1))

    async def fire() -> None:
        if args.fire_at is not None:
            await asyncio.sleep(args.fire_at)
            det.signal_handler.fire_detected()

    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.monotonic()
    th0 = thread_cpu()
    bg = [asyncio.create_task(c()) for c in (ticker, sampler, fire)]
    await det()
    wall = time.monotonic() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    th1 = thread_cpu()
    for t in bg:
        t.cancel()
    cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    result = {
        "label": args.label,
        "clip": Path(args.clip).name,
        "fps": cap.fps,
        "frames": cap.n,
        "wall_s": round(wall, 1),
        "cpu_cores_avg": round(cpu / wall, 3),
        "cpu_user_s": round(ru1.ru_utime - ru0.ru_utime, 1),
        "cpu_sys_s": round(ru1.ru_stime - ru0.ru_stime, 1),
        "peak_rss_mb": round(ru1.ru_maxrss / 1024, 1),
        "rss_series_mb": rss_series,
        "decode_ms_per_frame": round(1000 * cap.decode_s / max(cap.n, 1), 2),
        "late_pacing_s": round(cap.late_s, 2),
        "loop_lag_ms_max": round(1000 * lag_max, 1),
        "loop_lag_ms_mean": round(1000 * lag_sum / max(lag_n, 1), 2),
        "threads_cpu_s": {k: round(v - th0.get(k, 0.0), 1) for k, v in sorted(th1.items()) if v - th0.get(k, 0.0) >= 0.2},
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("VIDEO__", "CHECKS_PER", "DETECTION_STRATEGY", "FIRE_DETECTION"))
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
