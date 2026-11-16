"""Prototype of "archive without decoding, analyse separately" (#133, proposal 1). Prints one JSON line.

Archiving: one ``ffmpeg -c copy`` process writes the (paced, looped) source into a ring of short segments in a
tmpfs directory: no decode, no re-encode. Analysis: the real ``CPUFireDetector`` fed from a second capture that
``grab()``s every frame and ``retrieve()``s only the analysed ones (``CHECKS_PER_SECOND``). Total CPU and peak memory
come from the container's cgroup (so the ffmpeg child is included). After the run a "fire flush" is simulated:
the segments covering the last N seconds are concatenated with ``-c copy`` and probed.

    python -m tools.native_bench.split_e2e ARCHIVE_SRC ANALYSIS_SRC --seconds 90 --ring-dir /dev/shm/ring
"""

import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

SEGMENT_SECONDS = 2


def cgroup_cpu_s() -> float:
    for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
        if line.startswith("usage_usec"):
            return int(line.split()[1]) / 1e6
    return 0.0


def cgroup_peak_mb() -> float:
    try:
        return int(Path("/sys/fs/cgroup/memory.peak").read_text()) / 1e6
    except OSError:
        return -1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive_src")
    ap.add_argument("analysis_src")
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--buffer-seconds", type=int, default=60)
    ap.add_argument("--checks", type=float, default=1.0)
    ap.add_argument("--ring-dir", default="/dev/shm/ring")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    from is_goat_burning.fire_detection.detectors import CPUFireDetector

    ring = Path(args.ring_dir)
    ring.mkdir(parents=True, exist_ok=True)
    for old in ring.glob("*"):
        old.unlink()
    wrap = args.buffer_seconds // SEGMENT_SECONDS + 3
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", args.archive_src, "-map", "0:v:0", "-c", "copy"]
        + ["-f", "segment", "-segment_time", str(SEGMENT_SECONDS), "-segment_wrap", str(wrap), "-reset_timestamps", "1"]
        + ["-segment_format", "mpegts", str(ring / "seg%03d.ts")],
        stderr=subprocess.PIPE,
    )

    cap = cv2.VideoCapture(args.analysis_src, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    every = max(1, round(fps / args.checks))
    det = CPUFireDetector(
        margin=int(w * h * 0.1), lower=np.array([18, 50, 50], dtype="uint8"), upper=np.array([35, 255, 255], dtype="uint8")
    )

    cpu0, t0 = cgroup_cpu_s(), time.monotonic()
    n = checks = 0
    det_s = 0.0
    while time.monotonic() - t0 < args.seconds:
        due = t0 + n / fps  # pace the analysis capture to real time, as a live stream would
        if (delay := due - time.monotonic()) > 0:
            time.sleep(delay)
        if not cap.grab():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if n % every == 0:
            ok, frame = cap.retrieve()
            if ok:
                d0 = time.perf_counter()
                det._detect_logic(frame)
                det_s += time.perf_counter() - d0
                checks += 1
        n += 1
    wall = time.monotonic() - t0
    cpu = cgroup_cpu_s() - cpu0
    peak = cgroup_peak_mb()
    ring_bytes = sum(p.stat().st_size for p in ring.glob("seg*.ts"))
    segs = sorted(ring.glob("seg*.ts"), key=lambda p: p.stat().st_mtime)

    # Fire flush: concatenate the newest complete segments (skip the one being written) into an mp4 with -c copy.
    keep = segs[-1 - args.buffer_seconds // SEGMENT_SECONDS : -1]
    listing = ring / "list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in keep))
    out = ring / "pre_fire.mp4"
    f0 = time.perf_counter()
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)],
        check=True,
    )
    flush_s = time.perf_counter() - f0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,nb_frames", "-of", "json", str(out)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    ffmpeg.terminate()
    ffmpeg.wait(timeout=10)
    print(
        json.dumps(
            {
                "label": args.label,
                "archive_src": Path(args.archive_src).name,
                "analysis_src": Path(args.analysis_src).name,
                "analysis_res": f"{w}x{h}",
                "wall_s": round(wall, 1),
                "total_cpu_cores": round(cpu / wall, 3),
                "cgroup_peak_mb": round(peak, 1),
                "analysis_frames": n,
                "checks": checks,
                "detector_ms_mean": round(1000 * det_s / max(checks, 1), 1),
                "ring_mb": round(ring_bytes / 1e6, 1),
                "ring_segments": len(segs),
                "flush_s": round(flush_s, 2),
                "flush_probe": json.loads(probe) if probe else None,
            }
        )
    )


if __name__ == "__main__":
    main()
