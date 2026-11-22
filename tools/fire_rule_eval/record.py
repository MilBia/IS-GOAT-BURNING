"""Records clips from the live goat webcam at intervals and optionally extracts their features (#109).

Each clip gets a freshly resolved stream URL (resolved HLS URLs expire) and is stream-copied with ``ffmpeg``
(native fps, no re-encode), so both ``extract`` (1 Hz checks) and ``flicker`` (~30 fps) can use it. Neither the
source URL nor the resolved URL is ever logged: they carry tokens.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from is_goat_burning.stream import _URL_PATTERN
from is_goat_burning.stream import YouTubeStream
from tools.fire_rule_eval.features import ExtractOptions
from tools.fire_rule_eval.features import extract_features
from tools.fire_rule_eval.features import iter_check_frames
from tools.fire_rule_eval.flicker import extract_flicker_features
from tools.fire_rule_eval.flicker import iter_frames_30fps

logger = logging.getLogger(__name__)

FFMPEG_TIMEOUT_MARGIN_S = 120


def _resolve(source: str) -> str:
    """Resolves a fresh stream URL with the app's resolver, keeping its URL-bearing log lines quiet."""
    stream_logger = logging.getLogger("Stream")  # logs the source URL at INFO/DEBUG
    previous = stream_logger.level
    stream_logger.setLevel(logging.WARNING)
    try:
        return asyncio.run(YouTubeStream(source).resolve_url())
    finally:
        stream_logger.setLevel(previous)


def _record_clip(url: str, target: Path, seconds: float) -> bool:
    """Stream-copies ``seconds`` of ``url`` into ``target``; removes a partial file on failure or Ctrl+C."""
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", url, "-t", str(seconds), "-c", "copy", str(target)]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace", timeout=seconds + FFMPEG_TIMEOUT_MARGIN_S
        )
    except subprocess.TimeoutExpired:
        logger.error("%s: ffmpeg timed out", target.name)
        target.unlink(missing_ok=True)
        return False
    except BaseException:  # Ctrl+C: the killed ffmpeg leaves an unreadable mp4 (no moov atom)
        target.unlink(missing_ok=True)
        raise
    if result.returncode != 0 or not target.exists() or not target.stat().st_size:
        lines = (result.stderr or "").strip().splitlines()
        detail = _URL_PATTERN.sub("<url>", lines[-1]) if lines else ""
        logger.error("%s: ffmpeg failed (exit %s) %s", target.name, result.returncode, detail)
        target.unlink(missing_ok=True)
        return False
    return True


def _append_to_manifest(manifest_path: Path, entry: dict[str, str]) -> None:
    """Appends one clip entry (``scan`` format), writing aside and renaming so Ctrl+C never truncates it."""
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"clips": []}
    manifest["clips"].append(entry)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_path)


def _save(features: dict[str, np.ndarray], target: Path) -> int:
    """Stores features atomically; returns the row count (0 means nothing usable was decoded)."""
    rows = len(features["t"])
    if rows:
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **features)
        temporary.replace(target)
    return rows


def _feature_steps(video: Path, name: str, features_dir: Path | None, flicker_dir: Path | None) -> bool:
    """Runs the requested ``extract`` / ``flicker`` steps for one clip; True if every step stored rows."""
    ok = True
    steps = (
        (features_dir, "extract", lambda: extract_features(iter_check_frames(video, ExtractOptions()))),
        (flicker_dir, "flicker", lambda: extract_flicker_features(iter_frames_30fps(video))),
    )
    for out_dir, step, compute in steps:
        if out_dir is None:
            continue
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = _save(compute(), out_dir / f"{name}.npz")
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the recorder
            logger.error("%s: %s failed: %s: %s", name, step, type(exc).__name__, exc)
            ok = False
            continue
        if not rows:
            logger.error("%s: %s produced no rows", name, step)
            ok = False
        else:
            logger.info("%s: %s stored %d rows", name, step, rows)
    return ok


def _one_clip(args: argparse.Namespace) -> None:
    """Resolves, records, registers and (optionally) extracts one clip; failures are logged, never raised."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    video = Path(args.out) / f"{args.group}_{stamp}.mp4"
    try:
        url = _resolve(args.source)
    except Exception as exc:  # noqa: BLE001 - the message may carry the URL: log the class only
        logger.error("%s: stream URL could not be resolved (%s)", video.name, type(exc).__name__)
        return
    logger.info("%s: recording %s s", video.name, args.clip_seconds)
    if not _record_clip(url, video, args.clip_seconds):
        return
    name = f"{args.group}__{video.stem}"
    _append_to_manifest(Path(args.manifest), {"name": name, "path": str(video), "label": args.label, "group": args.group})
    logger.info("%s: recorded, added to %s", video, args.manifest)
    features_dir = Path(args.features) if args.features else None
    flicker_dir = Path(args.flicker) if args.flicker else None
    if not _feature_steps(video, name, features_dir, flicker_dir):
        logger.error("%s: kept (feature step failed)", video.name)
    elif args.delete_video:
        video.unlink()
        logger.info("%s: deleted (features stored)", video.name)


def cmd_record(args: argparse.Namespace) -> int:
    """Records ``--count`` clips (or until Ctrl+C), one every ``--interval-seconds`` (start to start)."""
    if args.delete_video and not (args.features or args.flicker):
        logger.error("--delete-video needs --features and/or --flicker (the video would be the only copy)")
        return 2
    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg not found on PATH")
        return 1
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    attempt = 0
    try:
        while args.count is None or attempt < args.count:
            started = time.monotonic()
            attempt += 1
            try:
                _one_clip(args)
            except Exception as exc:  # noqa: BLE001 - keep recording; the message may carry the URL: class only
                logger.error("attempt %d failed (%s)", attempt, type(exc).__name__)
            if args.count is not None and attempt >= args.count:
                break
            time.sleep(max(0.0, args.interval_seconds - (time.monotonic() - started)))
    except KeyboardInterrupt:
        logger.info("interrupted, stopping after %d attempt(s)", attempt)
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Registers the ``record`` subcommand."""
    record = sub.add_parser("record", help="record clips from the live stream at intervals (#109)")
    record.add_argument("--source", required=True, help="YouTube URL of the live stream")
    record.add_argument("--out", required=True, help="directory for the recorded videos")
    record.add_argument("--manifest", required=True)
    record.add_argument("--clip-seconds", type=float, default=600.0)
    record.add_argument("--interval-seconds", type=float, default=3600.0, help="start to start; default 3600")
    record.add_argument(
        "--count", type=int, default=None, help="clips to attempt, failed ones included (default: until Ctrl+C)"
    )
    record.add_argument("--label", default="neg", choices=["pos", "neg", "neg_hard"])
    record.add_argument("--group", default="goat")
    record.add_argument("--features", help="run `extract` for each new clip into this directory")
    record.add_argument("--flicker", help="run `flicker` for each new clip into this directory")
    record.add_argument("--delete-video", action="store_true", help="delete each clip once its feature steps succeeded")
    record.set_defaults(func=cmd_record)
