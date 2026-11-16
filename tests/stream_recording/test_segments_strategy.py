"""Tests for the 'segments' buffer strategy (ffmpeg stream-copy ring) and the sampled-decode streamer path."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from is_goat_burning.stream import VideoStreamer
from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver
from is_goat_burning.stream_recording.strategies import SegmentsBufferStrategy

URL = "https://example.invalid/live.m3u8"


@pytest.fixture
def ring(tmp_path: Path, mocker: MockerFixture) -> Path:
    ring_dir = tmp_path / "ring"
    ring_dir.mkdir()
    mocker.patch(
        "is_goat_burning.stream_recording.strategies.settings",
        **{
            "video.segments_directory": str(ring_dir),
            "video.segment_seconds": 2,
            "video.memory_buffer_seconds": 10,
            "video.record_during_fire": False,
        },
    )
    return ring_dir


@pytest.fixture
def context() -> MagicMock:
    ctx = MagicMock(spec=AsyncVideoChunkSaver)
    ctx.source_url = URL
    ctx.FILENAME_PREFIX = "goat-cam_"
    ctx.FILENAME_SUFFIX = ".mp4"
    ctx.chunk_length_seconds = 1
    ctx.chunks_to_keep_after_fire = 1
    ctx.signal_handler = MagicMock()
    ctx.signal_handler.is_fire_extinguished.return_value = True
    return ctx


def _segment(ring: Path, name: str, age_s: float) -> Path:
    path = ring / name
    path.write_bytes(b"x")
    import time

    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def test_ffmpeg_command_stream_copies_into_a_wrapping_ring(ring: Path, context: MagicMock) -> None:
    cmd = SegmentsBufferStrategy(context).ffmpeg_command()

    assert cmd[cmd.index("-i") + 1] == URL
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert cmd[cmd.index("-segment_wrap") + 1] == str(10 // 2 + SegmentsBufferStrategy.RING_EXTRA_SEGMENTS)
    assert cmd[-1] == str(ring / "seg%03d.ts")


@pytest.mark.usefixtures("ring")
def test_add_frame_and_reset_are_noops(context: MagicMock) -> None:
    strategy = SegmentsBufferStrategy(context)
    strategy.add_frame(np.zeros((2, 2, 3), dtype=np.uint8))
    strategy.reset()
    context.frame_queue.put_nowait.assert_not_called()


def test_complete_segments_skip_the_one_being_written_and_are_ordered(ring: Path, context: MagicMock) -> None:
    _segment(ring, "seg001.ts", 3)
    _segment(ring, "seg000.ts", 5)
    _segment(ring, "seg002.ts", 0.1)  # newest = still being written

    names = [p.name for _, p in SegmentsBufferStrategy(context)._complete_segments()]

    assert names == ["seg000.ts", "seg001.ts"]


def test_select_since_limits_pre_fire_window_and_resumes_after_mtime(ring: Path, context: MagicMock) -> None:
    _segment(ring, "seg000.ts", 100)  # older than the 10 s window (+ one segment)
    _segment(ring, "seg001.ts", 8)
    _segment(ring, "seg002.ts", 4)
    _segment(ring, "seg003.ts", 0.1)
    strategy = SegmentsBufferStrategy(context)

    window = strategy._select_since(None)
    resumed = strategy._select_since(window[0][0])

    assert [p.name for _, p in window] == ["seg001.ts", "seg002.ts"]
    assert [p.name for _, p in resumed] == ["seg002.ts"]


@pytest.mark.asyncio
async def test_handle_fire_event_writes_pre_and_post_fire_clips(
    ring: Path, context: MagicMock, tmp_path: Path, mocker: MockerFixture
) -> None:
    _segment(ring, "seg000.ts", 6)
    _segment(ring, "seg001.ts", 3)
    _segment(ring, "seg002.ts", 0.1)
    strategy = SegmentsBufferStrategy(context)
    clips: list[tuple[list[str], str]] = []

    def fake_concat(paths: list[Path], out: str) -> bool:
        clips.append(([p.name.split("_", 1)[1] for p in paths], out))
        return True

    mocker.patch.object(SegmentsBufferStrategy, "_concat_blocking", staticmethod(fake_concat))
    mocker.patch.object(SegmentsBufferStrategy, "COLLECT_INTERVAL_SECONDS", 0.01)
    context.chunks_to_keep_after_fire = 0  # no tail: the extinguished fire ends the recording at once
    context._publish_chunk_path = MagicMock()
    event_dir = tmp_path / "event"
    event_dir.mkdir()

    async def new_segment_appears(*_: object) -> None:
        _segment(ring, "seg003.ts", 0.05)

    mocker.patch("is_goat_burning.stream_recording.strategies.asyncio.sleep", AsyncMock(side_effect=new_segment_appears))
    mocker.patch.object(type(strategy), "_select_since", wraps=strategy._select_since)
    await strategy.handle_fire_event(str(event_dir))

    assert clips[0][0] == ["seg000.ts", "seg001.ts"]
    assert "pre-fire-buffer" in clips[0][1]
    assert context._publish_chunk_path.call_count >= 1
    assert not (event_dir / ".parts").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_concat_blocking_joins_real_segments(tmp_path: Path) -> None:
    parts = []
    for i in range(2):
        seg = tmp_path / f"s{i}.ts"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10", "-t", "1"]
            + ["-c:v", "libx264", "-g", "5", "-f", "mpegts", str(seg)],
            check=True,
        )
        parts.append(seg)
    out = tmp_path / "out.mp4"

    assert SegmentsBufferStrategy._concat_blocking(parts, str(out)) is True
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert float(probe) == pytest.approx(2.0, abs=0.3)


def test_concat_failure_returns_false(tmp_path: Path) -> None:
    missing = tmp_path / "nope.ts"
    assert SegmentsBufferStrategy._concat_blocking([missing], str(tmp_path / "o.mp4")) is False


# --- sampled decode in the streamer ---------------------------------------------------------------------------


def _streamer(skips: bool) -> VideoStreamer:
    streamer = VideoStreamer.__new__(VideoStreamer)
    streamer.cap = MagicMock()
    streamer.skips_decode = skips
    streamer._running = True
    streamer.backend = "cpu"
    streamer.video_saver = MagicMock()
    return streamer


@pytest.mark.asyncio
async def test_sampled_frames_decode_only_every_nth() -> None:
    streamer = _streamer(skips=True)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    grabs = {"n": 0}

    def grab() -> bool:
        grabs["n"] += 1
        return grabs["n"] <= 9

    streamer.cap.grab.side_effect = grab
    streamer.cap.retrieve.return_value = (True, frame)

    got = [f async for f in streamer.frames(every=3)]

    assert len(got) == 3  # grabs 1, 4, 7 decoded
    assert streamer.cap.retrieve.call_count == 3
    streamer.video_saver.assert_not_called()


@pytest.mark.asyncio
async def test_frames_ignore_every_when_decode_is_needed() -> None:
    streamer = _streamer(skips=False)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    streamer.cap.read.side_effect = [(True, frame), (True, frame), (False, None)]

    got = [f async for f in streamer.frames(every=5)]

    assert len(got) == 2
    assert streamer.video_saver.call_count == 2
    streamer.cap.grab.assert_not_called()


def test_saver_selects_segments_strategy(mocker: MockerFixture, tmp_path: Path) -> None:
    mocker.patch("is_goat_burning.stream_recording.save_stream_to_file.settings", **{"video.buffer_mode": "segments"})
    mocker.patch(
        "is_goat_burning.stream_recording.strategies.settings",
        **{"video.segments_directory": str(tmp_path / "r"), "video.segment_seconds": 2, "video.memory_buffer_seconds": 10},
    )

    saver = AsyncVideoChunkSaver(
        enabled=True,
        output_dir=str(tmp_path / "out"),
        chunk_length_seconds=5,
        max_chunks=1,
        chunks_to_keep_after_fire=1,
        source_url=URL,
    )

    assert isinstance(saver.strategy, SegmentsBufferStrategy)
    assert inspect.iscoroutinefunction(saver.strategy.handle_fire_event)
