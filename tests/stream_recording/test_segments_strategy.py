"""Tests for the 'segments' buffer strategy (ffmpeg stream-copy ring) and the sampled-decode streamer path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


def _encode(path: Path, size: str, seconds: float, age_s: float) -> None:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10", "-t", str(seconds)]
        + ["-c:v", "mpeg4", "-g", "5", "-f", "mpegts", str(path)],
        check=True,
    )
    import time

    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))


def _has_mpeg4_encoder() -> bool:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False).stdout
    return " mpeg4 " in encoders


@pytest.mark.skipif(not _has_mpeg4_encoder(), reason="ffmpeg/ffprobe with the mpeg4 encoder not installed")
def test_concat_blocking_joins_real_segments(tmp_path: Path) -> None:
    parts = []
    for i in range(2):
        seg = tmp_path / f"s{i}.ts"
        _encode(seg, "64x48", 1, 0)
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_concat_failure_returns_false(tmp_path: Path) -> None:
    missing = tmp_path / "nope.ts"
    assert SegmentsBufferStrategy._concat_blocking([missing], str(tmp_path / "o.mp4")) is False


# --- sampled decode in the streamer ---------------------------------------------------------------------------


def _streamer(skips: bool) -> VideoStreamer:
    streamer = VideoStreamer.__new__(VideoStreamer)
    streamer.cap = MagicMock()
    streamer._executor = ThreadPoolExecutor(max_workers=1)  # built by __init__, which this helper bypasses
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


def test_prepare_ring_warns_when_tmpfs_is_too_small(ring: Path, context: MagicMock, mocker: MockerFixture) -> None:
    mocker.patch("is_goat_burning.stream_recording.strategies.shutil.disk_usage", return_value=MagicMock(free=10 * 2**20))
    warn = mocker.patch("is_goat_burning.stream_recording.strategies.logger.warning")
    _segment(ring, "seg000.ts", 100)  # older than the pre-fire window

    SegmentsBufferStrategy(context)._prepare_ring()

    assert not list(ring.glob("seg*.ts"))  # stale segments are cleared
    warn.assert_called_once()
    assert "--shm-size" in warn.call_args.args[0]


# --- ring kept across reconnects (#160) -------------------------------------------------------------------------


def test_prepare_ring_keeps_young_segments_and_drops_old_or_out_of_range(ring: Path, context: MagicMock) -> None:
    _segment(ring, "seg000.ts", 100)  # older than the window: useless, freed at once
    _segment(ring, "seg001.ts", 6)  # from the previous connection, still inside the window: kept
    _segment(ring, "seg002.ts", 3)
    _segment(ring, "seg042.ts", 3)  # slot outside the current ring (other config): the ring bound would not hold
    _segment(ring, "seg_x.ts", 3)  # not a ring slot at all

    SegmentsBufferStrategy(context)._prepare_ring()

    assert sorted(p.name for p in ring.glob("seg*.ts")) == ["seg001.ts", "seg002.ts"]


def test_next_segment_number_continues_after_the_newest_slot(ring: Path, context: MagicMock) -> None:
    strategy = SegmentsBufferStrategy(context)  # ring of 10 // 2 + 4 = 9 slots
    assert strategy._next_segment_number() == 0  # empty ring

    _segment(ring, "seg003.ts", 6)
    _segment(ring, "seg004.ts", 3)  # newest: the previous run stopped here
    _segment(ring, "seg007.ts", 20)
    assert strategy._next_segment_number() == 5

    _segment(ring, "seg008.ts", 1)  # newest in the last slot: wrap around
    assert strategy._next_segment_number() == 0


@pytest.mark.usefixtures("ring")
def test_ffmpeg_command_starts_numbering_where_the_previous_run_stopped(context: MagicMock) -> None:
    cmd = SegmentsBufferStrategy(context).ffmpeg_command(start_number=5)

    assert cmd[cmd.index("-segment_start_number") + 1] == "5"
    assert cmd[cmd.index("-segment_wrap") + 1] == "9"


def test_newest_segment_of_a_previous_run_is_complete(ring: Path, context: MagicMock) -> None:
    import time

    _segment(ring, "seg000.ts", 6)
    _segment(ring, "seg001.ts", 3)  # last segment of the previous connection, finished when its ffmpeg stopped
    strategy = SegmentsBufferStrategy(context)
    strategy._run_started = time.time() - 1  # new ffmpeg started, has not written a segment yet

    assert [p.name for _, p in strategy._complete_segments()] == ["seg000.ts", "seg001.ts"]

    _segment(ring, "seg002.ts", 0.1)  # the new run's segment being written is still left out
    assert [p.name for _, p in strategy._complete_segments()] == ["seg000.ts", "seg001.ts"]


def _simulate_run(ring: Path, strategy: SegmentsBufferStrategy, segments: int, start_age: float) -> None:
    """Writes `segments` slots the way the segment muxer would, starting at the strategy's next number."""
    number = strategy._next_segment_number()
    for i in range(segments):
        _segment(ring, f"seg{(number + i) % strategy.ring_size:03d}.ts", start_age - 2 * i)


def test_ring_stays_bounded_across_many_reconnects(ring: Path, context: MagicMock) -> None:
    age = 1000.0
    for _ in range(6):  # six reconnects, each run writing more than a full ring
        strategy = SegmentsBufferStrategy(context)
        strategy._prepare_ring()
        _simulate_run(ring, strategy, segments=13, start_age=age)
        age -= 30

    files = list(ring.glob("seg*.ts"))
    assert len(files) <= strategy.ring_size
    assert all(p.name in {f"seg{i:03d}.ts" for i in range(strategy.ring_size)} for p in files)


@pytest.mark.asyncio
async def test_pre_fire_clip_spans_a_reconnect(ring: Path, context: MagicMock, tmp_path: Path, mocker: MockerFixture) -> None:
    import time

    # Old connection wrote seg003..seg005, then the URL expired; the new run continued at seg006.
    _segment(ring, "seg003.ts", 9)
    _segment(ring, "seg004.ts", 7)
    _segment(ring, "seg005.ts", 5)
    strategy = SegmentsBufferStrategy(context)
    strategy._run_started = time.time() - 3
    _segment(ring, "seg006.ts", 1)
    _segment(ring, "seg007.ts", 0.1)  # being written
    clips = _capture_clips(mocker, results=[True])
    mocker.patch.object(SegmentsBufferStrategy, "_probe_blocking", staticmethod(lambda _: "h264,High,1920,1080,yuv420p"))

    await strategy.handle_fire_event(str(_event_dir(tmp_path, context)))

    assert clips[0] == ["seg003.ts", "seg004.ts", "seg005.ts", "seg006.ts"]


@pytest.mark.asyncio
async def test_fire_right_after_reconnect_uses_previous_run(
    ring: Path, context: MagicMock, tmp_path: Path, mocker: MockerFixture
) -> None:
    import time

    _segment(ring, "seg003.ts", 6)
    _segment(ring, "seg004.ts", 4)
    strategy = SegmentsBufferStrategy(context)
    strategy._run_started = time.time() - 1  # no segment from the new connection yet
    clips = _capture_clips(mocker, results=[True])
    probe = MagicMock()
    mocker.patch.object(SegmentsBufferStrategy, "_probe_blocking", staticmethod(probe))

    await strategy.handle_fire_event(str(_event_dir(tmp_path, context)))

    assert clips[0] == ["seg003.ts", "seg004.ts"]
    probe.assert_not_called()  # nothing to compare against


@pytest.mark.asyncio
async def test_incompatible_previous_run_falls_back_to_current_segments(
    ring: Path, context: MagicMock, tmp_path: Path, mocker: MockerFixture
) -> None:
    import time

    _segment(ring, "seg003.ts", 9)
    _segment(ring, "seg004.ts", 7)
    strategy = SegmentsBufferStrategy(context)
    strategy._run_started = time.time() - 5
    _segment(ring, "seg005.ts", 3)
    _segment(ring, "seg006.ts", 1)
    _segment(ring, "seg007.ts", 0.1)
    clips = _capture_clips(mocker, results=[True])

    def probe(path: Path) -> str:
        return "h264,High,1280,720,yuv420p" if path.name.endswith(("seg003.ts", "seg004.ts")) else "h264,High,1920,1080,yuv420p"

    mocker.patch.object(SegmentsBufferStrategy, "_probe_blocking", staticmethod(probe))

    await strategy.handle_fire_event(str(_event_dir(tmp_path, context)))

    assert clips == [["seg005.ts", "seg006.ts"]]


@pytest.mark.asyncio
async def test_failed_probe_falls_back_to_current_segments(
    ring: Path, context: MagicMock, tmp_path: Path, mocker: MockerFixture
) -> None:
    import time

    _segment(ring, "seg003.ts", 9)
    strategy = SegmentsBufferStrategy(context)
    strategy._run_started = time.time() - 5
    _segment(ring, "seg004.ts", 3)
    _segment(ring, "seg005.ts", 0.1)
    clips = _capture_clips(mocker, results=[True])
    mocker.patch.object(SegmentsBufferStrategy, "_probe_blocking", staticmethod(lambda _: None))

    await strategy.handle_fire_event(str(_event_dir(tmp_path, context)))

    assert clips == [["seg004.ts"]]


@pytest.mark.asyncio
async def test_failed_concat_across_reconnect_retries_with_current_segments(
    ring: Path, context: MagicMock, tmp_path: Path, mocker: MockerFixture
) -> None:
    import time

    _segment(ring, "seg003.ts", 9)
    strategy = SegmentsBufferStrategy(context)
    strategy._run_started = time.time() - 5
    _segment(ring, "seg004.ts", 3)
    _segment(ring, "seg005.ts", 0.1)
    clips = _capture_clips(mocker, results=[False, True])
    mocker.patch.object(SegmentsBufferStrategy, "_probe_blocking", staticmethod(lambda _: "same"))

    await strategy.handle_fire_event(str(_event_dir(tmp_path, context)))

    assert clips == [["seg003.ts", "seg004.ts"], ["seg004.ts"]]
    assert context._publish_chunk_path.call_count == 1


@pytest.mark.asyncio
async def test_stop_drops_the_segment_cut_by_a_kill(ring: Path, context: MagicMock) -> None:
    _segment(ring, "seg000.ts", 6)
    _segment(ring, "seg001.ts", 0.1)
    strategy = SegmentsBufferStrategy(context)
    proc = MagicMock()
    proc.returncode = None
    proc.wait = AsyncMock(side_effect=[TimeoutError, 0])
    strategy._ffmpeg = proc

    await strategy._stop_ffmpeg()

    proc.kill.assert_called_once()
    assert [p.name for p in ring.glob("seg*.ts")] == ["seg000.ts"]


def _capture_clips(mocker: MockerFixture, results: list[bool]) -> list[list[str]]:
    """Patches the concat to record the ring names of each clip's parts; the pre-fire clip is the first one."""
    clips: list[list[str]] = []
    outcomes = iter(results)

    def fake_concat(paths: list[Path], _out: str) -> bool:
        clips.append([p.name.split("_", 1)[1] for p in paths])
        return next(outcomes, True)

    mocker.patch.object(SegmentsBufferStrategy, "_concat_blocking", staticmethod(fake_concat))
    mocker.patch("is_goat_burning.stream_recording.strategies.asyncio.sleep", AsyncMock())
    return clips


def _event_dir(tmp_path: Path, context: MagicMock) -> Path:
    context.chunks_to_keep_after_fire = 0  # the extinguished fire ends the post-fire recording at once
    context._publish_chunk_path = MagicMock()
    event_dir = tmp_path / "event"
    event_dir.mkdir()
    return event_dir


@pytest.mark.skipif(not _has_mpeg4_encoder(), reason="ffmpeg/ffprobe with the mpeg4 encoder not installed")
@pytest.mark.parametrize(("old_size", "expected_seconds"), [("64x48", 3.0), ("96x64", 1.0)])
@pytest.mark.asyncio
async def test_real_pre_fire_clip_across_reconnect(
    ring: Path,
    context: MagicMock,
    tmp_path: Path,
    mocker: MockerFixture,
    old_size: str,
    expected_seconds: float,
) -> None:
    import time

    _encode(ring / "seg003.ts", old_size, 1, 8)
    _encode(ring / "seg004.ts", old_size, 1, 6)
    strategy = SegmentsBufferStrategy(context)
    strategy._run_started = time.time() - 4
    _encode(ring / "seg005.ts", "64x48", 1, 2)
    _encode(ring / "seg006.ts", "64x48", 1, 0.1)
    mocker.patch("is_goat_burning.stream_recording.strategies.asyncio.sleep", AsyncMock())
    event_dir = _event_dir(tmp_path, context)

    await strategy.handle_fire_event(str(event_dir))

    (clip,) = event_dir.glob("*pre-fire-buffer*.mp4")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert float(probe) == pytest.approx(expected_seconds, abs=0.3)


def test_drop_truncated_segment_removes_only_the_newest(ring: Path, context: MagicMock) -> None:
    _segment(ring, "seg000.ts", 6)
    _segment(ring, "seg001.ts", 3)
    _segment(ring, "seg002.ts", 0.1)

    SegmentsBufferStrategy(context)._drop_truncated_segment()

    assert sorted(p.name for p in ring.glob("seg*.ts")) == ["seg000.ts", "seg001.ts"]


def test_drop_truncated_segment_spares_segments_of_earlier_runs(ring: Path, context: MagicMock) -> None:
    import time

    # The expired URL: each restart fails at input open without writing a segment; the earlier ones are complete.
    _segment(ring, "seg000.ts", 6)
    _segment(ring, "seg001.ts", 3)
    strategy = SegmentsBufferStrategy(context)
    strategy._run_started = time.time() - 1

    strategy._drop_truncated_segment()

    assert sorted(p.name for p in ring.glob("seg*.ts")) == ["seg000.ts", "seg001.ts"]


def _running_ffmpeg(strategy: SegmentsBufferStrategy) -> None:
    import time

    strategy._run_started = time.time() - 30
    strategy._ffmpeg = MagicMock(returncode=None)


@pytest.mark.asyncio
async def test_final_post_fire_clip_includes_the_segment_open_at_the_deadline(
    ring: Path, context: MagicMock, tmp_path: Path, mocker: MockerFixture
) -> None:
    _segment(ring, "seg000.ts", 6)
    _segment(ring, "seg001.ts", 3)
    _segment(ring, "seg002.ts", 0.1)  # open when the recording ends
    strategy = SegmentsBufferStrategy(context)
    _running_ffmpeg(strategy)
    clips = _capture_clips(mocker, results=[True, True])

    sleeps = 0

    async def next_segment_starts_after_the_deadline(*_: object) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:  # the first sleep is the collect interval; the second is the wait for the open segment
            _segment(ring, "seg003.ts", 0.0)

    mocker.patch(
        "is_goat_burning.stream_recording.strategies.asyncio.sleep",
        AsyncMock(side_effect=next_segment_starts_after_the_deadline),
    )

    await strategy.handle_fire_event(str(_event_dir(tmp_path, context)))

    assert clips[0] == ["seg000.ts", "seg001.ts"]
    assert clips[-1] == ["seg002.ts"]


@pytest.mark.asyncio
async def test_final_wait_is_bounded_when_the_open_segment_never_closes(
    ring: Path, context: MagicMock, tmp_path: Path, mocker: MockerFixture
) -> None:
    _segment(ring, "seg000.ts", 3)
    _segment(ring, "seg001.ts", 0.1)
    strategy = SegmentsBufferStrategy(context)
    _running_ffmpeg(strategy)
    mocker.patch.object(SegmentsBufferStrategy, "LAST_SEGMENT_MARGIN_SECONDS", 0.05)
    mocker.patch.object(SegmentsBufferStrategy, "LAST_SEGMENT_POLL_SECONDS", 0.01)
    mocker.patch.object(strategy, "segment_seconds", 0)
    clips = _capture_clips(mocker, results=[True])
    import asyncio

    real_sleep = asyncio.sleep
    mocker.patch("is_goat_burning.stream_recording.strategies.asyncio.sleep", lambda _s: real_sleep(0.01))

    await strategy.handle_fire_event(str(_event_dir(tmp_path, context)))

    assert clips == [["seg000.ts"]]  # only the pre-fire clip; nothing new to write after the deadline


@pytest.mark.asyncio
async def test_no_final_segment_when_ffmpeg_is_not_running(
    ring: Path, context: MagicMock, tmp_path: Path, mocker: MockerFixture
) -> None:
    _segment(ring, "seg000.ts", 3)
    _segment(ring, "seg001.ts", 0.1)
    strategy = SegmentsBufferStrategy(context)
    clips = _capture_clips(mocker, results=[True])
    wait = mocker.patch.object(strategy, "_await_open_segment_closed", wraps=strategy._await_open_segment_closed)

    await strategy.handle_fire_event(str(_event_dir(tmp_path, context)))

    wait.assert_awaited()
    assert clips == [["seg000.ts"]]
