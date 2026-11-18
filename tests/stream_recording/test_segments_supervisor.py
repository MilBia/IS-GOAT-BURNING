"""Tests for the ffmpeg supervisor of the 'segments' strategy, run against a fake process (no real ffmpeg)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import logging
import os
from pathlib import Path
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from is_goat_burning.stream_recording import strategies
from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver
from is_goat_burning.stream_recording.strategies import SegmentsBufferStrategy

URL = "https://example.invalid/live.m3u8"
MODULE = "is_goat_burning.stream_recording.strategies"


class FakeProcess:
    """Stands in for `asyncio.subprocess.Process`; `wait()` returns the next exit code (or blocks forever)."""

    def __init__(self, codes: list[int] | None = None, on_wait: object = None, stderr: bytes = b"") -> None:
        self.returncode: int | None = None
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()
        self._codes = codes
        self._on_wait = on_wait
        self.exited = asyncio.Event()
        self.terminate = MagicMock()
        self.kill = MagicMock()

    async def wait(self) -> int:
        if callable(self._on_wait):
            self._on_wait()
        if self._codes is None:
            await self.exited.wait()  # runs until cancelled or `terminate()` is called
            return self.returncode or 0
        code = self._codes.pop(0)
        self.returncode = code
        return code


@pytest.fixture
def ring(tmp_path: Path, mocker: MockerFixture) -> Path:
    ring_dir = tmp_path / "ring"
    ring_dir.mkdir()
    mocker.patch(
        f"{MODULE}.settings",
        **{
            "video.segments_directory": str(ring_dir),
            "video.segment_seconds": 2,
            "video.memory_buffer_seconds": 10,
            "video.record_during_fire": False,
        },
    )
    return ring_dir


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Captures the strategy's logger (it does not propagate, so the handler is attached directly)."""
    strategies_logger = logging.getLogger(strategies.logger.name)
    strategies_logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger=strategies_logger.name)
    yield caplog
    strategies_logger.removeHandler(caplog.handler)


@pytest.fixture
def context() -> MagicMock:
    ctx = MagicMock(spec=AsyncVideoChunkSaver)
    ctx.source_url = URL
    ctx.FILENAME_PREFIX = "goat-cam_"
    ctx.FILENAME_SUFFIX = ".mp4"
    ctx.chunk_length_seconds = 1
    ctx.chunks_to_keep_after_fire = 0
    ctx.signal_handler = MagicMock()
    ctx.signal_handler.is_fire_extinguished.return_value = True
    return ctx


@pytest.fixture
def strategy(ring: Path, context: MagicMock) -> SegmentsBufferStrategy:
    assert ring.exists()
    return SegmentsBufferStrategy(context)


def _segment(ring: Path, name: str, age_s: float) -> Path:
    path = ring / name
    path.write_bytes(b"x")
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def _fake_sleep(mocker: MockerFixture, limit: int) -> list[float]:
    """Patches the supervisor's sleep to record delays and cancel the supervisor after `limit` sleeps."""
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= limit:
            raise asyncio.CancelledError

    mocker.patch(f"{MODULE}.asyncio.sleep", sleep)
    return delays


@pytest.mark.asyncio
async def test_restart_uses_exponential_backoff_capped_at_max(strategy: SegmentsBufferStrategy, mocker: MockerFixture) -> None:
    mocker.patch(f"{MODULE}.asyncio.create_subprocess_exec", AsyncMock(side_effect=lambda *_a, **_k: FakeProcess(codes=[1])))
    mocker.patch.object(SegmentsBufferStrategy, "STABLE_RUN_SECONDS", 1e9)
    delays = _fake_sleep(mocker, limit=7)

    with pytest.raises(asyncio.CancelledError):
        await strategy._supervise_ffmpeg()

    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


@pytest.mark.asyncio
async def test_backoff_resets_after_a_stable_run(strategy: SegmentsBufferStrategy, mocker: MockerFixture) -> None:
    mocker.patch(f"{MODULE}.asyncio.create_subprocess_exec", AsyncMock(side_effect=lambda *_a, **_k: FakeProcess(codes=[1])))
    mocker.patch.object(SegmentsBufferStrategy, "STABLE_RUN_SECONDS", 0.0)  # every run counts as stable
    delays = _fake_sleep(mocker, limit=4)

    with pytest.raises(asyncio.CancelledError):
        await strategy._supervise_ffmpeg()

    assert delays == [1.0, 1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_start_failure_is_retried_with_backoff(strategy: SegmentsBufferStrategy, mocker: MockerFixture) -> None:
    spawn = AsyncMock(side_effect=OSError("no ffmpeg"))
    mocker.patch(f"{MODULE}.asyncio.create_subprocess_exec", spawn)
    mocker.patch.object(SegmentsBufferStrategy, "STABLE_RUN_SECONDS", 1e9)
    drop = mocker.patch.object(SegmentsBufferStrategy, "_drop_truncated_segment")
    delays = _fake_sleep(mocker, limit=3)

    with pytest.raises(asyncio.CancelledError):
        await strategy._supervise_ffmpeg()

    assert spawn.await_count == 3
    assert delays == [1.0, 2.0, 4.0]
    drop.assert_not_called()


@pytest.mark.asyncio
async def test_exit_drops_only_the_segment_cut_by_the_run(
    strategy: SegmentsBufferStrategy, ring: Path, mocker: MockerFixture
) -> None:
    _segment(ring, "seg000.ts", 100)  # complete, from before this run

    def write_segments() -> None:
        _segment(ring, "seg001.ts", 1)
        _segment(ring, "seg002.ts", 0)  # newest: was being written when ffmpeg died

    mocker.patch(
        f"{MODULE}.asyncio.create_subprocess_exec",
        AsyncMock(side_effect=lambda *_a, **_k: FakeProcess(codes=[1], on_wait=write_segments)),
    )
    _fake_sleep(mocker, limit=1)

    with pytest.raises(asyncio.CancelledError):
        await strategy._supervise_ffmpeg()

    assert sorted(p.name for p in ring.glob("seg*.ts")) == ["seg000.ts", "seg001.ts"]


@pytest.mark.asyncio
async def test_run_continues_slot_numbering_and_records_its_start(
    strategy: SegmentsBufferStrategy, ring: Path, mocker: MockerFixture
) -> None:
    _segment(ring, "seg002.ts", 3)
    spawn = AsyncMock(side_effect=lambda *_a, **_k: FakeProcess(codes=[0]))
    mocker.patch(f"{MODULE}.asyncio.create_subprocess_exec", spawn)
    _fake_sleep(mocker, limit=1)
    before = time.time()

    with pytest.raises(asyncio.CancelledError):
        await strategy._supervise_ffmpeg()

    cmd = list(spawn.await_args.args)
    assert cmd[cmd.index("-segment_start_number") + 1] == "3"
    assert strategy._run_started is not None
    assert strategy._run_started >= before


@pytest.mark.asyncio
async def test_stop_terminates_a_running_process_without_killing_it(strategy: SegmentsBufferStrategy) -> None:
    proc = FakeProcess(codes=[0])
    strategy._ffmpeg = proc  # type: ignore[assignment]

    await strategy._stop_ffmpeg()

    proc.terminate.assert_called_once()
    proc.kill.assert_not_called()


@pytest.mark.asyncio
async def test_stop_kills_after_terminate_times_out(
    strategy: SegmentsBufferStrategy,
) -> None:
    proc = FakeProcess()
    proc.wait = AsyncMock(side_effect=[TimeoutError, 0])  # type: ignore[method-assign]
    strategy._ffmpeg = proc  # type: ignore[assignment]
    order: list[str] = []
    proc.terminate.side_effect = lambda: order.append("terminate")
    proc.kill.side_effect = lambda: order.append("kill")

    await strategy._stop_ffmpeg()

    assert order == ["terminate", "kill"]


@pytest.mark.asyncio
async def test_stop_leaves_an_already_exited_process_alone(strategy: SegmentsBufferStrategy) -> None:
    proc = FakeProcess(codes=[0])
    proc.returncode = 0
    strategy._ffmpeg = proc  # type: ignore[assignment]

    await strategy._stop_ffmpeg()

    proc.terminate.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_supervisor_leaves_no_orphaned_child(strategy: SegmentsBufferStrategy, mocker: MockerFixture) -> None:
    """Cancelling while ffmpeg runs must still let `_stop_ffmpeg` terminate it (the child was stored before awaiting).

    The narrower window (cancelled between `create_subprocess_exec` returning and the process being stored) does not
    exist: the assignment follows the await with no suspension point in between, and a cancellation *inside*
    `create_subprocess_exec` is handled by asyncio itself, which closes the half-started transport.
    """
    proc = FakeProcess()  # wait() blocks until cancelled
    proc.terminate.side_effect = lambda: (setattr(proc, "returncode", -15), proc.exited.set())
    mocker.patch(f"{MODULE}.asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
    strategy._supervisor = asyncio.create_task(strategy._supervise_ffmpeg())
    async with asyncio.timeout(5):
        while strategy._ffmpeg is None:
            await asyncio.sleep(0.01)  # let the supervisor reach `wait()`
    assert strategy._ffmpeg is proc

    await strategy._stop_ffmpeg()

    assert strategy._supervisor.done()
    proc.terminate.assert_called_once()


SECRET = "SECRETTOKEN123"


@pytest.mark.asyncio
async def test_exit_logs_last_stderr_lines_with_urls_masked(
    strategy: SegmentsBufferStrategy, mocker: MockerFixture, logs: pytest.LogCaptureFixture
) -> None:
    lines = [f"noise {i}" for i in range(30)]
    lines.append(f"Error opening input: Server returned 403 Forbidden for https://rr1.example.invalid/v?sig={SECRET}&x=1")
    lines.append("x" * 5000)
    proc = FakeProcess(codes=[1], stderr=("\n".join(lines) + "\n").encode())
    mocker.patch(f"{MODULE}.asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
    _fake_sleep(mocker, limit=1)

    with pytest.raises(asyncio.CancelledError):
        await strategy._supervise_ffmpeg()

    text = logs.text
    assert "403 Forbidden" in text
    assert "<url>" in text
    assert SECRET not in text
    assert "noise 0" not in text  # only the last lines are kept
    assert "x" * 1000 not in text  # long lines are truncated


@pytest.mark.asyncio
async def test_stderr_is_piped_and_a_large_flood_keeps_only_the_last_lines(
    strategy: SegmentsBufferStrategy, mocker: MockerFixture
) -> None:
    flood = "".join(f"line {i:06d}\n" for i in range(20000)).encode()  # ~260 KB, far beyond a pipe buffer
    assert len(flood) > 64 * 1024
    spawn = AsyncMock(side_effect=lambda *_a, **_k: FakeProcess(codes=[0], stderr=flood))
    mocker.patch(f"{MODULE}.asyncio.create_subprocess_exec", spawn)
    _fake_sleep(mocker, limit=1)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(strategy._supervise_ffmpeg(), timeout=10)

    assert spawn.await_args.kwargs["stderr"] == asyncio.subprocess.PIPE
    assert list(strategy._stderr_tail) == [f"line {i:06d}" for i in range(20000 - 10, 20000)]


@pytest.mark.parametrize(
    "line",
    ["open HTTPS://h.invalid/p?token=SECRETTOKEN123 failed", "connect tcp://host.invalid:443?x=SECRETTOKEN123 refused"],
)
def test_url_masking_is_case_insensitive_and_scheme_agnostic(strategy: SegmentsBufferStrategy, line: str) -> None:
    masked = strategy._sanitize_stderr_line(line.encode())

    assert "SECRETTOKEN123" not in masked
    assert "<url>" in masked


@pytest.mark.asyncio
async def test_stderr_reader_is_cancelled_on_stop(strategy: SegmentsBufferStrategy, mocker: MockerFixture) -> None:
    proc = FakeProcess()
    proc.stderr = asyncio.StreamReader()  # never reaches EOF
    proc.terminate.side_effect = lambda: (setattr(proc, "returncode", -15), proc.exited.set())
    mocker.patch(f"{MODULE}.asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
    strategy._supervisor = asyncio.create_task(strategy._supervise_ffmpeg())
    async with asyncio.timeout(5):
        while strategy._stderr_reader is None:
            await asyncio.sleep(0.01)
    reader = strategy._stderr_reader

    await strategy._stop_ffmpeg()

    assert reader.done()


@pytest.mark.asyncio
async def test_missing_ffprobe_warns_once_at_startup(
    strategy: SegmentsBufferStrategy, context: MagicMock, mocker: MockerFixture, logs: pytest.LogCaptureFixture
) -> None:
    mocker.patch(f"{MODULE}.shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    mocker.patch.object(SegmentsBufferStrategy, "_supervise_ffmpeg", AsyncMock())
    context.signal_handler.fire_detected_event.wait = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await strategy._run_main_loop()

    warnings = [r for r in logs.records if "ffprobe" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelname == "WARNING"


def test_failed_concat_removes_the_partial_output(tmp_path: Path, mocker: MockerFixture) -> None:
    out = tmp_path / "clip.mp4"

    def fake_run(*_a: object, **_k: object) -> MagicMock:
        out.write_bytes(b"partial")
        return MagicMock(returncode=1, stderr=b"boom")

    mocker.patch(f"{MODULE}.subprocess.run", fake_run)
    part = tmp_path / "a.ts"
    part.write_bytes(b"x")

    assert SegmentsBufferStrategy._concat_blocking([part], str(out)) is False
    assert not out.exists()


def _capture_concat(mocker: MockerFixture) -> None:
    mocker.patch.object(SegmentsBufferStrategy, "_concat_blocking", staticmethod(lambda _paths, _out: True))


@pytest.mark.asyncio
async def test_cancelled_fire_event_cleans_up_staging(
    strategy: SegmentsBufferStrategy, ring: Path, context: MagicMock, tmp_path: Path, mocker: MockerFixture
) -> None:
    _segment(ring, "seg000.ts", 6)
    _segment(ring, "seg001.ts", 3)
    _segment(ring, "seg002.ts", 0.1)
    context._publish_chunk_path = MagicMock()
    _capture_concat(mocker)
    event_dir = tmp_path / "event"
    event_dir.mkdir()
    entered = asyncio.Event()

    async def sleep(_delay: float) -> None:
        entered.set()
        await asyncio.Event().wait()  # cancelled while collecting post-fire segments

    mocker.patch(f"{MODULE}.asyncio.sleep", sleep)
    task = asyncio.create_task(strategy.handle_fire_event(str(event_dir)))
    await asyncio.wait_for(entered.wait(), timeout=5)
    assert (event_dir / ".parts").exists() or context._publish_chunk_path.called

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not (event_dir / ".parts").exists()


@pytest.mark.asyncio
async def test_publishing_a_clip_survives_a_full_event_queue() -> None:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    queue.put_nowait("already-queued")
    saver = AsyncVideoChunkSaver(
        enabled=False,
        output_dir="x",
        chunk_length_seconds=1,
        max_chunks=1,
        chunks_to_keep_after_fire=1,
        event_queue=queue,
    )

    saver._publish_chunk_path("clip.mp4")  # must not raise

    assert queue.qsize() == 1
