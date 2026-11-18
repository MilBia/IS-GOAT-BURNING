"""Tests for the ffmpeg supervisor of the 'segments' strategy, run against a fake process (no real ffmpeg)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver
from is_goat_burning.stream_recording.strategies import SegmentsBufferStrategy

URL = "https://example.invalid/live.m3u8"
MODULE = "is_goat_burning.stream_recording.strategies"


class FakeProcess:
    """Stands in for `asyncio.subprocess.Process`; `wait()` returns the next exit code (or blocks forever)."""

    def __init__(self, codes: list[int] | None = None, on_wait: object = None) -> None:
        self.returncode: int | None = None
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
