"""Tests for the escalation log and its integration with HybridFireDetector."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from is_goat_burning.config import Settings
from is_goat_burning.fire_detection import escalation_log as el
from is_goat_burning.fire_detection.detectors import HybridFireDetector
from is_goat_burning.fire_detection.escalation_log import EscalationLog
from is_goat_burning.fire_detection.verifier_health import VerifierError

FRAME = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)


def _lines(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / el.INDEX_NAME).read_text().splitlines()]


def test_settings_default_disabled() -> None:
    cfg = Settings(_env_file=None, source="mock://source").escalation_log
    assert cfg.enabled is False
    assert cfg.max_mb == 2000
    assert cfg.jpeg_quality == 85


@pytest.mark.asyncio
async def test_records_frames_and_verdicts(tmp_path: Path) -> None:
    log = EscalationLog(tmp_path, max_mb=10)
    log.record(FRAME, "V", "fire", 0.91234, 12.34, policy=None)
    log.record(FRAME, "V", "error", None, 5.0, error_type="TimeoutError", policy="inconclusive")
    log.record_capped("V", "suppress")
    await log.aclose()

    lines = _lines(tmp_path)
    assert [x["verdict"] for x in lines] == ["fire", "error", "capped"]
    assert lines[0]["probability"] == 0.9123
    assert lines[1]["error_type"] == "TimeoutError"
    assert lines[2]["file"] is None
    assert len(list(tmp_path.glob("*.jpg"))) == 2
    assert all((tmp_path / x["file"]).is_file() for x in lines[:2])


@pytest.mark.asyncio
async def test_size_limit_deletes_oldest_jpegs_and_keeps_index(tmp_path: Path) -> None:
    log = EscalationLog(tmp_path, max_mb=1, jpeg_quality=100)
    for _ in range(30):
        log.record(FRAME, "V", "no_fire", None, 1.0)
        await log.aclose()
    log._max_bytes = 20_000
    log.record(FRAME, "V", "no_fire", None, 1.0)
    await log.aclose()

    total = sum(p.stat().st_size for p in tmp_path.iterdir())
    assert total <= 20_000
    assert _lines(tmp_path)  # index lines survive pruning of the JPEGs
    names = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert names and names[-1] == _lines(tmp_path)[-1]["file"]  # the newest frame is kept


@pytest.mark.asyncio
async def test_startup_scan_counts_existing_files(tmp_path: Path) -> None:
    old = tmp_path / "20200101T000000000000Z_0001_fire.jpg"
    old.write_bytes(b"x" * 5000)
    log = EscalationLog(tmp_path, max_mb=1)
    log._max_bytes = 3000
    log.record_capped("V", None)
    await log.aclose()
    assert not old.exists()


@pytest.mark.asyncio
async def test_index_rotation(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.object(el, "MAX_INDEX_BYTES", 300)
    log = EscalationLog(tmp_path, max_mb=10)
    for _ in range(6):
        log.record_capped("V", None)
        await log.aclose()
    assert (tmp_path / el.ROTATED_NAME).is_file()
    assert (tmp_path / el.INDEX_NAME).stat().st_size <= 600


@pytest.mark.asyncio
async def test_write_failure_never_raises_and_is_rate_limited(tmp_path: Path, mocker: MockerFixture) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log = EscalationLog(blocker / "sub", max_mb=1)  # cannot be created
    warn = mocker.patch.object(el.logger, "warning")
    for _ in range(3):
        log.record(FRAME, "V", "fire", None, 1.0)
        await log.aclose()
    assert warn.call_count == 1


@pytest.mark.asyncio
async def test_backlog_is_bounded(tmp_path: Path, mocker: MockerFixture) -> None:
    log = EscalationLog(tmp_path, max_mb=1)
    gate = asyncio.Event()

    async def slow(*_a: object, **_k: object) -> None:
        await gate.wait()

    mocker.patch.object(log, "_run", new=lambda *_: slow())
    for _ in range(el.MAX_IN_FLIGHT + 3):
        log.record(FRAME, "V", "fire", None, 1.0)
    assert len(log._pending) == el.MAX_IN_FLIGHT
    gate.set()
    await log.aclose()


@pytest.mark.asyncio
async def test_record_does_not_wait_for_the_write(tmp_path: Path) -> None:
    log = EscalationLog(tmp_path, max_mb=1)
    log.record(FRAME, "V", "fire", None, 1.0)
    assert not (tmp_path / el.INDEX_NAME).exists()  # scheduled, not yet written
    await log.aclose()
    assert (tmp_path / el.INDEX_NAME).exists()


def _hybrid(log: EscalationLog, verify: AsyncMock, **kwargs: object) -> HybridFireDetector:
    local = MagicMock()
    local.detect = AsyncMock(return_value=(True, FRAME))
    verifier = MagicMock()
    verifier.verify = verify
    verifier.last_probability = 0.7
    return HybridFireDetector(local, verifier, escalation_log=log, **kwargs)


@pytest.mark.asyncio
async def test_hybrid_records_one_entry_per_call_not_per_cooldown_reuse(tmp_path: Path) -> None:
    log = EscalationLog(tmp_path, max_mb=10)
    detector = _hybrid(log, AsyncMock(return_value=True), cooldown_seconds=3600, max_calls_per_hour=0)
    for _ in range(3):
        await detector.detect(FRAME)
    await log.aclose()
    lines = _lines(tmp_path)
    assert len(lines) == 1
    assert lines[0]["verdict"] == "fire"
    assert lines[0]["probability"] == 0.7
    assert lines[0]["verifier"] == "MagicMock"


@pytest.mark.asyncio
async def test_hybrid_records_error_without_message_and_cap(tmp_path: Path) -> None:
    log = EscalationLog(tmp_path, max_mb=10)
    err = VerifierError("secret-key-123")
    err.__cause__ = TimeoutError("secret-key-123")
    detector = _hybrid(
        log, AsyncMock(side_effect=[err, True]), cooldown_seconds=0, max_calls_per_hour=2, failure_policy="suppress"
    )
    for _ in range(3):
        await detector.detect(FRAME)
    await log.aclose()
    lines = _lines(tmp_path)
    assert [x["verdict"] for x in lines] == ["error", "fire", "capped"]
    assert lines[0]["error_type"] == "TimeoutError"
    assert lines[0]["policy"] == "inconclusive"
    assert lines[0]["probability"] is None
    assert lines[2]["policy"] == "suppress"
    assert "secret" not in (tmp_path / el.INDEX_NAME).read_text()


def test_get_escalation_log_disabled_returns_none() -> None:
    assert el.get_escalation_log() is None
