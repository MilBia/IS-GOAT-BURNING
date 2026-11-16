"""Tests for the verifier call limits of HybridFireDetector (cooldown with verdict reuse, hourly cap)."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from is_goat_burning.fire_detection import detectors
from is_goat_burning.fire_detection.detectors import HybridFireDetector

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def _hybrid(verdicts: list[bool], **limits: float) -> tuple[HybridFireDetector, AsyncMock]:
    local = MagicMock()
    local.detect = AsyncMock(return_value=(True, FRAME))
    verifier = MagicMock()
    verifier.detect = AsyncMock(side_effect=[(v, FRAME) for v in verdicts])
    return HybridFireDetector(local, verifier, **limits), verifier.detect


@pytest.mark.asyncio
async def test_cooldown_reuses_last_verdict_without_calling_again(mocker: MockerFixture) -> None:
    clock = MagicMock(monotonic=MagicMock(side_effect=[100.0, 110.0, 130.0, 161.0]))
    mocker.patch.object(detectors, "time", clock)
    detector, call = _hybrid([True, False], cooldown_seconds=60, max_calls_per_hour=0)

    results = [(await detector.detect(FRAME))[0] for _ in range(4)]

    assert results == [True, True, True, False]  # 3rd/2nd reuse True; at t=161 the API is asked again (False)
    assert call.await_count == 2
    assert clock.monotonic.call_count == 4


@pytest.mark.asyncio
async def test_hourly_cap_returns_no_fire_without_calling(mocker: MockerFixture) -> None:
    mocker.patch.object(detectors, "time", MagicMock(monotonic=MagicMock(side_effect=[0.0, 1.0, 2.0, 3601.0])))
    detector, call = _hybrid([True, True, True], cooldown_seconds=0, max_calls_per_hour=2)

    results = [(await detector.detect(FRAME))[0] for _ in range(4)]

    assert results == [True, True, False, True]  # 3rd is over the cap; after an hour the window frees up
    assert call.await_count == 3


@pytest.mark.asyncio
async def test_no_limits_calls_every_time() -> None:
    detector, call = _hybrid([True, True, True], cooldown_seconds=0, max_calls_per_hour=0)

    for _ in range(3):
        assert (await detector.detect(FRAME))[0] is True

    assert call.await_count == 3


@pytest.mark.asyncio
async def test_cap_warning_is_logged_once(mocker: MockerFixture) -> None:
    mocker.patch.object(detectors, "time", MagicMock(monotonic=MagicMock(side_effect=[0.0, 1.0, 2.0, 3.0])))
    detector, _ = _hybrid([True], cooldown_seconds=0, max_calls_per_hour=1)
    warn = mocker.patch.object(detector._logger, "warning")

    for _ in range(4):
        await detector.detect(FRAME)

    assert warn.call_count == 1
