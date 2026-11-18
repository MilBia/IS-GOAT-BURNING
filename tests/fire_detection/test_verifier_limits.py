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
    verifier.verify = AsyncMock(side_effect=verdicts)
    return HybridFireDetector(local, verifier, **limits), verifier.verify


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
async def test_hourly_cap_skips_the_call_and_applies_the_failure_policy(mocker: MockerFixture) -> None:
    mocker.patch.object(detectors, "time", MagicMock(monotonic=MagicMock(side_effect=[0.0, 1.0, 2.0, 3601.0])))
    detector, call = _hybrid([True, True, True], cooldown_seconds=0, max_calls_per_hour=2, failure_policy="suppress")

    results = [(await detector.detect(FRAME))[0] for _ in range(4)]

    assert results == [True, True, False, True]  # 3rd is over the cap (suppressed); after an hour the window frees up
    assert call.await_count == 3


@pytest.mark.asyncio
async def test_no_limits_calls_every_time() -> None:
    detector, call = _hybrid([True, True, True], cooldown_seconds=0, max_calls_per_hour=0)

    for _ in range(3):
        assert (await detector.detect(FRAME))[0] is True

    assert call.await_count == 3


@pytest.mark.asyncio
async def test_cap_critical_is_logged_once(mocker: MockerFixture) -> None:
    mocker.patch.object(detectors, "time", MagicMock(monotonic=MagicMock(side_effect=[0.0, 1.0, 2.0, 3.0])))
    detector, _ = _hybrid([True], cooldown_seconds=0, max_calls_per_hour=1)
    critical = mocker.patch.object(detector._logger, "critical")

    for _ in range(4):
        await detector.detect(FRAME)

    assert critical.call_count == 1


@pytest.mark.asyncio
async def test_failed_verifier_call_does_not_start_the_cooldown(mocker: MockerFixture) -> None:
    mocker.patch.object(detectors, "time", MagicMock(monotonic=MagicMock(side_effect=[100.0, 101.0])))
    detector, call = _hybrid([], cooldown_seconds=60, max_calls_per_hour=0)
    call.side_effect = [RuntimeError("boom"), True]

    with pytest.raises(RuntimeError):
        await detector.detect(FRAME)

    assert (await detector.detect(FRAME))[0] is True  # asked again immediately, not blocked for 60 s
    assert call.await_count == 2


# --- A negative verdict must not end an ongoing fire through the cooldown (#188, I5) ---


class _Clock:
    """A manually advanced clock shared by ``detectors`` and ``detector_core``."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def _verdicts(first: list[bool], then: bool) -> AsyncMock:
    """A verifier answering ``first`` in order, then ``then`` forever."""
    queue = list(first)

    async def verify(_frame: np.ndarray) -> bool:
        return queue.pop(0) if queue else then

    return AsyncMock(side_effect=verify)


async def _run_fire(
    mocker: MockerFixture, verify: AsyncMock, seconds: int, cooldown: float = 30.0, debounce: float = 10.0
) -> tuple[MagicMock, AsyncMock]:
    """Feeds one CV-positive frame per second through HybridFireDetector into StreamFireDetector's debounce."""
    from is_goat_burning.fire_detection import detector_core
    from is_goat_burning.fire_detection.detector_core import StreamFireDetector

    clock = _Clock()
    mocker.patch.object(detectors, "time", clock)
    mocker.patch.object(detector_core, "time", clock)
    core_settings = MagicMock(fire_detected_debounce_seconds=0.0, fire_extinguished_debounce_seconds=debounce)
    mocker.patch.object(detector_core, "settings", core_settings)

    local = MagicMock()
    local.detect = AsyncMock(return_value=(True, FRAME))
    verifier = MagicMock()
    verifier.verify = verify
    hybrid = HybridFireDetector(
        local, verifier, cooldown_seconds=cooldown, max_calls_per_hour=0, extinguish_debounce_seconds=debounce
    )
    on_extinguished = AsyncMock()
    stream = StreamFireDetector(
        on_fire_action=AsyncMock(), video_output=False, checks_per_second=1.0, on_extinguished_action=on_extinguished
    )
    stream.signal_handler = MagicMock()
    stream.fire_detector = hybrid
    for second in range(seconds):
        clock.now = float(second)
        fire, _ = await hybrid.detect(FRAME)
        await stream._handle_fire_detection(fire, FRAME)
    return on_extinguished, verify


@pytest.mark.asyncio
async def test_single_false_negative_after_the_cooldown_does_not_end_the_fire(mocker: MockerFixture) -> None:
    """Verified fire, then one false negative right after the cooldown: the fire state survives (re-asked at once)."""
    verify = _verdicts([True, False], then=True)  # t=0 fire, t=30 false negative, t=31 fire again

    on_extinguished, verify = await _run_fire(mocker, verify, seconds=70)

    on_extinguished.assert_not_awaited()
    assert verify.await_count == 4  # t=0, t=30 (no), t=31 (re-asked), t=61


@pytest.mark.asyncio
async def test_persistent_cv_false_positive_is_still_rate_limited_by_the_cooldown(mocker: MockerFixture) -> None:
    """No positive verdict at all: a negative is reused for the whole cooldown (API cost unchanged)."""
    on_extinguished, verify = await _run_fire(mocker, _verdicts([], then=False), seconds=121)

    assert verify.await_count == 5  # t=0, 30, 60, 90, 120
    on_extinguished.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_extinguish_still_ends_the_fire_with_bounded_calls(mocker: MockerFixture) -> None:
    """Verified fire, then the verifier keeps saying no: the fire ends after the debounce; extra calls are bounded."""
    on_extinguished, verify = await _run_fire(mocker, _verdicts([True], then=False), seconds=121)

    on_extinguished.assert_awaited_once()
    # t=0 (fire), every check from t=30 to t=39 (inside cooldown + debounce of the last positive), then once per
    # cooldown again: t=69, 99.
    assert verify.await_count == 1 + 10 + 2
