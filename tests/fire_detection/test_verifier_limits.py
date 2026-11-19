"""Tests for the verifier call limits of HybridFireDetector (cooldown with verdict reuse, hourly cap)."""

from __future__ import annotations

from collections.abc import Iterator
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


# --- The negative-verdict hold is anchored at the first negative after a fire (#191) ---


@pytest.fixture
def _clean_verifier_status() -> Iterator[None]:
    """Resets the shared verifier status that the outage scenarios publish."""
    from is_goat_burning.fire_detection.verifier_health import reset_verifier_status

    reset_verifier_status()
    yield
    reset_verifier_status()


def _sequence(answers: list[bool | Exception], then: bool) -> AsyncMock:
    """A verifier answering (or raising) ``answers`` in order, then ``then`` forever."""
    queue = list(answers)

    async def verify(_frame: np.ndarray) -> bool:
        if not queue:
            return then
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return AsyncMock(side_effect=verify)


async def _run_schedule(
    mocker: MockerFixture,
    verify: AsyncMock,
    times: list[int],
    cooldown: float = 30.0,
    debounce: float = 10.0,
    cv_negative: frozenset[int] = frozenset(),
    **kwargs: object,
) -> tuple[MagicMock, list[tuple[int, bool]]]:
    """Feeds a frame at each of ``times`` (seconds) through the hybrid detector and the stream debounce.

    The CV gate fires on every frame except at the seconds in ``cv_negative``.
    """
    from is_goat_burning.fire_detection import detector_core
    from is_goat_burning.fire_detection.detector_core import StreamFireDetector

    clock = _Clock()
    mocker.patch.object(detectors, "time", clock)
    mocker.patch.object(detector_core, "time", clock)
    core_settings = MagicMock(fire_detected_debounce_seconds=0.0, fire_extinguished_debounce_seconds=debounce)
    mocker.patch.object(detector_core, "settings", core_settings)

    local = MagicMock()
    local.detect = AsyncMock(side_effect=lambda _frame: (clock.now not in cv_negative, FRAME))
    verifier = MagicMock()
    verifier.verify = verify
    hybrid = HybridFireDetector(
        local, verifier, cooldown_seconds=cooldown, max_calls_per_hour=0, extinguish_debounce_seconds=debounce, **kwargs
    )
    on_extinguished = AsyncMock()
    stream = StreamFireDetector(
        on_fire_action=AsyncMock(), video_output=False, checks_per_second=1.0, on_extinguished_action=on_extinguished
    )
    stream.signal_handler = MagicMock()
    stream.fire_detector = hybrid
    calls: list[tuple[int, bool]] = []
    for second in times:
        clock.now = float(second)
        before = verify.await_count
        fire, _ = await hybrid.detect(FRAME)
        if verify.await_count > before:
            calls.append((second, fire))
        await stream._handle_fire_detection(fire, FRAME)
    return on_extinguished, calls


@pytest.mark.asyncio
@pytest.mark.usefixtures("_clean_verifier_status")
async def test_false_negative_after_a_verifier_outage_does_not_end_the_fire(mocker: MockerFixture) -> None:
    """Scenario 1: fire, outage (alert_unverified keeps the fire), recovery with one false negative: re-asked at once."""
    from is_goat_burning.fire_detection.verifier_health import VerifierError

    verify = _sequence([True, VerifierError("down"), VerifierError("down"), False], then=True)

    on_extinguished, calls = await _run_schedule(
        mocker, verify, list(range(125)), failure_policy="alert_unverified", errors_before_policy=1
    )

    on_extinguished.assert_not_awaited()
    # t=0 fire, probes t=30/60 fail (policy fire), t=90 false negative, t=91 re-asked (hold) -> fire, t=121 cooldown.
    assert calls == [(0, True), (30, True), (60, True), (90, False), (91, True), (121, True)]


@pytest.mark.asyncio
async def test_false_negative_at_the_end_of_the_188_window_is_held_after_the_window(mocker: MockerFixture) -> None:
    """Scenario 2: fire, no checks until just before cooldown + debounce, false negative: re-asked after the window."""
    verify = _sequence([True, False, False], then=True)
    times = list(range(30)) + list(range(39, 80))  # no CV-positive check between t=30 and t=38

    on_extinguished, calls = await _run_schedule(mocker, verify, times)

    on_extinguished.assert_not_awaited()
    # t=39 false negative (hold until 49), t=40 re-asked although the #188 window (until t=40) ended, t=41 fire.
    assert calls == [(0, True), (39, False), (40, False), (41, True), (71, True)]


@pytest.mark.asyncio
async def test_false_negative_after_a_cv_gate_gap_does_not_end_the_fire(mocker: MockerFixture) -> None:
    """Scenario 2 with CV-negative frames: a 9 s gate gap, then a false negative at the end of the #188 window."""
    gap = frozenset(range(30, 39))  # the gate sees nothing from t=30 to t=38; the extinguish debounce starts at t=30

    on_extinguished, calls = await _run_schedule(mocker, _sequence([True, False], then=True), list(range(80)), cv_negative=gap)

    on_extinguished.assert_not_awaited()
    # t=39 false negative (9 s of no-fire so far), t=40 re-asked (hold) -> fire resets the extinguish debounce.
    assert calls == [(0, True), (39, False), (40, True), (70, True)]


@pytest.mark.asyncio
async def test_inconclusive_error_before_the_false_negative_does_not_skip_the_hold(mocker: MockerFixture) -> None:
    """An inconclusive error is neither fire nor no-fire: the negative that follows it still starts the hold."""
    from is_goat_burning.fire_detection.verifier_health import VerifierError

    verify = _sequence([True, VerifierError("blip"), False], then=True)
    times = list(range(30)) + list(range(39, 80))  # no CV-positive check between t=30 and t=38

    on_extinguished, calls = await _run_schedule(mocker, verify, times, errors_before_policy=2)

    on_extinguished.assert_not_awaited()
    # t=39 inconclusive error, t=40 false negative (hold until 50, the #188 window has ended), t=41 re-asked -> fire.
    assert calls == [(0, True), (39, False), (40, False), (41, True), (71, True)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_clean_verifier_status")
async def test_real_extinguish_after_an_outage_ends_once_the_hold_expires(mocker: MockerFixture) -> None:
    """All negatives after recovery: the hold is not extended, the fire ends, then negatives are reused per cooldown."""
    from is_goat_burning.fire_detection.verifier_health import VerifierError

    verify = _sequence([True, VerifierError("down"), VerifierError("down")], then=False)

    on_extinguished, calls = await _run_schedule(
        mocker, verify, list(range(135)), failure_policy="alert_unverified", errors_before_policy=1
    )

    on_extinguished.assert_awaited_once()
    # Re-asked on every check of the hold (t=90..99, anchored at the first negative), then once per cooldown.
    assert [second for second, _ in calls] == [0, 30, 60, *range(90, 100), 129]


@pytest.mark.asyncio
async def test_persistent_cv_false_positive_without_fire_keeps_the_full_cooldown(mocker: MockerFixture) -> None:
    """No fire ever reported: a negative never starts a hold, calls stay one per cooldown."""
    on_extinguished, calls = await _run_schedule(mocker, _sequence([], then=False), list(range(121)))

    on_extinguished.assert_not_awaited()
    assert [second for second, _ in calls] == [0, 30, 60, 90, 120]
