"""Tests for HybridFireDetector's verifier failure policy (#169).

A verifier error ("the verifier did not answer") must never silently become "no fire": it is not cached for the
cooldown, it is re-asked on the next check, and after N consecutive errors (or at the hourly call cap) the
``VERIFIER_FAILURE_POLICY`` decides between an unverified alert and a service-only signal.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from is_goat_burning.config import settings
from is_goat_burning.fire_detection import detectors
from is_goat_burning.fire_detection.detector_core import StreamFireDetector
from is_goat_burning.fire_detection.detectors import HybridFireDetector
from is_goat_burning.fire_detection.verifier_health import VerifierError
from is_goat_burning.fire_detection.verifier_health import VerifierStatus
from is_goat_burning.fire_detection.verifier_health import add_verifier_status_listener
from is_goat_burning.fire_detection.verifier_health import remove_verifier_status_listener
from is_goat_burning.fire_detection.verifier_health import reset_verifier_status

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


@pytest.fixture
def statuses() -> Iterator[list[VerifierStatus]]:
    """Collects the verifier status notifications published during a test."""
    received: list[VerifierStatus] = []
    reset_verifier_status()
    add_verifier_status_listener(received.append)
    yield received
    remove_verifier_status_listener(received.append)
    reset_verifier_status()


def _hybrid(verdicts: list[bool | Exception], **kwargs: object) -> tuple[HybridFireDetector, AsyncMock]:
    """A hybrid detector whose gate always fires and whose verifier yields ``verdicts`` in order."""
    local = MagicMock()
    local.detect = AsyncMock(return_value=(True, FRAME))
    verifier = MagicMock()
    verifier.verify = AsyncMock(side_effect=verdicts)
    kwargs.setdefault("cooldown_seconds", 0)
    kwargs.setdefault("max_calls_per_hour", 0)
    return HybridFireDetector(local, verifier, **kwargs), verifier.verify


async def _run(detector: HybridFireDetector, checks: int) -> list[tuple[bool, bool]]:
    """Runs ``checks`` detections and returns ``(fire, unverified)`` for each."""
    results = []
    for _ in range(checks):
        fire, _ = await detector.detect(FRAME)
        results.append((fire, detector.last_result_unverified))
    return results


@pytest.mark.asyncio
async def test_errors_below_threshold_report_no_fire_and_reask_every_check() -> None:
    detector, verify = _hybrid([VerifierError("down")] * 2, errors_before_policy=3, failure_policy="alert_unverified")

    assert await _run(detector, 2) == [(False, False), (False, False)]
    assert verify.await_count == 2  # asked on every check, nothing cached


@pytest.mark.asyncio
async def test_alert_unverified_after_n_consecutive_errors(statuses: list[VerifierStatus]) -> None:
    detector, _ = _hybrid([VerifierError("down")] * 5, errors_before_policy=3, failure_policy="alert_unverified")

    results = await _run(detector, 5)

    assert results == [(False, False), (False, False), (True, True), (True, True), (True, True)]
    assert len(statuses) == 1  # signalled once per outage, not per frame
    assert statuses[0].available is False
    assert statuses[0].policy == "alert_unverified"


@pytest.mark.asyncio
async def test_suppress_policy_reports_no_fire_but_signals_and_logs_critical(
    statuses: list[VerifierStatus], mocker: MockerFixture
) -> None:
    detector, _ = _hybrid([VerifierError("down")] * 5, errors_before_policy=3, failure_policy="suppress")
    critical = mocker.patch.object(detector._logger, "critical")

    results = await _run(detector, 5)

    assert results == [(False, False)] * 5
    assert [s.available for s in statuses] == [False]
    critical.assert_called_once()


@pytest.mark.asyncio
async def test_recovery_resets_counter_and_signals_once(statuses: list[VerifierStatus]) -> None:
    errors: list[bool | Exception] = [VerifierError("down")] * 3
    detector, _ = _hybrid([*errors, False, VerifierError("blip")], errors_before_policy=3)

    results = await _run(detector, 5)

    assert results == [(False, False), (False, False), (True, True), (False, False), (False, False)]
    assert [s.available for s in statuses] == [False, True]


@pytest.mark.asyncio
async def test_error_is_not_cached_in_the_cooldown(mocker: MockerFixture) -> None:
    mocker.patch.object(detectors, "time", MagicMock(monotonic=MagicMock(side_effect=[100.0, 101.0, 102.0])))
    detector, verify = _hybrid([VerifierError("timeout"), True], cooldown_seconds=60)

    first = (await detector.detect(FRAME))[0]
    second = (await detector.detect(FRAME))[0]
    third = (await detector.detect(FRAME))[0]  # inside the cooldown of the successful call: reused

    assert (first, second, third) == (False, True, True)
    assert verify.await_count == 2  # re-asked one second after the error, not 60 s later


@pytest.mark.asyncio
async def test_error_does_not_overwrite_the_cached_verdict(mocker: MockerFixture) -> None:
    mocker.patch.object(detectors, "time", MagicMock(monotonic=MagicMock(side_effect=[0.0, 61.0, 62.0])))
    detector, verify = _hybrid([True, VerifierError("down")], cooldown_seconds=60, errors_before_policy=3)

    assert (await detector.detect(FRAME))[0] is True
    assert detector.last_result_inconclusive is False
    # Error below the threshold: inconclusive (the stream detector keeps its state), never the stale verdict.
    assert (await detector.detect(FRAME))[0] is False
    assert detector.last_result_inconclusive is True
    assert detector.last_result_unverified is False
    assert detector._last_verdict is True
    verify.side_effect = [True]
    assert (await detector.detect(FRAME))[0] is True
    assert detector.last_result_inconclusive is False
    assert verify.await_count == 3


@pytest.mark.asyncio
async def test_error_after_the_cap_outage_keeps_applying_the_policy(mocker: MockerFixture) -> None:
    """Once unavailable (cap), a single error is not reported as an inconclusive "no fire" (M1)."""
    mocker.patch.object(detectors, "time", MagicMock(monotonic=MagicMock(side_effect=[0.0, 1.0, 3601.0])))
    detector, _ = _hybrid([True, VerifierError("down")], max_calls_per_hour=1, errors_before_policy=3)

    results = await _run(detector, 3)

    assert results == [(True, False), (True, True), (True, True)]
    assert detector.last_result_inconclusive is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("policy", "expected"), [("alert_unverified", (True, True)), ("suppress", (False, False))])
async def test_hourly_cap_applies_the_failure_policy(
    mocker: MockerFixture, statuses: list[VerifierStatus], policy: str, expected: tuple[bool, bool]
) -> None:
    mocker.patch.object(detectors, "time", MagicMock(monotonic=MagicMock(side_effect=[0.0, 1.0, 2.0, 3601.0])))
    detector, verify = _hybrid([True, True], max_calls_per_hour=1, failure_policy=policy)

    results = await _run(detector, 4)

    assert results == [(True, False), expected, expected, (True, False)]
    assert verify.await_count == 2
    assert [s.available for s in statuses] == [False, True]  # cap hit once, recovered when the hour freed up
    assert statuses[0].reason.startswith("call limit")


@pytest.mark.asyncio
async def test_non_verifier_exception_still_propagates_without_starting_cooldown(mocker: MockerFixture) -> None:
    mocker.patch.object(detectors, "time", MagicMock(monotonic=MagicMock(side_effect=[100.0, 101.0])))
    detector, verify = _hybrid([RuntimeError("bug"), True], cooldown_seconds=60)

    with pytest.raises(RuntimeError):
        await detector.detect(FRAME)
    assert (await detector.detect(FRAME))[0] is True
    assert verify.await_count == 2


@pytest.mark.asyncio
async def test_gate_negative_clears_the_unverified_flag() -> None:
    detector, _ = _hybrid([VerifierError("down")], errors_before_policy=1)
    assert (await detector.detect(FRAME))[0] is True
    assert detector.last_result_unverified is True

    detector.local_detector.detect.return_value = (False, FRAME)
    assert (await detector.detect(FRAME))[0] is False
    assert detector.last_result_unverified is False


@pytest.mark.asyncio
async def test_recovery_after_a_reconnect_is_published(statuses: list[VerifierStatus]) -> None:
    """A detector rebuilt on reconnect reports the recovery of an outage announced by its predecessor, once."""
    first, _ = _hybrid([VerifierError("down")] * 3, errors_before_policy=3)
    await _run(first, 3)
    second, _ = _hybrid([VerifierError("down")] * 3, errors_before_policy=3)
    await _run(second, 3)  # the outage continues after a reconnect: not announced again
    third, _ = _hybrid([True], errors_before_policy=3)
    await _run(third, 1)  # recovered right after another reconnect: the new detector never saw the outage

    assert [s.available for s in statuses] == [False, True]


def test_defaults_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "verifier_failure_policy", "suppress")
    monkeypatch.setattr(settings, "verifier_errors_before_policy", 7)
    detector = HybridFireDetector(MagicMock(), MagicMock())
    assert detector.failure_policy == "suppress"
    assert detector.errors_before_policy == 7


def test_failing_listener_does_not_break_other_listeners() -> None:
    from is_goat_burning.fire_detection.verifier_health import publish_verifier_status

    def _broken(_status: VerifierStatus) -> None:
        raise RuntimeError("listener bug")

    received: list[VerifierStatus] = []
    add_verifier_status_listener(_broken)  # registered first, so it runs before the healthy one
    add_verifier_status_listener(received.append)
    status = VerifierStatus(available=False, verifier="X", reason="r", policy="suppress")
    try:
        publish_verifier_status(status)
    finally:
        remove_verifier_status_listener(_broken)
        remove_verifier_status_listener(received.append)

    assert received == [status]


def test_async_listener_is_rejected() -> None:
    async def _listener(_status: VerifierStatus) -> None:
        return None

    with pytest.raises(TypeError, match="synchronous"):
        add_verifier_status_listener(_listener)


def test_callable_object_with_async_call_is_rejected() -> None:
    """An object whose ``__call__`` is ``async def`` would never be awaited either (#171)."""

    class _AsyncCallable:
        async def __call__(self, _status: VerifierStatus) -> None:
            return None

    with pytest.raises(TypeError, match="synchronous"):
        add_verifier_status_listener(_AsyncCallable())


def test_callable_object_with_sync_call_is_accepted() -> None:
    """A plain callable object stays a valid listener."""

    class _SyncCallable:
        def __call__(self, _status: VerifierStatus) -> None:
            return None

    listener = _SyncCallable()
    add_verifier_status_listener(listener)
    remove_verifier_status_listener(listener)


# --- acceptance: a series of API errors behind an alarming gate ends in an alert, not silence ---


async def _stream_detector(policy: str, mocker: MockerFixture) -> tuple[StreamFireDetector, AsyncMock]:
    on_fire = AsyncMock()
    detector = StreamFireDetector(on_fire_action=on_fire, video_output=False, checks_per_second=None)
    detector.signal_handler = MagicMock()
    hybrid, _ = _hybrid([VerifierError("503")] * 6, errors_before_policy=3, failure_policy=policy)
    detector.fire_detector = hybrid
    mocker.patch.object(settings, "fire_detected_debounce_seconds", 0.0)
    for _ in range(6):
        fire, _ = await hybrid.detect(FRAME)
        await detector._handle_fire_detection(fire)
    return detector, on_fire


@pytest.mark.asyncio
async def test_api_outage_with_alarming_gate_sends_unverified_alert(
    mocker: MockerFixture, statuses: list[VerifierStatus]
) -> None:
    _, on_fire = await _stream_detector("alert_unverified", mocker)

    on_fire.assert_awaited_once_with(unverified=True)
    assert [s.available for s in statuses] == [False]


@pytest.mark.asyncio
async def test_api_outage_with_suppress_only_signals_service(mocker: MockerFixture, statuses: list[VerifierStatus]) -> None:
    _, on_fire = await _stream_detector("suppress", mocker)

    on_fire.assert_not_awaited()
    assert [s.available for s in statuses] == [False]


# --- inconclusive frames (errors below the threshold) keep the stream detector's debounce state ---


async def _stream_run(
    mocker: MockerFixture, verdicts: list[bool | Exception], times: list[float], *, detected: float, extinguished: float
) -> tuple[StreamFireDetector, AsyncMock, AsyncMock]:
    """Feeds one frame per verdict through a hybrid detector (gate always on) and the debounce logic.

    ``times`` are the clock readings of the debounce logic; an inconclusive frame must not read the clock at all.
    """
    on_fire, on_extinguished = AsyncMock(), AsyncMock()
    detector = StreamFireDetector(
        on_fire_action=on_fire, video_output=False, checks_per_second=None, on_extinguished_action=on_extinguished
    )
    detector.signal_handler = MagicMock()
    detector.fire_detector, _ = _hybrid(verdicts, errors_before_policy=3)
    mocker.patch.object(settings, "fire_detected_debounce_seconds", detected)
    mocker.patch.object(settings, "fire_extinguished_debounce_seconds", extinguished)
    from is_goat_burning.fire_detection import detector_core

    mocker.patch.object(detector_core, "time", MagicMock(monotonic=MagicMock(side_effect=times)))
    for _ in verdicts:
        fire, _ = await detector.fire_detector.detect(FRAME)
        await detector._handle_fire_detection(fire)
    return detector, on_fire, on_extinguished


@pytest.mark.asyncio
async def test_slow_verifier_errors_during_a_fire_do_not_extinguish_it(mocker: MockerFixture) -> None:
    """Verified fire, then two (10 s) errors: no EXTINGUISHED, no second alert."""
    detector, on_fire, on_extinguished = await _stream_run(
        mocker, [True, VerifierError("timeout"), VerifierError("timeout"), True], [0.0, 30.0], detected=0, extinguished=10
    )

    on_fire.assert_awaited_once_with()
    on_extinguished.assert_not_awaited()
    assert detector.fire_is_currently_detected is True


@pytest.mark.asyncio
async def test_verifier_errors_without_fire_do_not_raise_fire(mocker: MockerFixture) -> None:
    """No fire, then errors below the threshold: no FIRE."""
    detector, on_fire, _ = await _stream_run(
        mocker, [False, VerifierError("x"), VerifierError("x")], [0.0], detected=0, extinguished=10
    )

    on_fire.assert_not_awaited()
    assert detector.fire_is_currently_detected is False


@pytest.mark.asyncio
async def test_inconclusive_frame_does_not_reset_a_pending_fire_debounce(mocker: MockerFixture) -> None:
    """A verifier error between two confirmations neither starts nor resets the detection debounce."""
    _, on_fire, _ = await _stream_run(mocker, [True, VerifierError("x"), True], [0.0, 6.0], detected=5, extinguished=10)

    on_fire.assert_awaited_once_with()


# --- While the verifier is unavailable it is probed at most once per cooldown (#188) ---


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_unavailable_verifier_is_probed_once_per_cooldown(mocker: MockerFixture) -> None:
    """During an outage the policy result is returned between probes; the verifier is called once per cooldown."""
    clock = _Clock()
    mocker.patch.object(detectors, "time", clock)
    local = MagicMock()
    local.detect = AsyncMock(return_value=(True, FRAME))
    verifier = MagicMock()
    verifier.verify = AsyncMock(side_effect=VerifierError("down"))
    detector = HybridFireDetector(
        local, verifier, cooldown_seconds=30, max_calls_per_hour=0, failure_policy="alert_unverified", errors_before_policy=1
    )

    results = []
    for second in range(61):
        clock.now = float(second)
        results.append((await detector.detect(FRAME))[0])
        assert detector.last_result_unverified is True

    assert all(results)
    assert verifier.verify.await_count == 3  # t=0, 30, 60

    verifier.verify = AsyncMock(return_value=False)  # the verifier is back
    clock.now = 89.0
    assert (await detector.detect(FRAME))[0] is True  # still inside the probe interval: policy result
    clock.now = 90.0
    assert (await detector.detect(FRAME))[0] is False  # probed, real verdict
    assert verifier.verify.await_count == 1
    assert detector.last_result_unverified is False


@pytest.mark.asyncio
async def test_errors_below_the_policy_threshold_are_still_reasked_on_the_next_check(mocker: MockerFixture) -> None:
    """The probe interval applies only once the verifier is unavailable, not to inconclusive errors."""
    clock = _Clock()
    mocker.patch.object(detectors, "time", clock)
    local = MagicMock()
    local.detect = AsyncMock(return_value=(True, FRAME))
    verifier = MagicMock()
    verifier.verify = AsyncMock(side_effect=[VerifierError("a"), VerifierError("b"), True])
    detector = HybridFireDetector(local, verifier, cooldown_seconds=30, max_calls_per_hour=0, errors_before_policy=3)

    for second in range(3):
        clock.now = float(second)
        await detector.detect(FRAME)

    assert verifier.verify.await_count == 3
