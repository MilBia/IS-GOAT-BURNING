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

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


@pytest.fixture
def statuses() -> Iterator[list[VerifierStatus]]:
    """Collects the verifier status notifications published during a test."""
    received: list[VerifierStatus] = []
    add_verifier_status_listener(received.append)
    yield received
    remove_verifier_status_listener(received.append)


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
    assert (await detector.detect(FRAME))[0] is False  # error below threshold: "unknown", not the stale verdict
    assert detector._last_verdict is True
    verify.side_effect = [True]
    assert (await detector.detect(FRAME))[0] is True
    assert verify.await_count == 3


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


def test_defaults_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "verifier_failure_policy", "suppress")
    monkeypatch.setattr(settings, "verifier_errors_before_policy", 7)
    detector = HybridFireDetector(MagicMock(), MagicMock())
    assert detector.failure_policy == "suppress"
    assert detector.errors_before_policy == 7


def test_failing_listener_does_not_break_detection(statuses: list[VerifierStatus]) -> None:  # noqa: ARG001
    from is_goat_burning.fire_detection.verifier_health import publish_verifier_status

    def _broken(_status: VerifierStatus) -> None:
        raise RuntimeError("listener bug")

    add_verifier_status_listener(_broken)
    try:
        publish_verifier_status(VerifierStatus(available=False, verifier="X", reason="r", policy="suppress"))
    finally:
        remove_verifier_status_listener(_broken)


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
