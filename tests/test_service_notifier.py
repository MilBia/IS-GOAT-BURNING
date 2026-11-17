"""Tests for the service ("sign of life") notifier, with a mocked sender and a fake clock."""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np
import pytest
import pytest_asyncio

from is_goat_burning.fire_detection.detectors import clear_edge_verifier_cache
from is_goat_burning.fire_detection.detectors import create_fire_detector
from is_goat_burning.fire_detection.edge_detector import EdgeModelError
from is_goat_burning.fire_detection.verifier_health import VerifierStatus
from is_goat_burning.fire_detection.verifier_health import publish_verifier_status
from is_goat_burning.fire_detection.verifier_health import reset_verifier_status
from is_goat_burning.service_notifier import ServiceNotifier
from is_goat_burning.service_notifier import describe_source
from is_goat_burning.service_notifier import failure_reason
from is_goat_burning.service_notifier import format_duration

HOOKS = ["https://discord.example/api/webhooks/1/secret-token"]
THRESHOLD = 600.0
HEARTBEAT = 86400.0


class FakeClock:
    """A manually advanced monotonic clock."""

    def __init__(self) -> None:
        """Starts the clock at an arbitrary non-zero time."""
        self.now = 1000.0

    def __call__(self) -> float:
        """Returns the current fake time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Moves the clock forward."""
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    """Provides a fake clock."""
    return FakeClock()


@pytest.fixture
def send() -> AsyncMock:
    """Provides a mocked sender that reports successful delivery."""
    return AsyncMock(return_value=True)


@pytest_asyncio.fixture
async def notifier(clock: FakeClock, send: AsyncMock) -> AsyncGenerator[ServiceNotifier]:
    """Provides a started notifier; it is stopped (and its listener removed) afterwards."""
    reset_verifier_status()
    instance = ServiceNotifier(
        HOOKS,
        strategy="hybrid_edge",
        source="https://www.youtube.com/watch?v=abc123&token=xyz",
        heartbeat_interval=HEARTBEAT,
        stream_down_threshold=THRESHOLD,
        send=send,
        clock=clock,
    )
    instance.start()
    await instance.drain()
    send.reset_mock()
    yield instance
    await instance.stop()
    reset_verifier_status()


def sent(send: AsyncMock) -> list[str]:
    """Returns the texts passed to the mocked sender."""
    return [call.args[0] for call in send.await_args_list]


@pytest.mark.asyncio
async def test_start_sends_a_startup_message(clock: FakeClock, send: AsyncMock) -> None:
    """Starting the notifier sends one message naming the strategy and the source host + video id only."""
    instance = ServiceNotifier(
        HOOKS,
        strategy="hybrid_edge",
        source="https://www.youtube.com/watch?v=abc123&token=xyz",
        send=send,
        clock=clock,
    )
    instance.start()
    await instance.drain()
    await instance.stop()

    texts = sent(send)
    assert "started" in texts[0].lower()
    assert "hybrid_edge" in texts[0]
    assert "www.youtube.com" in texts[0]
    assert "abc123" in texts[0]
    assert "xyz" not in texts[0]


@pytest.mark.asyncio
async def test_long_stream_outage_warns_once_then_reports_recovery(
    notifier: ServiceNotifier, clock: FakeClock, send: AsyncMock
) -> None:
    """An outage longer than the threshold gives exactly one warning, then a recovery message."""
    notifier.stream_connected()
    clock.advance(100)
    notifier.stream_failed("TimeoutError")
    clock.advance(THRESHOLD / 2)
    notifier.stream_failed("ConnectionError")  # A failed reconnect attempt does not restart the outage.
    notifier.check_timers()
    await notifier.drain()
    assert sent(send) == []

    clock.advance(THRESHOLD / 2 + 1)
    notifier.check_timers()
    clock.advance(300)
    notifier.check_timers()
    notifier.stream_failed("ConnectionError")
    notifier.check_timers()
    await notifier.drain()
    texts = sent(send)
    assert len(texts) == 1
    assert "unavailable" in texts[0].lower()
    assert "ConnectionError" in texts[0]

    notifier.stream_connected()
    await notifier.drain()
    texts = sent(send)
    assert len(texts) == 2
    assert "recovered" in texts[1].lower()
    assert "15m" in texts[1]  # 300 + 301 + 300 s outage


@pytest.mark.asyncio
async def test_short_stream_outage_sends_nothing(notifier: ServiceNotifier, clock: FakeClock, send: AsyncMock) -> None:
    """An outage shorter than the threshold produces neither a warning nor a recovery message."""
    notifier.stream_connected()
    notifier.stream_failed("TimeoutError")
    clock.advance(THRESHOLD - 1)
    notifier.check_timers()
    notifier.stream_connected()
    clock.advance(THRESHOLD * 2)
    notifier.check_timers()
    await notifier.drain()
    assert sent(send) == []


@pytest.mark.asyncio
async def test_stream_never_connecting_warns(notifier: ServiceNotifier, clock: FakeClock, send: AsyncMock) -> None:
    """A stream that never comes up after startup is an outage too (measured from the start)."""
    clock.advance(THRESHOLD + 1)
    notifier.check_timers()
    await notifier.drain()
    assert len(sent(send)) == 1
    assert "unavailable" in sent(send)[0].lower()


@pytest.mark.asyncio
async def test_heartbeat_after_interval_reports_state_and_resets_counters(
    notifier: ServiceNotifier, clock: FakeClock, send: AsyncMock
) -> None:
    """The heartbeat is sent once per interval with uptime, stream state and fire-event counts."""
    notifier.stream_connected()
    notifier.fire_event()
    notifier.fire_event(unverified=True)
    clock.advance(HEARTBEAT - 1)
    notifier.check_timers()
    await notifier.drain()
    assert sent(send) == []

    clock.advance(1)
    notifier.check_timers()
    notifier.check_timers()
    await notifier.drain()
    texts = sent(send)
    assert len(texts) == 1
    assert "heartbeat" in texts[0].lower()
    assert "1d" in texts[0]
    assert "up" in texts[0].lower()
    assert "fire events since last heartbeat: 2 (unverified: 1)" in texts[0].lower()
    assert "verifier: available" in texts[0].lower()

    clock.advance(HEARTBEAT)
    notifier.check_timers()
    await notifier.drain()
    assert "fire events since last heartbeat: 0 (unverified: 0)" in sent(send)[1].lower()


@pytest.mark.asyncio
async def test_verifier_status_changes_are_sent(notifier: ServiceNotifier, clock: FakeClock, send: AsyncMock) -> None:
    """Verifier unavailable / recovered transitions from verifier_health become service messages."""
    notifier.stream_connected()
    publish_verifier_status(
        VerifierStatus(available=False, verifier="GeminiFireDetector", reason="3 consecutive errors", policy="suppress")
    )
    await notifier.drain()
    clock.advance(HEARTBEAT)
    notifier.check_timers()
    await notifier.drain()
    publish_verifier_status(
        VerifierStatus(available=True, verifier="GeminiFireDetector", reason="recovered", policy="suppress")
    )
    await notifier.drain()

    down, heartbeat, up = sent(send)
    assert "GeminiFireDetector" in down
    assert "unavailable" in down.lower()
    assert "3 consecutive errors" in down
    assert "suppress" in down.lower()
    assert "unavailable" in heartbeat.lower()
    assert "GeminiFireDetector" in up
    assert "recovered" in up.lower()


@pytest.mark.asyncio
async def test_listener_is_called_without_blocking(notifier: ServiceNotifier, send: AsyncMock) -> None:
    """The verifier listener only queues work: nothing is sent until the notifier task runs."""

    async def slow_send(text: str) -> bool:  # noqa: ARG001
        await asyncio.sleep(0)
        return True

    send.side_effect = slow_send
    notifier.on_verifier_status(VerifierStatus(available=False, verifier="V", reason="r", policy="alert_unverified"))
    assert send.await_count == 0
    await notifier.drain()
    assert send.await_count == 1


@pytest.mark.asyncio
async def test_stop_sends_a_shutdown_message_and_removes_listener(clock: FakeClock, send: AsyncMock) -> None:
    """Stopping sends a best-effort shutdown message and unregisters the verifier listener."""
    instance = ServiceNotifier(HOOKS, strategy="classic", source="mock://source", send=send, clock=clock)
    instance.start()
    await instance.stop()

    assert "stopped" in sent(send)[-1].lower()
    count = send.await_count
    publish_verifier_status(VerifierStatus(available=False, verifier="V", reason="r", policy="alert_unverified"))
    await asyncio.sleep(0)
    assert send.await_count == count  # The listener is gone: nothing is queued or sent after stop().


@pytest.mark.asyncio
async def test_stop_is_bounded_when_sending_hangs(clock: FakeClock) -> None:
    """A hanging webhook cannot delay shutdown beyond the stop timeout."""

    async def hanging_send(text: str) -> bool:  # noqa: ARG001
        await asyncio.sleep(3600)
        return True

    hang = AsyncMock(side_effect=hanging_send)
    instance = ServiceNotifier(HOOKS, strategy="classic", source="mock://source", send=hang, clock=clock, stop_timeout=0.05)
    instance.start()
    await asyncio.wait_for(instance.stop(), timeout=2)
    hang.assert_awaited()


@pytest.mark.asyncio
async def test_send_failures_never_propagate(notifier: ServiceNotifier, send: AsyncMock) -> None:
    """A failing or rejecting sender is logged and the notifier keeps working."""
    send.side_effect = [RuntimeError("boom"), False, True]
    notifier.fire_event()
    notifier.on_verifier_status(VerifierStatus(available=False, verifier="V", reason="r", policy="alert_unverified"))
    notifier.on_verifier_status(VerifierStatus(available=True, verifier="V", reason="recovered", policy="alert_unverified"))
    notifier.on_verifier_status(VerifierStatus(available=False, verifier="V", reason="r", policy="alert_unverified"))
    await notifier.drain()
    assert send.await_count == 3


@pytest.mark.asyncio
async def test_full_queue_drops_messages_instead_of_blocking(clock: FakeClock, send: AsyncMock) -> None:
    """With a full queue new messages are dropped (logged), never awaited by the caller."""
    instance = ServiceNotifier(HOOKS, strategy="classic", source="mock://source", send=send, clock=clock, queue_size=1)
    instance.start()  # The startup message fills the queue (the task has not run yet).
    for _ in range(5):
        instance.on_verifier_status(VerifierStatus(available=False, verifier="V", reason="r", policy="alert_unverified"))
    await instance.drain()
    await instance.stop()
    assert send.await_count == 2  # startup + shutdown


@pytest.mark.asyncio
async def test_disabled_without_hooks(clock: FakeClock, send: AsyncMock) -> None:
    """With no service webhooks nothing is started, registered or sent."""
    instance = ServiceNotifier([], strategy="classic", source="mock://source", send=send, clock=clock)
    assert instance.enabled is False
    instance.start()
    publish_verifier_status(VerifierStatus(available=False, verifier="V", reason="r", policy="alert_unverified"))
    instance.stream_failed("TimeoutError")
    clock.advance(THRESHOLD * 10)
    instance.check_timers()
    instance.stream_connected()
    instance.fire_event()
    await instance.drain()
    await instance.stop()
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_sender_uses_service_webhooks_without_frame(clock: FakeClock) -> None:
    """Without an injected sender, messages go through SendToDiscord to the service webhooks, text only."""
    with patch("is_goat_burning.service_notifier.SendToDiscord") as discord_cls:
        discord_cls.return_value = AsyncMock(return_value=True)
        instance = ServiceNotifier(
            HOOKS, strategy="classic", source="mock://source", clock=clock, send_options={"timeout": 3.0, "max_attempts": 2}
        )
        instance.start()
        await instance.drain()
        await instance.stop()

    kwargs = discord_cls.call_args_list[0].kwargs
    assert kwargs["webhooks"] == HOOKS
    assert kwargs["attach_frame"] is False
    assert kwargs["timeout"] == 3.0
    assert kwargs["max_attempts"] == 2
    assert "started" in kwargs["message"].lower()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("https://www.youtube.com/watch?v=abc123&t=5", "www.youtube.com (video abc123)"),
        ("https://youtu.be/abc123?si=token", "youtu.be (video abc123)"),
        ("https://www.youtube.com/live/abc123?si=token", "www.youtube.com (video abc123)"),
        ("https://example.com/stream.m3u8?key=secret", "example.com"),
        ("/home/user/videos/goat.mp4", "local source"),
    ],
)
def test_describe_source_keeps_only_host_and_video_id(source: str, expected: str) -> None:
    """The source is reduced to host + video id so tokens in the URL never reach the channel."""
    assert describe_source(source) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(42, "42s"), (305, "5m 5s"), (3600 * 5 + 60, "5h 1m"), (86400 + 3600 * 2 + 61, "1d 2h 1m")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    """Durations are short and human readable."""
    assert format_duration(seconds) == expected


@pytest.mark.asyncio
async def test_heartbeat_after_edge_fallback_reports_verifier_unavailable(
    notifier: ServiceNotifier, clock: FakeClock, send: AsyncMock
) -> None:
    """When the edge model fails to load, the heartbeat must not claim a healthy verifier."""
    clear_edge_verifier_cache()
    notifier.stream_connected()
    with patch(
        "is_goat_burning.fire_detection.edge_detector.EdgeInferenceDetector",
        side_effect=EdgeModelError("corrupt"),
    ):
        create_fire_detector(
            100, np.zeros(3, np.uint8), np.full(3, 255, np.uint8), use_open_cl=False, use_cuda=False, strategy="hybrid_edge"
        )
    clear_edge_verifier_cache()
    await notifier.drain()
    clock.advance(HEARTBEAT)
    notifier.check_timers()
    await notifier.drain()

    down, heartbeat = sent(send)
    assert "EdgeInferenceDetector" in down
    assert "unavailable" in down.lower()
    assert "verifier: unavailable" in heartbeat.lower()
    assert "EdgeModelError" in heartbeat


def test_failure_reason_names_the_first_real_exception() -> None:
    """The outage cause is the type name of the first non-cancellation exception, nested groups flattened."""
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [TimeoutError("x")]), ValueError("y")])
    assert failure_reason(nested) == "TimeoutError"
    assert failure_reason(ConnectionError("z")) == "ConnectionError"
    assert failure_reason(None) == "stream ended"
