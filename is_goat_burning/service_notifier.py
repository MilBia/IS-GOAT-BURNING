"""Service ("sign of life") notifications on a separate Discord channel.

Silence on the fire-alert channel must be distinguishable from a dead monitor, so the application reports its own
state to a separate list of webhooks (``DISCORD__SERVICE_HOOKS``): startup, a periodic heartbeat, a stream outage
longer than a threshold (once per outage) and the recovery after such an outage, verifier unavailable / recovered
(from :mod:`is_goat_burning.fire_detection.verifier_health`) and a best-effort shutdown message.

All public event methods are synchronous and never block: they update the state and put a message on a bounded
queue (a full queue drops the message with a warning). A single asyncio task delivers the messages and checks the
timers. Delivery failures are logged (never the webhook URLs) and never reach the detection loop.
"""

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
import threading
import time
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse

from is_goat_burning.fire_detection.verifier_health import STANDALONE_POLICY
from is_goat_burning.fire_detection.verifier_health import VerifierStatus
from is_goat_burning.fire_detection.verifier_health import add_verifier_status_listener
from is_goat_burning.fire_detection.verifier_health import remove_verifier_status_listener
from is_goat_burning.logger import get_logger
from is_goat_burning.on_fire_actions.send_to_discord import SendToDiscord

logger = get_logger("ServiceNotifier")

PREFIX = "[Goat monitor]"
_VIDEO_PATH_PREFIXES = ("live", "shorts", "embed", "v")
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400


def format_duration(seconds: float) -> str:
    """Formats a duration as a short human-readable string (``42s``, ``5m 5s``, ``5h 1m``, ``1d 2h 1m``).

    Args:
        seconds: The duration in seconds.

    Returns:
        The formatted duration.
    """
    total = max(0, int(seconds))
    days, rest = divmod(total, _SECONDS_PER_DAY)
    hours, rest = divmod(rest, _SECONDS_PER_HOUR)
    minutes, secs = divmod(rest, _SECONDS_PER_MINUTE)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def describe_source(source: str) -> str:
    """Reduces the stream source to its host and video id, so tokens or local paths never reach the channel.

    Args:
        source: The configured ``SOURCE`` (a URL or a local path).

    Returns:
        ``"<host> (video <id>)"`` for YouTube-style URLs, ``"<host>"`` for other URLs, ``"local source"`` otherwise.
    """
    parsed = urlparse(source)
    host = parsed.hostname
    if not host:
        return "local source"
    video_id = parse_qs(parsed.query).get("v", [None])[0]
    if video_id is None:
        parts = [part for part in parsed.path.split("/") if part]
        if host == "youtu.be" and len(parts) == 1:
            video_id = parts[0]
        elif len(parts) == 2 and parts[0] in _VIDEO_PATH_PREFIXES and "youtube" in host:  # noqa: PLR2004
            video_id = parts[1]
    return f"{host} (video {video_id})" if video_id else host


def failure_reason(exc: BaseException | None) -> str:
    """Names the cause of a stream stop for :meth:`ServiceNotifier.stream_failed` (no message text, no secrets).

    Args:
        exc: The exception (or exception group, flattened to its first leaf) that stopped the stream, or None when
            the stream ended without an error.

    Returns:
        The exception type name, or ``"stream ended"`` for None.
    """
    if exc is None:
        return "stream ended"
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return type(exc).__name__


class ServiceNotifier:
    """Sends service messages about the monitor's own health to dedicated webhooks.

    Construct it always; with no webhooks it is disabled and every method is a no-op, so callers need no checks.

    Stream state API (called from the reconnect loop, synchronous and non-blocking):
        * :meth:`stream_connected` after a stream connection was established;
        * :meth:`stream_failed` after a connection attempt failed or a running stream stopped.

    An outage starts at the first failure after a connection (or at :meth:`start` if the stream never connected);
    repeated failed reconnect attempts do not restart it. When it lasts longer than ``stream_down_threshold`` one
    warning is sent, and the next :meth:`stream_connected` sends a recovery message. Shorter outages are silent.

    Attributes:
        enabled: Whether service notifications are active (at least one webhook configured).
    """

    def __init__(
        self,
        webhooks: list[str],
        *,
        strategy: str,
        source: str,
        heartbeat_interval: float = 86400.0,
        stream_down_threshold: float = 600.0,
        send_options: dict[str, Any] | None = None,
        send: Callable[[str], Awaitable[bool]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        queue_size: int = 32,
        tick_seconds: float = 30.0,
        stop_timeout: float = 3.0,
    ) -> None:
        """Initializes the notifier.

        Args:
            webhooks: The service webhook URLs (``DISCORD__SERVICE_HOOKS``); empty disables the notifier.
            strategy: The detection strategy, named in the startup message.
            source: The stream source; only its host and video id are sent.
            heartbeat_interval: Seconds between heartbeats.
            stream_down_threshold: Seconds of stream outage before a warning is sent.
            send_options: Extra keyword arguments for :class:`SendToDiscord` (timeout, retries).
            send: An async callable delivering one text and returning success; defaults to :class:`SendToDiscord`
                on ``webhooks`` without a frame.
            clock: A monotonic clock (injectable for tests).
            queue_size: Maximum number of undelivered messages; further ones are dropped.
            tick_seconds: Maximum time between timer checks in the background task.
            stop_timeout: Maximum seconds :meth:`stop` waits for pending messages and the shutdown message.
        """
        self._webhooks = list(webhooks)
        self.enabled = bool(self._webhooks)
        self._strategy = strategy
        self._source = describe_source(source)
        self._heartbeat_interval = heartbeat_interval
        self._threshold = stream_down_threshold
        self._send_options = dict(send_options or {})
        self._send = send if send is not None else self._send_to_discord
        self._clock = clock
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_size)
        self._tick = tick_seconds
        self._stop_timeout = stop_timeout
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self._stopped = False

        self._started_at = clock()
        self._next_heartbeat = self._started_at + heartbeat_interval
        self._connected = False
        self._ever_connected = False
        self._connected_since = self._started_at
        self._down_since: float | None = self._started_at
        self._warned = False
        self._last_failure: str | None = None
        self._fire_events = 0
        self._unverified_events = 0
        self._verifiers_down: dict[str, VerifierStatus] = {}

    async def _send_to_discord(self, text: str) -> bool:
        """Delivers one text message to the service webhooks (no frame)."""
        return await SendToDiscord(webhooks=self._webhooks, message=text, attach_frame=False, **self._send_options)()

    # --- Lifecycle -------------------------------------------------------------------------------------------------

    def start(self) -> None:
        """Starts the delivery task, registers the verifier listener and queues the startup message.

        Must be called from the running event loop. A no-op when disabled or already started.
        """
        if not self.enabled or self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        now = self._clock()
        self._started_at = now
        self._next_heartbeat = now + self._heartbeat_interval
        if not self._connected:
            self._down_since = now
        add_verifier_status_listener(self.on_verifier_status)
        self._enqueue(
            f"{PREFIX} Monitoring started. Strategy: {self._strategy}. Source: {self._source}. "
            f"Heartbeat every {format_duration(self._heartbeat_interval)}; "
            f"stream-down warning after {format_duration(self._threshold)}."
        )
        self._task = asyncio.create_task(self._run(), name="service-notifier")

    async def drain(self) -> None:
        """Waits until every queued message has been handled (delivered or failed)."""
        if self._task is None:
            return
        await self._queue.join()

    async def stop(self) -> None:
        """Sends the shutdown message (best effort, bounded by ``stop_timeout``) and stops the task."""
        if self._task is None or self._stopped:
            return
        self._stopped = True
        remove_verifier_status_listener(self.on_verifier_status)
        self._enqueue(f"{PREFIX} Monitoring stopped after {format_duration(self._clock() - self._started_at)}.")
        try:
            await asyncio.wait_for(self._queue.join(), timeout=self._stop_timeout)
        except TimeoutError:
            logger.warning("Service notifications not delivered before shutdown (timed out); dropping them.")
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    # --- Events (synchronous, non-blocking) ------------------------------------------------------------------------

    def stream_connected(self) -> None:
        """Records an established stream connection; reports recovery after an outage that was warned about."""
        if not self.enabled:
            return
        now = self._clock()
        if self._down_since is not None and (self._warned or now - self._down_since >= self._threshold):
            what = "Stream recovered" if self._ever_connected else "Stream connected"
            self._enqueue(f"{PREFIX} {what} after an outage of {format_duration(now - self._down_since)}.")
        self._connected = True
        self._ever_connected = True
        self._connected_since = now
        self._down_since = None
        self._warned = False

    def stream_failed(self, reason: str) -> None:
        """Records a failed connection attempt or a stopped stream; starts an outage if the stream was up.

        Args:
            reason: A short reason without secrets, e.g. the exception type name or ``"stream ended"``.
        """
        if not self.enabled:
            return
        self._last_failure = reason
        if self._connected or self._down_since is None:
            self._down_since = self._clock()
            self._warned = False
        self._connected = False

    def fire_event(self, unverified: bool = False) -> None:
        """Counts a fire event for the next heartbeat.

        Args:
            unverified: Whether the fire was reported while the verifier was unavailable.
        """
        if not self.enabled:
            return
        self._fire_events += 1
        if unverified:
            self._unverified_events += 1

    def on_verifier_status(self, status: VerifierStatus) -> None:
        """Verifier health listener: queues a message on availability changes (never blocks the detect loop).

        Args:
            status: The verifier availability change published by ``verifier_health``.
        """
        if not self.enabled or self._loop is None:
            return
        if threading.get_ident() == self._loop_thread:
            self._handle_verifier_status(status)
        else:
            self._loop.call_soon_threadsafe(self._handle_verifier_status, status)

    def _handle_verifier_status(self, status: VerifierStatus) -> None:
        """Updates the verifier state and queues the message (runs on the event loop thread)."""
        if status.available:
            self._verifiers_down.pop(status.verifier, None)
            again = "detected" if status.policy == STANDALONE_POLICY else "verified"
            self._enqueue(f"{PREFIX} Verifier {status.verifier} recovered; fires are {again} again.")
            return
        self._verifiers_down[status.verifier] = status
        if status.policy == STANDALONE_POLICY:
            outcome = "its errors are treated as no fire, so fires cannot be detected"
        elif status.policy == "suppress":
            outcome = "fires seen by the local detector are SUPPRESSED (no alert)"
        else:
            outcome = "fires seen by the local detector are reported as UNVERIFIED"
        self._enqueue(
            f"{PREFIX} WARNING: verifier {status.verifier} unavailable ({status.reason}); "
            f"policy {status.policy}: {outcome} until it recovers."
        )

    # --- Timers ----------------------------------------------------------------------------------------------------

    def check_timers(self) -> None:
        """Queues the stream-down warning and the heartbeat when they are due."""
        if not self.enabled:
            return
        now = self._clock()
        if self._down_since is not None and not self._warned and now - self._down_since >= self._threshold:
            self._warned = True
            cause = f" (last error: {self._last_failure})" if self._last_failure else ""
            self._enqueue(
                f"{PREFIX} WARNING: the stream has been unavailable for {format_duration(now - self._down_since)}"
                f"{cause}. Fires cannot be detected until it is back."
            )
        if now >= self._next_heartbeat:
            self._next_heartbeat = now + self._heartbeat_interval
            self._enqueue(self._heartbeat_text(now))
            self._fire_events = 0
            self._unverified_events = 0

    def _heartbeat_text(self, now: float) -> str:
        """Builds the heartbeat message."""
        if self._connected:
            stream = f"up for {format_duration(now - self._connected_since)}"
        elif self._down_since is not None:
            state = "DOWN" if self._ever_connected else "not connected yet"
            cause = f", last error: {self._last_failure}" if self._last_failure else ""
            stream = f"{state} for {format_duration(now - self._down_since)}{cause}"
        else:
            stream = "unknown"
        if self._verifiers_down:
            verifier = "UNAVAILABLE: " + ", ".join(f"{s.verifier} ({s.reason})" for s in self._verifiers_down.values())
        elif self._strategy.startswith("hybrid"):
            verifier = "available"
        else:
            verifier = "n/a (no verifier stage)"
        return (
            f"{PREFIX} Heartbeat: running for {format_duration(now - self._started_at)}. Stream: {stream}. "
            f"Fire events since last heartbeat: {self._fire_events} (unverified: {self._unverified_events}). "
            f"Verifier: {verifier}."
        )

    # --- Delivery --------------------------------------------------------------------------------------------------

    def _enqueue(self, text: str) -> None:
        """Queues a message without blocking; drops it (with a warning) when the queue is full."""
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            logger.warning("Service notification queue is full; dropping a message.")

    def _safe_check_timers(self) -> None:
        """Runs :meth:`check_timers`, logging instead of raising."""
        try:
            self.check_timers()
        except Exception:
            logger.exception("Service notifier timer check failed.")

    async def _deliver(self, text: str) -> None:
        """Sends one message; failures are logged by type only (never the webhook URLs)."""
        try:
            delivered = await self._send(text)
        except Exception as exc:  # noqa: BLE001 - a service message must never break the monitor
            logger.error(f"Service notification failed: {type(exc).__name__}.")
            return
        if not delivered:
            logger.warning("Service notification was not accepted by any service webhook.")

    async def _run(self) -> None:
        """Delivers queued messages and checks the timers at least every ``tick_seconds``."""
        while True:
            try:
                text = await asyncio.wait_for(self._queue.get(), timeout=self._tick)
            except TimeoutError:
                self._safe_check_timers()
                continue
            try:
                await self._deliver(text)
            finally:
                self._queue.task_done()
            self._safe_check_timers()
