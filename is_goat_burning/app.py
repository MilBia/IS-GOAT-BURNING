"""The main application class that orchestrates the fire detection process."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any

from yt_dlp.utils import DownloadError

from is_goat_burning.config import settings
from is_goat_burning.fire_detection import StreamFireDetector
from is_goat_burning.fire_detection.escalation_log import check_escalation_log_directory
from is_goat_burning.fire_detection.escalation_log import close_escalation_log
from is_goat_burning.fire_detection.signal_handler import SignalHandler
from is_goat_burning.logger import get_logger
from is_goat_burning.on_fire_actions import FireEventAction
from is_goat_burning.on_fire_actions import OnceAction
from is_goat_burning.on_fire_actions import SendEmail
from is_goat_burning.on_fire_actions import SendToDiscord
from is_goat_burning.on_fire_actions import SendVideoToDiscord
from is_goat_burning.service_notifier import ServiceNotifier
from is_goat_burning.service_notifier import failure_reason

logger = get_logger("Application")

_SLEEP_SLICE_SECONDS = 1.0
_MAX_MESSAGE_LENGTH = 200
_URL_PATTERN = re.compile(r"https?://\S+")
_EXPECTED_MESSAGE_PREFIXES = ("Failed to resolve stream URL", "Could not extract stream URL", "Could not open video stream")


@dataclass(frozen=True, eq=False, slots=True)
class ActionEvent:
    """An item of the application's action queue.

    Attributes:
        kind: ``"FIRE"`` or ``"EXTINGUISHED"``.
        unverified: True if the fire was reported while the verifier was unavailable.
        frame: The host copy of the frame that triggered the fire (``np.ndarray``), or None.
    """

    kind: str
    unverified: bool = False
    frame: Any = None


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """Flattens an exception group into its leaf exceptions."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for inner in exc.exceptions for leaf in _leaf_exceptions(inner)]
    return [exc]


def _is_expected_failure(exc: BaseException) -> bool:
    """Whether the exception is a known, routine stream failure (offline stream, resolution error, timeout)."""
    if isinstance(exc, DownloadError | ConnectionError | TimeoutError):
        return True
    return isinstance(exc, ValueError | RuntimeError) and str(exc).startswith(_EXPECTED_MESSAGE_PREFIXES)


def _short_message(exc: BaseException) -> str:
    """A one-line, URL-free description of the failure (a yt-dlp cause is preferred: it says why)."""
    cause = exc.__cause__ if isinstance(exc.__cause__, DownloadError) else exc
    first_line = (str(cause).strip().splitlines() or [""])[0]
    text = _URL_PATTERN.sub("<url>", first_line)
    if len(text) > _MAX_MESSAGE_LENGTH:
        text = text[: _MAX_MESSAGE_LENGTH - 3] + "..."
    return text


class Application:
    """Encapsulates the fire detection application's state and lifecycle.

    This class initializes all major components of the application, including
    the signal handler and action handlers. It manages the main asyncio tasks,
    including the detector and action manager, and ensures a graceful shutdown
    and stream reconnection.

    Attributes:
        signal_handler (SignalHandler): The application-wide signal handler.
        action_queue (asyncio.Queue[ActionEvent]): A queue for communicating events (with the fire's frame).
        video_event_queue (asyncio.Queue[str]): A queue carrying the paths of
            saved video chunks to event-driven actions.
        on_fire_action_handler (OnceAction | None): The handler that sends one
            alert per fire event; re-armed on EXTINGUISHED events.
        video_event_action (FireEventAction | None): The persistent handler that
            forwards saved video chunk paths to its wrapped action.
        action_manager_task (asyncio.Task | None): The task managing the event
            queue and triggering actions.
        service_notifier (ServiceNotifier): Sends service messages (startup, heartbeat,
            stream outage, verifier health, shutdown) to `DISCORD__SERVICE_HOOKS`;
            a no-op when no service webhooks are configured.
    """

    def __init__(self) -> None:
        """Initializes the Application instance."""
        self.signal_handler = SignalHandler()
        self.action_queue: asyncio.Queue[ActionEvent] = asyncio.Queue()
        self.video_event_queue: asyncio.Queue[str] = asyncio.Queue()
        self.on_fire_action_handler: OnceAction | None = None
        self.video_event_action: FireEventAction | None = None
        self.action_manager_task: asyncio.Task | None = None
        self._setup_actions()
        self.service_notifier = ServiceNotifier(
            settings.discord.service_hooks,
            strategy=settings.detection_strategy,
            source=settings.source,
            heartbeat_interval=settings.discord.service_heartbeat_interval_seconds,
            stream_down_threshold=settings.discord.service_stream_down_warning_seconds,
            send_options={
                "timeout": settings.discord.timeout_seconds,
                "max_attempts": settings.discord.max_attempts,
                "base_delay": settings.discord.retry_base_delay_seconds,
                "max_delay": settings.discord.retry_max_delay_seconds,
            },
        )

    def _setup_actions(self) -> None:
        """Builds the list of notification actions based on settings."""
        actions: list[tuple[type | Callable[..., Any], dict[str, Any]]] = []
        if settings.email.use_emails:
            actions.append(
                (
                    SendEmail,
                    {
                        "sender": settings.email.sender,
                        "sender_password": settings.email.sender_password.get_secret_value(),
                        "recipients": settings.email.recipients,
                        "subject": settings.email.subject,
                        "message": settings.email.message,
                        "host": settings.email.email_host,
                        "port": settings.email.email_port,
                    },
                )
            )
        if settings.discord.use_discord:
            actions.append(
                (
                    SendToDiscord,
                    {
                        "message": settings.discord.message,
                        "webhooks": settings.discord.hooks,
                        "attach_frame": settings.discord.attach_frame,
                        "timeout": settings.discord.timeout_seconds,
                        "max_attempts": settings.discord.max_attempts,
                        "base_delay": settings.discord.retry_base_delay_seconds,
                        "max_delay": settings.discord.retry_max_delay_seconds,
                    },
                )
            )
        self.on_fire_action_handler = OnceAction(
            actions, cooldown_seconds=settings.alert_cooldown_seconds, retry_seconds=settings.alert_retry_seconds
        )

        if settings.discord.use_discord and settings.discord.send_video_chunks:
            self.video_event_action = FireEventAction(
                event_queue=self.video_event_queue,
                action=(
                    SendVideoToDiscord,
                    {
                        "webhooks": settings.discord.hooks,
                        "file_limit_mb": settings.discord.file_limit_mb,
                    },
                ),
            )

    @property
    def _video_chunks_enabled(self) -> bool:
        """Whether saved video chunks should be forwarded to an event action."""
        return self.video_event_action is not None

    async def _action_manager(self) -> None:
        """Manages the action queue, triggering handlers for received events."""
        while True:
            try:
                event = await self.action_queue.get()
                try:
                    if event.kind == "FIRE":
                        label = "an UNVERIFIED FIRE" if event.unverified else "FIRE"
                        logger.info(f"Action manager received {label} event. Triggering actions.")
                        if self.on_fire_action_handler:
                            await self.on_fire_action_handler(unverified=event.unverified, frame=event.frame)
                    elif event.kind == "EXTINGUISHED":
                        logger.info("Action manager received EXTINGUISHED event. Re-arming the fire alert.")
                        if self.on_fire_action_handler:
                            await self.on_fire_action_handler.rearm()
                finally:
                    self.action_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Action manager task is shutting down.")
                break
            except Exception:
                logger.exception("Action manager encountered an error, but will continue running.")

    async def _queue_fire_event(self, unverified: bool = False, frame: Any = None) -> None:
        """A callback for the detector to queue a fire event (``StreamFireDetector`` ``on_fire_action``).

        Args:
            unverified: True if the fire was reported while the verifier was unavailable.
            frame: The host copy of the triggering frame (``np.ndarray``), forwarded to the alert actions.
        """
        self.service_notifier.fire_event(unverified=unverified)
        await self.action_queue.put(ActionEvent("FIRE", unverified=unverified, frame=frame))

    async def _queue_extinguished_event(self) -> None:
        """A callback for the detector to queue a fire-extinguished event."""
        await self.action_queue.put(ActionEvent("EXTINGUISHED"))

    @staticmethod
    def _failure_log(group: BaseException) -> Callable[[float], None]:
        """Builds the logging call for a failed connection attempt or stream.

        Known failures (stream offline, URL resolution, connection, timeouts) are one WARNING line without a traceback
        or stream URL; anything else keeps its traceback at ERROR.

        Args:
            group: The exception group raised by the detector or the connection attempt.

        Returns:
            A function taking the delay before the next attempt and logging the failure.
        """
        leaves = _leaf_exceptions(group)
        if all(_is_expected_failure(leaf) for leaf in leaves):
            first = leaves[0]
            summary = f"{type(first).__name__}: {_short_message(first)}"
            return lambda delay: logger.warning(f"Stream unavailable ({summary}). Reconnecting in {delay}s...")
        return lambda delay: logger.error(f"Detector or connection attempt failed. Reconnecting in {delay}s...", exc_info=group)

    async def _backoff_sleep(self, delay: float) -> None:
        """Sleeps up to ``delay`` seconds, returning within about a second once a shutdown was requested.

        Args:
            delay: The wait in seconds.
        """
        remaining = delay
        while remaining > 0 and self.signal_handler.is_running():
            step = min(_SLEEP_SLICE_SECONDS, remaining)
            await asyncio.sleep(step)
            remaining -= step

    async def run(self) -> None:
        """Starts and manages the main application tasks with a retry loop."""
        self.action_manager_task = asyncio.create_task(self._action_manager())
        if self.video_event_action:
            self.video_event_action.start()
        self.service_notifier.start()
        await check_escalation_log_directory()

        delay = settings.reconnect_delay_seconds
        while self.signal_handler.is_running():
            detector = None
            failure = failure_reason(None)
            failure_log: Callable[[float], None] | None = None
            session_ok = False

            def on_first_frame() -> None:
                nonlocal session_ok
                session_ok = True
                self.service_notifier.stream_connected()

            try:
                logger.info(f"Attempting to start stream from source: {settings.source}")
                detector = await StreamFireDetector.create(
                    src=settings.source,
                    threshold=settings.fire_detection_threshold,
                    video_output=settings.video_output,
                    on_fire_action=self._queue_fire_event,
                    checks_per_second=settings.checks_per_second,
                    event_queue=self.video_event_queue if self._video_chunks_enabled else None,
                    on_extinguished_action=self._queue_extinguished_event,
                    on_first_frame=on_first_frame,
                )
                async with asyncio.TaskGroup() as tg:
                    detector_task = tg.create_task(detector())
                    self.signal_handler.set_main_task(detector_task)

            except* Exception as exc_group:
                # We catch all exceptions here. The loop's continuation is decided
                # by `signal_handler.is_running()` at the top of the loop.
                # `CancelledError` is a result of the signal handler setting the
                # running state to False, so we only need to log other errors.
                _, others = exc_group.split(asyncio.CancelledError)
                if others:
                    failure = failure_reason(others)
                    failure_log = self._failure_log(others)
            finally:
                if detector and detector.stream:
                    await detector.stream.stop()
                if detector is not None and detector.fire_is_currently_detected is True:
                    # The next detector starts fire-free and would never signal the end of this fire.
                    logger.info("Stream stopped during a fire. Re-arming the fire alert for the next connection.")
                    await self._queue_extinguished_event()

                if self.signal_handler.is_running():
                    if session_ok:
                        delay = settings.reconnect_delay_seconds  # The stream worked: start the backoff over.
                    await self._wait_before_reconnect(delay, failure, failure_log)
                    delay = min(delay * 2, settings.reconnect_max_delay_seconds)

    async def _wait_before_reconnect(self, delay: float, failure: str, failure_log: Callable[[float], None] | None) -> None:
        """Reports the stopped stream and waits ``delay`` seconds (cut short by a shutdown request).

        Args:
            delay: The wait before the next connection attempt.
            failure: The failure reason for the service notifier.
            failure_log: The logging call built by :meth:`_failure_log`, or None when the stream just ended.
        """
        self.service_notifier.stream_failed(failure)
        if failure_log is not None:
            failure_log(delay)
        else:
            logger.info(f"Stream has stopped. Reconnecting in {delay}s...")
        try:
            await self._backoff_sleep(delay)
        except asyncio.CancelledError:
            # If a shutdown signal arrives during sleep, we exit gracefully.
            logger.info("Reconnect wait cancelled by shutdown signal.")

    async def shutdown(self) -> None:
        """Gracefully shuts down all running application tasks."""
        logger.info("Initiating graceful shutdown.")
        if self.video_event_action:
            await self.video_event_action.stop()
        if self.on_fire_action_handler:
            await self.on_fire_action_handler.stop()

        tasks = []
        if self.action_manager_task and not self.action_manager_task.done():
            self.action_manager_task.cancel()
            tasks.append(self.action_manager_task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Last: the detector has stopped by now, so no further escalations can be queued.
        await close_escalation_log()
        # Last and bounded (stop_timeout): a slow service webhook must not eat the container's stop grace period.
        await self.service_notifier.stop()
        logger.info("Application has shut down.")
