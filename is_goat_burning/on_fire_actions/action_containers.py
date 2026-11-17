"""Provides container classes to manage and orchestrate on-fire actions."""

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import time
from typing import Any

from is_goat_burning.logger import get_logger

logger = get_logger("FireEventAction")


def _action_name(action: Callable[..., Any]) -> str:
    """Returns a log-safe name for an action (no bound arguments)."""
    func = action.func if isinstance(action, partial) else action
    return getattr(func, "__name__", type(func).__name__)


@dataclass(init=False, repr=False, eq=False, order=False, kw_only=True, slots=True)
class OnceAction:
    """A container that sends one alert per fire event, with a minimum spacing.

    The container starts *armed*. The first call while armed runs all wrapped
    actions concurrently; once the alert has been sent it is *disarmed*, so
    further calls for the same fire event are no-ops. `rearm` (called when the
    fire is extinguished, or when the stream reconnects mid-fire) arms it again
    for the next event.

    Consecutive alerts are at least `cooldown_seconds` apart. A call that arrives
    while armed but inside the cooldown is not dropped: it is deferred until the
    cooldown ends and sent then, unless `rearm` (the fire went out) cancels it
    first. A state flicker that settles inside the window thus yields one alert.

    Success contract: each wrapped action reports failure by raising or by
    returning ``False``; any other return value (``None``, ``True``) is success.
    The alert counts as sent (container disarmed, cooldown started) when at least
    one action succeeded; failed actions are logged. If every action failed, the
    container stays armed and the next call retries. The detector only signals
    on a no-fire -> fire transition, so during one ongoing fire that next call
    comes only after a re-detection (extinguish, then fire) or a stream
    reconnect, not on every frame.

    Attributes:
        was_performed (bool): True once the alert for the current fire event has
            been sent (the container is disarmed); reset by `rearm`.
        actions (list[Callable[[], Awaitable[Any]]]): A list of the awaitable
            actions to be executed.
        cooldown_seconds (float): The minimum number of seconds between alerts.
    """

    was_performed: bool
    actions: list[Callable[[], Awaitable[Any]]]
    cooldown_seconds: float
    _clock: Callable[[], float]
    _last_alert_time: float | None
    _deferred_task: asyncio.Task[None] | None
    _deferred_sending: bool
    _unverified: bool
    _lock: asyncio.Lock
    stop_timeout_seconds: float

    def __init__(
        self,
        actions: list[tuple[type | Callable[..., Any], dict[str, Any]]],
        cooldown_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        stop_timeout_seconds: float = 5.0,
    ) -> None:
        """Initializes the OnceAction container.

        Args:
            actions: A list of tuples, where each tuple contains an action
                (either a class or a function) and a dictionary of keyword
                arguments to initialize or call it with.
            cooldown_seconds: The minimum number of seconds between two alerts.
                ``0`` disables the spacing.
            clock: A monotonic clock returning seconds (injectable for tests).
            stop_timeout_seconds: How long `stop` waits for a deferred alert that
                is already being sent before cancelling it.
        """
        self.actions = []
        self.was_performed = False
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._last_alert_time = None
        self._deferred_task = None
        self._deferred_sending = False
        self._unverified = False
        self._lock = asyncio.Lock()
        self.stop_timeout_seconds = stop_timeout_seconds
        for action, kwargs in actions:
            if isinstance(action, type):
                self.actions.append(action(**kwargs))
            else:
                self.actions.append(partial(action, **kwargs))

    def _cooldown_remaining(self) -> float:
        """Returns the number of seconds left in the cooldown (``0`` if none)."""
        if self._last_alert_time is None:
            return 0.0
        return max(0.0, self._last_alert_time + self.cooldown_seconds - self._clock())

    async def __call__(self, *args: Any, unverified: bool = False, **kwargs: Any) -> None:
        """Sends the alert for the current fire event if it has not been sent yet.

        Inside the cooldown the alert is deferred until the cooldown ends.

        Args:
            *args: Ignored.
            unverified: True if the fire was reported while the verifier was unavailable. The latest value before
                the alert is sent wins; it is then passed to every action as ``unverified=True`` (actions must accept
                that keyword). Verified alerts call the actions without arguments.
            **kwargs: Ignored.
        """
        async with self._lock:
            if self.was_performed:
                return
            self._unverified = unverified
            remaining = self._cooldown_remaining()
            if remaining > 0:
                if self._deferred_task is None or self._deferred_task.done():
                    logger.info(f"Fire alert deferred: {remaining:.0f}s of the alert cooldown remain.")
                    self._deferred_task = asyncio.create_task(self._send_after(remaining))
                return
            await self._perform()

    async def _send_after(self, delay: float) -> None:
        """Sends the deferred alert once the cooldown has elapsed.

        Args:
            delay: Seconds to wait before sending.
        """
        await asyncio.sleep(delay)
        async with self._lock:
            # From here on `rearm` must not cancel the send: it waits on the lock instead.
            self._deferred_sending = True
            try:
                if not self.was_performed:
                    await self._perform()
            finally:
                self._deferred_sending = False

    async def _perform(self) -> None:
        """Runs all actions and records the alert as sent if any succeeded."""
        kwargs = {"unverified": True} if self._unverified else {}
        results = await asyncio.gather(*[action(**kwargs) for action in self.actions], return_exceptions=True)
        failures = 0
        for action, result in zip(self.actions, results, strict=True):
            # Log names and exception types only: reprs/messages may carry webhook URLs or credentials.
            if isinstance(result, BaseException):
                failures += 1
                logger.error(f"Alert action {_action_name(action)} failed with {type(result).__name__}.")
            elif result is False:
                failures += 1
                logger.error(f"Alert action {_action_name(action)} reported failure.")
        if self.actions and failures == len(self.actions):
            logger.error(
                "No alert action succeeded; the alert stays armed and is retried only after the fire is "
                "re-detected (extinguish, then fire) or the stream reconnects."
            )
            return
        self.was_performed = True
        self._last_alert_time = self._clock()

    async def rearm(self) -> None:
        """Arms the container for the next fire event.

        Cancels a pending deferred alert (the fire it was for is gone). An alert
        that is already being sent is allowed to finish first.
        """
        if self._deferred_task is not None and not self._deferred_sending:
            self._deferred_task.cancel()
            self._deferred_task = None
        async with self._lock:
            if self.was_performed:
                logger.info("Fire alert re-armed for the next fire event.")
            self.was_performed = False

    async def stop(self) -> None:
        """Cancels a pending deferred alert and waits for it to finish.

        A deferred alert that is still waiting for the cooldown is cancelled.
        One that is already being sent gets up to `stop_timeout_seconds` to
        finish; only if it is still running after that is it cancelled, so
        shutdown is never blocked indefinitely by a hanging send.
        """
        task = self._deferred_task
        self._deferred_task = None
        if task is not None and not task.done():
            if self._deferred_sending:
                await asyncio.wait({task}, timeout=self.stop_timeout_seconds)
                if task.done():
                    return
                logger.warning("Shutdown: the alert being sent did not finish in time and is cancelled.")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise


@dataclass(init=False, repr=False, eq=False, order=False, kw_only=True, slots=True)
class FireEventAction:
    """A container that drives a persistent, event-driven action from a queue.

    Unlike `OnceAction`, which fires its wrapped actions once per fire event, this
    container runs a long-lived background task that consumes items (e.g. video
    chunk file paths) from a shared `asyncio.Queue` and forwards each one to the
    wrapped action. It is intended for actions that must react to a stream of
    events over the lifetime of the application, such as uploading each new
    video chunk to Discord.

    Attributes:
        event_queue (asyncio.Queue[str]): The queue of items to forward to the
            wrapped action.
        action (Callable[[str], Awaitable[Any]]): The wrapped action, invoked
            once per item consumed from the queue.
    """

    event_queue: asyncio.Queue[str]
    action: Callable[[str], Awaitable[Any]]
    _task: asyncio.Task[None] | None

    def __init__(
        self,
        event_queue: asyncio.Queue[str],
        action: tuple[type | Callable[..., Any], dict[str, Any]],
    ) -> None:
        """Initializes the FireEventAction container.

        Args:
            event_queue: A shared queue from which items (file paths) are read.
            action: A tuple of an action (a class or a function) and a dictionary
                of keyword arguments used to initialize or bind it. The resulting
                callable is invoked with a single positional item per event.
        """
        self.event_queue = event_queue
        self._task = None
        act, kwargs = action
        if isinstance(act, type):
            self.action = act(**kwargs)
        else:
            self.action = partial(act, **kwargs)

    def start(self) -> None:
        """Starts the background task that consumes the event queue."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        """Consumes items from the queue and forwards each to the action."""
        while True:
            try:
                item = await self.event_queue.get()
                try:
                    await self.action(item)
                finally:
                    self.event_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Fire event action task is shutting down.")
                break
            except Exception:
                logger.exception("Fire event action encountered an error, but will continue running.")

    async def stop(self) -> None:
        """Cancels the background task and waits for it to finish cleanly."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                if not self._task.cancelled():
                    raise
                logger.debug("Fire event action task successfully cancelled.")
