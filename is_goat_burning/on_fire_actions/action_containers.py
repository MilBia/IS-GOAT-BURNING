"""Provides container classes to manage and orchestrate on-fire actions."""

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from is_goat_burning.logger import get_logger

logger = get_logger("FireEventAction")


@dataclass(init=False, repr=False, eq=False, order=False, kw_only=True, slots=True)
class OnceAction:
    """A container that ensures a set of actions are performed only once.

    This class wraps one or more awaitable actions and guarantees that they are
    executed only on the first `__call__`. Subsequent calls will have no effect.
    It supports initializing actions from classes or partial functions.

    Attributes:
        was_performed (bool): A flag indicating if the actions have been run.
        actions (list[Callable[[], Awaitable[Any]]]): A list of the awaitable
            actions to be executed.
    """

    was_performed: bool
    actions: list[Callable[[], Awaitable[Any]]]

    def __init__(self, actions: list[tuple[type | Callable[..., Any], dict[str, Any]]]) -> None:
        """Initializes the OnceAction container.

        Args:
            actions: A list of tuples, where each tuple contains an action
                (either a class or a function) and a dictionary of keyword
                arguments to initialize or call it with.
        """
        self.actions = []
        self.was_performed = False
        for action, kwargs in actions:
            if isinstance(action, type):
                self.actions.append(action(**kwargs))
            else:
                self.actions.append(partial(action, **kwargs))

    async def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Executes all contained actions if they have not yet been performed.

        This method is idempotent. It will run the actions on the first call
        and do nothing on all subsequent calls.

        Args:
            *args: Ignored.
            **kwargs: Ignored.
        """
        if self.was_performed:
            return
        await asyncio.gather(*[action() for action in self.actions])
        self.was_performed = True


@dataclass(init=False, repr=False, eq=False, order=False, kw_only=True, slots=True)
class FireEventAction:
    """A container that drives a persistent, event-driven action from a queue.

    Unlike `OnceAction`, which fires its wrapped actions a single time, this
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
