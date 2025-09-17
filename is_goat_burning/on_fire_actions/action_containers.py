"""Provides container classes to manage and orchestrate on-fire actions."""

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any


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
