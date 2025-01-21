import asyncio
from dataclasses import dataclass
from functools import partial
from typing import Callable


@dataclass(init=False, repr=False, eq=False, order=False, kw_only=True, slots=True)
class OnceAction:
    """
    Class contain provided method and allow it to perform only once at first call.
    """

    was_performed: bool
    actions: list[Callable]

    def __init__(self, actions: list[list[Callable, dict]]):
        self.actions = []
        self.was_performed = False
        for [action, kwargs] in actions:
            if isinstance(action, type):
                self.actions.append(action(**kwargs))
            else:
                self.actions.append(partial(action, **kwargs))

    async def __call__(self, *args, **kwargs):
        if self.was_performed:
            return
        await asyncio.gather(*[action() for action in self.actions])
        self.was_performed = True
