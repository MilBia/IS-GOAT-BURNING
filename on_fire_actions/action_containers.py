from functools import partial
from typing import Callable


class OnceAction:
    was_send: bool = False
    action: Callable

    def __init__(self, function, *args, **kwargs):
        self.action = partial(function, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.was_send:
            return
        self.action()
        self.was_send = True
