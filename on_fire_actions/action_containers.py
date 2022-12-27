from functools import partial
from typing import Callable


class OnceAction:
    """
    Class contain provided method and allow it to perform only once at first call.
    """

    was_performed: bool = False
    action: Callable

    def __init__(self, function, *args, **kwargs):
        self.action = partial(function, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.was_performed:
            return
        self.action()
        self.was_performed = True
