"""Initializes the fire_detection module, exporting the main detector class.

`StreamFireDetector` is resolved lazily (PEP 562). Importing it eagerly would pull in `is_goat_burning.stream`, which
imports `stream_recording`, which needs `fire_detection.signal_handler` -- a cycle that broke importing any of these
modules first (e.g. running a single test file).
"""

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from .detector_core import StreamFireDetector

__all__ = ["StreamFireDetector"]


def __getattr__(name: str) -> Any:
    """Lazily import `StreamFireDetector` on first access.

    Args:
        name: The attribute being looked up on the package.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If the attribute is not exported by this package.
    """
    if name == "StreamFireDetector":
        from .detector_core import StreamFireDetector

        return StreamFireDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
