"""Verifier error type and verifier health notifications.

Second-stage verifiers (Gemini, Decisions API, edge model) raise :class:`VerifierError` from ``verify()`` when they
could not produce a verdict, so ``HybridFireDetector`` can tell "the verifier said no" from "the verifier did not
answer" and apply ``VERIFIER_FAILURE_POLICY``.

When the verifier becomes unavailable (N consecutive errors or the hourly call cap) and when it recovers,
``HybridFireDetector`` publishes a :class:`VerifierStatus` to the listeners registered here. This is the hook for a
service/maintenance channel: register a listener with :func:`add_verifier_status_listener`. Listeners are called
synchronously from the detection loop, so they must be quick (e.g. put the status on a queue) and must not block.
A listener that raises is logged and skipped; it never affects detection.

The detector is rebuilt on every stream reconnect, so its error counters start from zero after a reconnect. The
last published availability is therefore kept here, per verifier class: listeners only receive real changes, so an
outage that spans a reconnect is not announced twice and its recovery (seen by the new detector) is not lost.
"""

from collections.abc import Callable
import contextlib
from dataclasses import dataclass
import inspect

from is_goat_burning.logger import get_logger

logger = get_logger("VerifierHealth")


class VerifierError(Exception):
    """Raised by a verifier's ``verify()`` when it could not produce a verdict (API error, timeout, bad answer)."""


@dataclass(frozen=True, slots=True)
class VerifierStatus:
    """A change of verifier availability.

    Attributes:
        available: ``False`` when the verifier became unavailable, ``True`` when it recovered.
        verifier: The verifier class name.
        reason: Why it is unavailable (e.g. ``"3 consecutive errors"``, ``"call limit 60/h reached"``); on recovery
            a short description of the recovery.
        policy: The active ``VERIFIER_FAILURE_POLICY`` (``alert_unverified`` or ``suppress``).
    """

    available: bool
    verifier: str
    reason: str
    policy: str


VerifierStatusListener = Callable[[VerifierStatus], object]

_listeners: list[VerifierStatusListener] = []
# Last published availability per verifier class name; a verifier not listed here is assumed available.
_last_available: dict[str, bool] = {}


def add_verifier_status_listener(listener: VerifierStatusListener) -> None:
    """Registers a listener for verifier availability changes.

    Args:
        listener: A quick, non-blocking callable taking a :class:`VerifierStatus`.

    Raises:
        TypeError: If ``listener`` is a coroutine function (it would never be awaited).
    """
    if inspect.iscoroutinefunction(listener):
        raise TypeError("Verifier status listeners must be synchronous; schedule async work yourself (e.g. queue.put_nowait).")
    _listeners.append(listener)


def remove_verifier_status_listener(listener: VerifierStatusListener) -> None:
    """Unregisters a listener; unknown listeners are ignored.

    Args:
        listener: A listener previously passed to :func:`add_verifier_status_listener`.
    """
    with contextlib.suppress(ValueError):
        _listeners.remove(listener)


def reset_verifier_status() -> None:
    """Forgets the last published availability of every verifier (all are assumed available again)."""
    _last_available.clear()


def publish_verifier_status(status: VerifierStatus) -> None:
    """Delivers an availability change to every listener, isolating listener failures.

    A status equal to the last one published for the same verifier is dropped, so callers may publish on every
    observation (e.g. each successful call) and listeners still see only transitions.

    Args:
        status: The verifier's current availability.
    """
    if _last_available.get(status.verifier, True) == status.available:
        return
    _last_available[status.verifier] = status.available
    for listener in list(_listeners):
        try:
            listener(status)
        except Exception:
            logger.exception("Verifier status listener failed.")
