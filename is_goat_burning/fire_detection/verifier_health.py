"""Verifier error type and verifier health notifications.

Second-stage verifiers (Gemini, Decisions API, edge model) raise :class:`VerifierError` from ``verify()`` when they
could not produce a verdict, so ``HybridFireDetector`` can tell "the verifier said no" from "the verifier did not
answer" and apply ``VERIFIER_FAILURE_POLICY``.

When the verifier becomes unavailable (N consecutive errors or the hourly call cap) and when it recovers,
``HybridFireDetector`` publishes a :class:`VerifierStatus` to the listeners registered here. This is the hook for a
service/maintenance channel: register a listener with :func:`add_verifier_status_listener`. Listeners are called
synchronously from the detection loop, so they must be quick (e.g. put the status on a queue) and must not block.
A listener that raises is logged and skipped; it never affects detection.

The detector is rebuilt on every stream reconnect, so its error counters start from zero after a reconnect; an
outage that spans a reconnect is announced again.
"""

from collections.abc import Callable
import contextlib
from dataclasses import dataclass

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


def add_verifier_status_listener(listener: VerifierStatusListener) -> None:
    """Registers a listener for verifier availability changes.

    Args:
        listener: A quick, non-blocking callable taking a :class:`VerifierStatus`.
    """
    _listeners.append(listener)


def remove_verifier_status_listener(listener: VerifierStatusListener) -> None:
    """Unregisters a listener; unknown listeners are ignored.

    Args:
        listener: A listener previously passed to :func:`add_verifier_status_listener`.
    """
    with contextlib.suppress(ValueError):
        _listeners.remove(listener)


def publish_verifier_status(status: VerifierStatus) -> None:
    """Delivers a status change to every listener, isolating listener failures.

    Args:
        status: The availability change to publish.
    """
    for listener in list(_listeners):
        try:
            listener(status)
        except Exception:
            logger.exception("Verifier status listener failed.")
