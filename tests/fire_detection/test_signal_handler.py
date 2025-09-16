"""Unit tests for the SignalHandler singleton.

These tests verify the singleton pattern, the state management of the fire
detection event, and the cancellation of the main task on graceful exit.
"""

import asyncio
import contextlib
from unittest.mock import Mock

import pytest

from is_goat_burning.fire_detection.signal_handler import SignalHandler


def test_signal_handler_is_singleton() -> None:
    """Verifies that SignalHandler consistently returns the same instance."""
    instance1 = SignalHandler()
    instance2 = SignalHandler()
    assert instance1 is instance2, "SignalHandler is not a singleton."


@pytest.mark.asyncio
async def test_fire_detection_event_logic() -> None:
    """Tests the set, check, and reset logic of the fire detected event."""
    handler = SignalHandler()
    handler.reset_fire_event()  # Ensure clean state
    assert handler.is_fire_detected() is False, "Event should be clear initially."

    handler.fire_detected()
    assert handler.is_fire_detected() is True, "Event should be set after fire_detected()."
    handler.fire_detected()
    assert handler.is_fire_detected() is True, "Event should remain set."

    handler.reset_fire_event()
    assert handler.is_fire_detected() is False, "Event should be clear after reset."


@pytest.mark.asyncio
async def test_exit_gracefully_cancels_main_task() -> None:
    """Verifies that exit_gracefully() cancels the registered main task."""
    handler = SignalHandler()

    async def dummy_task_func() -> None:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(30)

    main_task = asyncio.create_task(dummy_task_func())
    handler.set_main_task(main_task)
    handler.exit_gracefully(Mock(), Mock())
    await asyncio.sleep(0)  # Allow cancellation to propagate
    assert main_task.cancelled()
