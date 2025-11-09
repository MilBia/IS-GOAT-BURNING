"""Unit tests for the SignalHandler singleton.
These tests verify the singleton pattern, the state management of the fire
detection event, and the cancellation of the main task on graceful exit.
"""

import asyncio
import contextlib
from unittest.mock import Mock

import pytest

from is_goat_burning.fire_detection.signal_handler import SignalHandler


@pytest.fixture
def handler() -> SignalHandler:
    """Fixture to provide a clean SignalHandler instance for each test."""
    sh = SignalHandler()
    # Reset to a known clean state before each test to ensure isolation
    sh._running = True
    sh.main_task = None
    sh.reset_fire_event()
    sh.reset_fire_extinguished_event()
    return sh


def test_signal_handler_is_singleton(handler: SignalHandler) -> None:
    """Verifies that SignalHandler consistently returns the same instance."""
    instance1 = handler
    instance2 = SignalHandler()
    assert instance1 is instance2, "SignalHandler is not a singleton."


@pytest.mark.asyncio
async def test_fire_detection_event_logic(handler: SignalHandler) -> None:
    """Tests the set, check, and reset logic of the fire detected event."""
    assert handler.is_fire_detected() is False, "Event should be clear initially."

    handler.fire_detected()
    assert handler.is_fire_detected() is True, "Event should be set after fire_detected()."
    handler.fire_detected()
    assert handler.is_fire_detected() is True, "Event should remain set."

    handler.reset_fire_event()
    assert handler.is_fire_detected() is False, "Event should be clear after reset."


@pytest.mark.asyncio
async def test_fire_extinguished_event_logic(handler: SignalHandler) -> None:
    """Tests the set, check, and reset logic of the fire extinguished event."""
    assert handler.is_fire_extinguished() is False, "Event should be clear initially."

    handler.fire_extinguished()
    assert handler.is_fire_extinguished() is True, "Event should be set after fire_extinguished()."
    handler.fire_extinguished()
    assert handler.is_fire_extinguished() is True, "Event should remain set."

    handler.reset_fire_extinguished_event()
    assert handler.is_fire_extinguished() is False, "Event should be clear after reset."


@pytest.mark.asyncio
async def test_exit_gracefully_cancels_main_task(handler: SignalHandler) -> None:
    """Verifies that exit_gracefully() cancels the registered main task."""

    async def dummy_task_func() -> None:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(30)

    main_task = asyncio.create_task(dummy_task_func())
    handler.set_main_task(main_task)

    assert handler.is_running() is True
    handler.exit_gracefully(Mock(), Mock())
    await asyncio.sleep(0)  # Allow cancellation to propagate

    assert main_task.cancelled()
    assert handler.is_running() is False
