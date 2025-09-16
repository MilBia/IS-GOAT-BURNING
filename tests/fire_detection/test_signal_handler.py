import asyncio
import contextlib

import pytest

from is_goat_burning.fire_detection.signal_handler import SignalHandler


def test_signal_handler_is_singleton():
    """
    Tests that instantiating SignalHandler multiple times returns the exact same object.
    """
    instance1 = SignalHandler()
    instance2 = SignalHandler()

    assert instance1 is instance2, "SignalHandler is not a singleton; instances are not the same object."


@pytest.mark.asyncio
async def test_fire_detection_event_logic():
    """
    Tests the state management of the fire detected event.
    """
    handler = SignalHandler()

    # --- Initial State ---
    # We create a new instance to ensure its state is clean for the test.
    # Due to the singleton nature, this will be the same instance as any other.
    handler.reset_fire_event()  # Explicitly reset state before test
    assert handler.is_fire_detected() is False, "Event should be clear initially."

    # --- Set Event ---
    handler.fire_detected()
    assert handler.is_fire_detected() is True, "Event should be set after fire_detected() is called."
    # Calling it again should have no effect
    handler.fire_detected()
    assert handler.is_fire_detected() is True, "Event should remain set after a second call."

    # --- Reset Event ---
    handler.reset_fire_event()
    assert handler.is_fire_detected() is False, "Event should be clear after being reset."


@pytest.mark.asyncio
async def test_exit_gracefully_cancels_main_task():
    """
    Tests that the exit_gracefully method correctly cancels the main task.
    """
    handler = SignalHandler()

    # Create a dummy task that runs forever until cancelled
    async def dummy_task_func():
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(30)  # A long sleep that will be interrupted

    main_task = asyncio.create_task(dummy_task_func())
    handler.set_main_task(main_task)

    # Trigger the exit signal
    handler.exit_gracefully()

    # Allow the event loop to process the cancellation
    await asyncio.sleep(0)

    assert main_task.cancelled(), "The main task was not cancelled by exit_gracefully."
