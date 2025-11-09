import asyncio
import contextlib
from unittest.mock import Mock

import pytest

from is_goat_burning.fire_detection.signal_handler import SignalHandler


def test_signal_handler_is_singleton() -> None:
    instance1 = SignalHandler()
    instance2 = SignalHandler()
    assert instance1 is instance2


@pytest.mark.asyncio
async def test_fire_detection_event_logic() -> None:
    handler = SignalHandler()
    handler.reset_fire_event()
    assert not handler.is_fire_detected()
    handler.fire_detected()
    assert handler.is_fire_detected()
    handler.fire_detected()
    assert handler.is_fire_detected()
    handler.reset_fire_event()
    assert not handler.is_fire_detected()


@pytest.mark.asyncio
async def test_fire_extinguished_event_logic() -> None:
    handler = SignalHandler()
    handler.reset_fire_extinguished_event()
    assert not handler.is_fire_extinguished()
    handler.fire_extinguished()
    assert handler.is_fire_extinguished()
    handler.fire_extinguished()
    assert handler.is_fire_extinguished()
    handler.reset_fire_extinguished_event()
    assert not handler.is_fire_extinguished()


@pytest.mark.asyncio
async def test_exit_gracefully_cancels_main_task() -> None:
    handler = SignalHandler()
    handler._running = True

    async def dummy_task_func():
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(30)

    main_task = asyncio.create_task(dummy_task_func())
    handler.set_main_task(main_task)
    assert handler.is_running()
    handler.exit_gracefully(Mock(), Mock())
    await asyncio.sleep(0)
    assert main_task.cancelled()
    assert not handler.is_running()
