"""Unit tests for the action container classes.

These tests verify the core idempotency logic of the `OnceAction` class,
ensuring that wrapped actions are only executed once, and the persistent,
queue-driven behavior of the `FireEventAction` class.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from is_goat_burning.on_fire_actions.action_containers import FireEventAction
from is_goat_burning.on_fire_actions.action_containers import OnceAction


@pytest.mark.asyncio
async def test_once_action_calls_action_only_once() -> None:
    """Verifies that a single wrapped action is only called on the first invocation."""
    mock_action = AsyncMock()
    once_action = OnceAction(actions=[(mock_action, {})])

    # Call the container twice
    await once_action()
    await once_action()

    # Assert that the underlying action was only called once
    mock_action.assert_called_once()


@pytest.mark.asyncio
async def test_once_action_with_multiple_actions() -> None:
    """Verifies that all wrapped actions are called on the first invocation."""
    mock_action_1 = AsyncMock()
    mock_action_2 = AsyncMock()
    once_action = OnceAction(actions=[(mock_action_1, {}), (mock_action_2, {})])

    await once_action()

    # Assert that both actions were called
    mock_action_1.assert_called_once()
    mock_action_2.assert_called_once()


async def _wait_for(condition, timeout: float = 1.0) -> None:
    """Polls until ``condition()`` is truthy or the timeout elapses."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Condition was not met within the timeout.")


@pytest.mark.asyncio
async def test_fire_event_action_forwards_queue_items_to_action() -> None:
    """Verifies each item put on the queue is forwarded to the wrapped action."""
    mock_action = AsyncMock()
    queue: asyncio.Queue[str] = asyncio.Queue()
    container = FireEventAction(event_queue=queue, action=(mock_action, {}))
    container.start()

    await queue.put("/path/one.mp4")
    await queue.put("/path/two.mp4")

    await _wait_for(lambda: mock_action.call_count == 2)
    await container.stop()

    mock_action.assert_any_await("/path/one.mp4")
    mock_action.assert_any_await("/path/two.mp4")


@pytest.mark.asyncio
async def test_fire_event_action_instantiates_class_actions() -> None:
    """Verifies a class action is instantiated with the provided kwargs."""

    class RecordingAction:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.calls: list[str] = []

        async def __call__(self, item: str) -> None:
            self.calls.append(f"{self.tag}:{item}")

    queue: asyncio.Queue[str] = asyncio.Queue()
    container = FireEventAction(event_queue=queue, action=(RecordingAction, {"tag": "goat"}))
    container.start()

    await queue.put("/chunk.mp4")
    await _wait_for(lambda: container.action.calls == ["goat:/chunk.mp4"])
    await container.stop()


@pytest.mark.asyncio
async def test_fire_event_action_survives_action_errors() -> None:
    """Verifies a failing action does not stop the consumer task."""
    mock_action = AsyncMock(side_effect=[RuntimeError("boom"), None])
    queue: asyncio.Queue[str] = asyncio.Queue()
    container = FireEventAction(event_queue=queue, action=(mock_action, {}))
    container.start()

    await queue.put("/first.mp4")
    await queue.put("/second.mp4")

    await _wait_for(lambda: mock_action.call_count == 2)
    await container.stop()

    assert mock_action.await_count == 2


@pytest.mark.asyncio
async def test_fire_event_action_stop_cancels_task() -> None:
    """Verifies that stop() cleanly cancels the background task."""
    mock_action = AsyncMock()
    queue: asyncio.Queue[str] = asyncio.Queue()
    container = FireEventAction(event_queue=queue, action=(mock_action, {}))
    container.start()

    assert container._task is not None
    await container.stop()

    assert container._task.done()
