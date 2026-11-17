"""Unit tests for the action container classes.

These tests verify the per-fire-event logic of the `OnceAction` class (fire
once per event, re-arm after the fire is extinguished, a minimum spacing
between alerts, success-only bookkeeping), and the persistent, queue-driven
behavior of the `FireEventAction` class.
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


class FakeClock:
    """A manually advanced monotonic clock for cooldown tests."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_once_action_two_events_separated_by_extinguish_give_two_alerts() -> None:
    """Two fire events separated by an extinguish (past the cooldown) send two alerts."""
    mock_action = AsyncMock()
    clock = FakeClock()
    once_action = OnceAction(actions=[(mock_action, {})], cooldown_seconds=600.0, clock=clock)

    await once_action()
    await once_action()  # same event: no duplicate
    assert mock_action.await_count == 1

    clock.now += 30.0
    await once_action.rearm()  # fire extinguished
    clock.now += 600.0
    await once_action()  # a new fire event after the cooldown

    assert mock_action.await_count == 2
    await once_action.stop()


@pytest.mark.asyncio
async def test_once_action_flicker_within_cooldown_gives_one_alert() -> None:
    """Extinguish + re-detection inside the cooldown that settles inside it sends a single alert."""
    mock_action = AsyncMock()
    clock = FakeClock()
    once_action = OnceAction(actions=[(mock_action, {})], cooldown_seconds=600.0, clock=clock)

    await once_action()
    for _ in range(3):  # the state flickers several times within the window
        clock.now += 10.0
        await once_action.rearm()
        clock.now += 10.0
        await once_action()
    await once_action.rearm()  # ...and finally settles as extinguished
    await asyncio.sleep(0)

    assert mock_action.await_count == 1
    await once_action.stop()


@pytest.mark.asyncio
async def test_once_action_defers_alert_suppressed_by_cooldown() -> None:
    """A new fire inside the cooldown that keeps burning is alerted when the cooldown ends."""
    mock_action = AsyncMock()
    once_action = OnceAction(actions=[(mock_action, {})], cooldown_seconds=0.05)

    await once_action()
    await once_action.rearm()
    await once_action()  # still inside the cooldown: deferred, not dropped
    assert mock_action.await_count == 1

    await _wait_for(lambda: mock_action.await_count == 2)
    await asyncio.sleep(0.1)
    assert mock_action.await_count == 2
    await once_action.stop()


@pytest.mark.asyncio
async def test_once_action_deferred_alert_cancelled_by_extinguish() -> None:
    """A deferred alert is dropped if the fire is extinguished before the cooldown ends."""
    mock_action = AsyncMock()
    once_action = OnceAction(actions=[(mock_action, {})], cooldown_seconds=0.05)

    await once_action()
    await once_action.rearm()
    await once_action()  # deferred
    await once_action.rearm()  # extinguished before the cooldown ends
    await asyncio.sleep(0.1)

    assert mock_action.await_count == 1
    await once_action.stop()


@pytest.mark.asyncio
async def test_once_action_stays_armed_when_every_action_fails() -> None:
    """When no channel succeeds the alert is not marked sent and no cooldown starts."""
    failing = AsyncMock(side_effect=[RuntimeError("boom"), None])
    once_action = OnceAction(actions=[(failing, {})], cooldown_seconds=600.0, clock=FakeClock())

    await once_action()
    assert once_action.was_performed is False

    await once_action()  # retried on the next fire signal, not blocked by the cooldown
    assert failing.await_count == 2
    assert once_action.was_performed is True


@pytest.mark.asyncio
async def test_once_action_false_return_counts_as_failure() -> None:
    """An action returning ``False`` reports failure; ``None``/``True`` report success."""
    reporting_failure = AsyncMock(return_value=False)
    once_action = OnceAction(actions=[(reporting_failure, {})], clock=FakeClock())

    await once_action()
    assert once_action.was_performed is False


@pytest.mark.asyncio
async def test_once_action_partial_success_marks_sent_and_runs_all_actions() -> None:
    """One failing channel neither blocks the others nor causes duplicates to the successful ones."""
    failing = AsyncMock(side_effect=RuntimeError("discord down"))
    working = AsyncMock()
    once_action = OnceAction(actions=[(failing, {}), (working, {})], clock=FakeClock())

    await once_action()
    await once_action()

    assert working.await_count == 1
    assert failing.await_count == 1
    assert once_action.was_performed is True


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
