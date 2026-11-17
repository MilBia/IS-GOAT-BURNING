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


class BlockingAction:
    """An alert action that blocks until released, recording starts and completions."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = 0
        self.completed = 0

    async def __call__(self) -> None:
        self.started += 1
        await self.release.wait()
        self.completed += 1


async def _start_deferred_send(action: BlockingAction, **kwargs) -> OnceAction:
    """Sends a first alert, re-arms, and lets a deferred second alert start sending (and block)."""
    once_action = OnceAction(actions=[(action, {})], cooldown_seconds=0.05, **kwargs)
    action.release.set()
    await once_action()  # first alert, completes immediately
    action.release.clear()
    await once_action.rearm()
    await once_action()  # deferred
    await _wait_for(lambda: action.started == 2)  # the deferred send is now in flight
    return once_action


@pytest.mark.asyncio
async def test_once_action_stop_waits_for_in_flight_deferred_send() -> None:
    """Shutdown lets an alert that is already being sent finish (within the stop timeout)."""
    action = BlockingAction()
    once_action = await _start_deferred_send(action, stop_timeout_seconds=1.0)

    asyncio.get_running_loop().call_later(0.05, action.release.set)
    await once_action.stop()

    assert action.completed == 2
    assert once_action.was_performed is True


@pytest.mark.asyncio
async def test_once_action_stop_cancels_hanging_send_after_timeout() -> None:
    """A send that does not finish within the stop timeout is cancelled so shutdown is not blocked."""
    action = BlockingAction()
    once_action = await _start_deferred_send(action, stop_timeout_seconds=0.05)

    await asyncio.wait_for(once_action.stop(), timeout=1.0)

    assert action.completed == 1  # the hanging second send never completed
    assert once_action.was_performed is False


@pytest.mark.asyncio
async def test_once_action_rearm_waits_for_in_flight_deferred_send() -> None:
    """Re-arming during a deferred send neither cancels it nor is overwritten by it."""
    action = BlockingAction()
    once_action = await _start_deferred_send(action)

    rearm = asyncio.create_task(once_action.rearm())
    await asyncio.sleep(0.02)
    assert not rearm.done()  # waits for the in-flight send instead of cancelling it

    action.release.set()
    await asyncio.wait_for(rearm, timeout=1.0)

    assert action.completed == 2  # the deferred alert was delivered
    assert once_action.was_performed is False  # and the container is armed for the next event
    await once_action.stop()


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


# --- unverified alerts (#169) ---


@pytest.mark.asyncio
async def test_unverified_flag_is_forwarded_to_actions() -> None:
    """An unverified fire event passes ``unverified=True`` to every action; a verified one passes nothing."""
    action = AsyncMock()
    once_action = OnceAction(actions=[(action, {})])

    await once_action(unverified=True)
    action.assert_awaited_once_with(unverified=True)

    await once_action.rearm()
    await once_action()
    assert action.await_args_list[-1].kwargs == {}


@pytest.mark.asyncio
async def test_deferred_alert_keeps_the_unverified_flag() -> None:
    """A deferred alert (inside the cooldown) is still sent as unverified."""
    action = AsyncMock()
    clock = FakeClock()
    once_action = OnceAction(actions=[(action, {})], cooldown_seconds=0.05, clock=clock)
    await once_action()
    await once_action.rearm()

    await once_action(unverified=True)  # inside the cooldown: deferred
    await asyncio.sleep(0.1)

    assert action.await_args_list[-1].kwargs == {"unverified": True}


# --- frame of the triggering event (#177) ---


@pytest.mark.asyncio
async def test_frame_is_forwarded_to_actions_and_dropped_after_send() -> None:
    """The frame reaches every action as ``frame=``; it is released once the alert was sent."""
    action = AsyncMock()
    once_action = OnceAction(actions=[(action, {})])

    await once_action(frame="F1")
    assert action.await_args_list[-1].kwargs == {"frame": "F1"}
    assert once_action._frame is None

    await once_action.rearm()
    await once_action()
    assert action.await_args_list[-1].kwargs == {}


@pytest.mark.asyncio
async def test_deferred_alert_carries_the_latest_frame() -> None:
    """A deferred alert is sent with the frame of the latest event; only one frame is held."""
    action = AsyncMock()
    clock = FakeClock()
    once_action = OnceAction(actions=[(action, {})], cooldown_seconds=0.05, clock=clock)
    await once_action()
    await once_action.rearm()

    await once_action(frame="old")
    await once_action(frame="new")
    await asyncio.sleep(0.1)

    assert action.await_args_list[-1].kwargs == {"frame": "new"}


@pytest.mark.asyncio
async def test_frame_and_unverified_are_forwarded_together() -> None:
    """Both keywords are passed when both apply."""
    action = AsyncMock()
    once_action = OnceAction(actions=[(action, {})])
    await once_action(unverified=True, frame="F")
    action.assert_awaited_once_with(unverified=True, frame="F")
