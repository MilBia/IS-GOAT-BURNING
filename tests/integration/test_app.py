"""Integration tests for the main Application class.

These tests validate the end-to-end behavior of the Application's run
lifecycle, from fire detection signals to the triggering of action handlers,
using mocks for external dependencies like the fire detector itself.
"""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pytest
from pytest import MonkeyPatch

from is_goat_burning.app import Application
from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings
from is_goat_burning.on_fire_actions import OnceAction

TINY_RECONNECT_DELAY = 1


@pytest.fixture
def test_settings(monkeypatch: MonkeyPatch) -> Settings:
    """Provides a default, mocked Settings object for application tests."""
    settings = Settings(
        source="mock://source",
        log_level="CRITICAL",
        reconnect_delay_seconds=TINY_RECONNECT_DELAY,  # Use a tiny delay for tests
        email=EmailSettings(use_emails=False),
        discord=DiscordSettings(use_discord=False),
        video=VideoSettings(save_video_chunks=False),
        alert_cooldown_seconds=0.0,
    )
    monkeypatch.setattr("is_goat_burning.app.settings", settings)
    return settings


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_detects_fire_and_triggers_action(mock_detector_factory: AsyncMock, test_settings: Settings) -> None:  # noqa: ARG001
    """Verifies the app triggers the action handler when the detector signals fire."""
    app = Application()
    app.on_fire_action_handler = AsyncMock()
    # Mock the signal handler to stop the loop after the detector runs once.
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True

    mock_detector_instance = AsyncMock()

    async def detector_call_side_effect() -> None:
        """Simulate the detector running and then stopping the main loop."""
        on_fire_action_callback = mock_detector_factory.call_args.kwargs["on_fire_action"]
        await on_fire_action_callback()
        # After the action, signal the loop to stop.
        app.signal_handler.is_running.return_value = False

    mock_detector_instance.side_effect = detector_call_side_effect
    mock_detector_factory.return_value = mock_detector_instance

    await app.run()
    await app.shutdown()

    mock_detector_factory.assert_awaited_once()
    mock_detector_instance.assert_awaited_once()
    app.on_fire_action_handler.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_alerts_again_after_fire_is_extinguished(mock_detector_factory: AsyncMock, test_settings: Settings) -> None:  # noqa: ARG001
    """FIRE, EXTINGUISHED, FIRE through the action queue sends two alerts (re-armed between events)."""
    app = Application()
    alert = AsyncMock()
    app.on_fire_action_handler = OnceAction([(alert, {})], cooldown_seconds=0.0)
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True

    mock_detector_instance = AsyncMock()
    mock_detector_instance.fire_is_currently_detected = False

    async def detector_call_side_effect() -> None:
        kwargs = mock_detector_factory.call_args.kwargs
        await kwargs["on_fire_action"]()
        await kwargs["on_fire_action"]()  # duplicate signal within the same event
        await kwargs["on_extinguished_action"]()
        await kwargs["on_fire_action"]()
        await app.action_queue.join()
        app.signal_handler.is_running.return_value = False

    mock_detector_instance.side_effect = detector_call_side_effect
    mock_detector_factory.return_value = mock_detector_instance

    await app.run()
    await app.shutdown()

    assert alert.await_count == 2


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@patch("is_goat_burning.app.asyncio.sleep", new_callable=AsyncMock)
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_rearms_alert_when_stream_drops_mid_fire(
    mock_detector_factory: AsyncMock,
    mock_sleep: AsyncMock,  # noqa: ARG001
    test_settings: Settings,  # noqa: ARG001
) -> None:
    """A detector that stops while in the fire state re-arms the alert (the new detector starts fire-free)."""
    app = Application()
    app.on_fire_action_handler = MagicMock()
    app.on_fire_action_handler.rearm = AsyncMock()
    app.on_fire_action_handler.stop = AsyncMock()
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True

    mid_fire = AsyncMock()
    mid_fire.fire_is_currently_detected = True
    quiet = AsyncMock()
    quiet.fire_is_currently_detected = False

    async def stop_loop() -> None:
        await app.action_queue.join()
        app.signal_handler.is_running.return_value = False

    quiet.side_effect = stop_loop
    mock_detector_factory.side_effect = [mid_fire, quiet]

    await app.run()
    await app.shutdown()

    app.on_fire_action_handler.rearm.assert_awaited_once()


@pytest.mark.asyncio
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_does_not_detect_fire_and_remains_silent(mock_detector_factory: AsyncMock, test_settings: Settings) -> None:  # noqa: ARG001
    """Verifies the action handler is not called if the detector never signals fire."""
    app = Application()
    app.on_fire_action_handler = AsyncMock()
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True

    mock_detector_instance = AsyncMock()

    async def detector_call_side_effect() -> None:
        """Simulate the detector running and then stopping the main loop."""
        # Signal the loop to stop after one successful run.
        app.signal_handler.is_running.return_value = False

    mock_detector_instance.side_effect = detector_call_side_effect
    mock_detector_factory.return_value = mock_detector_instance

    await app.run()
    await app.shutdown()

    mock_detector_factory.assert_awaited_once()
    mock_detector_instance.assert_awaited_once()
    app.on_fire_action_handler.assert_not_called()


@pytest.mark.asyncio
@patch("is_goat_burning.app.asyncio.sleep", new_callable=AsyncMock)
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_run_loop_reconnects_after_detector_creation_failure(
    mock_detector_factory: AsyncMock,
    mock_sleep: AsyncMock,
    test_settings: Settings,
) -> None:
    """
    Arrange: Mock StreamFireDetector.create to fail once, then succeed.
    Act: Run the Application.
    Assert: The factory is called twice (initial + retry), and asyncio.sleep is
            called for the reconnect delay.
    """
    app = Application()
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True

    mock_detector_instance = AsyncMock()

    async def successful_detector_run() -> None:
        # On the successful run, stop the main loop.
        app.signal_handler.is_running.return_value = False

    mock_detector_instance.side_effect = successful_detector_run
    mock_detector_factory.side_effect = [Exception("Stream connection failed"), mock_detector_instance]

    await app.run()
    await app.shutdown()

    # The factory was called once, failed, then was called again on the next loop.
    assert mock_detector_factory.call_count == 2
    # The successful detector instance was awaited.
    mock_detector_instance.assert_awaited_once()
    # Sleep was called with the configured reconnect delay.
    mock_sleep.assert_awaited_once_with(test_settings.reconnect_delay_seconds)


@pytest.mark.asyncio
@patch("is_goat_burning.app.asyncio.sleep", new_callable=AsyncMock)
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_run_loop_handles_cancellation_during_sleep(
    mock_detector_factory: AsyncMock, mock_sleep: AsyncMock, test_settings: Settings
) -> None:
    """
    Arrange: Mock StreamFireDetector to fail, then mock asyncio.sleep to raise CancelledError.
    Act: Run the Application.
    Assert: The loop terminates gracefully without attempting a second connection.
    """
    app = Application()
    # The signal handler will be what stops the loop.
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True

    # Detector creation fails, triggering the reconnect sleep.
    mock_detector_factory.side_effect = Exception("Stream connection failed")

    async def sleep_and_stop_loop(*args, **kwargs) -> None:  # noqa: ARG001
        # When sleep is called, we stop the loop and raise the error.
        app.signal_handler.is_running.return_value = False
        raise asyncio.CancelledError

    mock_sleep.side_effect = sleep_and_stop_loop

    await app.run()
    await app.shutdown()

    # The factory was only called once, as the shutdown occurred before the retry.
    mock_detector_factory.assert_called_once()
    # Sleep was called, but was cancelled.
    mock_sleep.assert_awaited_once_with(test_settings.reconnect_delay_seconds)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_unverified_fire_event_reaches_the_alert_as_unverified(test_settings: Settings) -> None:  # noqa: ARG001
    """``on_fire_action(unverified=True)`` from the detector is delivered to the alert as unverified."""
    app = Application()
    alert = AsyncMock()
    app.on_fire_action_handler = OnceAction([(alert, {})], cooldown_seconds=0.0)
    manager = asyncio.create_task(app._action_manager())

    await app._queue_fire_event(unverified=True)
    await app.action_queue.join()
    manager.cancel()
    await asyncio.gather(manager, return_exceptions=True)

    alert.assert_awaited_once_with(unverified=True)


@pytest.mark.asyncio
@patch("is_goat_burning.app.asyncio.sleep", new_callable=AsyncMock)
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_run_loop_reports_stream_state_to_service_notifier(
    mock_detector_factory: AsyncMock,
    mock_sleep: AsyncMock,  # noqa: ARG001
    test_settings: Settings,  # noqa: ARG001
) -> None:
    """A failed connection is reported as `stream_failed(<exception type>)`, a successful one as `stream_connected()`,
    and a stream that ends without an error as `stream_failed("stream ended")`; the notifier is started and stopped.
    """
    app = Application()
    app.service_notifier = MagicMock()
    app.service_notifier.stop = AsyncMock()
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True
    runs = 0

    async def detector_run() -> None:
        nonlocal runs
        runs += 1
        if runs == 2:
            app.signal_handler.is_running.return_value = False

    detector = AsyncMock(side_effect=detector_run)
    mock_detector_factory.side_effect = [TimeoutError("resolve failed"), detector, detector]

    await app.run()
    await app.shutdown()

    notifier = app.service_notifier
    notifier.start.assert_called_once()
    assert [c for c in notifier.method_calls if c[0] in {"stream_connected", "stream_failed"}] == [
        call.stream_failed("TimeoutError"),
        call.stream_connected(),
        call.stream_failed("stream ended"),
        call.stream_connected(),
    ]
    notifier.stop.assert_awaited_once()
