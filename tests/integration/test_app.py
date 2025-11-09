"""Integration tests for the main Application class.

These tests validate the end-to-end behavior of the Application's run
lifecycle, from fire detection signals to the triggering of action handlers,
using mocks for external dependencies like the fire detector itself.
"""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pytest import MonkeyPatch

from is_goat_burning.app import Application
from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings


@pytest.fixture
def test_settings(monkeypatch: MonkeyPatch) -> Settings:
    """Provides a default, mocked Settings object for application tests."""
    settings = Settings(
        source="mock://source",
        log_level="CRITICAL",
        reconnect_delay_seconds=0.001,  # Use a tiny delay for tests
        email=EmailSettings(use_emails=False),
        discord=DiscordSettings(use_discord=False),
        video=VideoSettings(save_video_chunks=False),
    )
    monkeypatch.setattr("is_goat_burning.app.settings", settings)
    return settings


@pytest.mark.asyncio
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_detects_fire_and_triggers_action(mock_detector_factory: AsyncMock, test_settings: Settings) -> None:  # noqa: ARG001
    """Verifies the app triggers the action handler when the detector signals fire."""
    app = Application()
    app.on_fire_action_handler = AsyncMock()
    # Mock the signal handler to stop the loop after the detector runs once.
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True

    mock_detector_instance = AsyncMock()

    async def detector_call_side_effect():
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
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_does_not_detect_fire_and_remains_silent(mock_detector_factory: AsyncMock, test_settings: Settings) -> None:  # noqa: ARG001
    """Verifies the action handler is not called if the detector never signals fire."""
    app = Application()
    app.on_fire_action_handler = AsyncMock()
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True

    mock_detector_instance = AsyncMock()

    async def detector_call_side_effect():
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
    mock_detector_factory: AsyncMock, mock_sleep: AsyncMock, test_settings: Settings
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

    async def successful_detector_run():
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
