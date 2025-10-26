"""Integration tests for the main Application class.

These tests validate the end-to-end behavior of the Application's run
lifecycle, from fire detection signals to the triggering of action handlers,
using mocks for external dependencies like the fire detector itself.
"""

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


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_detects_fire_and_triggers_action(mock_detector_factory: AsyncMock, monkeypatch: MonkeyPatch) -> None:
    """Verifies the app triggers the action handler when the detector signals fire."""

    # --- Setup ---
    # This mock represents the detector instance returned by the factory.
    # Its __call__ method will be executed by the TaskGroup.
    mock_detector_instance = AsyncMock()

    async def detector_call_side_effect():
        """Simulate the detector running and then emitting a fire signal."""
        # The `on_fire_action` callback is passed to the factory during app.run()
        # We retrieve it from the factory's call arguments to invoke it.
        on_fire_action_callback = mock_detector_factory.call_args.kwargs["on_fire_action"]
        await on_fire_action_callback()

    mock_detector_instance.side_effect = detector_call_side_effect
    # The factory itself returns our mocked instance.
    mock_detector_factory.return_value = mock_detector_instance

    test_settings = Settings(
        source="mock://source",
        log_level="CRITICAL",
        reconnect_delay_seconds=0,
        email=EmailSettings(use_emails=False),
        discord=DiscordSettings(use_discord=False),
        video=VideoSettings(save_video_chunks=False),
    )
    monkeypatch.setattr("is_goat_burning.app.settings", test_settings)

    app = Application()
    app.on_fire_action_handler = AsyncMock()

    # Control the run loop to execute only once.
    run_count = [0]

    def is_running_side_effect():
        run_count[0] += 1
        return run_count[0] <= 1

    app.signal_handler.is_running = MagicMock(side_effect=is_running_side_effect)

    # --- Execution ---
    await app.run()
    await app.shutdown()

    # --- Assertion ---
    # Verify the factory was called to create the detector.
    mock_detector_factory.assert_awaited_once()
    # Verify the detector instance itself was called by the TaskGroup.
    mock_detector_instance.assert_awaited_once()
    # Verify the core logic: the on_fire_action handler was triggered.
    app.on_fire_action_handler.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_does_not_detect_fire_and_remains_silent(mock_detector_factory: AsyncMock, monkeypatch: MonkeyPatch) -> None:
    """Verifies the action handler is not called if the detector never signals fire."""

    # --- Setup ---
    # The detector instance will be called, but its side effect will do nothing.
    mock_detector_instance = AsyncMock()
    mock_detector_factory.return_value = mock_detector_instance

    test_settings = Settings(
        source="mock://source",
        log_level="CRITICAL",
        reconnect_delay_seconds=0,
        email=EmailSettings(use_emails=False),
        discord=DiscordSettings(use_discord=False),
        video=VideoSettings(save_video_chunks=False),
    )
    monkeypatch.setattr("is_goat_burning.app.settings", test_settings)

    app = Application()
    app.on_fire_action_handler = AsyncMock()

    # Control the run loop to execute only once.
    run_count = [0]

    def is_running_side_effect():
        run_count[0] += 1
        return run_count[0] <= 1

    app.signal_handler.is_running = MagicMock(side_effect=is_running_side_effect)

    # --- Execution ---
    await app.run()
    await app.shutdown()

    # --- Assertion ---
    mock_detector_factory.assert_awaited_once()
    mock_detector_instance.assert_awaited_once()
    # Verify the core logic: the on_fire_action handler was NOT triggered.
    app.on_fire_action_handler.assert_not_called()
