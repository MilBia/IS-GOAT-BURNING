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
    mock_detector_instance = AsyncMock()

    async def factory_side_effect(**kwargs):
        on_fire_action = kwargs.get("on_fire_action")
        mock_detector_instance.side_effect = on_fire_action
        return mock_detector_instance

    mock_detector_factory.side_effect = factory_side_effect

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

    # Control the run loop using a stateful callable to prevent StopIteration.
    # The list simulates a mutable "nonlocal" variable for the lambda.
    run_count = [0]

    def is_running_side_effect():
        run_count[0] += 1
        return run_count[0] <= 1  # True only on the first call

    app.signal_handler.is_running = MagicMock(side_effect=is_running_side_effect)

    # --- Execution ---
    await app.run()
    await app.shutdown()

    # --- Assertion ---
    mock_detector_factory.assert_awaited_once()
    mock_detector_instance.assert_awaited_once()
    app.on_fire_action_handler.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_does_not_detect_fire_and_remains_silent(mock_detector_factory: AsyncMock, monkeypatch: MonkeyPatch) -> None:
    """Verifies the action handler is not called if the detector never signals fire."""

    # --- Setup ---
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

    # Control the run loop using a stateful callable.
    run_count = [0]

    def is_running_side_effect():
        run_count[0] += 1
        return run_count[0] <= 1  # True only on the first call

    app.signal_handler.is_running = MagicMock(side_effect=is_running_side_effect)

    # --- Execution ---
    await app.run()
    await app.shutdown()

    # --- Assertion ---
    mock_detector_factory.assert_awaited_once()
    mock_detector_instance.assert_awaited_once()
    app.on_fire_action_handler.assert_not_called()
