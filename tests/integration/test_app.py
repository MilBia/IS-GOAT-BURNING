"""Integration tests for the main Application class.

These tests validate the end-to-end behavior of the Application's run
lifecycle, from fire detection signals to the triggering of action handlers,
using mocks for external dependencies like the fire detector itself.
"""

from unittest.mock import AsyncMock
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
@patch("is_goat_burning.app.OnceAction")
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_detects_fire_and_triggers_action(
    mock_detector_factory: AsyncMock, mock_once_action_class: AsyncMock, monkeypatch: MonkeyPatch
) -> None:
    """Verifies the app triggers the action handler when the detector signals fire."""

    # --- Setup ---
    # 1. Create a mock detector instance that will be "returned" by the factory.
    mock_detector_instance = AsyncMock()

    # 2. We need to capture the `on_fire_action` callback to configure the mock.
    # The factory itself will return our pre-configured mock detector instance.
    def factory_side_effect(**kwargs):
        on_fire_action = kwargs.get("on_fire_action")
        # Configure the mock's side_effect. When `mock_detector_instance()` is
        # awaited, it will execute the on_fire_action function.
        mock_detector_instance.side_effect = on_fire_action
        return mock_detector_instance

    mock_detector_factory.side_effect = factory_side_effect

    # 3. Create pure, in-memory settings for the test.
    test_settings = Settings(
        source="mock://source",
        log_level="CRITICAL",
        default_framerate=30.0,
        ytdlp_format="bestvideo/best",
        email=EmailSettings(use_emails=False),
        discord=DiscordSettings(use_discord=False),
        video=VideoSettings(save_video_chunks=False),
    )

    # 4. Patch settings and the action handler.
    monkeypatch.setattr("is_goat_burning.app.settings", test_settings)
    mock_action_handler_instance = AsyncMock()
    mock_once_action_class.return_value = mock_action_handler_instance

    # --- Execution ---
    app = await Application.create()
    await app.run()
    await app.shutdown()

    # --- Assertion ---
    # Assert that the detector instance itself was awaited.
    mock_detector_instance.assert_awaited_once()
    mock_action_handler_instance.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@patch("is_goat_burning.app.OnceAction")
@patch("is_goat_burning.app.StreamFireDetector.create")
async def test_app_does_not_detect_fire_and_remains_silent(
    mock_detector_factory: AsyncMock, mock_once_action_class: AsyncMock, monkeypatch: MonkeyPatch
) -> None:
    """Verifies the action handler is not called if the detector never signals fire."""

    # --- Setup ---
    # 1. This mock detector simulates NOT finding fire. A plain AsyncMock will
    # do nothing when awaited, which is the desired behavior.
    mock_detector_instance = AsyncMock()
    mock_detector_factory.return_value = mock_detector_instance

    # 2. Create pure settings.
    test_settings = Settings(
        source="mock://source",
        log_level="CRITICAL",
        default_framerate=30.0,
        ytdlp_format="bestvideo/best",
        email=EmailSettings(use_emails=False),
        discord=DiscordSettings(use_discord=False),
        video=VideoSettings(save_video_chunks=False),
    )

    # 3. Patch settings and the action handler.
    monkeypatch.setattr("is_goat_burning.app.settings", test_settings)
    mock_action_handler_instance = AsyncMock()
    mock_once_action_class.return_value = mock_action_handler_instance

    # --- Execution ---
    app = await Application.create()
    await app.run()
    await app.shutdown()

    # --- Assertion ---
    mock_detector_instance.assert_awaited_once()
    mock_action_handler_instance.assert_not_called()
