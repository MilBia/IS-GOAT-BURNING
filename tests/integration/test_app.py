import asyncio
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

# It is now safe to import at the top level because we will explicitly
# monkeypatch the settings object in the correct namespace within each test.
from is_goat_burning.app import Application
from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings


@pytest.mark.asyncio
@pytest.mark.timeout(5)  # The test should be very fast now
@patch("is_goat_burning.app.OnceAction")
async def test_app_detects_fire_and_triggers_action(mock_once_action_class, monkeypatch):
    """
    This integration test verifies that when the detector signals 'fire',
    the Application class correctly queues the event and triggers the action handler.

    It is fully isolated from .env files and mocks the external detector dependency.
    """

    # --- Setup ---
    # 1. Define a mock detector that simulates finding fire.
    class MockFireDetector:
        def __init__(self, *args, **kwargs):
            # Capture the callback function that Application passes to the detector.
            self.on_fire_action = kwargs.get("on_fire_action")

        async def __call__(self):
            # When run, this mock immediately calls the captured callback,
            # simulating a fire detection event.
            if self.on_fire_action:
                await self.on_fire_action()

    # 2. Create our pure, in-memory settings for the test.
    test_settings = Settings(
        source="mock://source",  # Source is irrelevant as the detector is mocked
        fire_detection_threshold=0.01,
        logging=False,
        video_output=False,
        checks_per_second=10,
        email=EmailSettings(use_emails=False),
        discord=DiscordSettings(use_discord=False),
        video=VideoSettings(save_video_chunks=False),
    )

    # 3. Apply all patches *before* instantiating the Application.
    monkeypatch.setattr("is_goat_burning.app.settings", test_settings)
    monkeypatch.setattr("is_goat_burning.app.YTCamGearFireDetector", MockFireDetector)
    mock_action_handler_instance = AsyncMock()
    mock_once_action_class.return_value = mock_action_handler_instance

    # --- Execution ---
    app = Application()
    await app.run()
    await app.shutdown()

    # --- Assertion ---
    mock_action_handler_instance.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@patch("is_goat_burning.app.OnceAction")
async def test_app_does_not_detect_fire_and_remains_silent(mock_once_action_class, monkeypatch):
    """
    This integration test verifies that if the detector runs and finishes
    without signaling 'fire', the action handler is never called.
    """

    # --- Setup ---
    # 1. Define a mock detector that simulates NOT finding fire.
    class MockNoFireDetector:
        def __init__(self, *args, **kwargs):
            # It still accepts the args, but will do nothing with the callback.
            pass

        async def __call__(self):
            # This detector finishes its "scan" without ever signaling a fire.
            await asyncio.sleep(0)  # Yield control once to simulate async work

    # 2. Create pure settings.
    test_settings = Settings(
        source="mock://source",
        fire_detection_threshold=0.01,
        logging=False,
        video_output=False,
        checks_per_second=10,
        email=EmailSettings(use_emails=False),
        discord=DiscordSettings(use_discord=False),
        video=VideoSettings(save_video_chunks=False),
    )

    # 3. Apply patches.
    monkeypatch.setattr("is_goat_burning.app.settings", test_settings)
    monkeypatch.setattr("is_goat_burning.app.YTCamGearFireDetector", MockNoFireDetector)
    mock_action_handler_instance = AsyncMock()
    mock_once_action_class.return_value = mock_action_handler_instance

    # --- Execution ---
    app = Application()
    await app.run()
    await app.shutdown()

    # --- Assertion ---
    mock_action_handler_instance.assert_not_called()
