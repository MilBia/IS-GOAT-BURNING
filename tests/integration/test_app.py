"""Integration tests for the main Application class.

These tests validate the end-to-end behavior of the Application's run
lifecycle, from fire detection signals to the triggering of action handlers,
using mocks for external dependencies like the fire detector itself.
"""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from pytest import MonkeyPatch

# It is now safe to import at the top level because we will explicitly
# monkeypatch the settings object in the correct namespace within each test.
from is_goat_burning.app import Application
from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@patch("is_goat_burning.app.OnceAction")
async def test_app_detects_fire_and_triggers_action(mock_once_action_class: AsyncMock, monkeypatch: MonkeyPatch) -> None:
    """Verifies that the app triggers the action handler when the detector signals fire.

    This test uses a mock detector that immediately calls its `on_fire_action`
    callback, simulating a fire event. It then asserts that the application's
    main action handler was called exactly once.

    Args:
        mock_once_action_class: A mock of the `OnceAction` class.
        monkeypatch: The pytest fixture for runtime patching.
    """

    # --- Setup ---
    # 1. Define a mock detector that simulates finding fire.
    class MockFireDetector:
        def __init__(self, *args, **kwargs):
            # Capture the callback function that Application passes to the detector.
            self.on_fire_action = kwargs.get("on_fire_action")

        async def __call__(self):
            # When run, this mock immediately calls the captured callback.
            if self.on_fire_action:
                await self.on_fire_action()

    # 2. Create pure, in-memory settings for the test.
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

    # 3. Apply all patches *before* instantiating the Application.
    monkeypatch.setattr("is_goat_burning.app.settings", test_settings)
    monkeypatch.setattr("is_goat_burning.app.StreamFireDetector", MockFireDetector)
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
async def test_app_does_not_detect_fire_and_remains_silent(mock_once_action_class: AsyncMock, monkeypatch: MonkeyPatch) -> None:
    """Verifies that the action handler is not called if the detector never signals fire.

    This test uses a mock detector that completes its run without ever calling
    the `on_fire_action` callback. It asserts that the application's main
    action handler was never called.

    Args:
        mock_once_action_class: A mock of the `OnceAction` class.
        monkeypatch: The pytest fixture for runtime patching.
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
    monkeypatch.setattr("is_goat_burning.app.StreamFireDetector", MockNoFireDetector)
    mock_action_handler_instance = AsyncMock()
    mock_once_action_class.return_value = mock_action_handler_instance

    # --- Execution ---
    app = Application()
    await app.run()
    await app.shutdown()

    # --- Assertion ---
    mock_action_handler_instance.assert_not_called()
