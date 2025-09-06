import asyncio
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from is_goat_burning.app import Application
from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture to create a mock Settings for testing."""
    return Settings(
        source="tests/assets/fire.mp4",
        fire_detection_threshold=0.5,
        logging=False,
        video_output=False,
        checks_per_second=10,
        open_cl=False,
        cuda=False,
        email=EmailSettings(),
        discord=DiscordSettings(use_discord=True, hooks=["http://fake.com"]),
        video=VideoSettings(
            save_video_chunks=False,
        ),
    )


@pytest.mark.asyncio
@patch("is_goat_burning.app.settings")
@patch("is_goat_burning.app.YTCamGearFireDetector")
async def test_app_fire_detection_flow(
    mock_yt_cam_gear_fire_detector,
    mock_app_settings,
    mock_settings: Settings,
):
    """Integration test for the main application fire detection loop."""
    for key, value in mock_settings.model_dump().items():
        if key == "email":
            value = EmailSettings(**value)
        elif key == "discord":
            value = DiscordSettings(**value)
        elif key == "video":
            value = VideoSettings(**value)
        setattr(mock_app_settings, key, value)

    on_fire_action_callback = None

    def get_on_fire_action_callback(**kwargs):
        nonlocal on_fire_action_callback
        on_fire_action_callback = kwargs["on_fire_action"]
        # Return a mock that we can await
        return AsyncMock()

    mock_yt_cam_gear_fire_detector.side_effect = get_on_fire_action_callback

    app = Application()

    # Mock the on_fire_action_handler to check if it's called
    app.on_fire_action_handler = AsyncMock()

    # Run the application for a short time
    async def run_and_shutdown():
        # In a real scenario, the detector would run and call the callback.
        # Here we call it directly to simulate the fire detection.
        if on_fire_action_callback:
            await on_fire_action_callback()
        await app.run()

    main_task = asyncio.create_task(run_and_shutdown())
    await asyncio.sleep(0.2)

    # Assert that the on_fire_action_handler was called
    app.on_fire_action_handler.assert_called_once()

    main_task.cancel()
    await app.shutdown()
