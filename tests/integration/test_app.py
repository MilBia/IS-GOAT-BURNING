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
    for key in Settings.model_fields:
        setattr(mock_app_settings, key, getattr(mock_settings, key))

    # The detector class mock should return an instance mock.
    # The instance is what will be awaited in app.run()
    mock_detector_instance = AsyncMock()
    mock_yt_cam_gear_fire_detector.return_value = mock_detector_instance
    app = Application()
    # After app initialization, we can inspect the call to the detector's constructor
    # to get the callback it was passed.
    mock_yt_cam_gear_fire_detector.assert_called_once()
    on_fire_action_callback = mock_yt_cam_gear_fire_detector.call_args.kwargs["on_fire_action"]

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
    for _ in range(20):  # Poll for up to 0.2 seconds
        if app.on_fire_action_handler.called:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("The on_fire_action_handler was not called within the timeout.")

    # Assert that the on_fire_action_handler was called
    app.on_fire_action_handler.assert_called_once()

    main_task.cancel()
    await app.shutdown()
