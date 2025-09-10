from unittest.mock import mock_open
from unittest.mock import patch

from pydantic import ValidationError
import pytest

from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings


def test_main_settings_load_from_env_file():
    """
    Tests that the main Settings object correctly loads values from a `.env` file.
    """
    test_env_content = "SOURCE=test_source_from_file\n"
    with patch("builtins.open", mock_open(read_data=test_env_content)):
        settings = Settings()
    assert settings.source == "test_source_from_file"


def test_email_enabled_with_defaults_raises_error():
    """
    Unit test: Directly tests the EmailSettings validator.
    """
    with pytest.raises(ValidationError) as excinfo:
        # Instantiate directly, which uses defaults for all unset fields (None, [])
        EmailSettings(use_emails=True)

    # The validator should fail on the first missing required field.
    assert "EMAIL__SENDER must be set when EMAIL__USE_EMAILS is true" in str(excinfo.value)


def test_discord_enabled_with_defaults_raises_error():
    """
    Unit test: Directly tests the DiscordSettings validator.
    """
    with pytest.raises(ValidationError) as excinfo:
        DiscordSettings(use_discord=True)  # `hooks` will default to []

    assert "DISCORD__HOOKS must be set when DISCORD__USE_DISCORD is true" in str(excinfo.value)


def test_video_saving_enabled_with_defaults_raises_error():
    """
    Unit test: Directly tests the VideoSettings validator.
    """
    with pytest.raises(ValidationError) as excinfo:
        VideoSettings(save_video_chunks=True)  # `video_output_directory` will be None

    assert "VIDEO__VIDEO_OUTPUT_DIRECTORY must be set when VIDEO__SAVE_VIDEO_CHUNKS is true" in str(excinfo.value)


def test_all_services_disabled_does_not_raise_error():
    """
    Unit test: Verifies that no validation error occurs when services are disabled.
    """
    try:
        # Directly instantiate with services disabled. This should always pass.
        EmailSettings(use_emails=False)
        DiscordSettings(use_discord=False)
        VideoSettings(save_video_chunks=False)
    except ValidationError:
        pytest.fail("ValidationError was raised unexpectedly when services are disabled.")
