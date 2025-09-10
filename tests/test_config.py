from unittest.mock import mock_open
from unittest.mock import patch

from pydantic import ValidationError
import pytest

from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings


def test_settings_loads_source_from_env_file():
    """
    Tests that the main Settings object correctly loads values from a `.env` file.
    """
    test_env_content = "SOURCE=test_source_from_file\n"
    with patch("builtins.open", mock_open(read_data=test_env_content)):
        settings = Settings()
    assert settings.source == "test_source_from_file"


def test_email_validator_raises_error_when_enabled_and_misconfigured():
    """
    Unit test: Directly tests the EmailSettings validator raises an error when
    `use_emails` is True but required fields are missing.
    """
    with pytest.raises(ValidationError) as excinfo:
        # Instantiate directly, which uses defaults for all unset fields (None, [])
        EmailSettings(use_emails=True)

    # The validator should fail on the first missing required field.
    assert "EMAIL__SENDER must be set when EMAIL__USE_EMAILS is true" in str(excinfo.value)


def test_discord_validator_raises_error_when_enabled_and_misconfigured():
    """
    Unit test: Directly tests the DiscordSettings validator raises an error when
    `use_discord` is True but `hooks` is empty.
    """
    with pytest.raises(ValidationError) as excinfo:
        DiscordSettings(use_discord=True)  # `hooks` will default to []

    assert "DISCORD__HOOKS must be set when DISCORD__USE_DISCORD is true" in str(excinfo.value)


def test_video_validator_raises_error_when_enabled_and_misconfigured():
    """
    Unit test: Directly tests the VideoSettings validator raises an error when
    `save_video_chunks` is True but `video_output_directory` is not set.
    """
    with pytest.raises(ValidationError) as excinfo:
        VideoSettings(save_video_chunks=True)  # `video_output_directory` will be None

    assert "VIDEO__VIDEO_OUTPUT_DIRECTORY must be set when VIDEO__SAVE_VIDEO_CHUNKS is true" in str(excinfo.value)


def test_validators_do_not_raise_error_when_disabled():
    """
    Unit test: Verifies that no validation error occurs when services are disabled,
    even with missing configuration.
    """
    try:
        # Directly instantiate with services disabled. This should always pass.
        EmailSettings(use_emails=False)
        DiscordSettings(use_discord=False)
        VideoSettings(save_video_chunks=False)
    except ValidationError:
        pytest.fail("ValidationError was raised unexpectedly when services are disabled.")
