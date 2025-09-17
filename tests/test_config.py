"""Unit tests for the Pydantic settings models in `is_goat_burning.config`.

These tests verify the loading of settings from mock `.env` files and the
behavior of the custom validators for each settings subgroup (Email, Discord,
Video).
"""

from unittest.mock import mock_open
from unittest.mock import patch

from pydantic import ValidationError
import pytest

from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings


def test_settings_loads_source_from_env_file() -> None:
    """Verifies that the main Settings object loads values from a `.env` file."""
    test_env_content = "SOURCE=test_source_from_file\n"
    with patch("builtins.open", mock_open(read_data=test_env_content)):
        settings = Settings()
    assert settings.source == "test_source_from_file"


def test_email_validator_raises_error_when_enabled_and_misconfigured() -> None:
    """Verifies EmailSettings validator fails if `use_emails` is True with missing fields."""
    with pytest.raises(ValidationError, match="must be set when EMAIL__USE_EMAILS is true"):
        EmailSettings(use_emails=True)


def test_discord_validator_raises_error_when_enabled_and_misconfigured() -> None:
    """Verifies DiscordSettings validator fails if `use_discord` is True with empty hooks."""
    with pytest.raises(ValidationError, match="must be set when DISCORD__USE_DISCORD is true"):
        DiscordSettings(use_discord=True)


def test_video_validator_raises_error_when_enabled_and_misconfigured() -> None:
    """Verifies VideoSettings validator fails if saving is enabled with no directory."""
    with pytest.raises(ValidationError, match="must be set when VIDEO__SAVE_VIDEO_CHUNKS is true"):
        VideoSettings(save_video_chunks=True)


def test_validators_do_not_raise_error_when_disabled() -> None:
    """Verifies validators do not raise errors when services are disabled."""
    try:
        EmailSettings(use_emails=False)
        DiscordSettings(use_discord=False)
        VideoSettings(save_video_chunks=False)
    except ValidationError:
        pytest.fail("ValidationError was raised unexpectedly when services are disabled.")
