"""Unit tests for the Pydantic settings models in `is_goat_burning.config`.

These tests verify the loading of settings from mock `.env` files and the
behavior of the custom validators for each settings subgroup (Email, Discord,
Video).
"""

import importlib
import os

from pydantic import ValidationError
import pytest

from is_goat_burning import config
from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings


def test_settings_loads_source_from_env_var(monkeypatch: MonkeyPatch) -> None:
    """Verifies that the main Settings object loads values from an environment variable."""
    # Arrange
    # Pydantic settings prioritizes environment variables. This is the most direct way to test loading.
    monkeypatch.setenv("SOURCE", "test_source_from_env")

    # Act
    # Re-initialize settings to pick up the new env var.
    settings = Settings()

    # Assert
    assert settings.source == "test_source_from_env"


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


def test_ffmpeg_capture_options_env_var_is_set_correctly(monkeypatch) -> None:
    """
    Arrange: Set the input environment variable to a custom value.
    Act: Reload the config module to trigger its top-level side effect.
    Assert: The output environment variable is set to the correct derived value.
    """
    # Arrange: Set the input environment variable that the module will read.
    monkeypatch.setenv("STREAM_INACTIVITY_TIMEOUT", "99")

    # Act: Reload the module to re-run its top-level statements.
    importlib.reload(config)

    # Assert: Check that the side effect (setting the other env var) happened correctly.
    expected_value = f"timeout;{99 * 1_000_000}"
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == expected_value

    # Cleanup: Restore the default value by unsetting the variable and reloading again.
    monkeypatch.delenv("STREAM_INACTIVITY_TIMEOUT")
    importlib.reload(config)
    assert "60" in os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
