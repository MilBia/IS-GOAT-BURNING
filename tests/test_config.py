"""Unit tests for the Pydantic settings models in `is_goat_burning.config`.

These tests verify the loading of settings from mock `.env` files and the
behavior of the custom validators for each settings subgroup (Email, Discord,
Video).
"""

import importlib
import os
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
from pydantic import ValidationError
import pytest

from is_goat_burning import config
from is_goat_burning.config import Accelerator
from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EdgeSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings

MICROSECONDS_PER_SECOND = 1_000_000
DEFAULT_INACTIVITY_TIMEOUT = 60
TEST_TIMEOUT_VAL = "99"


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


def test_ffmpeg_capture_options_env_var_is_set_correctly(monkeypatch: MonkeyPatch) -> None:
    """
    Arrange: Set the input environment variable to a custom value.
    Act: Reload the config module to trigger its top-level side effect.
    Assert: The output environment variable is set to the correct derived value.
    """
    # Arrange: Define constants and variables to avoid magic numbers.
    monkeypatch.setenv("STREAM_INACTIVITY_TIMEOUT", TEST_TIMEOUT_VAL)
    # Act: Reload the module to re-run its top-level statements.
    importlib.reload(config)
    # Assert: Check that the side effect (setting the other env var) happened correctly.
    expected_value = f"timeout;{int(TEST_TIMEOUT_VAL) * MICROSECONDS_PER_SECOND}"
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == expected_value
    # Cleanup: Restore the default value by unsetting the variable and reloading again.
    monkeypatch.delenv("STREAM_INACTIVITY_TIMEOUT")
    importlib.reload(config)
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == f"timeout;{DEFAULT_INACTIVITY_TIMEOUT * MICROSECONDS_PER_SECOND}"


def test_detection_strategy_hybrid_maps_to_hybrid_cloud(monkeypatch: MonkeyPatch) -> None:
    """Verifies the legacy `hybrid` strategy is auto-migrated to `hybrid_cloud`."""
    monkeypatch.setenv("DETECTION_STRATEGY", "hybrid")
    assert Settings().detection_strategy == "hybrid_cloud"


def test_detection_strategy_hybrid_cloud_is_accepted(monkeypatch: MonkeyPatch) -> None:
    """Verifies the new `hybrid_cloud` strategy is a valid value."""
    monkeypatch.setenv("DETECTION_STRATEGY", "hybrid_cloud")
    assert Settings().detection_strategy == "hybrid_cloud"


def test_detection_strategy_hybrid_edge_requires_model_path(monkeypatch: MonkeyPatch) -> None:
    """Verifies `hybrid_edge` fails validation when no model path is configured."""
    monkeypatch.setenv("DETECTION_STRATEGY", "hybrid_edge")
    monkeypatch.delenv("EDGE__MODEL_PATH", raising=False)
    with pytest.raises(ValidationError, match="EDGE__MODEL_PATH must be set"):
        Settings(_env_file=".env.tests")


def test_detection_strategy_hybrid_edge_requires_existing_model_file(monkeypatch: MonkeyPatch) -> None:
    """Verifies `hybrid_edge` fails validation when the model file does not exist."""
    monkeypatch.setenv("DETECTION_STRATEGY", "hybrid_edge")
    monkeypatch.setenv("EDGE__MODEL_PATH", "/nonexistent/path/to/model.onnx")
    with pytest.raises(ValidationError, match="EDGE__MODEL_PATH"):
        Settings()


def test_detection_strategy_hybrid_edge_with_valid_model_file(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Verifies `hybrid_edge` validates successfully when the model file exists."""
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"fake-model")
    monkeypatch.setenv("DETECTION_STRATEGY", "hybrid_edge")
    monkeypatch.setenv("EDGE__MODEL_PATH", str(model_file))
    settings = Settings()
    assert settings.detection_strategy == "hybrid_edge"
    assert settings.edge.model_path == str(model_file)


def test_hybrid_edge_validation_skipped_for_other_strategies(monkeypatch: MonkeyPatch) -> None:
    """Verifies the edge model-path check is only enforced for `hybrid_edge`."""
    # Default strategy is `classic` with no EDGE__MODEL_PATH; this must not raise.
    monkeypatch.setenv("DETECTION_STRATEGY", "classic")
    monkeypatch.delenv("EDGE__MODEL_PATH", raising=False)
    settings = Settings(_env_file=".env.tests")
    assert settings.detection_strategy == "classic"
    assert settings.edge.model_path is None


def test_edge_settings_defaults() -> None:
    """Verifies EdgeSettings applies the documented defaults."""
    edge = EdgeSettings()
    assert edge.model_path is None
    assert edge.confidence_threshold == 0.5
    assert edge.accelerator == Accelerator.AUTO


@pytest.mark.parametrize("value", ["auto", "cpu", "cuda", "opencl", "ncs2"])
def test_edge_accelerator_accepts_valid_values(value: str) -> None:
    """Verifies every documented accelerator value is accepted."""
    edge = EdgeSettings(accelerator=value)
    assert edge.accelerator == value


def test_edge_accelerator_rejects_invalid_value() -> None:
    """Verifies an unsupported accelerator value fails validation."""
    with pytest.raises(ValidationError):
        EdgeSettings(accelerator="quantum")


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_edge_confidence_threshold_rejects_out_of_range(value: float) -> None:
    """Verifies confidence_threshold is constrained to the [0.0, 1.0] range."""
    with pytest.raises(ValidationError):
        EdgeSettings(confidence_threshold=value)


@pytest.mark.parametrize(
    "input_str, expected_str",
    [
        ('"hello\\nworld"', "hello\nworld"),
        ("'hello\\nworld'", "hello\nworld"),
        ("no quotes\\n", "no quotes\n"),
        ('"unmatched quote', '"unmatched quote'),
        ('""', ""),
        ("''", ""),
        ('"\\n"', "\n"),
        ("just a string", "just a string"),
        ("", ""),
    ],
)
def test_message_cleaning(input_str: str, expected_str: str) -> None:
    """Verifies that message strings are cleaned correctly."""
    # Using DiscordSettings as an example to test the validator
    settings = DiscordSettings(message=input_str)
    assert settings.message == expected_str
