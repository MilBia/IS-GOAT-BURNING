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
from is_goat_burning.config import DecisionSettings
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


def test_alert_cooldown_defaults_to_600_seconds(monkeypatch: MonkeyPatch) -> None:
    """Verifies the alert cooldown default, isolated from any local ``.env``."""
    monkeypatch.delenv("ALERT_COOLDOWN_SECONDS", raising=False)
    assert Settings(_env_file=None, source="mock://source").alert_cooldown_seconds == 600.0


def test_alert_cooldown_is_read_from_env(monkeypatch: MonkeyPatch) -> None:
    """Verifies ``ALERT_COOLDOWN_SECONDS`` overrides the default."""
    monkeypatch.setenv("ALERT_COOLDOWN_SECONDS", "120")
    assert Settings(_env_file=None, source="mock://source").alert_cooldown_seconds == 120.0


def test_alert_cooldown_rejects_negative_values() -> None:
    """Verifies a negative alert cooldown fails validation."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, source="mock://source", alert_cooldown_seconds=-1)


def test_alert_retry_defaults_to_60_seconds(monkeypatch: MonkeyPatch) -> None:
    """Verifies the alert retry default, isolated from any local ``.env``."""
    monkeypatch.delenv("ALERT_RETRY_SECONDS", raising=False)
    assert Settings(_env_file=None, source="mock://source").alert_retry_seconds == 60.0


def test_alert_retry_is_read_from_env_and_rejects_negative(monkeypatch: MonkeyPatch) -> None:
    """Verifies ``ALERT_RETRY_SECONDS`` overrides the default and must not be negative."""
    monkeypatch.setenv("ALERT_RETRY_SECONDS", "15")
    assert Settings(_env_file=None, source="mock://source").alert_retry_seconds == 15.0
    with pytest.raises(ValidationError):
        Settings(_env_file=None, source="mock://source", alert_retry_seconds=-1)


def test_edge_settings_defaults() -> None:
    """Verifies EdgeSettings applies the documented defaults."""
    edge = EdgeSettings()
    assert edge.model_path is None
    assert edge.confidence_threshold == 0.5
    assert edge.accelerator == Accelerator.AUTO


@pytest.mark.parametrize("value", ["auto", "cpu", "cuda", "opencl"])
def test_edge_accelerator_accepts_valid_values(value: str) -> None:
    """Verifies every documented accelerator value is accepted."""
    edge = EdgeSettings(accelerator=value)
    assert edge.accelerator == value


@pytest.mark.parametrize("value", ["quantum", "ncs2"])
def test_edge_accelerator_rejects_invalid_value(value: str) -> None:
    """Verifies an unsupported accelerator value (including the dropped ``ncs2``) fails validation."""
    with pytest.raises(ValidationError):
        EdgeSettings(accelerator=value)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_edge_confidence_threshold_rejects_out_of_range(value: float) -> None:
    """Verifies confidence_threshold is constrained to the [0.0, 1.0] range."""
    with pytest.raises(ValidationError):
        EdgeSettings(confidence_threshold=value)


def test_decision_settings_defaults() -> None:
    """Verifies the Decisions API settings defaults."""
    decision = DecisionSettings()
    assert decision.model == "gemini-decision-1"
    assert decision.threshold == 0.5
    assert decision.timeout_seconds == 10.0


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_decision_threshold_rejects_out_of_range(value: float) -> None:
    """Verifies the decision threshold is constrained to the [0.0, 1.0] range."""
    with pytest.raises(ValidationError):
        DecisionSettings(threshold=value)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_decision_timeout_must_be_positive(value: float) -> None:
    """Verifies the request timeout must be strictly positive."""
    with pytest.raises(ValidationError):
        DecisionSettings(timeout_seconds=value)


@pytest.mark.parametrize("strategy", ["gemini_decision", "hybrid_decision"])
def test_decision_strategies_are_valid(monkeypatch: MonkeyPatch, strategy: str) -> None:
    """Verifies the Decisions API strategies are accepted without extra required settings."""
    monkeypatch.setenv("DETECTION_STRATEGY", strategy)
    assert Settings(_env_file=".env.tests").detection_strategy == strategy


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


def test_discord_retry_defaults_and_validation(monkeypatch: MonkeyPatch) -> None:
    """Verifies the Discord retry settings defaults, env override and bounds."""
    for key in list(os.environ):
        if key.startswith("DISCORD__"):
            monkeypatch.delenv(key)
    defaults = DiscordSettings()
    assert (defaults.timeout_seconds, defaults.max_attempts) == (10.0, 5)
    assert (defaults.retry_base_delay_seconds, defaults.retry_max_delay_seconds) == (1.0, 30.0)
    with pytest.raises(ValidationError):
        DiscordSettings(max_attempts=0)
    with pytest.raises(ValidationError):
        DiscordSettings(timeout_seconds=0)
    monkeypatch.setenv("SOURCE", "x")
    monkeypatch.setenv("DISCORD__MAX_ATTEMPTS", "3")
    assert Settings(_env_file=None).discord.max_attempts == 3


def test_verifier_failure_policy_defaults(monkeypatch: MonkeyPatch) -> None:
    """Verifies the verifier failure policy defaults (D2: alert unverified), isolated from any local ``.env``."""
    for name in ("VERIFIER_FAILURE_POLICY", "VERIFIER_ERRORS_BEFORE_POLICY", "GEMINI__TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, source="mock://source")
    assert settings.verifier_failure_policy == "alert_unverified"
    assert settings.verifier_errors_before_policy == 3
    assert settings.gemini.timeout_seconds == 10.0


def test_unverified_alert_note_default_and_env(monkeypatch: MonkeyPatch) -> None:
    """UNVERIFIED_ALERT_NOTE defaults to the English sentence and is read from the environment (#171)."""
    monkeypatch.delenv("UNVERIFIED_ALERT_NOTE", raising=False)
    assert Settings(_env_file=None, source="mock://source").unverified_alert_note.startswith("The fire verifier is unavailable")
    monkeypatch.setenv("UNVERIFIED_ALERT_NOTE", "Weryfikator niedostępny.")
    assert Settings(_env_file=None, source="mock://source").unverified_alert_note == "Weryfikator niedostępny."
    monkeypatch.setenv("UNVERIFIED_ALERT_NOTE", "n" * 501)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, source="mock://source")


def test_verifier_failure_policy_is_read_from_env(monkeypatch: MonkeyPatch) -> None:
    """Verifies the policy, error threshold and Gemini timeout are read from the environment."""
    monkeypatch.setenv("VERIFIER_FAILURE_POLICY", "suppress")
    monkeypatch.setenv("VERIFIER_ERRORS_BEFORE_POLICY", "5")
    monkeypatch.setenv("GEMINI__TIMEOUT_SECONDS", "2.5")
    settings = Settings(_env_file=None, source="mock://source")
    assert settings.verifier_failure_policy == "suppress"
    assert settings.verifier_errors_before_policy == 5
    assert settings.gemini.timeout_seconds == 2.5


@pytest.mark.parametrize(
    "overrides",
    [{"verifier_failure_policy": "ignore"}, {"verifier_errors_before_policy": 0}, {"gemini": {"timeout_seconds": 0}}],
)
def test_verifier_failure_settings_reject_invalid_values(overrides: dict) -> None:
    """Verifies an unknown policy, a zero error threshold and a non-positive timeout fail validation."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, source="mock://source", **overrides)


def test_discord_attach_frame_defaults_to_true(monkeypatch: MonkeyPatch) -> None:
    """Verifies ``DISCORD__ATTACH_FRAME`` defaults to true and can be disabled via the environment."""
    for key in list(os.environ):
        if key.startswith("DISCORD__"):
            monkeypatch.delenv(key)
    assert DiscordSettings().attach_frame is True
    monkeypatch.setenv("SOURCE", "x")
    monkeypatch.setenv("DISCORD__ATTACH_FRAME", "false")
    assert Settings().discord.attach_frame is False


def test_service_notification_defaults(monkeypatch: MonkeyPatch) -> None:
    """Service notifications are off by default (no hooks), with a daily heartbeat and a 10 min stream-down warning."""
    for name in (
        "DISCORD__SERVICE_HOOKS",
        "DISCORD__SERVICE_HEARTBEAT_INTERVAL_SECONDS",
        "DISCORD__SERVICE_STREAM_DOWN_WARNING_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    discord = Settings(_env_file=None, source="mock://source").discord
    assert discord.service_hooks == []
    assert discord.service_heartbeat_interval_seconds == 86400.0
    assert discord.service_stream_down_warning_seconds == 600.0


def test_service_settings_are_read_from_env(monkeypatch: MonkeyPatch) -> None:
    """Service settings use the DISCORD__ prefix and work without DISCORD__USE_DISCORD."""
    monkeypatch.setenv("DISCORD__USE_DISCORD", "false")
    monkeypatch.setenv("DISCORD__SERVICE_HOOKS", '["https://service.example/hook"]')
    monkeypatch.setenv("DISCORD__SERVICE_HEARTBEAT_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("DISCORD__SERVICE_STREAM_DOWN_WARNING_SECONDS", "120")
    discord = Settings(_env_file=None, source="mock://source").discord
    assert discord.service_hooks == ["https://service.example/hook"]
    assert discord.service_heartbeat_interval_seconds == 3600.0
    assert discord.service_stream_down_warning_seconds == 120.0


@pytest.mark.parametrize("field", ["service_heartbeat_interval_seconds", "service_stream_down_warning_seconds"])
def test_service_intervals_must_be_positive(field: str) -> None:
    """Zero or negative service intervals are rejected."""
    with pytest.raises(ValidationError):
        DiscordSettings(**{field: 0})


def test_service_hooks_must_not_repeat_alert_hooks() -> None:
    """Service messages must never land in the fire-alert channel; the error does not echo the URL."""
    with pytest.raises(ValidationError) as exc_info:
        DiscordSettings(use_discord=True, hooks=["https://a.example/secret"], service_hooks=["https://a.example/secret"])
    assert "DISCORD__SERVICE_HOOKS" in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


def test_service_hooks_overlap_error_hides_url_at_startup() -> None:
    """Through the real Settings path the overlap error does not echo the webhook URL either."""
    hooks = ["https://a.example/secret"]
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, source="mock://source", discord={"use_discord": True, "hooks": hooks, "service_hooks": hooks})
    assert "DISCORD__SERVICE_HOOKS" in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


def test_reconnect_max_delay_default_and_override(monkeypatch: MonkeyPatch) -> None:
    """RECONNECT_MAX_DELAY_SECONDS defaults to 300 and is read from the environment (isolated from a local .env)."""
    monkeypatch.delenv("RECONNECT_MAX_DELAY_SECONDS", raising=False)
    assert Settings(_env_file=None, source="mock://source").reconnect_max_delay_seconds == 300  # noqa: PLR2004
    monkeypatch.setenv("RECONNECT_MAX_DELAY_SECONDS", "42")
    assert Settings(_env_file=None, source="mock://source").reconnect_max_delay_seconds == 42  # noqa: PLR2004


def test_reconnect_max_delay_below_base_delay_is_rejected() -> None:
    """The cap must not be lower than the base delay."""
    with pytest.raises(ValidationError, match="RECONNECT_MAX_DELAY_SECONDS"):
        Settings(_env_file=None, source="mock://source", reconnect_delay_seconds=10, reconnect_max_delay_seconds=5)


def test_discord_file_limit_defaults_and_validation(monkeypatch: MonkeyPatch) -> None:
    """Verifies ``DISCORD__FILE_LIMIT_MB`` defaults to 18 MiB, can be overridden and must be positive."""
    for key in list(os.environ):
        if key.startswith("DISCORD__"):
            monkeypatch.delenv(key)
    assert DiscordSettings().file_limit_mb == 18.0
    monkeypatch.setenv("SOURCE", "x")
    monkeypatch.setenv("DISCORD__FILE_LIMIT_MB", "9.5")
    assert Settings().discord.file_limit_mb == 9.5
    with pytest.raises(ValidationError):
        DiscordSettings(file_limit_mb=0)
