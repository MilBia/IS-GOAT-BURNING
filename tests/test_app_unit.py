"""Unit tests for the Application class in `is_goat_burning.app`.

These tests focus on the `_create_actions` method in isolation, verifying
that it correctly constructs the list of action handlers based on mock
settings, without involving the full application lifecycle.
"""

from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pytest

from is_goat_burning.app import Application
from is_goat_burning.on_fire_actions import SendEmail
from is_goat_burning.on_fire_actions import SendToDiscord
from is_goat_burning.on_fire_actions import SendVideoToDiscord


def create_mock_settings(use_emails: bool = False, use_discord: bool = False, send_video_chunks: bool = False) -> MagicMock:
    """Creates a MagicMock to simulate the Settings object for testing.

    Args:
        use_emails: If True, the mock will be configured to enable emails.
        use_discord: If True, the mock will be configured to enable Discord.
        send_video_chunks: If True, the mock will be configured to enable
            sending saved video chunks to Discord.

    Returns:
        A MagicMock instance configured to simulate the global settings object.
    """
    mock_settings = MagicMock()
    mock_settings.email.use_emails = use_emails
    mock_settings.email.sender = "test_sender"
    mock_settings.email.sender_password.get_secret_value.return_value = "test_pass"
    mock_settings.email.recipients = ["test_rcpt"]
    mock_settings.email.subject = "subject"
    mock_settings.email.message = "message"
    mock_settings.email.email_host = "host"
    mock_settings.email.email_port = 123
    mock_settings.discord.use_discord = use_discord
    mock_settings.discord.message = "discord_message"
    mock_settings.discord.hooks = ["hook1"]
    mock_settings.discord.send_video_chunks = send_video_chunks
    mock_settings.discord.service_hooks = []
    mock_settings.source = "mock://source"
    mock_settings.detection_strategy = "classic"
    mock_settings.alert_cooldown_seconds = 600.0
    mock_settings.alert_retry_seconds = 60.0
    return mock_settings


@patch("is_goat_burning.app.OnceAction")
def test_setup_actions_instantiates_email_action_when_enabled(mock_once_action: MagicMock) -> None:
    """Verifies `_setup_actions` configures SendEmail when enabled."""
    mock_settings = create_mock_settings(use_emails=True)
    with patch("is_goat_burning.app.settings", mock_settings):
        Application()  # __init__ calls _setup_actions

    # Assert that OnceAction was called with the correct arguments
    mock_once_action.assert_called_once()
    actions_arg = mock_once_action.call_args[0][0]
    assert len(actions_arg) == 1
    action_class, kwargs = actions_arg[0]
    assert action_class is SendEmail
    assert kwargs["sender"] == "test_sender"


@patch("is_goat_burning.app.OnceAction")
def test_setup_actions_instantiates_discord_action_when_enabled(mock_once_action: MagicMock) -> None:
    """Verifies `_setup_actions` configures SendToDiscord when enabled."""
    mock_settings = create_mock_settings(use_discord=True)
    with patch("is_goat_burning.app.settings", mock_settings):
        Application()

    mock_once_action.assert_called_once()
    actions_arg = mock_once_action.call_args[0][0]
    assert len(actions_arg) == 1
    action_class, kwargs = actions_arg[0]
    assert action_class is SendToDiscord
    assert kwargs["webhooks"] == ["hook1"]


@patch("is_goat_burning.app.OnceAction")
def test_setup_actions_instantiates_both_actions_when_enabled(mock_once_action: MagicMock) -> None:
    """Verifies `_setup_actions` configures both actions when both are enabled."""
    mock_settings = create_mock_settings(use_emails=True, use_discord=True)
    with patch("is_goat_burning.app.settings", mock_settings):
        Application()

    mock_once_action.assert_called_once()
    actions_arg = mock_once_action.call_args[0][0]
    assert len(actions_arg) == 2
    action_classes = {action[0] for action in actions_arg}
    assert {SendEmail, SendToDiscord} == action_classes


@patch("is_goat_burning.app.OnceAction")
def test_setup_actions_instantiates_no_actions_when_disabled(mock_once_action: MagicMock) -> None:
    """Verifies `_setup_actions` configures an empty list when no actions are enabled."""
    mock_settings = create_mock_settings()
    with patch("is_goat_burning.app.settings", mock_settings):
        Application()

    mock_once_action.assert_called_once_with([], cooldown_seconds=600.0, retry_seconds=60.0)


@patch("is_goat_burning.app.OnceAction")
def test_setup_actions_wires_video_event_action_when_enabled(mock_once_action: MagicMock) -> None:  # noqa: ARG001
    """Verifies a FireEventAction is wired when Discord video chunks are enabled."""
    mock_settings = create_mock_settings(use_discord=True, send_video_chunks=True)
    with (
        patch("is_goat_burning.app.FireEventAction") as mock_fire_event_action,
        patch("is_goat_burning.app.settings", mock_settings),
    ):
        app = Application()

    mock_fire_event_action.assert_called_once()
    kwargs = mock_fire_event_action.call_args.kwargs
    assert kwargs["event_queue"] is app.video_event_queue
    action_class, action_kwargs = kwargs["action"]
    assert action_class is SendVideoToDiscord
    assert action_kwargs["webhooks"] == ["hook1"]
    assert action_kwargs["file_limit_mb"] == mock_settings.discord.file_limit_mb
    assert app._video_chunks_enabled is True


@patch("is_goat_burning.app.OnceAction")
def test_setup_actions_skips_video_event_action_when_discord_disabled(mock_once_action: MagicMock) -> None:  # noqa: ARG001
    """Verifies no FireEventAction is wired when Discord itself is disabled."""
    mock_settings = create_mock_settings(use_discord=False, send_video_chunks=True)
    with (
        patch("is_goat_burning.app.FireEventAction") as mock_fire_event_action,
        patch("is_goat_burning.app.settings", mock_settings),
    ):
        app = Application()

    mock_fire_event_action.assert_not_called()
    assert app.video_event_action is None
    assert app._video_chunks_enabled is False


@patch("is_goat_burning.app.OnceAction")
def test_setup_actions_skips_video_event_action_when_chunks_disabled(mock_once_action: MagicMock) -> None:  # noqa: ARG001
    """Verifies no FireEventAction is wired when video chunk sending is disabled."""
    mock_settings = create_mock_settings(use_discord=True, send_video_chunks=False)
    with (
        patch("is_goat_burning.app.FireEventAction") as mock_fire_event_action,
        patch("is_goat_burning.app.settings", mock_settings),
    ):
        app = Application()

    mock_fire_event_action.assert_not_called()
    assert app.video_event_action is None


@patch("is_goat_burning.app.OnceAction")
def test_setup_actions_passes_attach_frame_to_discord(mock_once_action: MagicMock) -> None:
    """Verifies the Discord action receives ``DISCORD__ATTACH_FRAME`` (#177)."""
    mock_settings = create_mock_settings(use_discord=True)
    mock_settings.discord.attach_frame = False
    with patch("is_goat_burning.app.settings", mock_settings):
        Application()
    _, kwargs = mock_once_action.call_args[0][0][0]
    assert kwargs["attach_frame"] is False


@patch("is_goat_burning.app.OnceAction")
def test_service_notifier_is_built_from_service_settings(mock_once_action: MagicMock) -> None:  # noqa: ARG001
    """The service notifier gets the separate service webhooks, never the alert webhooks."""
    mock_settings = create_mock_settings(use_discord=True)
    mock_settings.discord.service_hooks = ["service-hook"]
    mock_settings.discord.service_heartbeat_interval_seconds = 3600.0
    mock_settings.discord.service_stream_down_warning_seconds = 120.0
    mock_settings.detection_strategy = "classic"
    mock_settings.source = "mock://source"
    with (
        patch("is_goat_burning.app.ServiceNotifier") as mock_notifier,
        patch("is_goat_burning.app.settings", mock_settings),
    ):
        app = Application()

    assert app.service_notifier is mock_notifier.return_value
    args, kwargs = mock_notifier.call_args
    assert args[0] == ["service-hook"]
    assert kwargs["heartbeat_interval"] == 3600.0
    assert kwargs["stream_down_threshold"] == 120.0
    assert kwargs["strategy"] == "classic"


@pytest.mark.asyncio
@patch("is_goat_burning.app.OnceAction")
async def test_fire_events_are_counted_by_the_service_notifier(mock_once_action: MagicMock) -> None:  # noqa: ARG001
    """Each queued fire event is reported to the service notifier (for the heartbeat)."""
    with patch("is_goat_burning.app.settings", create_mock_settings()):
        app = Application()
    app.service_notifier = MagicMock()
    await app._queue_fire_event()
    await app._queue_fire_event(unverified=True)
    assert app.service_notifier.fire_event.call_args_list == [call(unverified=False), call(unverified=True)]
