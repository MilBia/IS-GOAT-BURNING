"""Unit tests for the Application class in `is_goat_burning.app`.

These tests focus on the `_create_actions` method in isolation, verifying
that it correctly constructs the list of action handlers based on mock
settings, without involving the full application lifecycle.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

from is_goat_burning.app import Application
from is_goat_burning.on_fire_actions import SendEmail
from is_goat_burning.on_fire_actions import SendToDiscord


def create_mock_settings(use_emails: bool = False, use_discord: bool = False) -> MagicMock:
    """Creates a MagicMock to simulate the Settings object for testing.

    Args:
        use_emails: If True, the mock will be configured to enable emails.
        use_discord: If True, the mock will be configured to enable Discord.

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

    mock_once_action.assert_called_once_with([])
