from unittest.mock import MagicMock
from unittest.mock import patch

from is_goat_burning.app import Application
from is_goat_burning.on_fire_actions import SendEmail
from is_goat_burning.on_fire_actions import SendToDiscord


def create_mock_settings(use_emails=False, use_discord=False) -> MagicMock:
    """
    Creates a MagicMock to simulate the Settings object.
    We do not use `spec` because of incompatibilities between unittest.mock's
    introspection and Pydantic's model structure.
    """
    # Create the main mock WITHOUT spec for maximum flexibility.
    mock_settings = MagicMock()

    # Configure the nested email settings. MagicMock creates these attributes on the fly.
    mock_settings.email.use_emails = use_emails
    mock_settings.email.sender = "test_sender"
    mock_settings.email.sender_password.get_secret_value.return_value = "test_pass"
    mock_settings.email.recipients = ["test_rcpt"]
    mock_settings.email.subject = "subject"
    mock_settings.email.message = "message"
    mock_settings.email.email_host = "host"
    mock_settings.email.email_port = 123

    # Configure the nested discord settings.
    mock_settings.discord.use_discord = use_discord
    mock_settings.discord.message = "discord_message"
    mock_settings.discord.hooks = ["hook1"]

    return mock_settings


def test_create_actions_instantiates_email_action_when_enabled():
    """
    Verifies that only the SendEmail action is created when only email is enabled.
    """
    mock_settings = create_mock_settings(use_emails=True)
    with patch("is_goat_burning.app.settings", mock_settings):
        actions = Application._create_actions()

    assert len(actions) == 1
    action_class, kwargs = actions[0]
    assert action_class is SendEmail
    assert kwargs["sender"] == "test_sender"
    assert kwargs["sender_password"] == "test_pass"  # Ensure the secret is unwrapped


def test_create_actions_instantiates_discord_action_when_enabled():
    """
    Verifies that only the SendToDiscord action is created when only Discord is enabled.
    """
    mock_settings = create_mock_settings(use_discord=True)
    with patch("is_goat_burning.app.settings", mock_settings):
        actions = Application._create_actions()

    assert len(actions) == 1
    action_class, kwargs = actions[0]
    assert action_class is SendToDiscord
    assert kwargs["webhooks"] == ["hook1"]


def test_create_actions_instantiates_both_actions_when_enabled():
    """
    Verifies that both actions are created when both email and Discord are enabled.
    """
    mock_settings = create_mock_settings(use_emails=True, use_discord=True)
    with patch("is_goat_burning.app.settings", mock_settings):
        actions = Application._create_actions()

    assert len(actions) == 2
    action_classes = {action[0] for action in actions}
    assert {SendEmail, SendToDiscord} == action_classes


def test_create_actions_instantiates_no_actions_when_disabled():
    """
    Verifies that no actions are created when neither service is enabled.
    """
    mock_settings = create_mock_settings()
    with patch("is_goat_burning.app.settings", mock_settings):
        actions = Application._create_actions()

    assert len(actions) == 0
