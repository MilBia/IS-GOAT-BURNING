from unittest.mock import patch

from is_goat_burning.app import Application
from is_goat_burning.on_fire_actions import SendEmail
from is_goat_burning.on_fire_actions import SendToDiscord


class MockSettings:
    """A flexible mock for the settings object to avoid depending on environment variables."""

    def __init__(self, use_emails=False, use_discord=False):
        self.email = self.Email(use_emails)
        self.discord = self.Discord(use_discord)

    class Email:
        def __init__(self, use_emails):
            self.use_emails = use_emails
            self.sender = "test_sender"
            self.sender_password = self.Secret("test_pass")
            self.recipients = ["test_rcpt"]
            self.subject = "subject"
            self.message = "message"
            self.email_host = "host"
            self.email_port = 123

        class Secret:
            def __init__(self, value):
                self._value = value

            def get_secret_value(self):
                return self._value

    class Discord:
        def __init__(self, use_discord):
            self.use_discord = use_discord
            self.message = "discord_message"
            self.hooks = ["hook1"]


def test_create_actions_email_only():
    """
    Verifies that only the SendEmail action is created when only email is enabled.
    """
    mock_settings = MockSettings(use_emails=True, use_discord=False)
    with patch("is_goat_burning.app.settings", mock_settings):
        actions = Application._create_actions()

    assert len(actions) == 1
    action_class, kwargs = actions[0]
    assert action_class is SendEmail
    assert kwargs["sender"] == "test_sender"
    assert "get_secret_value" not in kwargs["sender_password"]  # Ensure the secret is unwrapped


def test_create_actions_discord_only():
    """
    Verifies that only the SendToDiscord action is created when only Discord is enabled.
    """
    mock_settings = MockSettings(use_emails=False, use_discord=True)
    with patch("is_goat_burning.app.settings", mock_settings):
        actions = Application._create_actions()

    assert len(actions) == 1
    action_class, kwargs = actions[0]
    assert action_class is SendToDiscord
    assert kwargs["webhooks"] == ["hook1"]


def test_create_actions_both_enabled():
    """
    Verifies that both actions are created when both email and Discord are enabled.
    """
    mock_settings = MockSettings(use_emails=True, use_discord=True)
    with patch("is_goat_burning.app.settings", mock_settings):
        actions = Application._create_actions()

    assert len(actions) == 2
    action_classes = {action[0] for action in actions}
    assert {SendEmail, SendToDiscord} == action_classes


def test_create_actions_none_enabled():
    """
    Verifies that no actions are created when neither service is enabled.
    """
    mock_settings = MockSettings(use_emails=False, use_discord=False)
    with patch("is_goat_burning.app.settings", mock_settings):
        actions = Application._create_actions()

    assert len(actions) == 0
