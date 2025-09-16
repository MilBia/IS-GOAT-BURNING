"""Unit tests for the SendEmail action class."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from is_goat_burning.on_fire_actions.send_email import SendEmail


@pytest.mark.asyncio
async def test_send_email() -> None:
    """Verifies that the SendEmail class correctly interacts with smtplib.

    This test mocks the `smtplib.SMTP` class to ensure that the `SendEmail`
    action correctly calls the `starttls`, `login`, and `send_message` methods
    with the expected parameters.
    """
    email_sender = SendEmail(
        sender="test@example.com",
        sender_password="password",
        recipients=["recipient@example.com"],
        subject="Test Subject",
        message="Test Message",
        host="smtp.example.com",
        port=587,
    )

    with patch("smtplib.SMTP") as mock_smtp:
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance

        await email_sender()

        # Assert that the correct SMTP sequence was called
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("test@example.com", "password")
        mock_smtp_instance.send_message.assert_called_once()
