from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from on_fire_actions.send_email import SendEmail


@pytest.mark.asyncio
async def test_send_email():
    # Create an instance of the SendEmail class
    email_sender = SendEmail(
        sender="test@example.com",
        sender_password="password",
        recipients=["recipient@example.com"],
        subject="Test Subject",
        message="Test Message",
        host="smtp.example.com",
        port=587,
    )

    # Mock the smtplib.SMTP class
    with patch("smtplib.SMTP") as mock_smtp:
        # Create a mock instance of the SMTP class
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance

        # Call the __call__ method
        await email_sender()

        # Assert that the login and send_message methods were called
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("test@example.com", "password")
        mock_smtp_instance.send_message.assert_called_once()
