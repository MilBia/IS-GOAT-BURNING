import smtplib
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from is_goat_burning.on_fire_actions.send_email import SendEmail


@pytest.fixture
def email_sender() -> SendEmail:
    return SendEmail(
        sender="test@example.com",
        sender_password="password",
        recipients=["recipient@example.com"],
        subject="Test Subject",
        message="Test Message",
        host="smtp.example.com",
        port=587,
    )


@pytest.mark.asyncio
async def test_send_email_sequence(email_sender: SendEmail) -> None:
    with patch("smtplib.SMTP") as mock_smtp:
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance
        await email_sender()
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("test@example.com", "password")
        mock_smtp_instance.send_message.assert_called_once()
        msg = mock_smtp_instance.send_message.call_args[0][0]
        assert msg.get_content_type() == "text/plain"


@pytest.mark.asyncio
async def test_send_email_with_html(email_sender: SendEmail) -> None:
    email_sender.message_html = "<h1>Test HTML</h1>"
    with patch("smtplib.SMTP") as mock_smtp:
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance
        await email_sender()
        mock_smtp_instance.send_message.assert_called_once()
        msg = mock_smtp_instance.send_message.call_args[0][0]
        assert msg.is_multipart()
        payload_types = {part.get_content_type() for part in msg.get_payload()}
        assert "text/plain" in payload_types
        assert "text/html" in payload_types


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_to_raise",
    [
        smtplib.SMTPAuthenticationError(535, b"Authentication failed"),
        TimeoutError("Connection timed out"),
        Exception("Generic error"),
    ],
)
async def test_send_email_handles_exceptions_gracefully(email_sender: SendEmail, exception_to_raise: Exception) -> None:
    with patch("smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.side_effect = exception_to_raise
        await email_sender()
