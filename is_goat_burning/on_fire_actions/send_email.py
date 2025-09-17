"""Provides a class to send email notifications when a fire is detected."""

import asyncio
from dataclasses import dataclass
from email.message import EmailMessage
import logging as log
import smtplib
import ssl

from vidgear.gears.helper import logger_handler

logger = log.getLogger("EmailSender")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


@dataclass(init=True, repr=False, eq=False, order=False, kw_only=True, slots=True)
class SendEmail:
    """An awaitable class that sends an email notification.

    This class encapsulates the configuration and logic for sending an email
    via SMTP. The `__call__` method is awaitable and executes the blocking
    SMTP operations in a separate thread to avoid blocking the asyncio event loop.

    Attributes:
        sender (str): The email address of the sender.
        sender_password (str): The password or app-specific password for the
            sender's email account.
        recipients (list[str]): A list of recipient email addresses.
        subject (str): The subject line of the email.
        message (str): The plain text body of the email.
        message_html (str | None): An optional HTML version of the email body.
        host (str): The SMTP server host (e.g., "smtp.gmail.com").
        port (int): The port for the SMTP server (e.g., 587 for TLS).
    """

    sender: str
    sender_password: str
    recipients: list[str]
    subject: str
    message: str
    message_html: str | None = None
    host: str
    port: int

    def _send_email_blocking(self) -> None:
        """Constructs and sends the email using blocking I/O.

        This method contains the actual smtplib implementation and is intended
        to be run in a thread pool executor.
        """
        msg = EmailMessage()
        msg["Subject"] = self.subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(self.message)
        if self.message_html:
            msg.add_alternative(self.message_html, subtype="html")

        context = ssl.create_default_context()
        logger.info("Sending an email...")
        try:
            with smtplib.SMTP(host=self.host, port=self.port, timeout=5) as s:
                s.starttls(context=context)
                logger.info("Context loaded")
                s.login(self.sender, self.sender_password)
                logger.info("Logged in")
                s.send_message(msg)
                logger.info("Email sent")
        except smtplib.SMTPAuthenticationError:
            logger.error("Authentication failed")
        except TimeoutError:
            logger.error("Connection timed out")
        except Exception as e:
            logger.error(e)

    async def __call__(self) -> None:
        """Asynchronously sends the email notification.

        This method schedules the blocking `_send_email_blocking` method to run
        in the event loop's default executor.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._send_email_blocking)
