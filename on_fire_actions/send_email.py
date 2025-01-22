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
    sender: str
    sender_password: str
    recipients: list[str]
    subject: str
    message: str
    message_html: str | None = None
    host: str = None
    port: int = None

    async def __call__(self) -> None:
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
            with smtplib.SMTP(host=self.host, port=self.port, timeout=1) as s:
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
