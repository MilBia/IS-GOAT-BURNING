import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional


def send_email(
    sender: str,
    sender_password: str,
    recipients: list[str],
    subject: str,
    message: str,
    message_html: Optional[str] = None,
    host: str = None,
    port: int = None,
) -> None:
    """
    Sending an email, probably about burning goat.

    :param sender: email of sender
    :param sender_password: password of sender
    :param recipients: list to recipients emails
    :param subject: subject of email
    :param message: message to send
    :param message_html: html message version to send
    :param host: smtp host url
    :param port: smtp host port
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(message)
    if message_html:
        msg.add_alternative(message_html, subtype="html")

    context = ssl.create_default_context()
    print("Sending an email...")
    with smtplib.SMTP(host=host, port=port) as s:
        s.starttls(context=context)
        s.login(sender, sender_password)
        s.send_message(msg)
    print("Email sent")
