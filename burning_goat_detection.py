import asyncio

from config import settings
from fire_detection import detect_fire
from on_fire_actions import OnceAction
from on_fire_actions import SendEmail
from on_fire_actions import SendToDiscord


async def main():
    actions = []
    if settings.email.use_emails:
        actions.append(
            [
                SendEmail,
                {
                    "sender": settings.email.sender,
                    "sender_password": settings.email.sender_password,
                    "recipients": settings.email.recipients,
                    "subject": settings.email.subject,
                    "message": settings.email.message,
                    "host": settings.email.email_host,
                    "port": settings.email.email_port,
                },
            ]
        )
    if settings.discord.use_discord:
        actions.append(
            [
                SendToDiscord,
                {
                    "message": settings.discord.message,
                    "webhooks": settings.discord.hooks,
                },
            ]
        )
    on_fire_action = OnceAction(actions)
    await detect_fire(
        src=settings.source,
        threshold=0.1,
        logging=settings.logging,
        video_output=settings.video_output,
        on_fire_action=on_fire_action,
        checks_per_second=settings.checks_per_second,
    )


if __name__ == "__main__":
    asyncio.run(main())
