import asyncio

from fire_detection import detect_fire
from on_fire_actions import OnceAction, SendEmail, SendToDiscord
from setting import (
    SENDER,
    SENDER_PASSWORD,
    RECIPIENTS,
    LOGGING,
    EMAIL_HOST,
    EMAIL_PORT,
    VIDEO_OUTPUT,
    SOURCE,
    CHECKS_PER_SECOND,
    USE_EMAILS,
    USE_DISCORD,
    DISCORD_HOOKS,
)


async def main():
    actions = []
    if USE_EMAILS:
        actions.append(
            [
                SendEmail,
                {
                    "sender": SENDER,
                    "sender_password": SENDER_PASSWORD,
                    "recipients": RECIPIENTS,
                    "subject": "GOAT ON FIRE!",
                    "message": "Dear friend... Its time... Its time to Fight Fire With Fire!",
                    "host": EMAIL_HOST,
                    "port": EMAIL_PORT,
                },
            ]
        )
    if USE_DISCORD:
        actions.append(
            [
                SendToDiscord,
                {
                    "message": "Dear friend... Its time... Its time to Fight Fire With Fire!",
                    "webhooks": DISCORD_HOOKS,
                },
            ]
        )
    on_fire_action = OnceAction(actions)
    await detect_fire(
        src=SOURCE,
        threshold=0.1,
        logging=LOGGING,
        video_output=VIDEO_OUTPUT,
        on_fire_action=on_fire_action,
        checks_per_second=CHECKS_PER_SECOND,
    )


if __name__ == "__main__":
    asyncio.run(main())
