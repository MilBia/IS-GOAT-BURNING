import asyncio

from fire_detection import detect_fire
from on_fire_actions import OnceAction
from on_fire_actions import SendEmail
from on_fire_actions import SendToDiscord
from setting import CHECKS_PER_SECOND
from setting import DISCORD_HOOKS
from setting import EMAIL_HOST
from setting import EMAIL_PORT
from setting import LOGGING
from setting import RECIPIENTS
from setting import SENDER
from setting import SENDER_PASSWORD
from setting import SOURCE
from setting import USE_DISCORD
from setting import USE_EMAILS
from setting import VIDEO_OUTPUT


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
