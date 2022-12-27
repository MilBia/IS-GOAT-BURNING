import asyncio
import sys

sys.path.append("..")
from fire_detection import detect_fire  # noqa: E402
from on_fire_actions import OnceAction, send_email  # noqa: E402

"""
python online_video_with_output.py sender@gmail.com sender_password recipient1@gmail.com recipient2@gmail.com
"""


async def main(sender, sender_password, *recipients):
    on_fire_action = OnceAction(
        send_email,
        sender=sender,
        sender_password=sender_password,
        recipients=recipients,
        subject="GOAT ON FIRE!",
        message="Dear friend... Its time... Its time to Fight Fire With Fire.",
        host="smtp.gmail.com",
        port=587,
    )

    await detect_fire(
        "https://youtu.be/TqvguE5cKT0",
        threshold=0.1,
        logging=True,
        video_output=True,
        on_fire_action=on_fire_action,
        checks_per_second=1,
    )


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
