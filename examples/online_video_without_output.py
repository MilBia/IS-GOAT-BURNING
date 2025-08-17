import asyncio
import sys

sys.path.append("..")
from fire_detection import YTCamGearFireDetector  # noqa: E402
from on_fire_actions import OnceAction  # noqa: E402
from on_fire_actions import SendEmail  # noqa: E402


async def main(sender, sender_password, *recipients):
    """
    python online_video_with_output.py sender@gmail.com sender_password recipient1@gmail.com recipient2@gmail.com
    """
    on_fire_action = OnceAction(
        [
            [
                SendEmail,
                {
                    "sender": sender,
                    "sender_password": sender_password,
                    "recipients": recipients,
                    "subject": "GOAT ON FIRE!",
                    "message": "Dear friend... Its time... Its time to Fight Fire With Fire.",
                    "host": "smtp.gmail.com",
                    "port": 587,
                },
            ]
        ]
    )

    await YTCamGearFireDetector(
        src="https://youtu.be/6jge6uzRl-k",
        threshold=0.05,
        logging=True,
        video_output=False,
        on_fire_action=on_fire_action,
        checks_per_second=1,
    )()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
