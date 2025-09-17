"""An example of running the fire detector on a live stream without video output."""

import asyncio
import sys

sys.path.append("..")
from is_goat_burning.fire_detection import YTCamGearFireDetector  # noqa: E402
from is_goat_burning.on_fire_actions import OnceAction  # noqa: E402
from is_goat_burning.on_fire_actions import SendEmail  # noqa: E402


async def main(sender: str, sender_password: str, *recipients: str) -> None:
    """Runs the detector and sends an email upon fire detection.

    This example shows how to run the detector on a live YouTube stream in a
    headless mode (no video output).

    Usage:
        python examples/online_video_stream_without_output.py sender@example.com "pass" recipient@example.com
    """
    on_fire_action = OnceAction(
        [
            (
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
            )
        ]
    )

    detector = YTCamGearFireDetector(
        src="https://youtu.be/TqvguE5cKT0",
        threshold=0.1,
        logging=False,
        video_output=False,
        on_fire_action=on_fire_action,
        checks_per_second=1,
    )
    await detector()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
