import asyncio
import logging as log

from vidgear.gears.helper import logger_handler

from config import settings
from fire_detection import YTCamGearFireDetector
from fire_detection.signal_handler import SignalHandler
from on_fire_actions import OnceAction
from on_fire_actions import SendEmail
from on_fire_actions import SendToDiscord

logger = log.getLogger(__name__)
logger.propagate = False
logger.addHandler(logger_handler())


async def run_detector(detector: YTCamGearFireDetector):
    """The core detection loop, designed to be run as a supervised task."""
    try:
        await detector()
    except Exception:
        logger.exception("Fire detector task failed unexpectedly.")
    finally:
        logger.info("Fire detector task is shutting down.")


def create_actions():
    actions = []
    if settings.email.use_emails:
        actions.append(
            (
                SendEmail,
                {
                    "sender": settings.email.sender,
                    "sender_password": settings.email.sender_password.get_secret_value(),
                    "recipients": settings.email.recipients,
                    "subject": settings.email.subject,
                    "message": settings.email.message,
                    "host": settings.email.email_host,
                    "port": settings.email.email_port,
                },
            )
        )
    if settings.discord.use_discord:
        actions.append(
            (
                SendToDiscord,
                {
                    "message": settings.discord.message,
                    "webhooks": settings.discord.hooks,
                },
            )
        )
    return actions


async def main():
    signal_handler = SignalHandler()
    main_task = None

    actions = create_actions()
    on_fire_action = OnceAction(actions)
    detector = YTCamGearFireDetector(
        src=settings.source,
        threshold=settings.fire_detection_threshold,
        logging=settings.logging,
        video_output=settings.video_output,
        on_fire_action=on_fire_action,
        checks_per_second=settings.checks_per_second,
    )
    try:
        main_task = asyncio.create_task(run_detector(detector))
        signal_handler.set_main_task(main_task)
        await main_task
    except asyncio.CancelledError:
        logger.info("Main task was cancelled. Graceful shutdown initiated.")
    finally:
        if main_task and not main_task.done():
            main_task.cancel()
            await main_task
        logger.info("Application has shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
