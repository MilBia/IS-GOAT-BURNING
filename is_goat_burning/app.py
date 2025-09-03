import asyncio
import logging as log

from vidgear.gears.helper import logger_handler

from is_goat_burning.config import settings
from is_goat_burning.fire_detection import YTCamGearFireDetector
from is_goat_burning.fire_detection.signal_handler import SignalHandler
from is_goat_burning.on_fire_actions import OnceAction
from is_goat_burning.on_fire_actions import SendEmail
from is_goat_burning.on_fire_actions import SendToDiscord

logger = log.getLogger(__name__)
logger.propagate = False
logger.addHandler(logger_handler())


class Application:
    """Encapsulates the fire detection application's state and lifecycle."""

    def __init__(self):
        self.signal_handler = SignalHandler()
        self.action_queue = asyncio.Queue()

        # Setup actions
        on_fire_actions = self._create_actions()
        self.on_fire_action_handler = OnceAction(on_fire_actions)

        # Setup detector
        self.detector = YTCamGearFireDetector(
            src=settings.source,
            threshold=settings.fire_detection_threshold,
            logging=settings.logging,
            video_output=settings.video_output,
            on_fire_action=self._queue_fire_event,
            checks_per_second=settings.checks_per_second,
        )

        # Tasks
        self.detector_task: asyncio.Task | None = None
        self.action_manager_task: asyncio.Task | None = None

    @staticmethod
    def _create_actions():
        """Builds a list of notification actions based on settings."""
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

    async def _action_manager(self):
        """Waits for events on the queue and triggers the appropriate actions."""
        while True:
            try:
                event = await self.action_queue.get()
                if event == "FIRE":
                    logger.info("Action manager received FIRE event. Triggering actions.")
                    await self.on_fire_action_handler()
                self.action_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Action manager task is shutting down.")
                break
            except Exception:
                logger.exception("Action manager encountered an error, but will continue running.")

    async def _queue_fire_event(self):
        """Callback for the detector to queue a fire event."""
        await self.action_queue.put("FIRE")

    async def _run_detector(self):
        """Runs the core detection loop as a supervised task."""
        try:
            await self.detector()
        except Exception:
            logger.exception("Fire detector task failed unexpectedly.")
        finally:
            logger.info("Fire detector task is shutting down.")

    async def run(self):
        """Starts and runs the main application tasks."""
        self.action_manager_task = asyncio.create_task(self._action_manager())
        self.detector_task = asyncio.create_task(self._run_detector())

        self.signal_handler.set_main_task(self.detector_task)
        await self.detector_task

    async def shutdown(self):
        """Gracefully shuts down all application tasks."""
        logger.info("Initiating graceful shutdown.")

        tasks = []
        if self.action_manager_task:
            self.action_manager_task.cancel()
            tasks.append(self.action_manager_task)

        if self.detector_task and not self.detector_task.done():
            self.detector_task.cancel()
            tasks.append(self.detector_task)

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Application has shut down.")
