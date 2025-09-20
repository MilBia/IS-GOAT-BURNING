"""The main application class that orchestrates the fire detection process."""

import asyncio
from collections.abc import Callable
from typing import Any

from is_goat_burning.config import settings
from is_goat_burning.fire_detection import StreamFireDetector
from is_goat_burning.fire_detection.signal_handler import SignalHandler
from is_goat_burning.logger import get_logger
from is_goat_burning.on_fire_actions import OnceAction
from is_goat_burning.on_fire_actions import SendEmail
from is_goat_burning.on_fire_actions import SendToDiscord

logger = get_logger("Application")


class Application:
    """Encapsulates the fire detection application's state and lifecycle.

    This class initializes all major components of the application, including
    the signal handler, the fire detector, and the action handlers. It manages
    the main asyncio tasks and ensures a graceful shutdown process.

    Attributes:
        signal_handler (SignalHandler): The application-wide signal handler.
        action_queue (asyncio.Queue): A queue for communicating events between
            the detector and the action manager.
        on_fire_action_handler (OnceAction): The handler that executes
            notification actions only once per application run.
        detector (StreamFireDetector): The main fire detection instance.
        detector_task (asyncio.Task | None): The task running the detector loop.
        action_manager_task (asyncio.Task | None): The task managing the event
            queue and triggering actions.
    """

    def __init__(self) -> None:
        """Initializes the Application instance."""
        self.signal_handler = SignalHandler()
        self.action_queue: asyncio.Queue[str] = asyncio.Queue()

        # Setup actions
        on_fire_actions = self._create_actions()
        self.on_fire_action_handler = OnceAction(on_fire_actions)

        # Setup detector
        self.detector = StreamFireDetector(
            src=settings.source,
            threshold=settings.fire_detection_threshold,
            video_output=settings.video_output,
            on_fire_action=self._queue_fire_event,
            checks_per_second=settings.checks_per_second,
        )

        # Tasks
        self.detector_task: asyncio.Task | None = None
        self.action_manager_task: asyncio.Task | None = None

    @staticmethod
    def _create_actions() -> list[tuple[type | Callable[..., Any], dict[str, Any]]]:
        """Builds a list of notification actions based on settings.

        This static method reads from the global `settings` object to determine
        which notification actions (e.g., email, Discord) should be initialized.

        Returns:
            A list of tuples, where each tuple contains an action class and a
            dictionary of its initialization kwargs. This list is suitable for
            passing to the `OnceAction` constructor.
        """
        actions: list[tuple[type | Callable[..., Any], dict[str, Any]]] = []
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

    async def _action_manager(self) -> None:
        """Manages the action queue, triggering handlers for received events.

        This task runs in an infinite loop, waiting for events (like "FIRE")
        to be placed on the `action_queue`. When an event is received, it
        triggers the corresponding action handler.
        """
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

    async def _queue_fire_event(self) -> None:
        """A callback for the detector to queue a fire event."""
        await self.action_queue.put("FIRE")

    async def _run_detector(self) -> None:
        """Runs the core detection loop as a supervised task."""
        try:
            await self.detector()
        except Exception:
            logger.exception("Fire detector task failed unexpectedly.")
        finally:
            logger.info("Fire detector task is shutting down.")

    async def run(self) -> None:
        """Starts and manages the main application tasks.

        This method creates and starts the action manager and detector tasks,
        assigns the main task to the signal handler for graceful shutdown, and
        awaits the completion of the detector task.
        """
        self.action_manager_task = asyncio.create_task(self._action_manager())
        self.detector_task = asyncio.create_task(self._run_detector())

        self.signal_handler.set_main_task(self.detector_task)
        await self.detector_task

    async def shutdown(self) -> None:
        """Gracefully shuts down all running application tasks.

        This method cancels the main application tasks and waits for them to
        finish, ensuring a clean exit.
        """
        logger.info("Initiating graceful shutdown.")

        tasks = []
        if self.action_manager_task and not self.action_manager_task.done():
            self.action_manager_task.cancel()
            tasks.append(self.action_manager_task)

        if self.detector_task and not self.detector_task.done():
            self.detector_task.cancel()
            tasks.append(self.detector_task)

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Application has shut down.")
