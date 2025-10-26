"""The main application class that orchestrates the fire detection process."""

from __future__ import annotations

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
    the signal handler and action handlers. It manages the main asyncio tasks,
    including the detector and action manager, and ensures a graceful shutdown
    and stream reconnection.

    Attributes:
        signal_handler (SignalHandler): The application-wide signal handler.
        action_queue (asyncio.Queue[str]): A queue for communicating events.
        on_fire_action_handler (OnceAction | None): The handler that executes
            notification actions.
        action_manager_task (asyncio.Task | None): The task managing the event
            queue and triggering actions.
    """

    def __init__(self) -> None:
        """Initializes the Application instance."""
        self.signal_handler = SignalHandler()
        self.action_queue: asyncio.Queue[str] = asyncio.Queue()
        self.on_fire_action_handler: OnceAction | None = None
        self.action_manager_task: asyncio.Task | None = None
        self._setup_actions()

    def _setup_actions(self) -> None:
        """Builds the list of notification actions based on settings."""
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
        self.on_fire_action_handler = OnceAction(actions)

    async def _action_manager(self) -> None:
        """Manages the action queue, triggering handlers for received events."""
        while True:
            try:
                event = await self.action_queue.get()
                try:
                    if event == "FIRE":
                        logger.info("Action manager received FIRE event. Triggering actions.")
                        if self.on_fire_action_handler:
                            await self.on_fire_action_handler()
                finally:
                    self.action_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Action manager task is shutting down.")
                break
            except Exception:
                logger.exception("Action manager encountered an error, but will continue running.")

    async def _queue_fire_event(self) -> None:
        """A callback for the detector to queue a fire event."""
        await self.action_queue.put("FIRE")

    async def run(self) -> None:
        """Starts and manages the main application tasks with a retry loop."""
        self.action_manager_task = asyncio.create_task(self._action_manager())

        while self.signal_handler.is_running():
            detector = None
            try:
                try:
                    logger.info(f"Attempting to start stream from source: {settings.source}")
                    detector = await StreamFireDetector.create(
                        src=settings.source,
                        threshold=settings.fire_detection_threshold,
                        video_output=settings.video_output,
                        on_fire_action=self._queue_fire_event,
                        checks_per_second=settings.checks_per_second,
                    )
                    async with asyncio.TaskGroup() as tg:
                        detector_task = tg.create_task(detector())
                        self.signal_handler.set_main_task(detector_task)
                except* Exception as exc_group:
                    logger.error(f"Detector task group failed: {exc_group.exceptions}")

            except asyncio.CancelledError:
                logger.info("Main run loop cancelled by shutdown signal.")
                break  # Exit the while loop immediately.
            except Exception:
                # This catches errors from StreamFireDetector.create() and others.
                logger.exception("An unexpected error occurred during the connection attempt.")
            finally:
                # This block executes regardless of success or failure, ensuring cleanup and delay.
                if detector and detector.stream:
                    await detector.stream.stop()

                if self.signal_handler.is_running():
                    logger.info(f"Stream has stopped. Reconnecting in {settings.reconnect_delay_seconds} seconds...")
                    try:
                        await asyncio.sleep(settings.reconnect_delay_seconds)
                    except asyncio.CancelledError:
                        logger.info("Reconnect wait cancelled by shutdown signal.")
                        break  # Exit the while loop immediately.

    async def shutdown(self) -> None:
        """Gracefully shuts down all running application tasks."""
        logger.info("Initiating graceful shutdown.")

        tasks = []
        if self.action_manager_task and not self.action_manager_task.done():
            self.action_manager_task.cancel()
            tasks.append(self.action_manager_task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Application has shut down.")
