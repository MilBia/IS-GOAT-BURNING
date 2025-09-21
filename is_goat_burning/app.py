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
    """Encapsulates the fire detection application's state and lifecycle."""

    def __init__(self) -> None:
        """Initializes the Application instance."""
        self.signal_handler = SignalHandler()
        self.action_queue: asyncio.Queue[str] = asyncio.Queue()
        self.on_fire_action_handler: OnceAction | None = None
        self.detector_task: asyncio.Task | None = None
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
                if event == "FIRE":
                    logger.info("Action manager received FIRE event. Triggering actions.")
                    if self.on_fire_action_handler:
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

    async def run(self) -> None:
        """Starts and manages the main application tasks with a retry loop."""
        self.action_manager_task = asyncio.create_task(self._action_manager())

        while self.signal_handler.is_running():
            detector = None
            try:
                logger.info(f"Attempting to start stream from source: {settings.source}")
                detector = await StreamFireDetector.create(
                    src=settings.source,
                    threshold=settings.fire_detection_threshold,
                    video_output=settings.video_output,
                    on_fire_action=self._queue_fire_event,
                    checks_per_second=settings.checks_per_second,
                )
                self.detector_task = asyncio.create_task(detector())
                self.signal_handler.set_main_task(self.detector_task)
                await self.detector_task

            except asyncio.CancelledError:
                # This is caught when exit_gracefully is called.
                logger.info("Detector task cancelled by signal handler.")
                break  # Exit the while loop
            except Exception as e:
                logger.error(f"Detector failed with an unhandled exception: {e}")
            finally:
                if detector and detector.stream:
                    await detector.stream.stop()  # Ensure cleanup

            if self.signal_handler.is_running():
                logger.info(f"Stream has stopped unexpectedly. Reconnecting in {settings.reconnect_delay_seconds} seconds...")
                try:
                    await asyncio.sleep(settings.reconnect_delay_seconds)
                except asyncio.CancelledError:
                    logger.info("Reconnect wait cancelled by shutdown signal.")
                    break  # Exit the while loop immediately

    async def shutdown(self) -> None:
        """Gracefully shuts down all running application tasks."""
        logger.info("Initiating graceful shutdown.")

        tasks = []
        if self.action_manager_task and not self.action_manager_task.done():
            self.action_manager_task.cancel()
            tasks.append(self.action_manager_task)

        # The detector task is managed by the run loop, but we ensure it's
        # cancelled if it's still somehow running.
        if self.detector_task and not self.detector_task.done():
            self.detector_task.cancel()
            tasks.append(self.detector_task)

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Application has shut down.")
