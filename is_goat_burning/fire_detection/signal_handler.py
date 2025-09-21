"""Handles application-wide signals for graceful shutdown and custom events."""

import asyncio
import signal

from is_goat_burning.logger import get_logger

logger = get_logger("SignalHandler")


class SignalHandler:
    """A singleton to manage signals for graceful exit and custom events.

    This class ensures that there is only one instance handling signals like
    SIGINT and SIGTERM across the entire application. It also provides an
    asyncio.Event to signal custom occurrences, such as fire detection.

    Attributes:
        fire_detected_event (asyncio.Event): An event that is set when a fire
            is detected and cleared once handled.
        main_task (asyncio.Task | None): The main application task that should
            be cancelled upon receiving a termination signal.
    """

    _instance: "SignalHandler | None" = None

    def __new__(cls: type["SignalHandler"]) -> "SignalHandler":
        """Creates a new instance of the class if one does not already exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize state only once
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initializes the SignalHandler singleton instance.

        Sets up the signal handlers for SIGINT and SIGTERM and initializes the
        asyncio event. The `_initialized` flag prevents re-initialization on
        subsequent retrievals of the singleton instance.
        """
        if self._initialized:
            return

        self.fire_detected_event = asyncio.Event()
        self.main_task: asyncio.Task | None = None
        self._running = True

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        self._initialized = True
        logger.debug("SignalHandler singleton initialized.")

    def set_main_task(self, task: asyncio.Task) -> None:
        """Stores a reference to the main application task.

        Args:
            task: The primary asyncio.Task to be cancelled on graceful exit.
        """
        self.main_task = task

    def exit_gracefully(self, *args, **kwargs) -> None:
        """Cancels the main task and stops the main loop on termination signal.

        This method is registered as the handler for SIGINT and SIGTERM.
        """
        logger.info("Termination signal received. Requesting graceful exit.")
        self._running = False
        if self.main_task and not self.main_task.done():
            self.main_task.cancel()

    def is_running(self) -> bool:
        """Checks if the main application loop should continue running."""
        return self._running

    def fire_detected(self) -> None:
        """Sets the event to signal that a fire has been detected.

        This method is idempotent; calling it multiple times after the first
        has no additional effect until the event is cleared.
        """
        if not self.fire_detected_event.is_set():
            logger.warning("FIRE EVENT SIGNALED!")
            self.fire_detected_event.set()

    def is_fire_detected(self) -> bool:
        """Checks if the fire event has been triggered.

        Returns:
            True if the fire event is currently set, False otherwise.
        """
        return self.fire_detected_event.is_set()

    def reset_fire_event(self) -> None:
        """Resets the fire event after it has been handled."""
        self.fire_detected_event.clear()
        logger.info("Fire event has been handled and reset.")
