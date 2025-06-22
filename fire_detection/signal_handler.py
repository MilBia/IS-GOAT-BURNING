import asyncio
import logging as log
import signal

from vidgear.gears.helper import logger_handler

logger = log.getLogger("SignalHandler")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class SignalHandler:
    """
    A singleton to handle application-wide signals, including graceful exit
    and custom events like fire detection.
    """

    _instance = None

    KEEP_PROCESSING = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize state only once
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Ensure __init__ is only run once
        if self._initialized:
            return

        # --- State ---
        self.KEEP_PROCESSING = True
        self.fire_detected_event = asyncio.Event()

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        self._initialized = True
        logger.debug("SignalHandler singleton initialized.")

    def exit_gracefully(self, *args, **kwargs):
        """Called by SIGINT/SIGTERM to signal for a graceful shutdown."""
        logger.info("Termination signal received. Requesting graceful exit.")
        self.KEEP_PROCESSING = False

    def fire_detected(self):
        """Signals that a fire has been detected."""
        if not self.fire_detected_event.is_set():
            logger.warning("FIRE EVENT SIGNALED!")
            self.fire_detected_event.set()

    def is_fire_detected(self) -> bool:
        """Checks if the fire event has been triggered."""
        return self.fire_detected_event.is_set()

    def reset_fire_event(self):
        """Resets the fire event after it has been handled."""
        self.fire_detected_event.clear()
        logger.info("Fire event has been handled and reset.")
