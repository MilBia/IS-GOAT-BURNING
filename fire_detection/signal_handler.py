import signal
import logging as log

from vidgear.gears.helper import logger_handler

logger = log.getLogger("SignalHandler")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class SignalHandler:
    """
    Handle proper exit on quiting quieting CTRL+C
    """

    KEEP_PROCESSING = True

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.info("Terminating processes.")
        self.KEEP_PROCESSING = False
