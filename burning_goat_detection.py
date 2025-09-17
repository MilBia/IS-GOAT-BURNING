"""The main entry point for the Is the Gävle Goat Burning application."""

import asyncio
import logging as log

from vidgear.gears.helper import logger_handler

from is_goat_burning.app import Application

logger = log.getLogger(__name__)
logger.propagate = False
logger.addHandler(logger_handler())


async def main():
    """Initializes and runs the application."""
    app = Application()
    try:
        await app.run()
    except asyncio.CancelledError:
        logger.info("Main application task was cancelled by signal.")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
