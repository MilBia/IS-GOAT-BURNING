"""The main entry point for the Is the Gävle Goat Burning application."""

import asyncio

from is_goat_burning.app import Application
from is_goat_burning.logger import get_logger

logger = get_logger("Entrypoint")


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
