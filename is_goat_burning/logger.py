"""Provides a standardized logger for the entire application."""

import logging
import sys

from is_goat_burning.config import settings

LOG_LEVEL = "INFO" if settings.logging else "WARNING"

# The standard handler will be configured by the Application.
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))


def get_logger(name: str) -> logging.Logger:
    """Creates and configures a standard logger instance.

    Args:
        name: The name for the logger, typically `__name__`.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.addHandler(_handler)
    logger.propagate = False
    return logger
