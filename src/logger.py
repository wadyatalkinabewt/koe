"""
Centralized error logging for Koe.

Logs errors to file without console output (for background pythonw.exe processes).
"""

import logging
import os
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Log file path
LOG_FILE = LOGS_DIR / "koe_errors.log"


class KoeLogger:
    """Centralized logger for Koe application."""

    _instance = None
    _logger = None

    def __init__(self):
        """Initialize the logger (singleton)."""
        if KoeLogger._logger is None:
            KoeLogger._logger = self._setup_logger()

    @classmethod
    def get_logger(cls):
        """Get the singleton logger instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._logger

    def _setup_logger(self):
        """Set up the file logger with no console output."""
        logger = logging.getLogger('koe')
        logger.setLevel(logging.ERROR)

        # Remove any existing handlers
        logger.handlers = []

        # File handler only (no console output)
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.ERROR)

        # Format: [2024-01-15 14:30:25] ERROR - message
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger


# Convenience functions for logging
def log_error(message, exception=None):
    """
    Log an error message to file.

    Args:
        message: Error message string
        exception: Optional exception object to include traceback
    """
    logger = KoeLogger.get_logger()
    if exception:
        logger.error(f"{message}: {str(exception)}", exc_info=True)
    else:
        logger.error(message)


def log_exception(exception, context=""):
    """
    Log an exception with full traceback.

    Args:
        exception: Exception object
        context: Optional context string (e.g., "in transcription")
    """
    logger = KoeLogger.get_logger()
    if context:
        logger.error(f"Exception {context}", exc_info=exception)
    else:
        logger.error("Exception occurred", exc_info=exception)


# Initialize logger on import
KoeLogger.get_logger()
