"""
Console Formatter Module

Provides human-readable formatting for console output.
"""

import logging
import colorama
from datetime import datetime
from typing import Optional


class ConsoleFormatter(logging.Formatter):
    """
    Console formatter with colors and human-readable output.

    Formats log records for console display with color coding.
    """

    # Color mappings for different log levels
    COLORS = {
        logging.DEBUG: colorama.Fore.CYAN,
        logging.INFO: colorama.Fore.GREEN,
        logging.WARNING: colorama.Fore.YELLOW,
        logging.ERROR: colorama.Fore.RED,
        logging.CRITICAL: colorama.Fore.RED + colorama.Style.BRIGHT
    }

    def __init__(self, use_colors: bool = True, show_timestamp: bool = True):
        """
        Initialize the console formatter.

        Args:
            use_colors: Whether to use ANSI color codes
            show_timestamp: Whether to show timestamps
        """
        super().__init__()
        self.use_colors = use_colors
        self.show_timestamp = show_timestamp

        # Initialize colorama for Windows compatibility
        if self.use_colors:
            colorama.init(autoreset=True)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record for console output.

        Args:
            record: Log record to format

        Returns:
            Formatted log message
        """
        # Get the base message
        message = record.getMessage()

        # Build the formatted message
        parts = []

        # Add timestamp if requested
        if self.show_timestamp:
            timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            parts.append(f"[{timestamp}]")

        # Add level
        level_str = f"[{record.levelname}]"
        parts.append(level_str)

        # Add logger name (shortened)
        logger_name = self._shorten_logger_name(record.name)
        parts.append(f"[{logger_name}]")

        # Add the message
        parts.append(message)

        # Join parts
        formatted = " ".join(parts)

        # Add colors if enabled
        if self.use_colors:
            color = self.COLORS.get(record.levelno, colorama.Fore.WHITE)
            formatted = f"{color}{formatted}{colorama.Style.RESET_ALL}"

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted

    def _shorten_logger_name(self, name: str) -> str:
        """
        Shorten logger name for display.

        Args:
            name: Full logger name

        Returns:
            Shortened name
        """
        # Keep only the last two parts of the logger name
        parts = name.split('.')
        if len(parts) > 2:
            return '.'.join(parts[-2:])
        return name

    def formatException(self, ei) -> str:
        """
        Format exception information with colors.

        Args:
            ei: Exception info

        Returns:
            Formatted exception
        """
        formatted = super().formatException(ei)

        if self.use_colors:
            # Color the exception in red
            formatted = f"{colorama.Fore.RED}{formatted}{colorama.Style.RESET_ALL}"

        return formatted