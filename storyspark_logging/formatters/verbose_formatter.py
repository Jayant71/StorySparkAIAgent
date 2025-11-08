"""
Verbose Formatter Module

Provides detailed formatting for debugging and development.
"""

import logging
import inspect
import os
from datetime import datetime
from typing import Dict, Any, List


class VerboseFormatter(logging.Formatter):
    """
    Verbose formatter for detailed debugging output.

    Includes stack traces, variable values, and extensive context.
    """

    def __init__(self, include_stack: bool = True, include_locals: bool = False):
        """
        Initialize the verbose formatter.

        Args:
            include_stack: Whether to include stack trace information
            include_locals: Whether to include local variables (expensive)
        """
        super().__init__()
        self.include_stack = include_stack
        self.include_locals = include_locals

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with verbose details.

        Args:
            record: Log record to format

        Returns:
            Verbose formatted log message
        """
        # Start with timestamp and basic info
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        basic_info = f"[{timestamp}] {record.levelname} {record.name}"

        # Add location info
        location = f"{record.filename}:{record.lineno} in {record.funcName}"
        basic_info += f" ({location})"

        # Add the message
        message = record.getMessage()

        # Build the full output
        lines = [
            "=" * 80,
            basic_info,
            "-" * 80,
            f"Message: {message}",
        ]

        # Add exception info if present
        if record.exc_info:
            lines.extend(self._format_exception_verbose(record.exc_info))

        # Add stack info if requested
        if self.include_stack and record.stack_info:
            lines.extend([
                "",
                "Stack Info:",
                "-" * 40,
                record.stack_info
            ])

        # Add local variables if requested (expensive operation)
        if self.include_locals:
            lines.extend(self._format_locals(record))

        # Add extra fields
        extra = self._get_extra_fields(record)
        if extra:
            lines.extend([
                "",
                "Extra Fields:",
                "-" * 40,
                "\n".join(f"  {k}: {v}" for k, v in extra.items())
            ])

        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_exception_verbose(self, exc_info) -> List[str]:
        """
        Format exception information verbosely.

        Args:
            exc_info: Exception info tuple

        Returns:
            List of formatted lines
        """
        import traceback

        exc_type, exc_value, exc_traceback = exc_info

        lines = [
            "",
            "Exception Details:",
            "-" * 40,
            f"Type: {exc_type.__name__ if exc_type else 'Unknown'}",
            f"Message: {str(exc_value) if exc_value else 'None'}",
            "",
            "Traceback:",
            "-" * 40
        ]

        # Add formatted traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines.extend(tb_lines)

        return lines

    def _format_locals(self, record: logging.LogRecord) -> List[str]:
        """
        Format local variables from the calling frame.

        Args:
            record: Log record

        Returns:
            List of formatted lines
        """
        try:
            # Get the frame where the log call was made
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_frame = frame.f_back
                local_vars = caller_frame.f_locals

                lines = [
                    "",
                    "Local Variables:",
                    "-" * 40
                ]

                # Format variables (limit output size)
                for name, value in local_vars.items():
                    if not name.startswith('_'):  # Skip private variables
                        value_str = self._format_value(value)
                        lines.append(f"  {name}: {value_str}")

                        # Limit total output
                        if len(lines) > 50:  # Arbitrary limit
                            lines.append("  ... (truncated)")
                            break

                return lines
        except Exception:
            return ["", "Local Variables: (failed to retrieve)"]

        return []

    def _format_value(self, value: Any) -> str:
        """
        Format a value for display.

        Args:
            value: Value to format

        Returns:
            Formatted string representation
        """
        try:
            if isinstance(value, (str, int, float, bool, type(None))):
                return repr(value)
            elif isinstance(value, (list, tuple)):
                if len(value) > 5:
                    return f"{type(value).__name__} of {len(value)} items"
                return repr(value)
            elif isinstance(value, dict):
                if len(value) > 5:
                    return f"dict of {len(value)} items"
                return repr(value)
            else:
                return f"<{type(value).__name__} object>"
        except Exception:
            return "<error formatting value>"

    def _get_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Extract extra fields from the log record.

        Args:
            record: Log record

        Returns:
            Dictionary of extra fields
        """
        # Similar to JSONFormatter but for verbose output
        standard_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 'processName',
            'process', 'getMessage'
        }

        extra = {}
        for attr in dir(record):
            if not attr.startswith('_') and attr not in standard_fields:
                value = getattr(record, attr)
                if not callable(value):
                    extra[attr] = value

        return extra