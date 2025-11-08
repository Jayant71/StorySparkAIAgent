"""
JSON Formatter Module

Provides structured JSON logging format for production environments.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Formats log records as JSON objects with standardized fields.
    """

    def __init__(self, include_extra: bool = True):
        """
        Initialize the JSON formatter.

        Args:
            include_extra: Whether to include extra fields from the log record
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log entry
        """
        # Create the base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
            "thread_name": record.threadName
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self._format_exception(record.exc_info)

        # Add extra fields if requested
        if self.include_extra:
            extra_fields = self._get_extra_fields(record)
            if extra_fields:
                log_entry["extra"] = extra_fields

        # Convert to JSON
        return json.dumps(log_entry, default=str, ensure_ascii=False)

    def _format_exception(self, exc_info) -> Dict[str, Any]:
        """
        Format exception information.

        Args:
            exc_info: Exception info tuple

        Returns:
            Formatted exception data
        """
        import traceback

        exc_type, exc_value, exc_traceback = exc_info

        return {
            "type": exc_type.__name__ if exc_type else "Unknown",
            "message": str(exc_value) if exc_value else "",
            "traceback": traceback.format_exception(exc_type, exc_value, exc_traceback)
        }

    def _get_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Extract extra fields from the log record.

        Args:
            record: Log record

        Returns:
            Dictionary of extra fields
        """
        # Get all attributes that are not standard logging fields
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