"""
Console Handler Module

Provides stream handler for console output with enhanced features.
"""

import logging
import sys
from typing import TextIO, Optional


class ConsoleHandler(logging.StreamHandler):
    """
    Enhanced console handler with better stream management.

    Extends StreamHandler with additional features for console output.
    """

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        auto_flush: bool = True,
        encoding: Optional[str] = None
    ):
        """
        Initialize the console handler.

        Args:
            stream: Output stream (defaults to stdout)
            auto_flush: Whether to flush after each write
            encoding: Stream encoding
        """
        if stream is None:
            stream = sys.stdout

        super().__init__(stream)

        self.auto_flush = auto_flush
        self.encoding = encoding or getattr(stream, 'encoding', 'utf-8')

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record with enhanced error handling.

        Args:
            record: Log record to emit
        """
        try:
            super().emit(record)

            # Auto-flush if requested
            if self.auto_flush:
                self.flush()

        except Exception:
            # If console output fails, try stderr as fallback
            self._handle_emit_error(record)

    def _handle_emit_error(self, record: logging.LogRecord) -> None:
        """
        Handle emission errors by attempting fallback to stderr.

        Args:
            record: Log record that failed to emit
        """
        try:
            # Try to write to stderr
            fallback_msg = f"Console logging failed, falling back to stderr: {record.getMessage()}\n"
            sys.stderr.write(fallback_msg)
            sys.stderr.flush()
        except Exception:
            # Last resort: do nothing to avoid infinite loops
            pass

    def flush(self) -> None:
        """
        Flush the stream.
        """
        try:
            if hasattr(self.stream, 'flush'):
                self.stream.flush()
        except Exception:
            pass  # Silent failure

    @property
    def is_tty(self) -> bool:
        """
        Check if the stream is connected to a TTY.

        Returns:
            True if connected to a terminal
        """
        return hasattr(self.stream, 'isatty') and self.stream.isatty()

    @property
    def supports_color(self) -> bool:
        """
        Check if the stream supports ANSI color codes.

        Returns:
            True if colors are supported
        """
        return self.is_tty and self.encoding.lower() in ('utf-8', 'utf8')

    def set_stream(self, stream: TextIO) -> None:
        """
        Change the output stream.

        Args:
            stream: New output stream
        """
        self.stream = stream
        self.encoding = getattr(stream, 'encoding', 'utf-8')