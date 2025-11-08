"""
Performance Tracker Module

Provides a context manager for tracking execution times and performance metrics.
"""

import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager


class PerformanceTracker:
    """
    Context manager for tracking execution times and performance metrics.

    Automatically logs performance data when exiting the context.
    """

    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, log_level: int = logging.INFO):
        """
        Initialize the performance tracker.

        Args:
            operation: Name of the operation being tracked
            logger: Logger to use for performance logging (defaults to performance logger)
            log_level: Logging level for performance messages
        """
        self.operation = operation
        self.logger = logger or logging.getLogger("storyspark.performance")
        self.log_level = log_level
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metadata: Dict[str, Any] = {}

    def __enter__(self):
        """Enter the context and start timing."""
        self.start_time = time.perf_counter()
        self.logger.log(self.log_level, f"Started operation: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and log performance metrics."""
        self.end_time = time.perf_counter()

        if self.start_time is not None:
            duration = self.end_time - self.start_time
            duration_ms = duration * 1000

            # Prepare log message
            message = f"Completed operation: {self.operation} in {duration_ms:.2f}ms"

            # Add metadata if present
            if self.metadata:
                metadata_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
                message += f" ({metadata_str})"

            # Add exception info if an exception occurred
            if exc_type is not None:
                message += f" (FAILED: {exc_type.__name__})"
                self.logger.error(message)
            else:
                self.logger.log(self.log_level, message)

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to be included in the performance log.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_duration(self) -> Optional[float]:
        """
        Get the duration of the operation in seconds.

        Returns:
            Duration in seconds, or None if not yet completed
        """
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @staticmethod
    @contextmanager
    def track(operation: str, logger: Optional[logging.Logger] = None):
        """
        Static method to create a performance tracker context manager.

        Args:
            operation: Name of the operation
            logger: Logger to use

        Yields:
            PerformanceTracker instance
        """
        tracker = PerformanceTracker(operation, logger)
        with tracker:
            yield tracker