"""
Error Enhancer Module

Provides utilities for enhanced error logging with context and stack traces.
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Union
from pathlib import Path


class ErrorEnhancer:
    """
    Utility class for enhanced error logging with context and stack traces.
    """

    @staticmethod
    def log_error(
        logger: logging.Logger,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        include_stack_trace: bool = True,
        log_level: int = logging.ERROR
    ) -> None:
        """
        Log an error with enhanced context and stack trace information.

        Args:
            logger: Logger instance to use for logging
            error: The exception that occurred
            context: Additional context information
            include_stack_trace: Whether to include full stack trace
            log_level: Logging level to use
        """
        # Build error message
        error_message = f"Error: {type(error).__name__}: {str(error)}"

        # Add context if provided
        if context:
            context_str = ErrorEnhancer._format_context(context)
            error_message += f"\nContext: {context_str}"

        # Add stack trace if requested
        if include_stack_trace:
            stack_trace = ErrorEnhancer._get_enhanced_stack_trace(error)
            error_message += f"\nStack Trace:\n{stack_trace}"

        # Log the enhanced error message
        logger.log(log_level, error_message, exc_info=False)

    @staticmethod
    def log_unexpected_error(
        logger: logging.Logger,
        operation: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an unexpected error with operation context.

        Args:
            logger: Logger instance
            operation: Description of the operation that failed
            error: The exception
            context: Additional context
        """
        enhanced_context = {"operation": operation}
        if context:
            enhanced_context.update(context)

        ErrorEnhancer.log_error(
            logger=logger,
            error=error,
            context=enhanced_context,
            include_stack_trace=True,
            log_level=logging.CRITICAL
        )

    @staticmethod
    def log_validation_error(
        logger: logging.Logger,
        field: str,
        value: Any,
        expected_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a validation error with field information.

        Args:
            logger: Logger instance
            field: Field name that failed validation
            value: Invalid value
            expected_type: Expected type description
            context: Additional context
        """
        error = ValueError(f"Validation failed for field '{field}': expected {expected_type}, got {type(value).__name__}")

        enhanced_context = {
            "field": field,
            "value": str(value),
            "expected_type": expected_type
        }
        if context:
            enhanced_context.update(context)

        ErrorEnhancer.log_error(
            logger=logger,
            error=error,
            context=enhanced_context,
            include_stack_trace=False,
            log_level=logging.WARNING
        )

    @staticmethod
    def _format_context(context: Dict[str, Any]) -> str:
        """
        Format context dictionary into a readable string.

        Args:
            context: Context dictionary

        Returns:
            Formatted context string
        """
        formatted_items = []
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                formatted_items.append(f"{key}={value}")
            else:
                formatted_items.append(f"{key}={repr(value)}")
        return ", ".join(formatted_items)

    @staticmethod
    def _get_enhanced_stack_trace(error: Exception) -> str:
        """
        Get an enhanced stack trace with additional context.

        Args:
            error: The exception

        Returns:
            Formatted stack trace
        """
        # Get the full traceback
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)

        # Enhance with file and line information
        enhanced_lines = []
        for line in tb_lines:
            # Add relative path information if possible
            if "File " in line:
                try:
                    # Extract file path and make it relative if in project
                    parts = line.split('"')
                    if len(parts) >= 2:
                        file_path = parts[1]
                        path_obj = Path(file_path)
                        if path_obj.exists():
                            # Try to make relative to current working directory
                            cwd = Path.cwd()
                            try:
                                relative_path = path_obj.relative_to(cwd)
                                line = line.replace(file_path, str(relative_path))
                            except ValueError:
                                pass  # Keep absolute path if not relative
                except Exception:
                    pass  # Keep original line if enhancement fails

            enhanced_lines.append(line.rstrip())

        return "\n".join(enhanced_lines)

    @staticmethod
    def create_error_context(**kwargs) -> Dict[str, Any]:
        """
        Create a standardized error context dictionary.

        Args:
            **kwargs: Context key-value pairs

        Returns:
            Context dictionary
        """
        context = {}
        for key, value in kwargs.items():
            context[key] = value
        return context