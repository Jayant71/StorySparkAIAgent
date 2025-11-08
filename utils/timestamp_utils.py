"""
Timestamp Utilities Module

Provides utility functions for generating timestamps for various purposes,
including logging and file naming.
"""

import datetime
from typing import Optional


def generate_timestamp() -> str:
    """
    Generate a timestamp string in YYYYMMDD_HHMMSS format.

    This function is used for creating unique filenames with timestamps.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS

    Raises:
        RuntimeError: If timestamp generation fails
    """
    try:
        now = datetime.datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")
    except Exception as e:
        # Fallback in case of datetime errors
        raise RuntimeError(f"Failed to generate timestamp: {e}") from e


def generate_log_filename(base_name: str, timestamp: Optional[str] = None) -> str:
    """
    Generate a timestamped log filename.

    Args:
        base_name: Base name for the log file (e.g., 'storyspark')
        timestamp: Optional timestamp string. If None, generates current timestamp.

    Returns:
        Full filename with timestamp (e.g., 'logs/storyspark_20251108_123045.log')

    Raises:
        RuntimeError: If filename generation fails
    """
    try:
        if timestamp is None:
            timestamp = generate_timestamp()
        return f"logs/{base_name}_{timestamp}.log"
    except Exception as e:
        raise RuntimeError(f"Failed to generate log filename: {e}") from e


def generate_output_dir_path(timestamp: Optional[str] = None) -> str:
    """
    Generate a timestamped output directory path.

    Args:
        timestamp: Optional timestamp string. If None, generates current timestamp.

    Returns:
        Directory path in format 'output/YYYYMMDD_HHMMSS/'

    Raises:
        RuntimeError: If directory path generation fails
    """
    try:
        if timestamp is None:
            timestamp = generate_timestamp()
        return f"output/{timestamp}/"
    except Exception as e:
        raise RuntimeError(f"Failed to generate output directory path: {e}") from e