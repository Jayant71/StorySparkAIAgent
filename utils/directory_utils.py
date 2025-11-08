"""
Directory Management Utilities Module

Provides utility functions for creating and managing directories,
particularly for timestamped output directories.
"""

import os
from typing import Optional
from .timestamp_utils import generate_output_dir_path


def create_directory(dir_path: str) -> bool:
    """
    Create a directory if it doesn't exist.

    Args:
        dir_path: Path to the directory to create

    Returns:
        True if directory was created or already exists, False on failure

    Raises:
        OSError: If directory creation fails due to permission or other OS errors
    """
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        return True
    except OSError as e:
        # Re-raise with more context
        raise OSError(f"Failed to create directory '{dir_path}': {e}") from e


def ensure_output_directory(timestamp: Optional[str] = None) -> str:
    """
    Ensure the timestamped output directory exists and return its path.

    Args:
        timestamp: Optional timestamp string. If None, generates current timestamp.

    Returns:
        Path to the ensured output directory (e.g., 'output/20251108_123045/')

    Raises:
        OSError: If directory creation fails
    """
    dir_path = generate_output_dir_path(timestamp)
    create_directory(dir_path)
    return dir_path


# Global variable to store the current run's output directory
_current_output_dir: Optional[str] = None


def get_current_output_dir() -> str:
    """
    Get the current run's output directory path.

    Returns:
        Path to the current output directory

    Raises:
        RuntimeError: If no output directory has been set for this run
    """
    if _current_output_dir is None:
        raise RuntimeError("Output directory has not been initialized for this run")
    return _current_output_dir


def set_current_output_dir(timestamp: Optional[str] = None) -> str:
    """
    Set and create the current run's output directory.

    This should be called once at application startup.

    Args:
        timestamp: Optional timestamp string. If None, generates current timestamp.

    Returns:
        Path to the set output directory

    Raises:
        OSError: If directory creation fails
    """
    global _current_output_dir
    _current_output_dir = ensure_output_directory(timestamp)
    return _current_output_dir