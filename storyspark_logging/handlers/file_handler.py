"""
File Handler Module

Provides rotating file handler for log files with size and backup management.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class FileHandler(logging.handlers.RotatingFileHandler):
    """
    Enhanced rotating file handler with better file management.

    Extends RotatingFileHandler with additional features like
    automatic directory creation and better error handling.
    """

    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        maxBytes: int = 0,
        backupCount: int = 0,
        encoding: Optional[str] = None,
        delay: bool = False
    ):
        """
        Initialize the file handler.

        Args:
            filename: Log file path
            mode: File mode ('a' for append, 'w' for write)
            maxBytes: Maximum file size before rotation (0 = no rotation)
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Whether to delay file opening
        """
        # Ensure the directory exists
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename=str(log_path),
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay
        )

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record with enhanced error handling.

        Args:
            record: Log record to emit
        """
        try:
            super().emit(record)
        except Exception:
            # If writing to file fails, try to write to a fallback location
            self._handle_emit_error(record)

    def _handle_emit_error(self, record: logging.LogRecord) -> None:
        """
        Handle emission errors by attempting fallback logging.

        Args:
            record: Log record that failed to emit
        """
        try:
            # Try to write to a fallback file
            fallback_path = Path(self.baseFilename).parent / "logging_error.log"
            with open(fallback_path, 'a', encoding='utf-8') as f:
                f.write(f"Failed to write log: {record.getMessage()}\n")
        except Exception:
            # Last resort: do nothing to avoid infinite loops
            pass

    def doRollover(self) -> None:
        """
        Perform log rotation with enhanced error handling.
        """
        try:
            super().doRollover()
        except Exception as e:
            # Log rotation failed, try to continue with current file
            error_msg = f"Log rotation failed: {e}"
            try:
                with open(self.baseFilename, 'a', encoding='utf-8') as f:
                    f.write(f"WARNING: {error_msg}\n")
            except Exception:
                pass  # Silent failure

    @property
    def current_file_size(self) -> int:
        """
        Get the current size of the log file.

        Returns:
            File size in bytes
        """
        try:
            return Path(self.baseFilename).stat().st_size
        except (OSError, FileNotFoundError):
            return 0

    @property
    def backup_files(self) -> list:
        """
        Get list of backup files.

        Returns:
            List of backup file paths
        """
        base_path = Path(self.baseFilename)
        backups = []

        for i in range(1, self.backupCount + 1):
            backup_path = base_path.parent / f"{base_path.name}.{i}"
            if backup_path.exists():
                backups.append(str(backup_path))

        return backups

    def cleanup_old_backups(self, max_backups: Optional[int] = None) -> None:
        """
        Clean up old backup files beyond the specified limit.

        Args:
            max_backups: Maximum number of backups to keep (defaults to backupCount)
        """
        if max_backups is None:
            max_backups = self.backupCount

        base_path = Path(self.baseFilename)

        # Get all backup files
        all_backups = []
        for i in range(1, 100):  # Arbitrary upper limit
            backup_path = base_path.parent / f"{base_path.name}.{i}"
            if backup_path.exists():
                all_backups.append((backup_path, i))
            else:
                break

        # Remove excess backups
        if len(all_backups) > max_backups:
            to_remove = all_backups[max_backups:]
            for backup_path, _ in to_remove:
                try:
                    backup_path.unlink()
                except Exception:
                    pass  # Silent failure