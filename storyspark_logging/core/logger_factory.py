"""
Logger Factory Module

Provides a singleton LoggerFactory for creating and managing loggers
with verbose control and component-specific overrides.
"""

import logging
import logging.config
import json
import threading
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any

# Ensure the project root is in sys.path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.timestamp_utils import generate_log_filename
except ImportError:
    # Fallback if utils not available
    def generate_log_filename(base_name: str) -> str:
        return f"logs/{base_name}.log"


class LoggerFactory:
    """
    Singleton factory for creating and managing loggers with verbose control.

    This class provides centralized logger management with support for:
    - Global verbose levels (minimal, normal, verbose, debug)
    - Component-specific overrides
    - Thread-safe operations
    - Automatic configuration loading
    """

    _instance: Optional['LoggerFactory'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'LoggerFactory':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._loggers: Dict[str, logging.Logger] = {}
        self._verbose_level = "normal"
        self._component_overrides: Dict[str, str] = {}
        self._config_loaded = False

        # Load configuration
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load logging configuration from config files."""
        try:
            config_dir = Path(__file__).parent.parent / "config"

            # Load log levels
            levels_file = config_dir / "log_levels.json"
            if levels_file.exists():
                with open(levels_file, 'r') as f:
                    levels_config = json.load(f)
                    self._verbose_level = levels_config.get("default_level", "normal")
                    self._component_overrides = levels_config.get("component_overrides", {})

            # Set up basic logging configuration
            self._setup_basic_logging()
            self._config_loaded = True

        except Exception as e:
            # Fallback to basic configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            print(f"Warning: Failed to load logging configuration: {e}")

    def _setup_basic_logging(self) -> None:
        """Set up basic logging configuration with our custom components."""
        # Import our custom components
        from ..formatters.json_formatter import JSONFormatter
        from ..formatters.console_formatter import ConsoleFormatter
        from ..formatters.verbose_formatter import VerboseFormatter
        from ..handlers.file_handler import FileHandler
        from ..handlers.console_handler import ConsoleHandler

        # Create formatters
        json_formatter = JSONFormatter()
        console_formatter = ConsoleFormatter()
        verbose_formatter = VerboseFormatter()

        # Create handlers
        console_handler = ConsoleHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.DEBUG)

        try:
            storyspark_filename = generate_log_filename("storyspark")
            verbose_filename = generate_log_filename("storyspark_verbose")
        except Exception as e:
            # Fallback to non-timestamped filenames if timestamp generation fails
            print(f"Warning: Failed to generate timestamped log filenames: {e}. Using default names.")
            storyspark_filename = "logs/storyspark.log"
            verbose_filename = "logs/storyspark_verbose.log"

        file_handler = FileHandler(storyspark_filename, maxBytes=10485760, backupCount=5)
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(logging.DEBUG)

        verbose_file_handler = FileHandler(verbose_filename, maxBytes=5242880, backupCount=3)
        verbose_file_handler.setFormatter(verbose_formatter)
        verbose_file_handler.setLevel(logging.DEBUG)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        root_logger.addHandler(console_handler)

        # Configure storyspark logger
        storyspark_logger = logging.getLogger("storyspark")
        storyspark_logger.setLevel(logging.DEBUG)
        storyspark_logger.addHandler(console_handler)
        storyspark_logger.addHandler(file_handler)

        # Configure other loggers
        agent_logger = logging.getLogger("storyspark.agent")
        agent_logger.setLevel(logging.INFO)

        tool_logger = logging.getLogger("storyspark.tool")
        tool_logger.setLevel(logging.WARNING)

        model_logger = logging.getLogger("storyspark.model")
        model_logger.setLevel(logging.DEBUG)
        model_logger.addHandler(verbose_file_handler)

        perf_logger = logging.getLogger("storyspark.performance")
        perf_logger.setLevel(logging.INFO)
        perf_logger.addHandler(file_handler)

    def set_verbose_level(self, level: str) -> None:
        """
        Set the global verbose level.

        Args:
            level: Verbose level ('minimal', 'normal', 'verbose', 'debug')
        """
        valid_levels = ['minimal', 'normal', 'verbose', 'debug']
        if level not in valid_levels:
            raise ValueError(f"Invalid verbose level: {level}. Must be one of {valid_levels}")

        self._verbose_level = level
        # Update existing loggers
        for logger_name in self._loggers:
            self._apply_verbose_level(self._loggers[logger_name], logger_name)

    def get_verbose_level(self) -> str:
        """Get the current global verbose level."""
        return self._verbose_level

    def set_component_override(self, component: str, level: str) -> None:
        """
        Set verbose level override for a specific component.

        Args:
            component: Component name (e.g., 'storyspark.agent')
            level: Verbose level for this component
        """
        self._component_overrides[component] = level
        # Update logger if it exists
        if component in self._loggers:
            self._apply_verbose_level(self._loggers[component], component)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name.

        Args:
            name: Logger name (typically module or component name)

        Returns:
            Configured logger instance
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
            self._apply_verbose_level(logger, name)

        return self._loggers[name]

    def _apply_verbose_level(self, logger: logging.Logger, name: str) -> None:
        """Apply the appropriate verbose level to a logger."""
        # Determine level for this component
        level = self._component_overrides.get(name, self._verbose_level)

        # Map verbose level to logging level
        level_mapping = {
            'minimal': logging.ERROR,
            'normal': logging.INFO,
            'verbose': logging.DEBUG,
            'debug': logging.DEBUG
        }

        logger.setLevel(level_mapping.get(level, logging.INFO))

    def get_all_loggers(self) -> Dict[str, logging.Logger]:
        """Get all managed loggers."""
        return self._loggers.copy()

    def shutdown(self) -> None:
        """Shutdown all loggers and clean up resources."""
        for logger in self._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        self._loggers.clear()
        logging.shutdown()