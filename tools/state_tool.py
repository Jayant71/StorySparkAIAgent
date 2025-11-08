from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from storyspark_logging.core.logger_factory import LoggerFactory
from storyspark_logging.core.performance_tracker import PerformanceTracker
from storyspark_logging.core.error_enhancer import ErrorEnhancer
from config.config import Config
from typing import Any, Optional
import json
import os
from datetime import datetime


class StateSaverInput(BaseModel):
    state_data: str = Field(description="JSON string of state data to save")
    state_file_path: str = Field(description="Path to save the state file")


class StateSaverTool(BaseTool):
    name: str = "StateSaverTool"
    description: str = "Saves novel generation state to JSON file for memory efficiency"
    args_schema: type[BaseModel] = StateSaverInput
    logger: Any = Field(default=None, exclude=True)

    def __init__(self, config: Config):
        super().__init__()
        self.__dict__['_config'] = config
        self.logger = LoggerFactory().get_logger("storyspark.tools.state")
        self.logger.info("StateSaverTool initialized")

    def _run(self, state_data: str, state_file_path: str) -> str:
        self.logger.info("StateSaverTool._run() invoked", extra={
            "state_file_path": state_file_path,
            "state_data_length": len(state_data)
        })

        try:
            with PerformanceTracker("state_saving", self.logger) as tracker:
                # Parse and validate state data
                self.logger.debug("Parsing state data JSON")
                try:
                    state = json.loads(state_data)
                except json.JSONDecodeError as e:
                    # Try to handle double-encoded JSON (common LLM issue)
                    self.logger.debug("Initial JSON parsing failed, trying to handle escaped JSON", extra={
                        "error": str(e),
                        "state_data_preview": state_data[:200] + "..." if len(state_data) > 200 else state_data
                    })
                    try:
                        # If it's a string that contains escaped JSON, unescape it
                        if state_data.startswith('{"') and state_data.endswith('"}'):
                            # It might be a JSON string wrapped in another JSON string
                            inner_json = state_data[1:-1]  # Remove outer quotes
                            # Unescape the inner content
                            import codecs
                            unescaped = codecs.decode(inner_json, 'unicode_escape')
                            state = json.loads(unescaped)
                            self.logger.debug("Successfully parsed double-encoded JSON")
                        else:
                            raise e
                    except Exception as inner_e:
                        ErrorEnhancer.log_error(
                            self.logger,
                            e,
                            context={
                                "operation": "json_parsing",
                                "state_data_length": len(state_data),
                                "state_data_preview": state_data[:200] + "..." if len(state_data) > 200 else state_data,
                                "inner_error": str(inner_e)
                            }
                        )
                        raise e

                parsed_state_size = len(json.dumps(state))
                self.logger.debug("State data parsed successfully", extra={
                    "parsed_state_size_bytes": parsed_state_size,
                    "has_title": "title" in state,
                    "chapter_count": len(state.get("chapters", []))
                })

                # Add timestamp
                timestamp = datetime.now().isoformat()
                state['last_updated'] = timestamp
                self.logger.debug("Timestamp added to state", extra={
                    "timestamp": timestamp
                })

                # Prepare file operations
                directory = os.path.dirname(state_file_path)
                self.logger.debug("Extracted directory from path", extra={
                    "state_file_path": state_file_path,
                    "directory": directory,
                    "directory_is_empty": directory == "",
                    "is_absolute_path": os.path.isabs(state_file_path),
                    "output_directory": self.__dict__.get('_config').output_dir if self.__dict__.get('_config') else None
                })

                # Handle case where no directory is specified in the path
                if not directory:
                    # Use timestamped output directory if no directory is specified
                    config = self.__dict__.get('_config')
                    if config is None:
                        raise ValueError("Config is required for StateSaverTool")
                    directory = config.output_dir
                    self.logger.info("No directory specified in state_file_path, using timestamped output directory", extra={
                        "state_file_path": state_file_path,
                        "default_directory": directory
                    })
                    # Update the file path to include the directory
                    state_file_path = os.path.join(directory, os.path.basename(state_file_path))
                    self.logger.debug("Updated state_file_path with directory", extra={
                        "new_state_file_path": state_file_path
                    })

                try:
                    os.makedirs(directory, exist_ok=True)
                    self.logger.debug("Directory ensured successfully", extra={
                        "directory": directory,
                        "directory_exists": os.path.exists(directory)
                    })
                except Exception as e:
                    ErrorEnhancer.log_error(
                        self.logger,
                        e,
                        context={
                            "operation": "directory_creation",
                            "directory": directory,
                            "state_file_path": state_file_path
                        }
                    )
                    raise

                # Save to file
                self.logger.debug("Writing state to file", extra={
                    "state_file_path": state_file_path
                })

                try:
                    with open(state_file_path, 'w') as f:
                        json.dump(state, f, indent=2)
                except Exception as e:
                    ErrorEnhancer.log_error(
                        self.logger,
                        e,
                        context={
                            "operation": "file_writing",
                            "file_path": state_file_path,
                            "file_permissions": oct(os.stat(directory).st_mode) if os.path.exists(directory) else "unknown"
                        }
                    )
                    raise

                # Get final file size
                final_file_size = os.path.getsize(state_file_path)
                tracker.add_metadata("state_data_length", len(state_data))
                tracker.add_metadata("parsed_state_size_bytes", parsed_state_size)
                tracker.add_metadata("final_file_size_bytes", final_file_size)

                self.logger.info("State saving completed successfully", extra={
                    "state_file_path": state_file_path,
                    "final_file_size_bytes": final_file_size,
                    "timestamp": timestamp
                })

                return f"State saved successfully to: {state_file_path}"

        except Exception as e:
            ErrorEnhancer.log_unexpected_error(
                self.logger,
                "state saving",
                e,
                context={
                    "state_file_path": state_file_path,
                    "state_data_length": len(state_data)
                }
            )
            raise
