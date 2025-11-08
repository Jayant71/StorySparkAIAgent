import os
from langchain_ollama import ChatOllama
from storyspark_logging.core.logger_factory import LoggerFactory
from storyspark_logging.core.performance_tracker import PerformanceTracker
from storyspark_logging.core.error_enhancer import ErrorEnhancer


def create_ollama_model(model_name: str = "ollama-model", temperature: float = 0.7) -> ChatOllama:
    """
    Creates and returns a ChatOllama model instance.

    Args:
        model_name: The name of the Ollama model to use (default: "ollama-model")
        temperature: Controls randomness in the output (default: 0.7)
    Returns:

        ChatOllama: Configured Ollama model instance
    """
    # Initialize logging components
    logger = LoggerFactory().get_logger("storyspark.models.ollama")

    # Log function entry with parameters
    logger.debug(f"create_ollama_model called with model_name='{model_name}', temperature={temperature}")

    with PerformanceTracker("ollama_model_creation", logger) as tracker:
        # Base URL configuration
        base_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        logger.debug(f"Using base URL: {base_url}")

        # Model name validation
        if not model_name or model_name == "ollama-model":
            logger.warning(f"Using default model name '{model_name}' - consider specifying a specific model")

        tracker.add_metadata("operation", "model_initialization")
        logger.debug(f"Initializing ChatOllama with base_url='{base_url}', model='{model_name}', temperature={temperature}")

        try:
            model = ChatOllama(
                base_url=base_url,
                model=model_name,
                temperature=temperature,
            )

            logger.info(f"Ollama model '{model_name}' created successfully at {base_url}")
            logger.debug(f"Model configuration: temperature={temperature}")

            tracker.add_metadata("model_name", model_name)
            tracker.add_metadata("base_url", base_url)
            tracker.add_metadata("temperature", temperature)

            return model

        except Exception as e:
            logger.error(f"Failed to create Ollama model '{model_name}' at {base_url}: {str(e)}")
            ErrorEnhancer.log_error(
                logger=logger,
                error=e,
                context={
                    "model_name": model_name,
                    "base_url": base_url,
                    "temperature": temperature,
                    "operation": "model_creation"
                }
            )
            raise


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    ollama_model = create_ollama_model(model_name="gemma3:4b", temperature=0.7)
    response = ollama_model.invoke("Hello, how are you?")
    print(response.content)
