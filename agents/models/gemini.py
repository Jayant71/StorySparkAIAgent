import os
from langchain_google_genai import ChatGoogleGenerativeAI
from storyspark_logging.core.logger_factory import LoggerFactory
from storyspark_logging.core.performance_tracker import PerformanceTracker
from storyspark_logging.core.error_enhancer import ErrorEnhancer
from pydantic import BaseModel
from typing import Optional, Union, Any


def create_gemini_model(model_name: str = "gemini-flash-latest", temperature: float = 0.7, schema: Optional[type[BaseModel]] = None) -> Union[ChatGoogleGenerativeAI, Any]:
    """
    Creates and returns a ChatGoogleGenerativeAI model instance, optionally configured for structured output.

    Args:
        model_name: The name of the Gemini model to use (default: "gemini-flash-latest")
        temperature: Controls randomness in the output (default: 0.7)
        schema: Optional Pydantic model class for structured output

    Returns:
        ChatGoogleGenerativeAI or structured output chain: Configured Gemini model instance

    Raises:
        ValueError: If GOOGLE_API_KEY environment variable is not set
    """
    # Initialize logging components
    logger = LoggerFactory().get_logger("storyspark.models.gemini")

    # Log function entry with parameters
    schema_name = schema.__name__ if schema else None
    logger.debug(f"create_gemini_model called with model_name='{model_name}', temperature={temperature}, schema={schema_name}")

    with PerformanceTracker("gemini_model_creation", logger) as tracker:
        # API key validation
        tracker.add_metadata("operation", "api_key_validation")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable is not set")
            ErrorEnhancer.log_error(
                logger=logger,
                error=ValueError("GOOGLE_API_KEY environment variable must be set"),
                context={
                    "model_name": model_name,
                    "temperature": temperature,
                    "schema": schema_name,
                    "api_key_present": False
                }
            )
            raise ValueError("GOOGLE_API_KEY environment variable must be set")

        logger.info("API key validation successful")

        # Model creation process
        tracker.add_metadata("operation", "model_initialization")
        logger.debug(f"Initializing ChatGoogleGenerativeAI with model='{model_name}', temperature={temperature}")

        try:
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )

            logger.info(f"Gemini model '{model_name}' created successfully")
            logger.debug(f"Model configuration: temperature={temperature}, max_retries=2")

            tracker.add_metadata("model_name", model_name)
            tracker.add_metadata("temperature", temperature)
            tracker.add_metadata("schema", schema_name)

            # Configure for structured output if schema provided
            if schema:
                logger.debug(f"Configuring model for structured output with schema '{schema_name}'")
                structured_model = model.with_structured_output(schema)
                logger.info(f"Structured output configured for schema '{schema_name}'")
                return structured_model
            else:
                return model

        except Exception as e:
            logger.error(f"Failed to create Gemini model '{model_name}': {str(e)}")
            ErrorEnhancer.log_error(
                logger=logger,
                error=e,
                context={
                    "model_name": model_name,
                    "temperature": temperature,
                    "schema": schema_name,
                    "operation": "model_creation"
                }
            )
            raise


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    gemini_model = create_gemini_model()
    response = gemini_model.invoke("Hello, how are you?")
    print(response.content)
