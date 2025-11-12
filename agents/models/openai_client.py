import os
from langchain_openai import ChatOpenAI
from storyspark_logging.core.logger_factory import LoggerFactory
from storyspark_logging.core.performance_tracker import PerformanceTracker
from storyspark_logging.core.error_enhancer import ErrorEnhancer
from pydantic import BaseModel, SecretStr
from typing import Optional, Union, Any


def create_openai_model(model_name: str = "zai-org/GLM-4.6", temperature: float = 0.7, schema: Optional[type[BaseModel]] = None) -> Union[ChatOpenAI, Any]:
    """
    Creates and returns a ChatOpenAI model instance, optionally configured for structured output.

    Args:
        model_name: The name of the openai model to use (default: "zai-org/GLM-4.6")
        temperature: Controls randomness in the output (default: 0.7)
        schema: Optional Pydantic model class for structured output

    Returns:
        ChatOpenAI or structured output chain: Configured openai model instance

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    # Initialize logging components
    logger = LoggerFactory().get_logger("storyspark.models.openai")

    # Log function entry with parameters
    schema_name = schema.__name__ if schema else None
    logger.debug(
        f"create_openai_model called with model_name='{model_name}', temperature={temperature}, schema={schema_name}")

    with PerformanceTracker("openai_model_creation", logger) as tracker:
        # API key validation
        tracker.add_metadata("operation", "api_key_validation")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE_URL",
                             "https://api.openai.com/v1")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            ErrorEnhancer.log_error(
                logger=logger,
                error=ValueError(
                    "OPENAI_API_KEY environment variable must be set"),
                context={
                    "model_name": model_name,
                    "temperature": temperature,
                    "schema": schema_name,
                    "api_key_present": False
                }
            )
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        logger.info("API key validation successful")

        # Model creation process
        tracker.add_metadata("operation", "model_initialization")
        logger.debug(
            f"Initializing ChatGoogleGenerativeAI with model='{model_name}', temperature={temperature}")

        try:
            model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,  # type: ignore
                base_url=base_url,
                max_retries=2
            )

            logger.info(f"openai model '{model_name}' created successfully")
            logger.debug(
                f"Model configuration: temperature={temperature}, max_retries=2")

            tracker.add_metadata("model_name", model_name)
            tracker.add_metadata("temperature", temperature)
            tracker.add_metadata("schema", schema_name)

            # Configure for structured output if schema provided
            if schema:
                logger.debug(
                    f"Configuring model for structured output with schema '{schema_name}'")
                structured_model = model.with_structured_output(schema)
                logger.info(
                    f"Structured output configured for schema '{schema_name}'")
                return structured_model
            else:
                return model

        except Exception as e:
            logger.error(
                f"Failed to create openai model '{model_name}': {str(e)}")
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
    from schemas.story_schemas import StoryOutline, ChapterContent
    dotenv.load_dotenv()

    # # Test with StoryOutline schema
    # print("Testing OpenAI model with StoryOutline schema...")
    # story_model = create_openai_model(
    #     model_name="gpt-4.1-2025-04-14",
    #     temperature=0.7,
    #     schema=StoryOutline
    # )

    # story_prompt = """Create a story outline for a children's adventure novel aimed at 6-8 year olds.
    # Theme: Friendship and courage
    # Number of chapters: 3

    # Provide:
    # 1. Story title
    # 2. Main characters (2-3) with detailed descriptions
    # 3. Chapter-by-chapter outline with titles and summaries"""

    # try:
    #     story_response = story_model.invoke(story_prompt)
    #     print("\nStory Outline Response:")
    #     print(f"Type of response: {type(story_response)}")
    #     print(story_response.model_dump_json(indent=2))

    # except Exception as e:
    #     print(f"Error testing StoryOutline schema: {e}")

    # Test with ChapterContent schema
    print("\n\nTesting OpenAI model with ChapterContent schema...")
    chapter_model = create_openai_model(
        model_name="gpt-4.1-2025-04-14",
        temperature=0.7,
        schema=ChapterContent
    )

    chapter_prompt = """Write Chapter 1 for a children's story aimed at 6-8 year olds.

    Chapter Outline: A young squirrel named Squeaky discovers a magical nut that can talk.

    Characters: Squeaky - a curious young squirrel, Nutmeg - a wise magical nut

    Previous Chapters: None

    Requirements:
    - Age-appropriate language and themes
    - 500-800 words
    - Include descriptive scenes for potential illustrations
    - IDENTIFY EXACTLY 2-3 STRATEGIC ILLUSTRATION OPPORTUNITIES by marking them with [ILLUSTRATION: brief description] tags

    Provide the chapter title and complete chapter content."""

    try:
        chapter_response = chapter_model.invoke(chapter_prompt)
        print("\nChapter Content Response:")
        print(f"Type of response: {type(chapter_response)}")
        print(chapter_response.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error testing ChapterContent schema: {e}")
