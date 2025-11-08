from langchain.tools import BaseTool
from agents.models.gemini import create_gemini_model
from pydantic import BaseModel, Field, ConfigDict
from storyspark_logging.core.logger_factory import LoggerFactory
from storyspark_logging.core.performance_tracker import PerformanceTracker
from storyspark_logging.core.error_enhancer import ErrorEnhancer
from schemas.story_schemas import StoryOutline, ChapterContent
from typing import Any
import json
import re


class StoryOutlinerInput(BaseModel):
    theme: str = Field(description="The main theme of the story")
    age_group: str = Field(description="Target age group (e.g., 6-8)")
    genre: str = Field(description="Story genre")
    num_chapters: int = Field(description="Number of chapters")


class StoryOutlinerTool(BaseTool):
    name: str = "StoryOutlinerTool"
    description: str = "Creates a complete story outline with chapter summaries and character descriptions"
    args_schema: type[BaseModel] = StoryOutlinerInput
    logger: Any = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory().get_logger("storyspark.tools.story.outliner")
        self.logger.info("StoryOutlinerTool initialized")

    def _run(self, theme: str, age_group: str, genre: str, num_chapters: int) -> str:
        self.logger.info("StoryOutlinerTool._run() invoked", extra={
            "theme": theme,
            "age_group": age_group,
            "genre": genre,
            "num_chapters": num_chapters
        })

        try:
            with PerformanceTracker("story_outline_generation", self.logger) as tracker:
                # Create LLM model with structured output
                self.logger.debug("Creating Gemini model with structured output for story outlining")
                llm = create_gemini_model(schema=StoryOutline)  # FIXED: Added schema parameter
                self.logger.debug("Structured Gemini model created successfully")

                # Construct prompt
                prompt = f'''Create a story outline for a {genre} novel aimed at {age_group} year olds.
Theme: {theme}
Number of chapters: {num_chapters}

Provide:
1. Story title
2. Main characters (3-5) with detailed descriptions
3. Chapter-by-chapter outline with titles and summaries'''

                prompt_length = len(prompt)
                self.logger.debug("Prompt constructed", extra={
                    "prompt_length": prompt_length,
                    "theme": theme,
                    "genre": genre,
                    "age_group": age_group,
                    "num_chapters": num_chapters
                })

                # Make API call with structured output and retry logic
                self.logger.info("Invoking LLM for structured story outline generation")
                max_retries = 3
                structured_outline = None

                for attempt in range(max_retries):
                    try:
                        self.logger.debug(f"Structured outline generation attempt {attempt + 1}")
                        structured_outline = llm.invoke(prompt)
                        if structured_outline is not None:
                            break  # Success, exit retry loop
                        else:
                            self.logger.warning(f"LLM returned None on attempt {attempt + 1}")
                    except Exception as e:
                        self.logger.warning(f"Structured outline generation attempt {attempt + 1} failed", extra={
                            "error": str(e),
                            "attempt": attempt + 1,
                            "max_retries": max_retries
                        })
                        if attempt == max_retries - 1:
                            raise e  # Re-raise on last attempt
                        # Wait a bit before retry (simple backoff)
                        import time
                        time.sleep(2 * (attempt + 1))

                # Validate structured output response
                if structured_outline is None:
                    raise ValueError("LLM returned None response")
                
                if not isinstance(structured_outline, StoryOutline):
                    raise ValueError(f"Expected StoryOutline object, got {type(structured_outline).__name__}")
                
                self.logger.debug("Structured outline received directly from model", extra={
                    "outline_type": type(structured_outline).__name__,
                    "has_title": bool(structured_outline.title),
                    "character_count": len(structured_outline.characters),
                    "chapter_count": len(structured_outline.chapters)
                })

                self.logger.debug("Structured outline parsed successfully", extra={
                    "outline_type": type(structured_outline).__name__,
                    "has_title": bool(structured_outline.title),
                    "character_count": len(structured_outline.characters),
                    "chapter_count": len(structured_outline.chapters)
                })

                # Validate structured output
                if not structured_outline.title or not structured_outline.characters or not structured_outline.chapters:
                    raise ValueError("Incomplete structured outline received from LLM")

                # Convert to JSON string for backward compatibility
                outline_json = json.dumps(structured_outline.model_dump(), indent=2)

                self.logger.info("Story outline generation completed successfully", extra={
                    "title": structured_outline.title,
                    "character_count": len(structured_outline.characters),
                    "chapter_count": len(structured_outline.chapters),
                    "json_length": len(outline_json)
                })

                tracker.add_metadata("character_count", len(structured_outline.characters))
                tracker.add_metadata("chapter_count", len(structured_outline.chapters))
                tracker.add_metadata("prompt_length", prompt_length)
                tracker.add_metadata("json_length", len(outline_json))

                return outline_json

        except Exception as e:
            ErrorEnhancer.log_unexpected_error(
                self.logger,
                "story outline generation",
                e,
                context={
                    "theme": theme,
                    "age_group": age_group,
                    "genre": genre,
                    "num_chapters": num_chapters,
                    "operation": "structured_output"
                }
            )
            raise



class ChapterWriterInput(BaseModel):
    chapter_number: int = Field(description="Chapter number to write")
    chapter_outline: str = Field(description="Outline for this chapter")
    previous_chapters: str = Field(
        description="Content of previous chapters for consistency")
    characters: str = Field(description="Character information")
    age_group: str = Field(description="Target age group")


class ChapterWriterTool(BaseTool):
    name: str = "ChapterWriterTool"
    description: str = "Writes a complete chapter based on the outline and previous content"
    args_schema: type[BaseModel] = ChapterWriterInput
    logger: Any = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory().get_logger("storyspark.tools.story.writer")
        self.logger.info("ChapterWriterTool initialized")

    def _run(self, chapter_number: int, chapter_outline: str,
              previous_chapters: str, characters: str, age_group: str) -> str:
        self.logger.info("ChapterWriterTool._run() invoked", extra={
            "chapter_number": chapter_number,
            "age_group": age_group,
            "chapter_outline_length": len(chapter_outline),
            "characters_length": len(characters),
            "previous_chapters_length": len(previous_chapters)
        })

        try:
            with PerformanceTracker("chapter_writing", self.logger) as tracker:
                # Create LLM model with structured output
                self.logger.debug("Creating Gemini model with structured output for chapter writing")
                llm = create_gemini_model(schema=ChapterContent)
                self.logger.debug("Structured Gemini model created successfully")

                # Construct prompt
                prompt = f'''Write Chapter {chapter_number} for a story aimed at {age_group} year olds.

Chapter Outline: {chapter_outline}

Characters: {characters}

Previous Chapters Summary: {previous_chapters}

Requirements:
- Age-appropriate language and themes
- 500-800 words
- Consistent with previous chapters
- Include descriptive scenes for potential illustrations
- Mark scenes that would benefit from illustrations with [ILLUSTRATION: brief description]

Provide the chapter title and complete chapter content.'''

                prompt_length = len(prompt)
                self.logger.debug("Prompt constructed for chapter writing", extra={
                    "chapter_number": chapter_number,
                    "prompt_length": prompt_length,
                    "chapter_outline_length": len(chapter_outline),
                    "characters_length": len(characters),
                    "previous_chapters_length": len(previous_chapters)
                })

                # Make API call with structured output and retry logic
                self.logger.info("Invoking LLM for structured chapter generation")
                max_retries = 3
                chapter_data = None

                for attempt in range(max_retries):
                    try:
                        chapter_data = llm.invoke(prompt)
                        break  # Success, exit retry loop
                    except Exception as e:
                        self.logger.warning(f"Chapter generation attempt {attempt + 1} failed", extra={
                            "error": str(e),
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "chapter_number": chapter_number
                        })
                        if attempt == max_retries - 1:
                            raise e  # Re-raise on last attempt
                        # Wait a bit before retry (simple backoff)
                        import time
                        time.sleep(1 * (attempt + 1))

                if chapter_data is None or not isinstance(chapter_data, ChapterContent):
                    raise ValueError(f"Structured chapter generation failed - Expected ChapterContent, got {type(chapter_data).__name__ if chapter_data else 'None'}")

                self.logger.debug("Structured chapter data received", extra={
                    "chapter_number": chapter_data.chapter_number,
                    "title": chapter_data.title,
                    "content_length": len(chapter_data.content),
                    "word_count": chapter_data.word_count,
                    "illustration_count": len(chapter_data.illustrations)
                })

                # Validate structured output
                if not chapter_data.content:
                    raise ValueError("Empty chapter content received from LLM")

                # Extract illustrations from content if not provided in structured output
                if not chapter_data.illustrations:
                    illustration_pattern = r'\[ILLUSTRATION:\s*(.*?)\]'
                    matches = re.findall(illustration_pattern, chapter_data.content, re.IGNORECASE)
                    chapter_data.illustrations = matches

                # Update word count if not accurate
                actual_word_count = len(chapter_data.content.split())
                if abs(chapter_data.word_count - actual_word_count) > 50:  # Allow some tolerance
                    chapter_data.word_count = actual_word_count

                self.logger.info("Chapter content generated and validated", extra={
                    "chapter_number": chapter_data.chapter_number,
                    "title": chapter_data.title,
                    "word_count": chapter_data.word_count,
                    "illustration_count": len(chapter_data.illustrations),
                    "content_length": len(chapter_data.content)
                })

                tracker.add_metadata("chapter_number", chapter_number)
                tracker.add_metadata("word_count", chapter_data.word_count)
                tracker.add_metadata("illustration_count", len(chapter_data.illustrations))
                tracker.add_metadata("content_length", len(chapter_data.content))
                tracker.add_metadata("prompt_length", prompt_length)

                self.logger.info("Chapter writing completed successfully", extra={
                    "chapter_number": chapter_number
                })

                # Return the chapter content for backward compatibility
                return chapter_data.content

        except Exception as e:
            ErrorEnhancer.log_unexpected_error(
                self.logger,
                "chapter writing",
                e,
                context={
                    "chapter_number": chapter_number,
                    "age_group": age_group,
                    "chapter_outline_preview": chapter_outline[:100] + "..." if len(chapter_outline) > 100 else chapter_outline,
                    "operation": "structured_output"
                }
            )
            raise
