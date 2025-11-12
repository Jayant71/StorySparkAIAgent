from langchain.tools import BaseTool
from agents.models.gemini import create_gemini_model
from agents.models.openai_client import create_openai_model
from pydantic import BaseModel, Field, ConfigDict
from storyspark_logging.core.logger_factory import LoggerFactory
from storyspark_logging.core.performance_tracker import PerformanceTracker
from storyspark_logging.core.error_enhancer import ErrorEnhancer
from schemas.story_schemas import StoryOutline, ChapterContent
from schemas.state_schemas import NovelState, Character, Chapter
from typing import Any, Optional, List, Dict
import json
import re
import os
from datetime import datetime
from utils.timestamp_utils import generate_output_dir_path
from schemas.state_schemas import IllustrationReference
import re


def load_novel_state(state_file_path: str) -> NovelState:
    """
    Helper function to load novel state and return as Pydantic model
    
    Args:
        state_file_path: Path to the state file
        
    Returns:
        NovelState: Loaded novel state as Pydantic model
    """
    logger = LoggerFactory().get_logger("storyspark.tools.state")
    
    logger.info("Loading novel state", extra={
        "state_file_path": state_file_path
    })
    
    try:
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
        
        # Handle different state formats and convert to expected schema
        if 'characters' in state_dict and isinstance(state_dict['characters'], dict):
            logger.debug("Converting characters from dict to list format")
            characters_list = []
            for name, details in state_dict['characters'].items():
                if isinstance(details, dict):
                    char_data = {'name': name}
                    char_data.update(details)
                    characters_list.append(char_data)
            state_dict['characters'] = characters_list
        
        # Handle chapters - might be in chapters_outline or chapters_content
        chapters_list = []

        # First check if there's chapters_content with actual chapter data
        if 'chapters_content' in state_dict and isinstance(state_dict['chapters_content'], dict):
            logger.debug("Processing chapters from chapters_content")
            for number, content in state_dict['chapters_content'].items():
                if isinstance(content, dict):
                    ch_data = {
                        'chapter_number': int(number) if number.isdigit() else number,
                        'title': content.get('title', f'Chapter {number}'),
                        'content': content.get('content'),
                        'word_count': content.get('word_count'),
                        'illustrations': content.get('illustrations', []),
                        'images': content.get('images', [])
                    }
                    chapters_list.append(ch_data)

        # If no chapters_content, try to use chapters_outline
        elif 'chapters_outline' in state_dict and isinstance(state_dict['chapters_outline'], list):
            logger.debug("Processing chapters from chapters_outline")
            for ch in state_dict['chapters_outline']:
                if isinstance(ch, dict):
                    ch_data = {
                        'chapter_number': ch.get('number', 0),
                        'title': ch.get('title', 'Untitled'),
                        'summary': ch.get('summary'),
                        'content': None,
                        'word_count': None,
                        'illustrations': [],
                        'images': []
                    }
                    chapters_list.append(ch_data)

        # Handle direct chapters array (current format)
        elif 'chapters' in state_dict and isinstance(state_dict['chapters'], list):
            logger.debug("Processing chapters from direct chapters array")
            chapters_list = state_dict['chapters']
        
        # Sort chapters by number
        chapters_list.sort(key=lambda x: x.get('chapter_number', 0))
        state_dict['chapters'] = chapters_list
        
        # Remove old format fields to avoid confusion
        state_dict.pop('chapters_outline', None)
        state_dict.pop('chapters_content', None)
        state_dict.pop('character_images', None)
        state_dict.pop('num_chapters', None)
        state_dict.pop('output_directory', None)
        
        # Validate with Pydantic schema
        novel_state = NovelState(**state_dict)
        
        logger.info("Novel state loaded successfully", extra={
            "state_file_path": state_file_path,
            "title": novel_state.title,
            "chapter_count": len(novel_state.chapters)
        })
        
        return novel_state
        
    except Exception as e:
        ErrorEnhancer.log_unexpected_error(
            logger,
            "state loading",
            e,
            context={
                "state_file_path": state_file_path
            }
        )
        raise


def extract_illustration_metadata(content: str, characters: List[str] = None) -> List[IllustrationReference]:
    """
    Extract illustration metadata from chapter content and create enhanced illustration references.

    Args:
        content: Chapter text content
        characters: List of character names for character focus detection

    Returns:
        List of IllustrationReference objects with extracted metadata
    """
    if not content:
        return []

    illustrations = []
    character_names = set(characters or [])

    # Find all illustration tags in content
    illustration_pattern = r'\[ILLUSTRATION:\s*(.*?)\]'
    matches = re.finditer(illustration_pattern, content, re.IGNORECASE)

    for i, match in enumerate(matches):
        description = match.group(1).strip()
        position = match.start()

        # Extract context around the illustration tag
        context_start = max(0, position - 200)
        context_end = min(len(content), position + len(match.group(0)) + 200)
        context_text = content[context_start:context_end].strip()

        # Determine paragraph number
        paragraph_number = content[:position].count('\n\n') + 1

        # Analyze description for metadata
        metadata = _analyze_illustration_description(description, context_text, character_names)

        # Create illustration reference
        illustration = IllustrationReference(
            id=f"illustration_{i+1}",
            description=description,
            content_position=position,
            paragraph_number=paragraph_number,
            context_text=context_text[:500] if context_text else None,  # Limit context length
            **metadata
        )

        illustrations.append(illustration)

    return illustrations


def _analyze_illustration_description(description: str, context: str, character_names: set) -> Dict[str, Any]:
    """
    Analyze illustration description and context to extract metadata.

    Args:
        description: The illustration description text
        context: Surrounding context text
        character_names: Set of character names in the story

    Returns:
        Dictionary of metadata fields
    """
    metadata = {}
    desc_lower = description.lower()
    context_lower = context.lower()

    # Detect character focus
    character_focus = []
    for char_name in character_names:
        if char_name.lower() in desc_lower or char_name.lower() in context_lower:
            character_focus.append(char_name)
    if character_focus:
        metadata['character_focus'] = character_focus

    # Detect scene type
    if any(word in desc_lower for word in ['action', 'running', 'fighting', 'chasing', 'adventure']):
        metadata['scene_type'] = 'action'
    elif any(word in desc_lower for word in ['talking', 'speaking', 'conversation', 'dialogue']):
        metadata['scene_type'] = 'dialogue'
    elif any(word in desc_lower for word in ['forest', 'woods', 'mountain', 'castle', 'setting', 'landscape']):
        metadata['scene_type'] = 'establishing'
    elif any(word in desc_lower for word in ['happy', 'sad', 'excited', 'scared', 'emotional', 'surprised']):
        metadata['scene_type'] = 'emotional'

    # Detect mood
    if any(word in desc_lower for word in ['happy', 'joyful', 'excited', 'celebrating']):
        metadata['mood'] = 'happy'
    elif any(word in desc_lower for word in ['mysterious', 'dark', 'scary', 'ominous']):
        metadata['mood'] = 'mysterious'
    elif any(word in desc_lower for word in ['adventurous', 'exciting', 'thrilling']):
        metadata['mood'] = 'adventurous'
    elif any(word in desc_lower for word in ['peaceful', 'calm', 'serene']):
        metadata['mood'] = 'peaceful'

    # Detect setting
    setting_keywords = {
        'forest': ['forest', 'woods', 'trees', 'enchanted woods'],
        'castle': ['castle', 'palace', 'tower', 'kingdom'],
        'mountain': ['mountain', 'peak', 'cliff', 'summit'],
        'village': ['village', 'town', 'home', 'house'],
        'river': ['river', 'stream', 'water', 'lake'],
        'cave': ['cave', 'underground', 'dark tunnel']
    }

    for setting, keywords in setting_keywords.items():
        if any(keyword in desc_lower for keyword in keywords):
            metadata['setting'] = setting
            break

    # Suggest style based on content
    if 'magical' in desc_lower or 'enchanted' in desc_lower:
        metadata['style'] = 'watercolor'
    elif 'cartoon' in desc_lower or 'funny' in desc_lower:
        metadata['style'] = 'cartoon'
    elif 'realistic' in desc_lower or 'detailed' in desc_lower:
        metadata['style'] = 'realistic'

    # Suggest size based on description
    if 'full page' in desc_lower or 'large' in desc_lower:
        metadata['size'] = 'large'
    elif 'small' in desc_lower or 'thumbnail' in desc_lower:
        metadata['size'] = 'small'

    return metadata


class StoryOutlinerInput(BaseModel):
    theme: str = Field(description="The main theme of the story")
    age_group: str = Field(description="Target age group (e.g., 6-8)")
    genre: str = Field(description="Story genre")
    num_chapters: int = Field(description="Number of chapters")
    output_directory: str = Field(description="Output directory path where state file should be saved")


class StoryOutlinerTool(BaseTool):
    name: str = "StoryOutlinerTool"
    description: str = "Creates a complete story outline with chapter summaries and character descriptions"
    args_schema: type[BaseModel] = StoryOutlinerInput
    logger: Any = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__(llm=None)
        self.logger = LoggerFactory().get_logger("storyspark.tools.story.outliner")
        self.logger.info("StoryOutlinerTool initialized")

    def _run(self, theme: str, age_group: str, genre: str, num_chapters: int, output_directory: str) -> str:
        self.logger.info("StoryOutlinerTool._run() invoked", extra={
            "theme": theme,
            "age_group": age_group,
            "genre": genre,
            "num_chapters": num_chapters,
            "output_directory": output_directory
        })

        try:
            with PerformanceTracker("story_outline_generation", self.logger) as tracker:
                # Create LLM model with structured output
                self.logger.debug(
                    "Creating OpenAI model with structured output for story outlining")
                # FIXED: Added schema parameter
                llm = create_openai_model(
                    model_name="gpt-4.1-2025-04-14", schema=StoryOutline)
                self.logger.debug(
                    "Structured OpenAI model created successfully")

                # Construct prompt
                prompt = f'''Create a story outline for a {genre} novel aimed at {age_group} year olds.
Theme: {theme}
Number of chapters: {num_chapters}

Provide:
1. Story title
2. Main characters (3-5) with detailed descriptions including:
   - Name
   - Age (if applicable)
   - Role in the story
   - Personality traits
   - Physical appearance
   - Background story
3. Chapter-by-chapter outline with titles and summaries'''

                prompt_length = len(prompt)
                self.logger.debug("Prompt constructed", extra={
                    "prompt_length": prompt_length,
                    "theme": theme,
                    "genre": genre,
                    "age_group": age_group,
                    "num_chapters": num_chapters,
                    "output_directory": output_directory
                })

                # Make API call with structured output and retry logic
                self.logger.info(
                    "Invoking LLM for structured story outline generation")
                max_retries = 3
                structured_outline = None

                for attempt in range(max_retries):
                    try:
                        self.logger.debug(
                            f"Structured outline generation attempt {attempt + 1}")
                        structured_outline = llm.invoke(prompt)
                        if structured_outline is not None:
                            break  # Success, exit retry loop
                        else:
                            self.logger.warning(
                                f"LLM returned None on attempt {attempt + 1}")
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
                    raise ValueError(
                        f"Expected StoryOutline object, got {type(structured_outline).__name__}")

                self.logger.debug("Structured outline received directly from model", extra={
                    "outline_type": type(structured_outline).__name__,
                    "has_title": bool(structured_outline.title),
                    "character_count": len(structured_outline.characters),
                    "chapter_count": len(structured_outline.chapters)
                })

                # Validate structured output
                if not structured_outline.title or not structured_outline.characters or not structured_outline.chapters:
                    raise ValueError(
                        "Incomplete structured outline received from LLM")

                # Convert StoryOutline to NovelState and save directly
                # Convert characters from story_schemas.Character to state_schemas.Character
                characters = [
                    Character(
                        name=char.name,
                        description=char.description,
                        age=char.age,
                        role=char.role,
                        personality=char.personality,
                        appearance=char.appearance,
                        background=char.background
                    )
                    for char in structured_outline.characters
                ]

                novel_state = NovelState(
                    title=structured_outline.title,
                    characters=characters,
                    chapters=[
                        Chapter(
                            chapter_number=ch.number,
                            title=ch.title,
                            summary=ch.summary,
                            content=None,
                            word_count=None,
                            illustrations=[],
                            images=[]
                        )
                        for ch in structured_outline.chapters
                    ],
                    last_updated=datetime.now().isoformat()
                )

                # Save the novel state directly
                output_dir = output_directory
                state_file_path = os.path.join(output_dir, "novel_state.json")

                os.makedirs(output_dir, exist_ok=True)
                with open(state_file_path, 'w', encoding='utf-8') as f:
                    json.dump(novel_state.model_dump(), f, indent=2, ensure_ascii=False)

                save_result = f"State saved successfully to: {state_file_path}"

                # Also return JSON for backward compatibility
                outline_json = json.dumps(
                    structured_outline.model_dump(), indent=2)

                self.logger.info("Story outline generation and save completed successfully", extra={
                    "title": structured_outline.title,
                    "character_count": len(structured_outline.characters),
                    "chapter_count": len(structured_outline.chapters)
                })

                tracker.add_metadata("character_count", len(
                    structured_outline.characters))
                tracker.add_metadata("chapter_count", len(
                    structured_outline.chapters))
                tracker.add_metadata("prompt_length", prompt_length)
                tracker.add_metadata("json_length", len(outline_json))

                return f"{outline_json}\n\n{save_result}"

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
                    "output_directory": output_directory,
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
    output_directory: str = Field(description="Output directory path where state file should be saved")


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
             previous_chapters: str, characters: str, age_group: str, output_directory: str) -> str:
        self.logger.info("ChapterWriterTool._run() invoked", extra={
            "chapter_number": chapter_number,
            "age_group": age_group,
            "chapter_outline_length": len(chapter_outline),
            "characters_length": len(characters),
            "previous_chapters_length": len(previous_chapters),
            "output_directory": output_directory
        })

        try:
            with PerformanceTracker("chapter_writing", self.logger) as tracker:
                # Create LLM model with structured output
                self.logger.debug(
                    "Creating Gemini model with structured output for chapter writing")
                llm = create_gemini_model(schema=ChapterContent)
                # llm = create_openai_model(
                #     model_name="gpt-4.1-2025-04-14", schema=ChapterContent)
                self.logger.debug(
                    "Structured Gemini model created successfully")

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
- IDENTIFY EXACTLY 2-3 STRATEGIC ILLUSTRATION OPPORTUNITIES per chapter by marking them with [ILLUSTRATION: brief description] tags
- Focus on key moments that advance the plot, show character emotions, establish new locations, or depict important actions
- Place illustration tags at the END of the paragraph where the scene occurs
- Each illustration should be a complete scene that can stand alone as a children's book illustration

Strategic illustration types to consider:
- Character action scenes (important plot moments)
- Environmental establishing shots (new locations)
- Emotional moments (character reactions, important dialogues)
- Key plot points (story turning points)

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
                self.logger.info(
                    "Invoking LLM for structured chapter generation")
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
                    raise ValueError(
                        f"Structured chapter generation failed - Expected ChapterContent, got {type(chapter_data).__name__ if chapter_data else 'None'}")

                # Ensure chapter number matches input
                chapter_data.chapter_number = chapter_number

                # Validate structured output
                if not chapter_data.content:
                    raise ValueError("Empty chapter content received from LLM")

                # Extract illustrations from content if not provided in structured output
                if not chapter_data.illustrations:
                    illustration_pattern = r'\[ILLUSTRATION:\s*(.*?)\]'
                    matches = re.findall(
                        illustration_pattern, chapter_data.content, re.IGNORECASE)
                    chapter_data.illustrations = matches

                # Extract enhanced illustration metadata
                # Get character names from the state file for metadata extraction
                character_names = []
                try:
                    if os.path.exists(state_file_path):
                        existing_state = load_novel_state(state_file_path)
                        character_names = [char.name for char in existing_state.characters]
                except Exception:
                    # If we can't load state, continue without character names
                    pass

                enhanced_illustrations = extract_illustration_metadata(
                    chapter_data.content, character_names)

                # Update word count if not accurate
                actual_word_count = len(chapter_data.content.split())
                if abs(chapter_data.word_count - actual_word_count) > 50:  # Allow some tolerance
                    chapter_data.word_count = actual_word_count

                self.logger.info("Chapter content generated and validated", extra={
                    "chapter_number": chapter_data.chapter_number,
                    "word_count": chapter_data.word_count,
                    "illustration_count": len(chapter_data.illustrations)
                })

                tracker.add_metadata("chapter_number", chapter_number)
                tracker.add_metadata("word_count", chapter_data.word_count)
                tracker.add_metadata("illustration_count",
                                     len(chapter_data.illustrations))
                tracker.add_metadata(
                    "content_length", len(chapter_data.content))
                tracker.add_metadata("prompt_length", prompt_length)

                self.logger.info("Chapter writing completed successfully", extra={
                    "chapter_number": chapter_number
                })

                # Save the chapter to the novel state
                # Use the provided output directory instead of generating a new one
                output_dir = output_directory
                state_file_path = os.path.join(output_dir, "novel_state.json")
                
                self.logger.info("Attempting to save chapter to state", extra={
                    "chapter_number": chapter_data.chapter_number,
                    "output_dir": output_dir,
                    "state_file_path": state_file_path,
                    "state_file_exists": os.path.exists(state_file_path)
                })
                
                try:
                    # Load existing state if it exists
                    if os.path.exists(state_file_path):
                        self.logger.info("Loading existing state file", extra={
                            "state_file_path": state_file_path
                        })
                        novel_state = load_novel_state(state_file_path)
                        
                        # Find and update the chapter
                        for i, ch in enumerate(novel_state.chapters):
                            if ch.chapter_number == chapter_data.chapter_number:
                                # Update the chapter with new content
                                novel_state.chapters[i] = Chapter(
                                    chapter_number=chapter_data.chapter_number,
                                    title=chapter_data.title,
                                    content=chapter_data.content,
                                    summary=ch.summary,  # Keep existing summary
                                    word_count=chapter_data.word_count,
                                    illustrations=enhanced_illustrations,  # Use enhanced illustrations
                                    images=ch.images  # Keep existing images
                                )
                                break
                        else:
                            # Chapter not found, add it
                            novel_state.chapters.append(Chapter(
                                chapter_number=chapter_data.chapter_number,
                                title=chapter_data.title,
                                content=chapter_data.content,
                                summary=None,
                                word_count=chapter_data.word_count,
                                illustrations=enhanced_illustrations,  # Use enhanced illustrations
                                images=[]
                            ))
                            # Sort chapters by number
                            novel_state.chapters.sort(key=lambda x: x.chapter_number)
                        
                        # Save the updated state directly
                        novel_state.last_updated = datetime.now().isoformat()
                        self.logger.info("Saving updated state", extra={
                            "chapter_number": chapter_data.chapter_number,
                            "total_chapters": len(novel_state.chapters),
                            "chapter_content_length": len(chapter_data.content) if chapter_data.content else 0
                        })
                        with open(state_file_path, 'w', encoding='utf-8') as f:
                            json.dump(novel_state.model_dump(), f, indent=2, ensure_ascii=False)
                        save_result = f"State saved successfully to: {state_file_path}"
                        self.logger.info("Chapter saved to novel state", extra={
                            "chapter_number": chapter_data.chapter_number,
                            "state_file_path": state_file_path,
                            "content_saved": bool(chapter_data.content)
                        })
                    else:
                        self.logger.warning("No existing state file found, chapter not saved", extra={
                            "state_file_path": state_file_path,
                            "output_dir": output_dir,
                            "files_in_output_dir": os.listdir(output_dir) if os.path.exists(output_dir) else []
                        })
                except Exception as e:
                    self.logger.error("Failed to save chapter to state", extra={
                        "error": str(e),
                        "chapter_number": chapter_data.chapter_number
                    })
                    # Don't raise error, just log it and continue
                
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
