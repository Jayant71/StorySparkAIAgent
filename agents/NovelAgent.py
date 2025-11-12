from langchain.agents import create_agent
from agents.models.gemini import create_gemini_model
from agents.models.openai_client import create_openai_model
from tools.story_tool import StoryOutlinerTool, ChapterWriterTool
from tools.image_generation import CharacterImageGeneratorTool, ChapterImageGeneratorTool
from tools.pdf_generator import LaTeXGeneratorTool, PDFCompilerTool
from tools.story_tool import load_novel_state
import json
import os
from storyspark_logging.core.logger_factory import LoggerFactory
from storyspark_logging.core.performance_tracker import PerformanceTracker
from storyspark_logging.core.error_enhancer import ErrorEnhancer


class NovelGenerationAgent:
    def __init__(self, config):
        logger = LoggerFactory().get_logger("storyspark.agents.novel")

        try:
            logger.info("Initializing NovelGenerationAgent", extra={
                "config": {
                    "age_group": config.age_group,
                    "genre": config.genre,
                    "num_chapters": config.num_chapters,
                    "output_dir": config.output_dir
                }
            })

            self.config = config
            # self.llm = create_openai_model(
            #     model_name="gpt-4.1-2025-04-14", temperature=0.9)
            self.llm = create_gemini_model()
            self.state_file = None

            # Initialize tools
            self.tools = [
                StoryOutlinerTool(),
                ChapterWriterTool(),
                CharacterImageGeneratorTool(),
                ChapterImageGeneratorTool(),
                LaTeXGeneratorTool(config),
                PDFCompilerTool(config)
            ]

            # Log tool loading status
            tool_names = [type(tool).__name__ for tool in self.tools]
            logger.debug("Tools initialized", extra={
                "tool_count": len(self.tools),
                "tool_names": tool_names
            })

            # Create agent
            with PerformanceTracker("agent_initialization", logger) as tracker:
                self.agent = self._create_agent()

            logger.info("NovelGenerationAgent initialized successfully", extra={
                "initialization_time_seconds": tracker.get_duration()
            })

        except Exception as e:
            ErrorEnhancer.log_unexpected_error(
                logger=logger,
                operation="NovelGenerationAgent_initialization",
                error=e,
                context={
                    "config_age_group": config.age_group,
                    "config_genre": config.genre
                }
            )
            raise

    def _create_agent(self):
        logger = LoggerFactory().get_logger("storyspark.agents.novel")

        # Define system prompt for ReAct-style reasoning
        system_prompt = """You are an expert novel generation assistant. You have access to specialized tools for creating complete novels.

When given a task, think step-by-step and use the available tools systematically:
1. Analyze what needs to be done
2. Choose the appropriate tool for each step
3. Execute actions in logical sequence
4. Verify results before moving to the next step

Available tools and their purposes:
- StoryOutlinerTool: Create novel outlines and character descriptions
- ChapterWriterTool: Write detailed chapter content
- CharacterImageGeneratorTool: Generate character reference images
- ChapterImageGeneratorTool: Generate strictly consistent images for chapter illustrations
- LaTeXGeneratorTool: Generate LaTeX documents from content
- PDFCompilerTool: Compile LaTeX into PDF

Always break down complex tasks into smaller steps and use tools efficiently."""

        logger.debug("Creating agent with system prompt", extra={
            "system_prompt_length": len(system_prompt),
            "model_type": type(self.llm).__name__
        })

        # Create agent using LangChain 1.0 API
        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

        logger.debug("Agent created successfully", extra={
            "system_prompt_length": len(system_prompt)
        })

        return agent

    def generate_novel(self, theme, title):
        logger = LoggerFactory().get_logger("storyspark.agents.novel")

        try:
            logger.info("Starting novel generation", extra={
                "theme": theme,
                "title": title,
                "config": {
                    "age_group": self.config.age_group,
                    "genre": self.config.genre,
                    "num_chapters": self.config.num_chapters
                }
            })

            # Create workflow instructions as a user message
            workflow = f"""Generate a complete novel with the following specifications:
- Title: {title}
- Theme: {theme}
- Age Group: {self.config.age_group}
- Genre: {self.config.genre}
- Number of Chapters: {self.config.num_chapters}
- Output Directory: {self.config.output_dir}

Follow these steps:
1. Use StoryOutlinerTool to create the novel outline and character descriptions
   - Parameters: theme={theme}, age_group={self.config.age_group}, genre={self.config.genre}, num_chapters={self.config.num_chapters}, output_directory={self.config.output_dir}
2. Use CharacterImageGeneratorTool to create reference images for main characters (save as 'character_name_reference.png')
"""

            # Add explicit steps for each chapter
            for i in range(1, self.config.num_chapters + 1):
                workflow += f"""
{i+2}. Generate Chapter {i}:
   - Use ChapterWriterTool to write the chapter content with exactly 2-3 strategic illustration tags
     - Parameters: chapter_number={i}, chapter_outline=(from outline), previous_chapters=(from previous chapters), characters=(from character list), age_group={self.config.age_group}, output_directory={self.config.output_dir}
   - Use ChapterImageGeneratorTool to extract illustrations and generate strictly consistent images for each tag
   - Images will be automatically saved with proper naming and character consistency
   - Note: ChapterWriterTool automatically saves the chapter to the state file in {self.config.output_dir}
"""

            workflow += f"""
{self.config.num_chapters + 3}. Use LaTeXGeneratorTool to generate the complete LaTeX document in the output directory
{self.config.num_chapters + 4}. Use PDFCompilerTool to compile the final PDF in the output directory

CRITICAL REQUIREMENTS:
- Each chapter must have minimum 2 and maximum 5 illustrations generated based on the chapter content context
- Use character reference images (created in step 2) to maintain visual consistency
- Save all images in the images subdirectory of the output directory
- Track all generated images in the state JSON under each chapter's 'images' field
- Ensure illustration prompts include character descriptions, scene details, and emotional context

All files will be automatically saved to the timestamped output directory: {self.config.output_dir}
Note: The StoryOutlinerTool and ChapterWriterTool now automatically save progress to JSON format."""
    #    - Use CharacterImageGeneratorTool to create scene illustrations

            logger.debug("Workflow instructions generated", extra={
                "workflow_length": len(workflow)
            })

            # Execute agent with new v1.0 message format
            with PerformanceTracker("novel_generation", logger) as tracker:
                logger.info("Invoking agent for novel generation")
                result = self.agent.invoke({
                    "messages": [
                        {"role": "user", "content": workflow}
                    ],
                }, {
                    "recursion_limit": 100
                })

            logger.warning("Novel generation completed", extra={
                "generation_time_seconds": tracker.get_duration(),
                "result_type": type(result).__name__,
                "has_messages": "messages" in result if isinstance(result, dict) else False
            })

            return result

        except Exception as e:
            ErrorEnhancer.log_unexpected_error(
                logger=logger,
                operation="novel_generation",
                error=e,
                context={
                    "theme": theme,
                    "title": title,
                    "config_age_group": self.config.age_group,
                    "config_genre": self.config.genre
                }
            )
            raise

    def continue_novel(self, state_file_path: str):
        logger = LoggerFactory().get_logger("storyspark.agents.novel")

        try:
            logger.warning("Starting novel continuation", extra={
                "state_file_path": state_file_path
            })

            # Load existing state
            with PerformanceTracker("state_loading", logger) as tracker:
                state = load_novel_state(state_file_path)
                state_dict = state.model_dump()

            logger.info("State loaded successfully", extra={
                "title": state.title,
                "existing_chapters": len(state.chapters)
            })

            # Determine next chapter to generate
            chapters = state.chapters
            completed_chapters = len(chapters)
            next_chapter_num = completed_chapters + 1

            if next_chapter_num > self.config.num_chapters:
                logger.info("Novel is already complete", extra={
                    "total_chapters": self.config.num_chapters,
                    "completed_chapters": completed_chapters
                })
                return {"message": "Novel is already complete", "chapters_generated": 0}

            # Extract novel metadata
            title = state.title
            characters = state.characters
            # Default theme, could be enhanced
            theme = "mysterious adventure with ancient artifacts"

            logger.info("Continuing novel generation", extra={
                "next_chapter": next_chapter_num,
                "total_chapters": self.config.num_chapters,
                "title": title
            })

            # Generate remaining chapters
            chapters_generated = 0
            for chapter_num in range(next_chapter_num, self.config.num_chapters + 1):
                logger.info(f"Generating chapter {chapter_num}")

                # Create continuation workflow
                continuation_workflow = f"""Continue generating the novel "{title}" from chapter {chapter_num}.

Existing context:
- Total chapters planned: {self.config.num_chapters}
- Chapters already completed: {completed_chapters}
- Characters: {json.dumps(characters, indent=2)}
- Theme: {theme}

Previous chapters summary:
{self._get_previous_chapters_summary(chapters)}

For chapter {chapter_num}:
1. Use ChapterWriterTool to write the chapter content with exactly 2-3 strategic illustration tags
   - Parameters: chapter_number={chapter_num}, chapter_outline=(from outline), previous_chapters=(from previous chapters), characters=(from character list), age_group={self.config.age_group}, output_directory={os.path.dirname(state_file_path)}
2. Use ChapterImageGeneratorTool to extract illustrations and generate images for each tag
Note: ChapterWriterTool automatically saves the chapter to the state file

CRITICAL REQUIREMENTS:
- Each chapter must have exactly 2-3 illustrations generated
- Use existing character descriptions for visual consistency
- Save all images in the images subdirectory
- Track all generated images in the chapter's 'images' field
- Ensure illustration prompts include character descriptions, scene details, and emotional context
- Continue the story logically from the previous chapters

Save progress to: {state_file_path}"""

                # Execute agent for this chapter
                with PerformanceTracker(f"chapter_{chapter_num}_generation", logger) as tracker:
                    result = self.agent.invoke({
                        "messages": [
                            {"role": "user", "content": continuation_workflow}
                        ],
                    }, {
                        "recursion_limit": 100
                    })

                chapters_generated += 1
                logger.info(f"Chapter {chapter_num} completed", extra={
                    "chapters_generated": chapters_generated
                })

            # Final generation of LaTeX and PDF if all chapters are done
            if next_chapter_num + chapters_generated - 1 >= self.config.num_chapters:
                logger.info("All chapters completed, generating final PDF")

                final_workflow = f"""Generate the complete LaTeX document and PDF for the novel "{title}".

The novel is now complete with all {self.config.num_chapters} chapters.

Use LaTeXGeneratorTool to generate the LaTeX document
Use PDFCompilerTool to compile the final PDF

Output directory: {self.config.output_dir}"""

                result = self.agent.invoke({
                    "messages": [
                        {"role": "user", "content": final_workflow}
                    ],
                }, {
                    "recursion_limit": 50
                })

            logger.warning("Novel continuation completed", extra={
                "chapters_generated": chapters_generated,
                "total_chapters_now": completed_chapters + chapters_generated
            })

            return {
                "message": f"Successfully continued novel generation",
                "chapters_generated": chapters_generated,
                "total_chapters": completed_chapters + chapters_generated
            }

        except Exception as e:
            ErrorEnhancer.log_unexpected_error(
                logger=logger,
                operation="novel_continuation",
                error=e,
                context={
                    "state_file_path": state_file_path,
                    "config_num_chapters": self.config.num_chapters
                }
            )
            raise

    def _get_previous_chapters_summary(self, chapters):
        """Generate a summary of previous chapters for context"""
        if not chapters:
            return "No previous chapters."

        summary = ""
        for chapter in chapters[-3:]:  # Last 3 chapters for context
            summary += f"Chapter {chapter['number']}: {chapter['title']}\n"
            summary += f"Summary: {chapter['summary']}\n\n"

        return summary
