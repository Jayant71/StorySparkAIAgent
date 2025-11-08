from langchain.agents import create_agent
# from agents.models.gemini import create_gemini_model
from agents.models.openai_client import create_openai_model
from tools.story_tool import StoryOutlinerTool, ChapterWriterTool
# from tools.image_generation import CharacterImageGeneratorTool
from tools.pdf_generator import LaTeXGeneratorTool, PDFCompilerTool
from tools.state_tool import StateSaverTool
import json
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
            self.llm = create_openai_model()
            self.state_file = None

            # Initialize tools
            self.tools = [
                StoryOutlinerTool(),
                ChapterWriterTool(),
                # CharacterImageGeneratorTool(),
                LaTeXGeneratorTool(config),
                PDFCompilerTool(config),
                StateSaverTool(config)
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
- LaTeXGeneratorTool: Generate LaTeX documents from content
- PDFCompilerTool: Compile LaTeX into PDF
- StateSaverTool: Save progress to JSON format

Always break down complex tasks into smaller steps and use tools efficiently."""
# - CharacterImageGeneratorTool: Generate character and scene illustrations

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

        logger.info("Agent created successfully", extra={
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
2. Use CharacterImageGeneratorTool to create reference images for main characters (if available)
3. For each chapter:
     - Use ChapterWriterTool to write the chapter content
     - Use StateSaverTool to save the chapter progress to JSON in the output directory
4. Use LaTeXGeneratorTool to generate the complete LaTeX document in the output directory
5. Use PDFCompilerTool to compile the final PDF in the output directory

All files will be automatically saved to the timestamped output directory: {self.config.output_dir}
Store all progress in JSON format for memory efficiency."""
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

            logger.info("Novel generation completed", extra={
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
