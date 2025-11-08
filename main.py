from agents.NovelAgent import NovelGenerationAgent
from config.config import Config
import dotenv
from storyspark_logging.core.logger_factory import LoggerFactory
from storyspark_logging.core.performance_tracker import PerformanceTracker
from storyspark_logging.core.error_enhancer import ErrorEnhancer
from utils.directory_utils import set_current_output_dir, get_current_output_dir
import time
import argparse
import sys

dotenv.load_dotenv()

# Set up the timestamped output directory for this run
set_current_output_dir()


def parse_arguments():
    """
    Parse command-line arguments for the StorySpark application.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="StorySpark AI Agent - Generate complete novels with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --age-group "6-8" --genre "adventure" --chapters 5 --theme "A brave cat exploring a magical forest" --title "Luna's Magical Adventure"
  python main.py --help
        """
    )

    parser.add_argument(
        "--age-group",
        choices=["4-6", "6-8", "9-12", "13+"],
        default="6-8",
        help="Target age group for the story (default: 6-8)"
    )

    parser.add_argument(
        "--genre",
        default="adventure",
        help="Story genre (default: adventure)"
    )

    parser.add_argument(
        "--chapters",
        type=int,
        default=5,
        help="Number of chapters to generate (default: 5)"
    )

    parser.add_argument(
        "--theme",
        default="A brave cat exploring a magical forest",
        help="Main theme of the story"
    )

    parser.add_argument(
        "--title",
        default="Luna's Magical Adventure",
        help="Title of the novel"
    )

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize logging
    logger = LoggerFactory().get_logger("storyspark.main")

    try:
        # Get the timestamped output directory
        output_dir = get_current_output_dir()

        # Log application startup with output directory info
        logger.info("StorySpark application starting up", extra={
            "config_params": {
                "age_group": args.age_group,
                "genre": args.genre,
                "num_chapters": args.chapters,
                "output_directory": output_dir
            }
        })

        print(f"StorySpark AI Agent - Generating novel in directory: {output_dir}")

        # Start performance tracking for total execution
        with PerformanceTracker("total_execution", logger) as tracker:
            # Initialize configuration
            config = Config(
                age_group=args.age_group,
                genre=args.genre,
                num_chapters=args.chapters
            )
            logger.debug("Configuration initialized successfully", extra={
                "output_directory": config.output_dir
            })

            # Create agent
            logger.info("Initializing NovelGenerationAgent")
            agent = NovelGenerationAgent(config)
            logger.info("NovelGenerationAgent initialized successfully")

            # Generate novel
            theme = args.theme
            title = args.title
            logger.info("Starting novel generation", extra={
                "theme": theme,
                "title": title,
                "output_directory": config.output_dir
            })

            print(f"Generating novel: '{title}' with theme: '{theme}'")
            print(f"All files will be saved to: {config.output_dir}")

            result = agent.generate_novel(theme=theme, title=title)

            # Log completion
            logger.info("Novel generation completed successfully", extra={
                "result_keys": list(result.keys()) if isinstance(result, dict) else str(type(result)),
                "output_directory": config.output_dir
            })

            print(f"\nNovel generation completed successfully!")
            print(f"All output files are saved in: {config.output_dir}")
            print(f"- Novel state: novel_state.json")
            print(f"- LaTeX document: {title.replace(' ', '_')}.tex")
            print(f"- Final PDF: {title.replace(' ', '_')}.pdf")

        # Log total execution time
        execution_time = tracker.get_duration()
        logger.info("Total execution completed", extra={
            "total_execution_time_seconds": execution_time,
            "output_directory": output_dir
        })

    except Exception as e:
        # Enhance error with context
        ErrorEnhancer.log_unexpected_error(
            logger=logger,
            operation="main_execution",
            error=e,
            context={
                "config": {
                    "age_group": args.age_group,
                    "genre": args.genre,
                    "num_chapters": args.chapters,
                    "output_directory": get_current_output_dir()
                }
            }
        )
        raise


if __name__ == "__main__":
    main()
