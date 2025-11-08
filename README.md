# StorySpark AI Agent

An AI-powered children's novel generation system that creates complete, age-appropriate stories with professional PDF output. Using advanced LangChain agents with OpenAI's API and the gpt-4.1 model, StorySpark generates multi-chapter novels tailored for different age groups, complete with character development and illustration markers.

## Features

- **AI-Powered Novel Generation**: Leverages OpenAI's API with the gpt-4.1 model for intelligent story creation
- **Age-Appropriate Content**: Supports four age groups (4-6, 6-8, 9-12, 13+) with tailored language and themes
- **Multi-Chapter Stories**: Generates complete novels with structured chapters and character arcs
- **Professional PDF Output**: Uses LaTeX for beautifully formatted, print-ready documents
- **Flexible Configuration**: Command-line interface with extensive customization options
- **Comprehensive Logging**: Built-in performance tracking and error enhancement
- **State Persistence**: Saves progress in JSON format for reliability and debugging
- **Modular Architecture**: Extensible design with specialized tools for each generation step

## Quick Start

1. **Clone and Setup**:

   ```bash
   git clone <repository-url>
   cd StorySparkAIAgent
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Generate Your First Novel**:
   ```bash
   python main.py --title "My First Story" --theme "A magical adventure in the forest"
   ```

## Installation

### Prerequisites

- **Python 3.13+**: Required runtime environment
- **LaTeX Distribution**: For PDF generation (TeX Live, MiKTeX, or MacTeX recommended)
- **API Keys**: OpenAI API key (required for story generation)

### Setup Steps

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd StorySparkAIAgent
   ```

2. **Create Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   # For basic usage
   pip install -r requirements.txt

   # For development (includes additional tools like langsmith for monitoring)
   pip install -e .[dev]
   ```

   **Note**: `requirements.txt` contains core dependencies for basic functionality. `pyproject.toml` defines the full project dependencies including optional development tools like `langchain-openai` for OpenAI API integration and `langsmith` for monitoring. Use `pip install -e .[dev]` for development work.

4. **Verify LaTeX Installation**:
   ```bash
   pdflatex --version
   ```

## Configuration

Create a `.env` file in the project root based on `.env.example`:

```bash
cp .env.example .env
```

### Required Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required for story generation)

### Optional Environment Variables

- `OPENAI_API_BASE_URL`: Custom API base URL for using alternative OpenAI-compatible endpoints (optional, defaults to https://api.openai.com/v1)

- `LANGSMITH_TRACING`: Enable LangSmith tracing (true/false)
- `LANGSMITH_ENDPOINT`: LangSmith endpoint URL
- `LANGSMITH_API_KEY`: LangSmith API key for advanced monitoring
- `LANGSMITH_PROJECT`: LangSmith project name

## Usage

### Basic Usage

Generate a novel with default settings:

```bash
python main.py
```

### Command-Line Parameters

| Parameter     | Type   | Default                                  | Description                           |
| ------------- | ------ | ---------------------------------------- | ------------------------------------- |
| `--age-group` | choice | 6-8                                      | Target age group: 4-6, 6-8, 9-12, 13+ |
| `--genre`     | string | adventure                                | Story genre (fantasy, mystery, etc.)  |
| `--chapters`  | int    | 5                                        | Number of chapters to generate        |
| `--theme`     | string | "A brave cat exploring a magical forest" | Main story theme                      |
| `--title`     | string | "Luna's Magical Adventure"               | Novel title                           |

### Advanced Examples

**Young Children's Story**:

```bash
python main.py \
  --age-group "4-6" \
  --genre "fantasy" \
  --chapters 3 \
  --theme "A friendly dragon learning to fly" \
  --title "Danny the Dragon"
```

**Teen Adventure**:

```bash
python main.py \
  --age-group "13+" \
  --genre "mystery" \
  --chapters 8 \
  --theme "Solving ancient riddles in a hidden temple" \
  --title "The Lost Temple Mystery"
```

**Custom Configuration**:

```bash
python main.py \
  --age-group "9-12" \
  --genre "science-fiction" \
  --chapters 6 \
  --theme "Exploring distant planets with alien friends" \
  --title "Star Explorers"
```

### Output

All files are saved to a timestamped directory (e.g., `output/20251108_144716/`):

- `novel_state.json`: Complete story data and metadata
- `{title}.tex`: LaTeX source document
- `{title}.pdf`: Final formatted PDF
- `{title}.aux`, `{title}.log`: LaTeX compilation artifacts

## Architecture

StorySpark uses a modular agent-based architecture built on LangChain:

### Core Components

- **NovelGenerationAgent**: Main orchestration agent using ReAct reasoning
- **Model Clients**: Currently supports OpenAI API with gpt-4.1 model (Google Gemini and Ollama support planned for future releases)
- **Specialized Tools**:
  - `StoryOutlinerTool`: Creates novel structure and character profiles
  - `ChapterWriterTool`: Generates individual chapter content
  - `LaTeXGeneratorTool`: Converts content to LaTeX format
  - `PDFCompilerTool`: Compiles LaTeX to PDF
  - `StateSaverTool`: Persists progress to JSON

### Data Flow

1. **Input Processing**: Parse CLI arguments and load configuration
2. **Agent Orchestration**: Use LangChain agent to coordinate generation steps
3. **Content Generation**: Create outlines, chapters, and character descriptions
4. **Document Creation**: Generate LaTeX markup with proper formatting
5. **PDF Compilation**: Produce final professional document
6. **State Persistence**: Save all progress for debugging and recovery

### Logging System

Comprehensive logging with:

- Performance tracking for each operation
- Error enhancement with contextual information
- Multiple output formats (console, JSON, verbose)
- Configurable log levels

## Output Structure

Each generation run creates a timestamped output directory containing:

```
output/YYYYMMDD_HHMMSS/
├── novel_state.json      # Complete story data
├── {title}.tex          # LaTeX source
├── {title}.pdf          # Final PDF document
├── {title}.aux          # LaTeX auxiliary file
└── {title}.log          # LaTeX compilation log
```

### novel_state.json Structure

```json
{
  "title": "Story Title",
  "theme": "Main theme",
  "age_group": "6-8",
  "genre": "adventure",
  "chapters": [
    {
      "chapter_number": 1,
      "title": "Chapter Title",
      "content": "Chapter text...",
      "word_count": 450,
      "illustrations": ["Scene description for illustration"]
    }
  ],
  "characters": [
    {
      "name": "Character Name",
      "description": "Character details",
      "age": 8,
      "role": "protagonist"
    }
  ]
}
```

## Customization

### Adding Age Groups

Modify `config/config.py` to add new age group configurations:

```python
AGE_GROUP_CONFIGS = {
    "4-6": {...},
    "6-8": {...},
    "9-12": {...},
    "13+": {...},
    "custom": {
        "vocabulary_level": "intermediate",
        "complexity": "medium",
        "themes": ["friendship", "discovery"]
    }
}
```

### Modifying Templates

Edit LaTeX templates in `templates/main.tex` to customize document formatting, fonts, and layout.

### Extending Models

Add new AI model support in `agents/models/`:

1. Create new model client (e.g., `claude_client.py`)
2. Implement model creation function
3. Update `NovelAgent.py` to support the new model

**Note**: Currently, only OpenAI API integration is fully implemented. Google Gemini and Ollama support are planned for future releases.

### Custom Tools

Create new tools by extending the base tool classes in `tools/`:

```python
from langchain.tools import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Description of custom functionality"

    def _run(self, input_str: str) -> str:
        # Implementation
        return result
```

## Current Limitations

- **Model Support**: Currently only supports OpenAI API with gpt-4.1 model. Google Gemini and Ollama integrations are planned for future releases.
- **API Dependency**: Requires active OpenAI API key with sufficient credits for story generation.
- **LaTeX Requirement**: Professional PDF output requires LaTeX distribution installation.
- **Structured Output**: Some advanced features depend on model support for structured JSON output.

## Troubleshooting

### API Key Verification

After configuring your `.env` file, verify your OpenAI API key:

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); from agents.models.openai_client import create_openai_model; model = create_openai_model(); print('API key verified successfully')"
```

### Common Issues

**LaTeX Compilation Fails**

- Ensure LaTeX distribution is installed and `pdflatex` is in PATH
- Check `{title}.log` for specific error messages
- Verify all required LaTeX packages are installed

**API Key Errors**

- Confirm `OPENAI_API_KEY` is set in `.env` file
- Check API key validity and quota limits on OpenAI dashboard
- Verify network connectivity to API endpoints
- Use the verification command above to test API access

**Structured Output Validation Errors**

- Ensure the gpt-4.1 model supports structured output (function calling)
- Check logs for schema validation failures
- Reduce complexity of prompts if validation consistently fails

**Memory Issues**

- Reduce chapter count for long stories
- Ensure sufficient RAM (4GB+ recommended)
- Check `novel_state.json` for incomplete generations

**Model-Specific Errors**

- OpenAI/gpt-4.1: Check API key and billing status
- Custom API endpoints: Verify `OPENAI_API_BASE_URL` is correct and accessible

### Debug Mode

Enable verbose logging:

```bash
export LOG_LEVEL=DEBUG
python main.py --title "Debug Story"
```

Check logs in `logs/` directory for detailed execution information.

### Recovery from Failures

If generation fails mid-process:

1. Check `novel_state.json` for saved progress
2. Resume from last completed chapter
3. Manually run PDF compilation if content is complete

---
