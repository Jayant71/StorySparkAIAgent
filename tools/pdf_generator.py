from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from pylatex import Document, Section, Subsection, Figure
from pylatex.utils import NoEscape, italic, bold
from storyspark_logging.core.logger_factory import LoggerFactory
from storyspark_logging.core.performance_tracker import PerformanceTracker
from storyspark_logging.core.error_enhancer import ErrorEnhancer
from config.config import Config
from typing import Any
import subprocess
import json
import os
import argparse
import sys


class LaTeXGeneratorInput(BaseModel):
    state_json_path: str = Field(
        description="Path to the novel state JSON file")
    output_tex_path: str = Field(description="Where to save the LaTeX file")


class LaTeXGeneratorTool(BaseTool):
    name: str = "LaTeXGeneratorTool"
    description: str = "Generates a complete LaTeX document from the novel state JSON"
    args_schema: type[BaseModel] = LaTeXGeneratorInput
    logger: Any = Field(default=None, exclude=True)

    def __init__(self, config: Config = None):
        super().__init__()
        self.__dict__['_config'] = config
        self.logger = LoggerFactory().get_logger("storyspark.tools.pdf.latex")
        self.logger.info("LaTeXGeneratorTool initialized")

    def _generate_chapters_latex(self, chapters, output_directory):
        """Generate LaTeX content for all chapters with embedded images."""
        self.logger.debug("Generating chapters LaTeX content", extra={
            "chapter_count": len(chapters)
        })

        chapters_content = []

        for i, chapter in enumerate(chapters, 1):
            chapter_title = chapter.get('title', f'Chapter {i}')
            chapter_content = chapter.get('content', '')

            if chapter_content:
                content_length = len(chapter_content)
            else:
                content_length = 0
                self.logger.warning(f"Empty content for chapter {i}")

            self.logger.debug("Processing chapter for LaTeX", extra={
                "chapter_number": i,
                "chapter_title": chapter_title,
                "content_length": content_length
            })

            # Process illustrations in content
            processed_content = self._process_illustrations(
                chapter_content, chapter.get('images', []), output_directory)

            # Generate chapter LaTeX
            chapter_latex = f"""
\\chapter{{{chapter_title}}}

{processed_content}
"""
            chapters_content.append(chapter_latex)

        return '\n'.join(chapters_content)

    def _process_illustrations(self, content, images, output_directory):
        """Replace [ILLUSTRATION: description] placeholders with actual LaTeX image commands."""
        import re

        # Handle different image formats with improved path resolution
        image_paths = []
        
        # Ensure images directory exists
        images_dir = os.path.join(output_directory, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
        
        if images and isinstance(images[0], dict):
            # Images is list of dicts
            for img in images:
                path = img.get('path', img.get('filename', ''))
                if path:
                    resolved_path = self._resolve_image_path(path, output_directory, images_dir)
                    if resolved_path:  # Only append if path is valid (not None)
                        image_paths.append(resolved_path)
        elif images and isinstance(images[0], str):
            # Images is list of strings (paths)
            for path in images:
                resolved_path = self._resolve_image_path(path, output_directory, images_dir)
                if resolved_path:  # Only append if path is valid (not None)
                    image_paths.append(resolved_path)

        self.logger.debug("Processed image paths", extra={
            "image_count": len(image_paths),
            "image_paths": image_paths
        })

        # Collect all illustration descriptions in order
        illustrations = []
        pattern = r'\[ILLUSTRATION:\s*(.*?)\]'
        for match in re.finditer(pattern, content, flags=re.IGNORECASE):
            desc = match.group(1).strip()
            illustrations.append(desc)

        self.logger.debug("Found illustrations in content", extra={
            "illustration_count": len(illustrations),
            "illustrations": illustrations
        })

        # Assign images to illustrations in order
        image_index = 0
        def replace_illustration(match):
            nonlocal image_index
            desc = match.group(1).strip()

            if image_index < len(image_paths):
                img_path = image_paths[image_index]
                image_index += 1
                self.logger.debug("Assigned image to illustration", extra={
                    "description": desc,
                    "image_path": img_path,
                    "image_index": image_index - 1
                })
                return f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{img_path}}}
\\caption{{{desc}}}
\\end{{figure}}
"""
            else:
                # Check if we can find any missing images in the images directory
                if image_index == 0 and os.path.exists(images_dir):
                    # Try to find any images that might match this chapter
                    import glob
                    chapter_pattern = f"chapter_*_illustration_{image_index + 1}.png"
                    found_images = glob.glob(os.path.join(images_dir, chapter_pattern))
                    if found_images:
                        img_path = os.path.relpath(found_images[0], output_directory).replace('\\', '/')
                        image_index += 1
                        self.logger.info(f"Found missing image for illustration: {img_path}", extra={
                            "description": desc,
                            "image_path": img_path
                        })
                        return f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{img_path}}}
\\caption{{{desc}}}
\\end{{figure}}
"""
                
                self.logger.warning("No more images available for illustration", extra={
                    "description": desc,
                    "available_images": len(image_paths),
                    "images_dir_exists": os.path.exists(images_dir)
                })
                return f"\\textit{{[Illustration: {desc}]}}"

        # Replace all illustrations
        processed_content = re.sub(pattern, replace_illustration, content, flags=re.IGNORECASE)

        return processed_content
    
    def _resolve_image_path(self, path, output_directory, images_dir):
        """Resolve image path with multiple fallback strategies"""
        try:
            # If path is already a relative path from output_directory, use it directly
            if not os.path.isabs(path) and not path.startswith('images/'):
                # Try direct path first
                direct_path = os.path.join(output_directory, path)
                if os.path.exists(direct_path):
                    rel_path = os.path.relpath(direct_path, output_directory)
                    return rel_path.replace('\\', '/')
            
            # Try the path as given
            if os.path.exists(path):
                rel_path = os.path.relpath(path, output_directory)
                return rel_path.replace('\\', '/')
            
            # Try in images directory
            filename = os.path.basename(path)
            images_path = os.path.join(images_dir, filename)
            if os.path.exists(images_path):
                return f"images/{filename}"
            
            # Try common naming patterns
            base_name = os.path.splitext(filename)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(test_path):
                    return f"images/{base_name + ext}"
            
            # If all else fails, return None to indicate no valid image found
            self.logger.warning(f"Image file not found: {filename}", extra={
                "original_path": path,
                "tried_paths": [
                    path,
                    os.path.join(output_directory, path),
                    images_path
                ]
            })
            return None
            
        except Exception as e:
            self.logger.error(f"Error resolving image path: {e}", extra={
                "path": path
            })
            # Return None to indicate failure
            return None

    def _run(self, state_json_path: str, output_tex_path: str) -> str:
        self.logger.info("LaTeXGeneratorTool._run() invoked", extra={
            "state_json_path": state_json_path,
            "output_tex_path": output_tex_path
        })

        try:
            with PerformanceTracker("latex_document_generation", self.logger) as tracker:
                # Handle output path logic
                directory = os.path.dirname(output_tex_path)
                self.logger.debug("Extracted directory from output_tex_path", extra={
                    "output_tex_path": output_tex_path,
                    "directory": directory,
                    "directory_is_empty": directory == "",
                    "is_absolute_path": os.path.isabs(output_tex_path),
                    "output_directory": self.__dict__.get('_config').output_dir if self.__dict__.get('_config') else None
                })

                # Handle case where no directory is specified in the path
                if not directory:
                    # Use timestamped output directory if no directory is specified and config is available
                    config = self.__dict__.get('_config')
                    if config is not None:
                        directory = config.output_dir
                        self.logger.info("No directory specified in output_tex_path, using timestamped output directory", extra={
                            "output_tex_path": output_tex_path,
                            "default_directory": directory
                        })
                        # Update the file path to include the directory
                        output_tex_path = os.path.join(
                            directory, os.path.basename(output_tex_path))
                        self.logger.debug("Updated output_tex_path with directory", extra={
                            "new_output_tex_path": output_tex_path
                        })
                    else:
                        # Use current directory if no config
                        directory = os.getcwd()
                        self.logger.info("No directory specified in output_tex_path and no config available, using current directory", extra={
                            "output_tex_path": output_tex_path,
                            "default_directory": directory
                        })
                        # Update the file path to include the directory
                        output_tex_path = os.path.join(
                            directory, os.path.basename(output_tex_path))
                        self.logger.debug("Updated output_tex_path with current directory", extra={
                            "new_output_tex_path": output_tex_path
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
                            "output_tex_path": output_tex_path
                        }
                    )
                    raise
                # Load and validate state file
                self.logger.debug("Loading novel state from JSON file", extra={
                    "state_json_path": state_json_path
                })

                try:
                    with open(state_json_path, 'r', encoding='utf-8') as f:
                        novel_state = json.load(f)
                except FileNotFoundError as e:
                    ErrorEnhancer.log_error(
                        self.logger,
                        e,
                        context={
                            "operation": "state_file_loading",
                            "file_path": state_json_path
                        }
                    )
                    raise
                except json.JSONDecodeError as e:
                    ErrorEnhancer.log_error(
                        self.logger,
                        e,
                        context={
                            "operation": "json_parsing",
                            "file_path": state_json_path
                        }
                    )
                    raise

                state_size = len(json.dumps(novel_state))
                # Use chapters for content generation
                written_chapters = novel_state.get('chapters', [])
                chapter_count = len(written_chapters)
                self.logger.info("Novel state loaded successfully", extra={
                    "state_size_bytes": state_size,
                    "chapter_count": chapter_count,
                    "title": novel_state.get('title', 'Unknown')
                })

                # Read LaTeX template
                template_path = os.path.join(os.path.dirname(
                    __file__), '..', 'templates', 'main.tex')
                self.logger.debug("Reading LaTeX template", extra={
                    "template_path": template_path
                })

                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        template_content = f.read()
                except FileNotFoundError as e:
                    ErrorEnhancer.log_error(
                        self.logger,
                        e,
                        context={
                            "operation": "template_reading",
                            "template_path": template_path
                        }
                    )
                    raise

                # Replace title placeholder
                template_content = template_content.replace(
                    'STORY_TITLE', novel_state['title'])

                # Generate chapters content
                chapters_latex = self._generate_chapters_latex(
                    written_chapters, directory)
                template_content = template_content.replace(
                    'CHAPTERS_CONTENT', chapters_latex)

                # Count total images
                total_images = sum(len(chapter.get('images', []))
                                   for chapter in written_chapters)

                # Write the LaTeX file
                self.logger.debug("Writing LaTeX file", extra={
                    "output_path": output_tex_path
                })

                try:
                    with open(output_tex_path, 'w', encoding='utf-8') as f:
                        f.write(template_content)
                except Exception as e:
                    ErrorEnhancer.log_error(
                        self.logger,
                        e,
                        context={
                            "operation": "latex_file_writing",
                            "output_path": output_tex_path
                        }
                    )
                    raise

                tracker.add_metadata("chapter_count", chapter_count)
                tracker.add_metadata("total_images", total_images)
                tracker.add_metadata("state_size_bytes", state_size)

                self.logger.info("LaTeX document generation completed successfully", extra={
                    "output_tex_path": output_tex_path,
                    "chapter_count": chapter_count,
                    "total_images": total_images
                })

                return f"LaTeX file generated: {output_tex_path}"

        except Exception as e:
            ErrorEnhancer.log_unexpected_error(
                self.logger,
                "LaTeX document generation",
                e,
                context={
                    "state_json_path": state_json_path,
                    "output_tex_path": output_tex_path
                }
            )
            raise


class PDFCompilerInput(BaseModel):
    tex_file_path: str = Field(description="Path to the LaTeX file to compile")


class PDFCompilerTool(BaseTool):
    name: str = "PDFCompilerTool"
    description: str = "Compiles a LaTeX file to PDF using pdflatex"
    args_schema: type[BaseModel] = PDFCompilerInput
    logger: Any = Field(default=None, exclude=True)

    def __init__(self, config: Config = None):
        super().__init__()
        self.__dict__['_config'] = config
        self.logger = LoggerFactory().get_logger("storyspark.tools.pdf.compiler")
        self.logger.info("PDFCompilerTool initialized")

    def _check_pdflatex_availability(self) -> bool:
        """
        Check if pdflatex is available in the system PATH.

        Returns:
            bool: True if pdflatex is found, False otherwise
        """
        self.logger.debug("Checking pdflatex availability")

        try:
            # Try to find pdflatex using shutil.which (cross-platform)
            import shutil
            pdflatex_path = shutil.which('pdflatex')

            if pdflatex_path:
                self.logger.info("pdflatex found in system PATH", extra={
                    "pdflatex_path": pdflatex_path
                })
                return True
            else:
                self.logger.warning("pdflatex not found in system PATH", extra={
                    "platform": sys.platform,
                    "path_env": os.environ.get('PATH', 'PATH not set')
                })
                return False

        except Exception as e:
            self.logger.error("Error checking pdflatex availability", extra={
                "error": str(e)
            })
            return False

    def _get_safe_working_directory(self, tex_file_path: str) -> str:
        """
        Get a safe working directory for pdflatex execution.

        Args:
            tex_file_path: Path to the LaTeX file

        Returns:
            str: Safe working directory path
        """
        working_dir = os.path.dirname(tex_file_path)

        # If working_dir is empty, use current directory
        if not working_dir:
            working_dir = os.getcwd()
            self.logger.debug("Using current directory as working directory", extra={
                "original_working_dir": "",
                "fallback_working_dir": working_dir
            })

        # Ensure the directory exists
        if not os.path.exists(working_dir):
            error_msg = f"Working directory does not exist: {working_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        return working_dir

    def _run(self, tex_file_path: str) -> str:
        self.logger.info("PDFCompilerTool._run() invoked", extra={
            "tex_file_path": tex_file_path
        })

        try:
            with PerformanceTracker("pdf_compilation", self.logger) as tracker:
                # Handle tex file path logic
                directory = os.path.dirname(tex_file_path)
                self.logger.debug("Extracted directory from tex_file_path", extra={
                    "tex_file_path": tex_file_path,
                    "directory": directory,
                    "directory_is_empty": directory == "",
                    "is_absolute_path": os.path.isabs(tex_file_path),
                    "output_directory": self.__dict__.get('_config').output_dir if self.__dict__.get('_config') else None
                })

                # Handle case where no directory is specified in the path
                if not directory:
                    # Use timestamped output directory if no directory is specified and config is available
                    config = self.__dict__.get('_config')
                    if config is not None:
                        directory = config.output_dir
                        self.logger.info("No directory specified in tex_file_path, using timestamped output directory", extra={
                            "tex_file_path": tex_file_path,
                            "default_directory": directory
                        })
                        # Update the file path to include the directory
                        tex_file_path = os.path.join(
                            directory, os.path.basename(tex_file_path))
                        self.logger.debug("Updated tex_file_path with directory", extra={
                            "new_tex_file_path": tex_file_path
                        })
                    else:
                        # Use current directory if no config
                        directory = os.getcwd()
                        self.logger.info("No directory specified in tex_file_path and no config available, using current directory", extra={
                            "tex_file_path": tex_file_path,
                            "default_directory": directory
                        })
                        # Update the file path to include the directory
                        tex_file_path = os.path.join(
                            directory, os.path.basename(tex_file_path))
                        self.logger.debug("Updated tex_file_path with current directory", extra={
                            "new_tex_file_path": tex_file_path
                        })

                # Validate input file
                if not os.path.exists(tex_file_path):
                    error_msg = f"LaTeX file not found: {tex_file_path}"
                    self.logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

                file_size = os.path.getsize(tex_file_path)
                self.logger.debug("LaTeX file validated", extra={
                    "tex_file_path": tex_file_path,
                    "file_size_bytes": file_size
                })

                # Prepare compilation parameters with safe working directory
                working_dir = self._get_safe_working_directory(tex_file_path)
                tex_filename = os.path.basename(tex_file_path)
                pdf_path = tex_file_path.replace('.tex', '.pdf')

                # DEBUG LOGGING: Add diagnostic information
                self.logger.debug("DEBUG: Compilation parameters", extra={
                    "working_directory": working_dir,
                    "working_directory_is_empty": working_dir == "",
                    "working_directory_is_none": working_dir is None,
                    "tex_filename": tex_filename,
                    "expected_pdf_path": pdf_path,
                    "current_working_directory": os.getcwd(),
                    "os_path_separator": os.sep,
                    "platform": sys.platform
                })

                # Check if pdflatex is available before attempting compilation
                pdflatex_available = self._check_pdflatex_availability()
                if not pdflatex_available:
                    error_msg = (
                        "pdflatex executable not found. Please install a LaTeX distribution "
                        "(such as MiKTeX, TeX Live, or MacTeX) and ensure it's in your system PATH. "
                        "For Windows: Download and install MiKTeX from https://miktex.org/download"
                    )
                    self.logger.error(error_msg, extra={
                        "platform": sys.platform,
                        "path_env": os.environ.get('PATH', 'PATH not set')
                    })
                    raise RuntimeError(error_msg)

                self.logger.info("Starting PDF compilation process", extra={
                    "working_directory": working_dir,
                    "tex_filename": tex_filename,
                    "expected_pdf_path": pdf_path,
                    "pdflatex_available": True
                })

                # Run pdflatex twice for proper references
                compilation_results = []
                for run_number in range(1, 3):
                    self.logger.debug("Executing pdflatex run", extra={
                        "run_number": run_number,
                        "command": f"pdflatex -interaction=nonstopmode {tex_filename}"
                    })

                    try:
                        # Prepare command with proper Windows handling
                        cmd = ['pdflatex',
                               '-interaction=nonstopmode', tex_filename]

                        # On Windows, we might need shell=True for some LaTeX installations
                        # but we'll try without first for security
                        shell = False
                        if sys.platform == 'win32':
                            # Try to detect if we need shell mode
                            import shutil
                            pdflatex_path = shutil.which('pdflatex')
                            if pdflatex_path and pdflatex_path.lower().endswith('.bat'):
                                shell = True
                                self.logger.debug("Using shell=True for .bat file execution", extra={
                                    "pdflatex_path": pdflatex_path
                                })

                        self.logger.debug("Executing pdflatex command", extra={
                            "command": cmd,
                            "shell": shell,
                            "working_directory": working_dir
                        })

                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            cwd=working_dir,
                            timeout=300,  # 5 minute timeout
                            shell=shell
                        )

                        compilation_results.append({
                            'run_number': run_number,
                            'return_code': result.returncode,
                            'stdout_length': len(result.stdout),
                            'stderr_length': len(result.stderr)
                        })

                        # Log compilation output for debugging
                        if result.returncode != 0:
                            self.logger.warning("pdflatex run completed with warnings/errors", extra={
                                "run_number": run_number,
                                "return_code": result.returncode,
                                "stdout_preview": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
                                "stderr_preview": result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                            })
                        else:
                            self.logger.debug("pdflatex run completed successfully", extra={
                                "run_number": run_number,
                                "return_code": result.returncode
                            })

                    except subprocess.TimeoutExpired as e:
                        ErrorEnhancer.log_error(
                            self.logger,
                            e,
                            context={
                                "operation": "pdflatex_execution",
                                "run_number": run_number,
                                "timeout_seconds": 300
                            }
                        )
                        raise
                    except FileNotFoundError as e:
                        # This specifically catches the case where pdflatex executable is not found
                        error_msg = (
                            f"pdflatex executable not found in system PATH. "
                            f"Please install a LaTeX distribution (MiKTeX, TeX Live, or MacTeX). "
                            f"For Windows: Download MiKTeX from https://miktex.org/download"
                        )
                        self.logger.error(error_msg, extra={
                            "platform": sys.platform,
                            "path_env": os.environ.get('PATH', 'PATH not set'),
                            "working_directory": working_dir,
                            "original_error": str(e)
                        })
                        raise RuntimeError(error_msg) from e
                    except PermissionError as e:
                        # Handle Windows permission issues
                        error_msg = (
                            f"Permission denied when executing pdflatex. "
                            f"Please check that you have sufficient permissions to execute programs "
                            f"and write to the output directory."
                        )
                        self.logger.error(error_msg, extra={
                            "working_directory": working_dir,
                            "original_error": str(e)
                        })
                        raise RuntimeError(error_msg) from e
                    except OSError as e:
                        # Handle Windows-specific OS errors
                        if "WinError 123" in str(e):
                            error_msg = (
                                f"Windows path syntax error. This usually happens when the working "
                                f"directory is invalid or contains unsupported characters. "
                                f"Working directory: '{working_dir}'"
                            )
                        else:
                            error_msg = (
                                f"System error when executing pdflatex: {str(e)}. "
                                f"Please ensure LaTeX is properly installed and accessible."
                            )
                        self.logger.error(error_msg, extra={
                            "working_directory": working_dir,
                            "platform": sys.platform,
                            "error_code": getattr(e, 'errno', 'unknown'),
                            "original_error": str(e)
                        })
                        raise RuntimeError(error_msg) from e
                    except Exception as e:
                        ErrorEnhancer.log_error(
                            self.logger,
                            e,
                            context={
                                "operation": "pdflatex_execution",
                                "run_number": run_number,
                                "working_directory": working_dir
                            }
                        )
                        raise

                # Check if PDF was generated
                if not os.path.exists(pdf_path):
                    error_msg = f"PDF compilation failed - output file not found: {pdf_path}"
                    self.logger.error(error_msg, extra={
                        "compilation_results": compilation_results
                    })
                    raise RuntimeError(error_msg)

                pdf_size = os.path.getsize(pdf_path)
                tracker.add_metadata("tex_file_size_bytes", file_size)
                tracker.add_metadata("pdf_file_size_bytes", pdf_size)
                tracker.add_metadata("compilation_runs",
                                     len(compilation_results))

                self.logger.info("PDF compilation completed successfully", extra={
                    "pdf_path": pdf_path,
                    "pdf_size_bytes": pdf_size,
                    "compilation_runs": len(compilation_results)
                })

                return f"PDF compiled successfully: {pdf_path}"

        except Exception as e:
            ErrorEnhancer.log_unexpected_error(
                self.logger,
                "PDF compilation",
                e,
                context={
                    "tex_file_path": tex_file_path,
                    "working_directory": os.path.dirname(tex_file_path) if 'tex_file_path' in locals() else None
                }
            )
            raise


def main():
    """
    Main function to run PDF generator tools from command line.

    Usage examples:
    python tools/pdf_generator.py latex-generator --state-json-path novel_state.json --output-tex-path output.tex
    python tools/pdf_generator.py pdf-compiler --tex-file-path output.tex
    """
    parser = argparse.ArgumentParser(description="PDF Generator Tools")
    subparsers = parser.add_subparsers(dest='tool', help='Tool to run')

    # LaTeX generator subcommand
    latex_parser = subparsers.add_parser(
        'latex-generator', help='Generate LaTeX from novel state')
    latex_parser.add_argument(
        '--state-json-path', required=True, help='Path to novel state JSON file')
    latex_parser.add_argument(
        '--output-tex-path', required=True, help='Output LaTeX file path')

    # PDF compiler subcommand
    pdf_parser = subparsers.add_parser(
        'pdf-compiler', help='Compile LaTeX to PDF')
    pdf_parser.add_argument(
        '--tex-file-path', required=True, help='Path to LaTeX file')

    args = parser.parse_args()

    if not args.tool:
        parser.print_help()
        sys.exit(1)

    try:
        if args.tool == 'latex-generator':
            tool = LaTeXGeneratorTool()
            result = tool._run(args.state_json_path, args.output_tex_path)
            print(result)
        elif args.tool == 'pdf-compiler':
            tool = PDFCompilerTool()
            result = tool._run(args.tex_file_path)
            print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
