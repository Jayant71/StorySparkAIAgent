from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from storyspark_logging.core.logger_factory import LoggerFactory
from storyspark_logging.core.performance_tracker import PerformanceTracker
from storyspark_logging.core.error_enhancer import ErrorEnhancer
import aiohttp
import asyncio
import os
from PIL import Image
import io
from typing import Optional, Any
import base64
import threading
import time
from typing import cast

# Module-level API lock shared across all CharacterImageGeneratorTool instances
_api_lock = threading.Lock()


class CharacterImageInput(BaseModel):
    character_description: str = Field(
        description="Detailed character description")
    scene_context: str = Field(description="Scene or context for the image")
    reference_image_path: Optional[str] = Field(
        default=None, description="Path to reference image for consistency")
    output_path: str = Field(description="Where to save the generated image")


class CharacterImageGeneratorTool(BaseTool):
    name: str = "CharacterImageGeneratorTool"
    description: str = "Generates consistent character images using Chutes API"
    args_schema: type[BaseModel] = CharacterImageInput
    logger: Any = None
    api_token: Optional[str] = None


    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory().get_logger("storyspark.tools.image")
        self.logger.info("CharacterImageGeneratorTool initialized")

        # Load API token from environment
        self.api_token = os.getenv("CHUTES_API_TOKEN")
        if not self.api_token:
            error_msg = "CHUTES_API_TOKEN environment variable not set"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.logger.info("Chutes API token loaded successfully")

    def _run(self, character_description: str, scene_context: str,
             reference_image_path: Optional[str] = None, output_path: Optional[str] = None) -> str:
        if not output_path:
            raise ValueError("output_path is required")

        self.logger.info("CharacterImageGeneratorTool._run() invoked", extra={
            "character_description_length": len(character_description),
            "scene_context_length": len(scene_context),
            "has_reference_image": reference_image_path is not None,
            "output_path": output_path
        })

        try:
            with PerformanceTracker("image_generation", self.logger) as tracker:
                # Modify output path to include images subdirectory if not already present
                output_dir = os.path.dirname(output_path)
                if os.path.basename(output_dir) != "images":
                    images_dir = os.path.join(output_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    output_path = os.path.join(
                        images_dir, os.path.basename(output_path))
                else:
                    # If "images" is already in the path, ensure the directory exists
                    os.makedirs(output_dir, exist_ok=True)

                # Create and validate prompt
                prompt = f"{character_description}, {scene_context}, children's book illustration style, highly detailed, vibrant colors"
                prompt_length = len(prompt)

                # Construct API request body - using only required parameters
                body = {
                    "model": "FLUX.1-dev",
                    "prompt": prompt,
                    "guidance_scale": 9.0,
                    "width": 512,
                    "height": 512,
                    "num_inference_steps": 30
                }

                headers = {
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json"
                }

                self.logger.info("Starting image generation via Chutes API", extra={
                    "api_url": "https://image.chutes.ai/generate",
                    "body": body
                })

                # Make async API call
                self.logger.info("Acquiring API lock for image generation")
                with _api_lock:
                    self.logger.info("API lock acquired, starting image generation")

                    async def generate_image():
                        async with aiohttp.ClientSession() as session:
                            self.logger.debug("Making API request", extra={
                                "url": "https://image.chutes.ai/generate",
                                "headers": {k: v if k != "Authorization" else "***REDACTED***" for k, v in headers.items()},
                                "request_body": body
                            })

                            async with session.post(
                                "https://image.chutes.ai/generate",
                                headers=headers,
                                json=body
                            ) as response:
                                self.logger.debug("API response received", extra={
                                    "status_code": response.status,
                                    "content_type": response.headers.get('Content-Type', 'Unknown'),
                                    "response_headers": dict(response.headers)
                                })

                                # Check for error status first
                                if response.status >= 400:
                                    # Read error response for debugging
                                    error_text = await response.text()
                                    self.logger.error("API returned error response", extra={
                                        "status_code": response.status,
                                        "error_response": error_text[:1000] + "..." if len(error_text) > 1000 else error_text
                                    })
                                    response.raise_for_status()

                                content_type = response.headers.get(
                                    'Content-Type', '')

                                # Handle direct image response (JPEG/PNG)
                                if content_type.startswith('image/'):
                                    self.logger.debug("Processing direct image response", extra={
                                        "content_type": content_type
                                    })
                                    image_bytes = await response.read()
                                    self.logger.debug("Successfully read image bytes", extra={
                                        "image_bytes_length": len(image_bytes)
                                    })
                                    return image_bytes

                                # Handle JSON response
                                elif content_type.startswith('application/json'):
                                    self.logger.debug("Processing JSON response")
                                    response_json = await response.json()
                                    self.logger.debug("Parsed JSON response", extra={
                                        "response_keys": list(response_json.keys()) if isinstance(response_json, dict) else "Not a dict"
                                    })

                                    # Handle different possible JSON response formats
                                    if 'image' in response_json:
                                        image_data = response_json['image']
                                    elif 'data' in response_json:
                                        image_data = response_json['data']
                                    elif isinstance(response_json, str):
                                        image_data = response_json
                                    else:
                                        self.logger.error("Unexpected JSON response format", extra={
                                            "response": response_json
                                        })
                                        raise ValueError(
                                            "Unexpected JSON response format")

                                    # Decode base64 image data to bytes
                                    try:
                                        image_bytes = base64.b64decode(image_data)
                                        self.logger.debug("Successfully decoded base64 image data", extra={
                                            "decoded_bytes_length": len(image_bytes)
                                        })
                                    except Exception as e:
                                        self.logger.error(f"Failed to decode base64 data: {e}", extra={
                                            "image_data_type": type(image_data),
                                            "image_data_length": len(image_data) if image_data else 0
                                        })
                                        raise

                                    return image_bytes

                                # Handle streaming response (original implementation)
                                else:
                                    self.logger.debug(
                                        "Processing streaming response")
                                    image_data = ""
                                    async for line in response.content:
                                        line = line.decode("utf-8").strip()
                                        self.logger.debug("Received streaming line", extra={
                                            "line": line[:200] + "..." if len(line) > 200 else line
                                        })
                                        if line.startswith("data: "):
                                            data = line[6:]
                                            if data == "[DONE]":
                                                break
                                            try:
                                                chunk = data.strip()
                                                if chunk:
                                                    image_data += chunk
                                            except Exception as e:
                                                self.logger.error(
                                                    f"Error parsing chunk: {e}")
                                                raise

                                    # Decode base64 image data to bytes
                                    try:
                                        image_bytes = base64.b64decode(image_data)
                                        self.logger.debug("Successfully decoded base64 image data", extra={
                                            "decoded_bytes_length": len(image_bytes)
                                        })
                                    except Exception as e:
                                        self.logger.error(f"Failed to decode base64 data: {e}", extra={
                                            "image_data_type": type(image_data),
                                            "image_data_length": len(image_data) if image_data else 0
                                        })
                                        raise

                                    return image_bytes

                    image_bytes = asyncio.run(generate_image())

                    self.logger.info("Image generation completed, releasing API lock")

                # Process image
                image = Image.open(io.BytesIO(image_bytes))

                # Save image
                try:
                    image.save(output_path)
                    image_size = os.path.getsize(output_path)
                except Exception as e:
                    ErrorEnhancer.log_error(
                        self.logger,
                        e,
                        context={
                            "operation": "image_saving",
                            "output_path": output_path,
                            "image_dimensions": f"{image.width}x{image.height}"
                        }
                    )
                    raise

                tracker.add_metadata("prompt_length", prompt_length)
                tracker.add_metadata("image_width", image.width)
                tracker.add_metadata("image_height", image.height)
                tracker.add_metadata("image_size_bytes", image_size)
                tracker.add_metadata(
                    "inference_steps", body.get("num_inference_steps", "N/A"))

                self.logger.info("Image generated and saved", extra={
                    "output_path": output_path,
                    "image_dimensions": f"{image.width}x{image.height}"
                })

                return f"Image generated and saved to: {output_path}"

        except Exception as e:
            ErrorEnhancer.log_unexpected_error(
                self.logger,
                "image generation",
                e,
                context={
                    "character_description_length": len(character_description),
                    "scene_context_length": len(scene_context),
                    "output_path": output_path
                }
            )
            raise


class ChapterImageInput(BaseModel):
    chapter_number: int = Field(description="Chapter number for image naming")
    chapter_content: str = Field(
        description="Complete chapter content with illustration tags")
    characters: str = Field(
        description="Character descriptions for consistency")
    output_directory: str = Field(
        description="Base output directory for saving images")


class ChapterImageGeneratorTool(BaseTool):
    name: str = "ChapterImageGeneratorTool"
    description: str = "Extracts illustration tags from chapter content and generates images for each strategic illustration opportunity"
    args_schema: type[BaseModel] = ChapterImageInput
    logger: Any = None

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory().get_logger("storyspark.tools.chapter_image")
        self.logger.info("ChapterImageGeneratorTool initialized")

    def _run(self, chapter_number: int, chapter_content: str, characters: str, output_directory: str) -> str:
        import re
        import json

        self.logger.info("ChapterImageGeneratorTool._run() invoked", extra={
            "chapter_number": chapter_number,
            "chapter_content_length": len(chapter_content),
            "characters_length": len(characters),
            "output_directory": output_directory
        })

        try:
            with PerformanceTracker("chapter_image_generation", self.logger) as tracker:
                # Extract illustration descriptions from chapter content
                illustration_pattern = r'\[ILLUSTRATION:\s*(.*?)\]'
                matches = re.findall(illustration_pattern,
                                     chapter_content, re.IGNORECASE | re.DOTALL)

                self.logger.debug("Extracted illustrations from chapter", extra={
                    "chapter_number": chapter_number,
                    "illustration_count": len(matches),
                    "illustrations": matches
                })

                if not matches:
                    self.logger.warning("No illustrations found in chapter content", extra={
                        "chapter_number": chapter_number
                    })
                    return json.dumps({
                        "chapter_number": chapter_number,
                        "illustrations_found": 0,
                        "images_generated": []
                    })

                # Parse characters for reference image paths
                self.logger.debug("Attempting to parse characters JSON", extra={
                    "characters_length": len(characters),
                    "characters_content": characters[:200] + "..." if len(characters) > 200 else characters,
                    "characters_is_empty": len(characters.strip()) == 0 if characters else True
                })
                
                characters_data = []
                character_refs = {}
                
                try:
                    # Check if characters is empty or None
                    if not characters or characters.strip() == "":
                        raise ValueError("Characters parameter is empty or None")
                    
                    # Try to parse as JSON first
                    try:
                        characters_data = json.loads(characters)
                        self.logger.debug("Successfully parsed characters as JSON", extra={
                            "character_count": len(characters_data)
                        })
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to extract from state file immediately
                        raise ValueError("Not valid JSON, loading from state file")
                    
                    # Build character references
                    for char in characters_data:
                        char_name = char.get('name', '').lower().replace(' ', '_')
                        ref_path = f"{output_directory}/images/{char_name}_reference.png"
                        if os.path.exists(ref_path):
                            character_refs[char['name']] = ref_path
                            self.logger.debug(
                                f"Found reference image for {char['name']}: {ref_path}")
                                
                except Exception as e:
                    self.logger.warning("Could not parse characters JSON, attempting to load from state file", extra={
                        "error": str(e),
                        "characters_received": characters[:100] + "..." if characters and len(characters) > 100 else characters
                    })
                    
                    # Fallback: Try to load characters from the state file
                    characters_data = []
                    character_refs = {}
                    
                    try:
                        # Look for novel_state.json in the output directory
                        state_file_path = None
                        if output_directory:
                            # Check if output_directory already contains a timestamp
                            if "novel_state.json" in output_directory:
                                state_file_path = output_directory
                            else:
                                # Look for novel_state.json in the output directory
                                import glob
                                state_files = glob.glob(f"{output_directory}/**/novel_state.json", recursive=True)
                                if state_files:
                                    state_file_path = state_files[0]
                                else:
                                    # Try direct path in output_directory
                                    direct_path = os.path.join(output_directory, "novel_state.json")
                                    if os.path.exists(direct_path):
                                        state_file_path = direct_path
                        
                        if state_file_path and os.path.exists(state_file_path):
                            self.logger.info(f"Loading characters from state file: {state_file_path}")
                            with open(state_file_path, 'r', encoding='utf-8') as f:
                                state_data = json.load(f)
                                characters_data = state_data.get('characters', [])
                                self.logger.info(f"Successfully loaded {len(characters_data)} characters from state file")
                                
                                # Rebuild character refs with loaded data
                                for char in characters_data:
                                    char_name = char.get('name', '').lower().replace(' ', '_')
                                    # Check multiple possible reference image paths
                                    ref_paths = [
                                        f"{output_directory}/images/{char_name}_reference.png",
                                        f"{output_directory}/images/{char['name']}_reference.png",
                                        f"{output_directory}/{char_name}_reference.png"
                                    ]
                                    for ref_path in ref_paths:
                                        if os.path.exists(ref_path):
                                            character_refs[char['name']] = ref_path
                                            self.logger.debug(f"Found reference image for {char['name']}: {ref_path}")
                                            break
                        else:
                            self.logger.warning("No state file found, proceeding without character data", extra={
                                "output_directory": output_directory,
                                "tried_paths": [
                                    f"{output_directory}/**/novel_state.json",
                                    f"{output_directory}/novel_state.json"
                                ]
                            })
                            
                    except Exception as fallback_error:
                        self.logger.error("Failed to load characters from state file", extra={
                            "fallback_error": str(fallback_error),
                            "output_directory": output_directory
                        })

                generated_images = []

                # Create single image tool instance for sequential processing
                image_tool = CharacterImageGeneratorTool()

                # Generate image for each illustration
                for i, illustration_desc in enumerate(matches, 1):
                    try:
                        self.logger.info(f"Generating image {i} for chapter {chapter_number}", extra={
                            "chapter_number": chapter_number,
                            "illustration_number": i,
                            "illustration_description": illustration_desc[:100] + "..." if len(illustration_desc) > 100 else illustration_desc
                        })

                        # Create enhanced prompt with character context
                        enhanced_prompt = self._create_scene_prompt(
                            illustration_desc, characters_data)

                        # Determine reference image (use first character's reference if available)
                        reference_image = None
                        if character_refs:
                            # Try to find a character mentioned in the illustration
                            for char_name, ref_path in character_refs.items():
                                if char_name.lower() in illustration_desc.lower():
                                    reference_image = ref_path
                                    self.logger.debug(
                                        f"Using reference image for {char_name}: {ref_path}")
                                    break
                            # If no specific character found, use the first available reference for consistency
                            if not reference_image:
                                reference_image = list(
                                    character_refs.values())[0]
                                self.logger.debug(
                                    f"Using default reference image: {reference_image}")

                        # Generate the image
                        output_filename = f"chapter_{chapter_number}_illustration_{i}.png"
                        
                        # Ensure images directory exists and create proper path
                        images_dir = os.path.join(output_directory, "images")
                        os.makedirs(images_dir, exist_ok=True)
                        output_path = os.path.join(images_dir, output_filename)

                        result = image_tool._run(
                            character_description=enhanced_prompt,
                            scene_context="",  # Scene context is included in the enhanced prompt
                            reference_image_path=reference_image,
                            output_path=output_path
                        )

                        generated_images.append({
                            "filename": output_filename,
                            "description": illustration_desc.strip(),
                            "path": output_path
                        })
                        
                        # Update the state file with the generated image
                        self._update_chapter_images_in_state(
                            output_directory, chapter_number, generated_images)

                        # Add delay between API calls to respect rate limits
                        if i < len(matches):
                            self.logger.info("Waiting 1 second before next API call to respect rate limits")
                            time.sleep(1)

                        self.logger.info(f"Successfully generated image {i} for chapter {chapter_number}", extra={
                            "chapter_number": chapter_number,
                            "illustration_number": i,
                            "output_path": output_path,
                            "used_reference": reference_image is not None
                        })

                    except Exception as e:
                        self.logger.error(f"Failed to generate image {i} for chapter {chapter_number}", extra={
                            "chapter_number": chapter_number,
                            "illustration_number": i,
                            "error": str(e)
                        })
                        continue

                tracker.add_metadata("chapter_number", chapter_number)
                tracker.add_metadata("illustrations_found", len(matches))
                tracker.add_metadata("images_generated", len(generated_images))

                self.logger.info("Chapter image generation completed", extra={
                    "chapter_number": chapter_number,
                    "total_illustrations": len(matches),
                    "successful_generations": len(generated_images)
                })

                return json.dumps({
                    "chapter_number": chapter_number,
                    "illustrations_found": len(matches),
                    "images_generated": generated_images
                })

        except Exception as e:
            ErrorEnhancer.log_unexpected_error(
                self.logger,
                "chapter image generation",
                e,
                context={
                    "chapter_number": chapter_number,
                    "chapter_content_length": len(chapter_content),
                    "output_directory": output_directory
                }
            )
            raise

    def _create_scene_prompt(self, illustration_desc: str, characters: list) -> str:
        """Create an enhanced prompt for scene illustration including character details"""
        prompt_parts = [illustration_desc]

        # Add character details that might be relevant to the scene
        relevant_chars = []
        for char in characters:
            char_name = char.get('name', '')
            if char_name.lower() in illustration_desc.lower():
                relevant_chars.append(char)

        # If multiple characters, describe them together
        if len(relevant_chars) > 1:
            char_descriptions = []
            for char in relevant_chars:
                appearance = char.get('appearance', char.get('description', ''))
                char_descriptions.append(f"{char['name']}: {appearance}")
            prompt_parts.append(f"Characters: {', '.join(char_descriptions)}")
        elif relevant_chars:
            char = relevant_chars[0]
            appearance = char.get('appearance', char.get('description', ''))
            prompt_parts.append(f"{char['name']}: {appearance}")

        prompt_parts.append(
            "children's book illustration style, highly detailed, vibrant colors, storybook art")

        return ", ".join(prompt_parts)

    def _update_chapter_images_in_state(self, output_directory: str, chapter_number: int, generated_images: list):
        """Update the novel state file with generated images for a chapter"""
        import json
        import glob
        
        # Find the state file with improved path resolution
        state_file_path = None
        if output_directory:
            # Try multiple approaches to find the state file
            search_patterns = [
                f"{output_directory}/novel_state.json",
                f"{output_directory}/**/novel_state.json",
                f"{output_directory}/../novel_state.json"
            ]
            
            for pattern in search_patterns:
                state_files = glob.glob(pattern, recursive=True)
                if state_files:
                    state_file_path = state_files[0]
                    self.logger.debug(f"Found state file using pattern: {pattern}")
                    break
        
        if not state_file_path:
            self.logger.warning("Could not find state file to update with generated images", extra={
                "output_directory": output_directory,
                "searched_patterns": [
                    f"{output_directory}/novel_state.json",
                    f"{output_directory}/**/novel_state.json",
                    f"{output_directory}/../novel_state.json"
                ]
            })
            return False
        
        # Load the current state with error handling
        try:
            with open(state_file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
        except json.JSONDecodeError as jde:
            self.logger.error(f"Invalid JSON in state file: {jde}")
            return False
        except Exception as fe:
            self.logger.error(f"Failed to read state file: {fe}")
            return False
        
        # Find the chapter and update its images
        chapters = state_data.get('chapters', [])
        chapter_found = False
        for chapter in chapters:
            if chapter.get('chapter_number') == chapter_number:
                # Update the images field - ensure it's a list of dicts with proper structure
                chapter['images'] = generated_images
                chapter_found = True
                self.logger.info(f"Updated chapter {chapter_number} with {len(generated_images)} generated images")
                break
        
        if not chapter_found:
            self.logger.warning(f"Chapter {chapter_number} not found in state file", extra={
                "available_chapters": [ch.get('chapter_number') for ch in chapters]
            })
            return False
        
        # Save the updated state with backup
        try:
            import shutil
            # Create backup before saving
            backup_path = state_file_path + '.backup'
            if os.path.exists(state_file_path):
                shutil.copy2(state_file_path, backup_path)
            
            with open(state_file_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug("Successfully updated state file with generated images")
            return True
            
        except Exception as se:
            self.logger.error(f"Failed to save updated state file: {se}")
            # Attempt to restore from backup if it exists
            if 'backup_path' in locals() and os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, state_file_path)
                    self.logger.info("Restored state file from backup after save failure")
                except:
                    self.logger.error("Failed to restore from backup")
            return False


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    tool = CharacterImageGeneratorTool()
    result = tool._run(
        character_description="A brave young knight with shining armor",
        scene_context="Standing in an enchanted forest",
        output_path="output/test_image.png"
    )
    print(result)
