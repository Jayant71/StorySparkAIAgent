from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, ForwardRef, Union
from datetime import datetime
import uuid
import re
import logging

logger = logging.getLogger(__name__)


def safe_get(data, key, default=None):
    """Unified helper for type-safe access to both dict and object attributes"""
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


class IllustrationReference(BaseModel):
    """Enhanced schema for illustration references with rich metadata"""

    # Basic identification
    id: str = Field(description="Unique identifier for the illustration")
    description: str = Field(description="Brief description of the illustration scene")

    # Content positioning
    content_position: Optional[int] = Field(default=None, description="Character position in content where illustration should appear")
    paragraph_number: Optional[int] = Field(default=None, description="Paragraph number where illustration is referenced")
    context_text: Optional[str] = Field(default=None, description="Surrounding text context for the illustration")

    # Illustration metadata
    size: Optional[str] = Field(default="medium", description="Illustration size (small, medium, large, full-page)")
    style: Optional[str] = Field(default=None, description="Artistic style (watercolor, cartoon, realistic, etc.)")
    mood: Optional[str] = Field(default=None, description="Emotional mood (happy, mysterious, adventurous, etc.)")

    # Character and scene focus
    character_focus: List[str] = Field(default_factory=list, description="List of character names featured in illustration")
    scene_type: Optional[str] = Field(default=None, description="Type of scene (action, dialogue, establishing, emotional)")
    setting: Optional[str] = Field(default=None, description="Location/setting of the illustration")

    # Multi-character scene support
    multi_character_scene: Optional["MultiCharacterScene"] = Field(default=None, description="Detailed multi-character scene configuration")
    character_interactions: List[str] = Field(default_factory=list, description="Specific interactions between characters")

    # Generation parameters
    generation_params: Optional["ImageGenerationParameters"] = Field(default=None, description="AI generation parameters")
    prompt_enhancement: Optional[str] = Field(default=None, description="Enhanced prompt for better generation")
    character_references: Dict[str, str] = Field(default_factory=dict, description="Character name to reference image mapping")

    # File references
    image_path: Optional[str] = Field(default=None, description="Path to generated image file")
    thumbnail_path: Optional[str] = Field(default=None, description="Path to thumbnail version")

    # Status and metadata
    status: Optional[str] = Field(default="pending", description="Generation status (pending, generating, completed, failed)")
    generated_at: Optional[str] = Field(default=None, description="Timestamp when image was generated")
    generation_attempts: Optional[int] = Field(default=0, description="Number of generation attempts")

    @validator('id', pre=True)
    def generate_id_if_missing(cls, v):
        if not v:
            return str(uuid.uuid4())
        return v

    @validator('size')
    def validate_size(cls, v):
        if v and v not in ['small', 'medium', 'large', 'full-page']:
            raise ValueError("Size must be one of: small, medium, large, full-page")
        return v

    @validator('status')
    def validate_status(cls, v):
        if v and v not in ['pending', 'generating', 'completed', 'failed']:
            raise ValueError("Status must be one of: pending, generating, completed, failed")
        return v


class CharacterAppearance(BaseModel):
    """Flexible nested appearance model that adapts to different character types"""

    # Core attributes applicable to all characters
    species: Optional[str] = Field(default=None, description="Species type (human, dog, cat, owl, etc.)")
    primary_colors: List[str] = Field(default_factory=list, description="Main colors of the character")
    secondary_colors: List[str] = Field(default_factory=list, description="Secondary/accent colors")
    size_category: Optional[str] = Field(default=None, description="Size category (tiny, small, medium, large, giant)")

    # Physical attributes
    body_type: Optional[str] = Field(default=None, description="Body shape/type")
    fur_skin_texture: Optional[str] = Field(default=None, description="Fur, skin, or surface texture")
    eye_color: Optional[str] = Field(default=None, description="Eye color")
    special_features: List[str] = Field(default_factory=list, description="Distinguishing features (scars, markings, etc.)")

    # Clothing and accessories
    clothing: List[str] = Field(default_factory=list, description="Items of clothing")
    accessories: List[str] = Field(default_factory=list, description="Accessories, jewelry, tools")

    # Character-specific attributes (flexible)
    species_attributes: Dict[str, Any] = Field(default_factory=dict, description="Species-specific attributes")
    custom_attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom attributes for unique characters")

    # Visual style for generation
    art_style_notes: Optional[str] = Field(default=None, description="Specific art style notes for this character")
    consistency_keywords: List[str] = Field(default_factory=list, description="Keywords for maintaining visual consistency")

    @validator('size_category')
    def validate_size_category(cls, v):
        if v and v not in ['tiny', 'small', 'medium', 'large', 'giant']:
            raise ValueError("Size category must be one of: tiny, small, medium, large, giant")
        return v


class CharacterState(BaseModel):
    """Timeline-based character state for tracking evolution across chapters"""

    state_id: str = Field(description="Unique identifier for this character state")
    character_name: str = Field(description="Name of the character this state belongs to")

    # Timeline information
    applicable_from_chapter: int = Field(description="Chapter number where this state becomes active")
    applicable_to_chapter: Optional[int] = Field(default=None, description="Chapter number where this state ends (None for current)")
    story_event_trigger: Optional[str] = Field(default=None, description="Story event that triggers this state change")

    # State changes
    appearance_changes: Optional[CharacterAppearance] = Field(default=None, description="Modified appearance attributes")
    personality_changes: Optional[str] = Field(default=None, description="Changes to personality traits")
    temporary_features: List[str] = Field(default_factory=list, description="Temporary features (injuries, costumes, etc.)")

    # Metadata
    change_reason: Optional[str] = Field(default=None, description="Reason for the character change")
    created_at: Optional[str] = Field(default=None, description="Timestamp when this state was created")

    @validator('state_id', pre=True)
    def generate_state_id_if_missing(cls, v):
        if not v:
            return str(uuid.uuid4())
        return v

    @validator('applicable_to_chapter')
    def validate_chapter_range(cls, v, values):
        if v and 'applicable_from_chapter' in values:
            if v < values['applicable_from_chapter']:
                raise ValueError("applicable_to_chapter must be greater than or equal to applicable_from_chapter")
        return v


class CharacterRelationship(BaseModel):
    """Model for tracking emotional and spatial relationships between characters"""

    character_a: str = Field(description="First character name")
    character_b: str = Field(description="Second character name")

    # Emotional relationship
    relationship_type: str = Field(description="Type of relationship (friend, enemy, family, mentor, etc.)")
    relationship_strength: Optional[float] = Field(default=None, ge=0, le=10, description="Strength of relationship (0-10)")
    emotional_state: Optional[str] = Field(default=None, description="Current emotional state between characters")
    relationship_history: List[str] = Field(default_factory=list, description="Key events in relationship development")

    # Spatial preferences for scenes
    default_positioning: Optional[str] = Field(default=None, description="Typical positioning (side-by-side, opposite, etc.)")
    proximity_preference: Optional[str] = Field(default=None, description="Preferred proximity (close, medium, distant)")
    interaction_patterns: List[str] = Field(default_factory=list, description="Common interaction patterns")

    # Scene-specific guidance
    scene_prompts: List[str] = Field(default_factory=list, description="Prompt suggestions for scenes with both characters")

    @validator('relationship_type')
    def validate_relationship_type(cls, v):
        allowed_types = ['friend', 'enemy', 'family', 'mentor', 'student', 'rival', 'neutral', 'romantic', 'ally']
        if v not in allowed_types:
            raise ValueError(f"Relationship type must be one of: {allowed_types}")
        return v


class CharacterPosition(BaseModel):
    """Position and role of a character in a multi-character scene"""

    character_name: str = Field(description="Name of the character")
    spatial_position: Optional[str] = Field(default=None, description="Position in scene (left, right, center, foreground, background)")
    distance_from_viewer: Optional[str] = Field(default=None, description="Distance from viewer (close, medium, far)")
    pose: Optional[str] = Field(default=None, description="Suggested pose or action")
    emotional_state: Optional[str] = Field(default=None, description="Emotional state in this scene")
    focus_level: Optional[str] = Field(default=None, description="Focus level (main, secondary, background)")

    @validator('spatial_position')
    def validate_spatial_position(cls, v):
        if v and v not in ['left', 'right', 'center', 'foreground', 'background', 'top', 'bottom']:
            raise ValueError("Spatial position must be one of: left, right, center, foreground, background, top, bottom")
        return v

    @validator('distance_from_viewer')
    def validate_distance(cls, v):
        if v and v not in ['close', 'medium', 'far']:
            raise ValueError("Distance from viewer must be one of: close, medium, far")
        return v

    @validator('focus_level')
    def validate_focus_level(cls, v):
        if v and v not in ['main', 'secondary', 'background']:
            raise ValueError("Focus level must be one of: main, secondary, background")
        return v


class MultiCharacterScene(BaseModel):
    """Model for handling multi-character scenes with relationship mapping"""

    scene_id: str = Field(description="Unique identifier for the scene configuration")
    scene_description: str = Field(description="Description of the scene")

    # Character positioning
    character_positions: List[CharacterPosition] = Field(description="Positions and roles of all characters")
    character_interactions: List[str] = Field(default_factory=list, description="Interactions between characters")

    # Scene composition
    setting: Optional[str] = Field(default=None, description="Scene setting/background")
    mood: Optional[str] = Field(default=None, description="Overall mood of the scene")
    action_level: Optional[str] = Field(default=None, description="Level of action (static, moderate, dynamic)")

    # Visual composition
    composition_type: Optional[str] = Field(default=None, description="Composition type (group shot, dialogue, action, etc.)")
    camera_angle: Optional[str] = Field(default=None, description="Suggested camera angle")
    lighting: Optional[str] = Field(default=None, description="Lighting description")

    # Generation guidance
    prompt_enhancements: List[str] = Field(default_factory=list, description="Additional prompt elements for generation")
    negative_prompts: List[str] = Field(default_factory=list, description="Elements to avoid in generation")

    @validator('scene_id', pre=True)
    def generate_scene_id_if_missing(cls, v):
        if not v:
            return str(uuid.uuid4())
        return v

    @validator('action_level')
    def validate_action_level(cls, v):
        if v and v not in ['static', 'moderate', 'dynamic']:
            raise ValueError("Action level must be one of: static, moderate, dynamic")
        return v


class CharacterReference(BaseModel):
    """Model for tracking character reference images and consistency parameters"""

    character_name: str = Field(description="Name of the character")
    reference_image_path: Optional[str] = Field(default=None, description="Path to primary reference image")

    # Reference image variants
    reference_variants: Dict[str, str] = Field(default_factory=dict, description="Alternative reference images for different poses/moods")

    # Consistency parameters
    seed_value: Optional[int] = Field(default=None, description="Seed value for consistent generation")
    guidance_scale: Optional[float] = Field(default=None, description="Guidance scale for this character")
    generation_params: Dict[str, Any] = Field(default_factory=dict, description="Character-specific generation parameters")

    # Visual consistency keywords
    core_keywords: List[str] = Field(default_factory=list, description="Core keywords that must be included")
    style_keywords: List[str] = Field(default_factory=list, description="Style-specific keywords")
    negative_keywords: List[str] = Field(default_factory=list, description="Keywords to avoid")

    # Usage tracking
    usage_count: int = Field(default=0, description="Number of times this reference has been used")
    last_used: Optional[str] = Field(default=None, description="Last usage timestamp")
    success_rate: Optional[float] = Field(default=None, description="Success rate of generations with this reference")


class ImageGenerationParameters(BaseModel):
    """Model for consistent visual style parameters across character images"""

    # Base generation parameters
    model: str = Field(default="FLUX.1-dev", description="AI model to use for generation")
    width: int = Field(default=512, description="Image width")
    height: int = Field(default=512, description="Image height")
    guidance_scale: float = Field(default=9.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(default=30, description="Number of inference steps")

    # Style parameters
    art_style: str = Field(default="children's book illustration", description="Art style for images")
    quality_level: str = Field(default="high", description="Quality level (low, medium, high, ultra)")
    color_palette: Optional[str] = Field(default=None, description="Color palette preference")
    lighting_style: Optional[str] = Field(default=None, description="Lighting style preference")

    # Consistency parameters
    style_seed: Optional[int] = Field(default=None, description="Seed for style consistency")
    character_consistency_weight: float = Field(default=0.8, description="Weight for character consistency")
    scene_consistency_weight: float = Field(default=0.6, description="Weight for scene consistency")

    # Content parameters
    include_background: bool = Field(default=True, description="Whether to include detailed backgrounds")
    background_complexity: str = Field(default="medium", description="Background complexity (simple, medium, complex)")
    character_focus: str = Field(default="balanced", description="Character focus level (character, balanced, scene)")

    # Enhancement parameters
    auto_enhance_prompt: bool = Field(default=True, description="Whether to automatically enhance prompts")
    include_emotional_keywords: bool = Field(default=True, description="Whether to include emotional keywords")
    include_action_keywords: bool = Field(default=True, description="Whether to include action keywords")

    @validator('quality_level')
    def validate_quality_level(cls, v):
        allowed_levels = ['low', 'medium', 'high', 'ultra']
        if v not in allowed_levels:
            raise ValueError(f"Quality level must be one of: {allowed_levels}")
        return v


class CharacterValidationRules:
    """Comprehensive validation rules for character models"""

    @staticmethod
    def validate_character_consistency(character: "Character", novel_state: "NovelState") -> List[str]:
        """Validate character consistency across the novel"""
        errors = []

        # Check for duplicate names
        names = [char.name for char in novel_state.characters]
        if names.count(character.name) > 1:
            errors.append(f"Duplicate character name: {character.name}")

        # Validate character states timeline
        for i, state in enumerate(character.character_states):
            for j, other_state in enumerate(character.character_states):
                if i != j:
                    if (state.applicable_from_chapter >= other_state.applicable_from_chapter and
                        state.applicable_from_chapter <= (other_state.applicable_to_chapter or float('inf'))):
                        errors.append(f"Overlapping character states: {state.state_id} and {other_state.state_id}")

        # Validate relationship consistency
        for rel in character.relationships:
            if rel.character_a not in names or rel.character_b not in names:
                errors.append(f"Relationship references non-existent character: {rel.character_a}/{rel.character_b}")

        return errors

    @staticmethod
    def validate_scene_composition(scene: "MultiCharacterScene", characters: List["Character"]) -> List[str]:
        """Validate multi-character scene composition"""
        errors = []
        character_names = [char.name for char in characters]

        # Check all characters in scene exist
        for pos in scene.character_positions:
            if pos.character_name not in character_names:
                errors.append(f"Scene references non-existent character: {pos.character_name}")

        # Validate spatial positioning
        positions = [pos.spatial_position for pos in scene.character_positions if pos.spatial_position]
        if positions.count("center") > 1:
            errors.append("Multiple characters positioned at center")

        return errors

    @validator('background_complexity')
    def validate_background_complexity(cls, v):
        allowed_levels = ['simple', 'medium', 'complex']
        if v not in allowed_levels:
            raise ValueError(f"Background complexity must be one of: {allowed_levels}")
        return v

    @validator('character_focus')
    def validate_character_focus(cls, v):
        allowed_levels = ['character', 'balanced', 'scene']
        if v not in allowed_levels:
            raise ValueError(f"Character focus must be one of: {allowed_levels}")
        return v


# Update forward references for models that reference others defined later
IllustrationReference.update_forward_refs()
CharacterState.update_forward_refs()
CharacterRelationship.update_forward_refs()
CharacterPosition.update_forward_refs()
MultiCharacterScene.update_forward_refs()
CharacterReference.update_forward_refs()
ImageGenerationParameters.update_forward_refs()


class Character(BaseModel):
    """Enhanced schema for story characters with comprehensive tracking"""

    # Basic information
    name: str = Field(description="Character's name")
    description: str = Field(description="Detailed description of the character")
    age: Optional[int] = Field(default=None, description="Character's age if applicable")
    role: Optional[str] = Field(default=None, description="Character's role in the story")

    # Enhanced personality and background
    personality: Optional[str] = Field(default=None, description="Character's personality traits")
    background: Optional[str] = Field(default=None, description="Character's background story")

    # Enhanced appearance system (now optional with smart defaults)
    base_appearance: Optional[CharacterAppearance] = Field(default=None, description="Base appearance of the character")

    # Character evolution tracking
    character_states: List[CharacterState] = Field(default_factory=list, description="Timeline of character changes")

    # Reference tracking (now optional with smart defaults)
    reference_images: Optional[CharacterReference] = Field(default=None, description="Reference image information")

    # Relationships
    relationships: List[CharacterRelationship] = Field(default_factory=list, description="Relationships with other characters")

    # Story integration
    first_appearance_chapter: Optional[int] = Field(default=None, description="Chapter where character first appears")
    character_arc_summary: Optional[str] = Field(default=None, description="Summary of character development arc")

    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Character name cannot be empty")
        return v.strip()

    @root_validator(pre=True)
    def populate_missing_fields(cls, values):
        """Auto-populate base_appearance and reference_images from description if missing"""
        # Extract basic fields
        name = values.get('name', '')
        description = values.get('description', '')
        appearance_text = values.get('appearance', '')  # Legacy field support

        # Auto-populate base_appearance if missing or invalid
        base_appearance = values.get('base_appearance')
        needs_reparse = False
        
        if not base_appearance:
            needs_reparse = True
        elif isinstance(base_appearance, dict):
            # Check if base_appearance is effectively empty or has missing critical data
            if (not base_appearance.get('species') and
                not base_appearance.get('primary_colors')):
                needs_reparse = True
        
        if needs_reparse:
            # Combine description and legacy appearance field for better parsing
            combined_text = f"{description} {appearance_text}".strip()
            values['base_appearance'] = parse_appearance_text(combined_text)

        # Auto-populate reference_images if missing or invalid
        reference_images = values.get('reference_images')
        if not reference_images or (isinstance(reference_images, dict) and not reference_images):
            values['reference_images'] = cls._generate_character_reference(name, description)

        return values

    @classmethod
    def _generate_character_reference(cls, name: str, description: str) -> dict:
        """Generate CharacterReference from character name and description"""
        return {
            'character_name': name,
            'reference_image_path': None,  # Will be set later when images are generated
            'reference_variants': {},
            'seed_value': None,
            'guidance_scale': None,
            'generation_params': {},
            'core_keywords': extract_keywords_from_description(description),
            'style_keywords': ['childrens book illustration', 'storybook art'],
            'negative_keywords': [],
            'usage_count': 0,
            'last_used': None,
            'success_rate': None
        }

    def get_appearance_for_chapter(self, chapter_number: int) -> CharacterAppearance:
        """Get character appearance for a specific chapter, considering evolution"""
        current_appearance = self.base_appearance

        # Apply any relevant state changes
        for state in self.character_states:
            if (state.applicable_from_chapter <= chapter_number and
                (state.applicable_to_chapter is None or state.applicable_to_chapter >= chapter_number)):
                if state.appearance_changes:
                    # Merge appearance changes with base appearance
                    current_appearance = self._merge_appearance(current_appearance, state.appearance_changes)

        return current_appearance

    def _merge_appearance(self, base: CharacterAppearance, changes: CharacterAppearance) -> CharacterAppearance:
        """Merge appearance changes with base appearance"""
        # Create a merged appearance starting with base
        merged = base.copy()

        # Override non-null fields from changes
        for field_name, field_value in changes.dict(exclude_unset=True).items():
            if field_value is not None:
                setattr(merged, field_name, field_value)

        # Handle list fields specially - extend rather than replace
        list_fields = ['primary_colors', 'secondary_colors', 'special_features',
                      'clothing', 'accessories', 'consistency_keywords']
        for field_name in list_fields:
            changes_list = getattr(changes, field_name, [])
            if changes_list:
                base_list = getattr(merged, field_name, [])
                # Remove duplicates while preserving order
                combined = base_list + [item for item in changes_list if item not in base_list]
                setattr(merged, field_name, combined)

        # Handle dict fields specially - update rather than replace
        dict_fields = ['species_attributes', 'custom_attributes']
        for field_name in dict_fields:
            changes_dict = getattr(changes, field_name, {})
            if changes_dict:
                base_dict = getattr(merged, field_name, {})
                base_dict.update(changes_dict)
                setattr(merged, field_name, base_dict)

        return merged


class ImageData(BaseModel):
    """Model for image data with filename, path, and description"""
    filename: str
    path: Optional[str] = None
    description: Optional[str] = None


class Chapter(BaseModel):
    """Enhanced schema for chapter in novel state with improved illustration references"""
    chapter_number: int = Field(description="Chapter number")
    title: str = Field(description="Chapter title")
    content: Optional[str] = Field(default=None, description="Chapter text content")
    summary: Optional[str] = Field(default=None, description="Brief summary of the chapter")
    word_count: Optional[int] = Field(default=None, description="Word count of the chapter")

    # Enhanced illustration fields
    illustrations: List[IllustrationReference] = Field(default_factory=list, description="Enhanced illustration references with metadata")
    images: List[Union[str, ImageData]] = Field(default_factory=list, description="List of image filenames or image objects")

    @validator('illustrations', pre=True)
    def migrate_legacy_illustrations(cls, v):
        """Migrate legacy string illustrations to enhanced format"""
        if isinstance(v, list) and v and isinstance(v[0], str):
            # Convert legacy string format to enhanced format
            enhanced_illustrations = []
            for i, desc in enumerate(v):
                enhanced_illustrations.append(IllustrationReference(
                    id=f"legacy_{i+1}",
                    description=desc,
                    status="pending"
                ))
            return enhanced_illustrations
        return v

    @validator('chapter_number', pre=True)
    def parse_chapter_number(cls, v):
        """Handle both int and string chapter numbers"""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return 0
        return v


# Migration and validation utilities
def migrate_novel_state_data(novel_state_data: dict) -> dict:
    """Migrate legacy novel state data to handle various format issues"""

    # Migrate characters
    characters = safe_get(novel_state_data, 'characters')
    if characters:
        logger.debug(f"Migrating {len(characters)} characters")
        migrated_characters = []
        for i, legacy_char in enumerate(characters):
            logger.debug(f"Migrating character {i}: type={type(legacy_char)}")
            try:
                migrated_char = migrate_legacy_character(legacy_char)
                migrated_characters.append(migrated_char)
            except Exception as e:
                # New migration logic
                try:
                    # Handle both dict and Pydantic model inputs
                    name = safe_get(legacy_char, 'name', 'Unknown')
                    logger.debug(f"Failed to migrate character {name}: {e}")
                except Exception as print_error:
                    logger.debug(f"Failed to get character name during migration: {print_error}")
                migrated_characters.append(legacy_char)

        novel_state_data['characters'] = migrated_characters

    # Handle chapters field - convert empty dict to empty list
    chapters = safe_get(novel_state_data, 'chapters')
    if chapters is not None:
        if isinstance(chapters, dict) and not chapters:
            # Empty dict should be empty list
            novel_state_data['chapters'] = []
        elif isinstance(chapters, dict):
            # If it's a non-empty dict, try to convert to list format
            # This handles cases where chapters might be stored as {chapter_id: chapter_data}
            chapters_list = []
            for chapter_id, chapter_data in chapters.items():
                if isinstance(chapter_data, dict):
                    # Ensure chapter has chapter_number
                    if safe_get(chapter_data, 'chapter_number') is None and safe_get(chapter_data, 'number') is not None:
                        chapter_data['chapter_number'] = chapter_data['number']
                        del chapter_data['number']
                    chapters_list.append(chapter_data)
            novel_state_data['chapters'] = chapters_list

    # Migrate images in chapters to support both string filenames and ImageData objects
    chapters = safe_get(novel_state_data, 'chapters', [])
    if isinstance(chapters, list):
        logger.debug(f"Migrating images in {len(chapters)} chapters")
        for i, chapter in enumerate(chapters):
            logger.debug(f"Migrating chapter {i}: type={type(chapter)}")
            images = safe_get(chapter, 'images')
            if images:
                migrated_images = []
                for img in images:
                    if isinstance(img, str):
                        migrated_images.append(img)
                    elif isinstance(img, dict):
                        # Convert legacy image objects to new format
                        migrated_images.append(ImageData(**img))
                chapter['images'] = migrated_images

    return novel_state_data

def migrate_legacy_character(legacy_data: Union[dict, Character]) -> Character:
    """Migrate legacy character data to the current schema."""
    logger.debug(f"Migrating character: type={type(legacy_data)}")
    # Handle both dict and Character inputs
    if isinstance(legacy_data, dict):
        name = safe_get(legacy_data, 'name', '')
        description = safe_get(legacy_data, 'description', '')
        age = safe_get(legacy_data, 'age')
        role = safe_get(legacy_data, 'role')
        personality = safe_get(legacy_data, 'personality')
        background = safe_get(legacy_data, 'background')
        appearance_text = safe_get(legacy_data, 'appearance', '')
        base_appearance = parse_appearance_text(appearance_text)
        character_states = []
        reference_images = {
            'character_name': name,
            'reference_image_path': f"images/{name.lower().replace(' ', '_')}_reference.png",
            'reference_variants': {},
            'core_keywords': extract_keywords_from_description(description + ' ' + appearance_text),
            'style_keywords': ['childrens book illustration', 'storybook art'],
            'usage_count': 0
        }
        relationships = []
        first_appearance_chapter = None
        character_arc_summary = None
    else:
        name = safe_get(legacy_data, 'name')
        description = safe_get(legacy_data, 'description')
        age = safe_get(legacy_data, 'age')
        role = safe_get(legacy_data, 'role')
        personality = safe_get(legacy_data, 'personality')
        background = safe_get(legacy_data, 'background')
        base_appearance = safe_get(legacy_data, 'base_appearance')
        character_states = safe_get(legacy_data, 'character_states', [])
        reference_images = safe_get(legacy_data, 'reference_images')
        relationships = safe_get(legacy_data, 'relationships', [])
        first_appearance_chapter = safe_get(legacy_data, 'first_appearance_chapter')
        character_arc_summary = safe_get(legacy_data, 'character_arc_summary')

    return Character(
        name=name,
        description=description,
        age=age,
        role=role,
        personality=personality,
        background=background,
        base_appearance=base_appearance,
        character_states=character_states,
        reference_images=reference_images,
        relationships=relationships,
        first_appearance_chapter=first_appearance_chapter,
        character_arc_summary=character_arc_summary
    )


def parse_appearance_text(appearance_text: str) -> dict:
    """Parse free-text appearance description into structured format"""
    if not appearance_text:
        return CharacterAppearance().dict()

    appearance = CharacterAppearance()

    # Enhanced parsing logic for better extraction from description
    text_lower = appearance_text.lower()

    # Extract colors with better pattern matching
    color_keywords = [
        'golden', 'silver', 'bronze', 'copper', 'chestnut', 'auburn',  # Additional colors
        'brown', 'white', 'black', 'gray', 'grey', 'red', 'blue', 'green',
        'yellow', 'orange', 'purple', 'pink', 'turquoise', 'violet'
    ]
    primary_colors = []
    for color in color_keywords:
        if color in text_lower:
            primary_colors.append(color)
    appearance.primary_colors = primary_colors[:3]  # Allow up to 3 primary colors

    # Extract secondary colors (patterns, markings)
    secondary_patterns = ['spotted', 'striped', 'patched', 'speckled', 'mottled']
    secondary_colors = []
    for pattern in secondary_patterns:
        if pattern in text_lower:
            secondary_colors.append(pattern)
    appearance.secondary_colors = secondary_colors

    # Enhanced species extraction with more comprehensive list and better matching
    species_keywords = [
        ('golden retriever', 'dog'), ('labrador', 'dog'), ('poodle', 'dog'), ('beagle', 'dog'),
        ('dog', 'dog'), ('cat', 'cat'), ('kitten', 'cat'), ('puppy', 'dog'),
        ('owl', 'owl'), ('rabbit', 'rabbit'), ('bunny', 'rabbit'), ('squirrel', 'squirrel'),
        ('fox', 'fox'), ('mouse', 'mouse'), ('turtle', 'turtle'), ('human', 'human'),
        ('boy', 'human'), ('girl', 'human'), ('man', 'human'), ('woman', 'human'),
        ('child', 'human'), ('baby', 'human'), ('bear', 'bear'), ('wolf', 'wolf'),
        ('deer', 'deer'), ('frog', 'frog'), ('bird', 'bird'), ('dragon', 'dragon'),
        ('unicorn', 'unicorn'), ('fairy', 'fairy'), ('elf', 'elf'), ('gnome', 'gnome'),
        ('troll', 'troll'), ('giant', 'giant'), ('chipmunk', 'squirrel')
    ]
    
    # Check for multi-word species first
    for species_term, species_category in species_keywords:
        if species_term in text_lower:
            appearance.species = species_category
            break
    
    # If no multi-word match found, try single words
    if not appearance.species:
        single_word_species = ['dog', 'cat', 'owl', 'rabbit', 'squirrel', 'fox', 'mouse',
                              'turtle', 'human', 'bear', 'wolf', 'deer', 'frog', 'bird']
        for species in single_word_species:
            # Use word boundaries to avoid partial matches
            import re
            if re.search(r'\b' + re.escape(species) + r'\b', text_lower):
                appearance.species = species
                break

    # Enhanced special features extraction
    feature_keywords = [
        'floppy ears', 'shiny nose', 'bushy tail', 'large eyes', 'small', 'tiny',
        'large', 'fluffy', 'curly hair', 'straight hair', 'wavy hair', 'long hair',
        'short hair', 'pointed ears', 'rounded ears', 'big eyes', 'small eyes',
        'bright eyes', 'sparkling eyes', 'wise eyes', 'gentle eyes', 'sharp beak',
        'soft fur', 'rough fur', 'smooth scales', 'bumpy shell', 'shiny feathers',
        'colorful feathers', 'iridescent wings', 'translucent wings'
    ]
    special_features = []
    for feature in feature_keywords:
        if feature in text_lower:
            special_features.append(feature)
    appearance.special_features = special_features

    # Enhanced clothing and accessories extraction
    clothing_keywords = [
        'scarf', 'hat', 'shirt', 'dress', 'jacket', 'collar', 'tie', 'bowtie',
        'crown', 'tiara', 'glasses', 'spectacles', 'bandana', 'ribbon',
        'belt', 'boots', 'shoes', 'sandals', 'gloves', 'mittens', 'cap',
        'helmet', 'armor', 'robe', 'cloak', 'vest', 'jumper', 'sweater'
    ]
    clothing = []
    for item in clothing_keywords:
        if item in text_lower:
            clothing.append(item)
    appearance.clothing = clothing

    # Extract accessories
    accessory_keywords = [
        'necklace', 'bracelet', 'ring', 'earrings', 'watch', 'bag', 'backpack',
        'basket', 'lantern', 'staff', 'wand', 'sword', 'shield', 'key',
        'bottle', 'potion', 'book', 'scroll', 'map', 'compass'
    ]
    accessories = []
    for item in accessory_keywords:
        if item in text_lower:
            accessories.append(item)
    appearance.accessories = accessories

    # Determine size category
    size_keywords = {
        'tiny': ['tiny', 'miniature', 'little'],
        'small': ['small', 'petite'],
        'medium': ['medium', 'average', 'normal'],
        'large': ['large', 'big', 'huge'],
        'giant': ['giant', 'enormous', 'massive']
    }
    for size_cat, keywords in size_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                appearance.size_category = size_cat
                break
        if appearance.size_category:
            break

    # Set consistency keywords from the full text
    appearance.consistency_keywords = extract_keywords_from_description(appearance_text)

    return appearance.dict()


def extract_keywords_from_description(description: str) -> List[str]:
    """Extract visual keywords from character description"""
    keywords = []

    # Common visual keywords for children's books
    visual_keywords = [
        'floppy', 'bushy', 'shiny', 'bright', 'expressive', 'large', 'small',
        'fluffy', 'soft', 'sparkling', 'magical', 'friendly', 'curious'
    ]

    desc_lower = description.lower()
    for keyword in visual_keywords:
        if keyword in desc_lower:
            keywords.append(keyword)

    return keywords


class NovelState(BaseModel):
    """Schema for complete novel generation state"""
    title: str = Field(description="Story title")
    theme: Optional[str] = Field(default=None, description="Main theme of the story")
    age_group: Optional[str] = Field(default=None, description="Target age group")
    genre: Optional[str] = Field(default=None, description="Story genre")
    characters: List[Character] = Field(description="List of main characters")
    chapters: List[Chapter] = Field(description="List of chapters")
    last_updated: Optional[str] = Field(default=None, description="Timestamp of last update")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @validator('chapters')
    def validate_chapters(cls, v):
        """Ensure chapters are properly ordered"""
        if v:
            chapter_numbers = [ch.chapter_number for ch in v]
            if chapter_numbers != sorted(chapter_numbers):
                raise ValueError("Chapters must be in sequential order")
        return v

    @validator('characters')
    def validate_characters(cls, v):
        """Ensure character names are unique"""
        if v:
            names = [ch.name for ch in v]
            if len(names) != len(set(names)):
                raise ValueError("Character names must be unique")
        return v
    
    @root_validator(pre=True)
    def migrate_legacy_data(cls, values):
        """Apply migration to legacy data before validation"""
        # Apply the migration function to handle various format issues
        return migrate_novel_state_data(values)