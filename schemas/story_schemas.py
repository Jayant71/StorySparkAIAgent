from pydantic import BaseModel, Field
from typing import List, Optional


class Character(BaseModel):
    """Schema for story characters"""
    name: str = Field(description="Character's name")
    description: str = Field(description="Detailed description of the character")
    age: Optional[int] = Field(default=None, description="Character's age if applicable")
    role: Optional[str] = Field(default=None, description="Character's role in the story")


class Chapter(BaseModel):
    """Schema for chapter outline"""
    number: int = Field(description="Chapter number")
    title: str = Field(description="Chapter title")
    summary: str = Field(description="Brief summary of the chapter")


class StoryOutline(BaseModel):
    """Schema for complete story outline"""
    title: str = Field(description="Story title")
    characters: List[Character] = Field(description="List of main characters")
    chapters: List[Chapter] = Field(description="List of chapter outlines")


class ChapterContent(BaseModel):
    """Schema for written chapter content"""
    chapter_number: int = Field(description="Chapter number")
    title: str = Field(description="Chapter title")
    content: str = Field(description="Complete chapter text content")
    word_count: int = Field(description="Approximate word count")
    illustrations: List[str] = Field(default_factory=list, description="List of illustration suggestions extracted from content")