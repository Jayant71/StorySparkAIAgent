from dataclasses import dataclass
from utils.directory_utils import get_current_output_dir


@dataclass
class Config:
    age_group: str  # "4-6", "6-8", "9-12", "13+"
    genre: str
    num_chapters: int

    @property
    def output_dir(self) -> str:
        """
        Get the current run's output directory path.

        Returns:
            Path to the timestamped output directory
        """
        return get_current_output_dir()

    # Age-appropriate parameters
    @property
    def words_per_chapter(self):
        age_ranges = {
            "4-6": (200, 400),
            "6-8": (500, 800),
            "9-12": (1000, 1500),
            "13+": (2000, 3000)
        }
        return age_ranges.get(self.age_group, (500, 800))

    @property
    def vocabulary_level(self):
        levels = {
            "4-6": "simple",
            "6-8": "elementary",
            "9-12": "intermediate",
            "13+": "advanced"
        }
        return levels.get(self.age_group, "elementary")
