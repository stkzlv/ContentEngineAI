# src/video/producer/constants.py
"""Constants for the video producer module.

Contains platform definitions, text processing constants, and default visual settings.
"""

from enum import Enum


class Platform(str, Enum):
    """Supported social media platforms for video publishing."""

    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"


# List of supported platform values for iteration
SUPPORTED_PLATFORMS: list[str] = [p.value for p in Platform]

# Skip words for hashtag extraction - common words that don't make good hashtags
HASHTAG_SKIP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "you",
        "your",
        "our",
        "can",
        "will",
        "has",
        "have",
        "been",
        "only",
        "also",
    }
)

# Default visual bounds for subtitle positioning (percentage of video dimensions)
DEFAULT_VIDEO_TOP_POSITION: float = 0.07  # 7% from top
DEFAULT_VIDEO_HEIGHT: float = 0.8  # 80% of frame height
DEFAULT_VIDEO_WIDTH: float = 0.9  # 90% of frame width
