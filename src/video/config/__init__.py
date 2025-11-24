# src/video/config/__init__.py
"""
Video configuration module - Modular structure with backward compatibility.

All classes are re-exported from this module to maintain backward compatibility
with existing imports like: from src.video.video_config import VideoConfig
"""

# Re-export all constants
from src.video.config.constants import *  # noqa: F403, F401

# Re-export audio models
from src.video.config.audio_models import (  # noqa: F401
    AudioProcessingSettings,
    AudioSettings,
    CoquiTTSSettings,
    GoogleCloudSTTSettings,
    GoogleCloudTTSSettings,
    GoogleCloudVoiceCriteria,
    TTSConfig,
)

# Re-export visual models
from src.video.config.visual_models import (  # noqa: F401
    CTADetectionSettings,
    MediaSettings,
    MediaValidationSettings,
    StockMediaSettings,
    VideoProcessingSettings,
    VideoProfile,
    VideoSettings,
)

# Re-export subtitle models
from src.video.config.subtitle_models import (  # noqa: F401
    SubtitleEffectsSettings,
    SubtitleSegmentationSettings,
)

# Re-export core models (including VideoConfig)
from src.video.config.core_models import (  # noqa: F401
    ApiSettings,
    AttributionSettings,
    CleanupConfig,
    CleanupSettings,
    DebugSettings,
    DescriptionSettings,
    FFmpegSettings,
    FilesystemSettings,
    GlobalDirs,
    LLMSettings,
    LLMValidationSettings,
    OptimizationSettings,
    OutputStructure,
    PathConfig,
    ProductFiles,
    ProductSubdirs,
    ProductTempFiles,
    ScraperTimingSettings,
    TextProcessingSettings,
    TextRenderingSettings,
    URLShortenerSettings,
    VideoConfig,
    WhisperSettings,
    load_video_config,
)

__all__ = [
    # Constants (exported via *)
    # Audio models
    "AudioProcessingSettings",
    "AudioSettings",
    "CoquiTTSSettings",
    "GoogleCloudSTTSettings",
    "GoogleCloudTTSSettings",
    "GoogleCloudVoiceCriteria",
    "TTSConfig",
    # Visual models
    "CTADetectionSettings",
    "MediaSettings",
    "MediaValidationSettings",
    "StockMediaSettings",
    "VideoProcessingSettings",
    "VideoProfile",
    "VideoSettings",
    # Subtitle models
    "SubtitleEffectsSettings",
    "SubtitleSegmentationSettings",
    # Core models
    "ApiSettings",
    "AttributionSettings",
    "CleanupConfig",
    "CleanupSettings",
    "DebugSettings",
    "DescriptionSettings",
    "FFmpegSettings",
    "FilesystemSettings",
    "GlobalDirs",
    "LLMSettings",
    "LLMValidationSettings",
    "OptimizationSettings",
    "OutputStructure",
    "PathConfig",
    "ProductFiles",
    "ProductSubdirs",
    "ProductTempFiles",
    "ScraperTimingSettings",
    "TextProcessingSettings",
    "TextRenderingSettings",
    "URLShortenerSettings",
    "VideoConfig",
    "WhisperSettings",
    "load_video_config",
]
