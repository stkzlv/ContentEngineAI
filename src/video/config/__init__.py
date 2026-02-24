# src/video/config/__init__.py
"""Video configuration module - Modular structure with backward compatibility.

All classes are re-exported from this module to maintain backward compatibility
with existing imports like: from src.video.config import VideoConfig
"""

# Re-export all constants
# Re-export audio models
from src.video.config.audio_models import (  # noqa: F401
    AudioProcessingSettings,
    AudioSettings,
    CoquiTTSSettings,
    GoogleCloudSTTSettings,
    GoogleCloudTTSSettings,
    GoogleCloudVoiceCriteria,
    TextMarkupRule,
    TTSConfig,
    VoiceProfileConfig,
)
from src.video.config.constants import *  # noqa: F403, F401

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

# Re-export LLM settings (separate module to avoid circular imports)
from src.video.config.llm_settings import LLMSettings  # noqa: F401

# Re-export subtitle models
from src.video.config.subtitle_models import (  # noqa: F401
    SubtitleEffectsSettings,
    SubtitleSegmentationSettings,
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

# Initialize global config singleton for backward compatibility
# Use load_video_config_modular to properly instantiate VideoConfig Pydantic model
from src.video.config_adapter import load_video_config_modular

config: VideoConfig = load_video_config_modular()

__all__ = [
    # Global config singleton
    "config",
    # Constants (exported via *)
    # Audio models
    "AudioProcessingSettings",
    "AudioSettings",
    "CoquiTTSSettings",
    "GoogleCloudSTTSettings",
    "GoogleCloudTTSSettings",
    "GoogleCloudVoiceCriteria",
    "TextMarkupRule",
    "TTSConfig",
    "VoiceProfileConfig",
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
