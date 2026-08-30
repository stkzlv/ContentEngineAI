# src/video/config/__init__.py
"""Video configuration module - Modular structure with backward compatibility.

All classes are re-exported from this module to maintain backward compatibility
with existing imports like: from src.video.config import VideoConfig
"""

from typing import TYPE_CHECKING, Any

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
    VideoConfig,
    WhisperSettings,
    load_video_config,
)

# Re-export LLM settings (separate module to avoid circular imports)
from src.video.config.llm_settings import LLMSettings  # noqa: F401

# Re-export subtitle models
from src.video.config.subtitle_models import (  # noqa: F401
    ColorPoolEntry,
    FontPoolEntry,
    PartialSubtitleSettings,
    PlatformSafeZone,
    Position,
    PositionAnchor,
    StylePreset,
    StylePresetConfig,
    SubtitleEffectsSettings,
    SubtitleSegmentationSettings,
    SubtitleSettings,
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

# Global config singleton, resolved on first access rather than at import.
#
# Two reasons it cannot be eager. It closes an import cycle: the loader lives
# in `src.video.config_adapter`, which imports `VideoConfig` back out of this
# package, so importing the adapter first fails with a partially-initialised
# module -- which is why `tools/cleanup_outputs.py` had never run. And it read
# five cwd-relative YAML files at import time, so merely importing any
# submodule of this package could fail on a machine with an unrelated config
# error, or from any directory but the repo root.
#
# PEP 562: `from src.video.config import config` still works unchanged.
#
# The TYPE_CHECKING block below is not decoration. A module `__getattr__`
# returns `Any`, so without it mypy sees `config` as `Any` at all nine import
# sites and stops reporting `attr-defined` on anything read from it -- a
# typo'd attribute would then pass CI and raise at render time. Declaring the
# names here restores the checking the eager assignment used to give, and
# costs nothing at runtime. The redundant `as` alias is what marks the
# re-export explicit to mypy.
if TYPE_CHECKING:
    from src.video.config_adapter import (
        load_video_config_modular as load_video_config_modular,
    )

    config: VideoConfig

_config: "VideoConfig | None" = None


def __getattr__(name: str) -> Any:
    """Resolve the `config` singleton and the loader lazily.

    `load_video_config_modular` was re-exported as a side effect of the
    module-level import this replaces, and callers rely on that spelling, so
    it is served here rather than dropped.
    """
    if name in ("config", "load_video_config_modular"):
        from src.video.config_adapter import load_video_config_modular

        if name == "load_video_config_modular":
            return load_video_config_modular

        global _config
        if _config is None:
            _config = load_video_config_modular()
        return _config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Global config singleton, resolved by `__getattr__` above rather than
    # being a module attribute, which is what F405 is reacting to.
    "config",  # noqa: F405
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
    "ColorPoolEntry",
    "FontPoolEntry",
    "PartialSubtitleSettings",
    "PlatformSafeZone",
    "Position",
    "PositionAnchor",
    "StylePreset",
    "StylePresetConfig",
    "SubtitleEffectsSettings",
    "SubtitleSegmentationSettings",
    "SubtitleSettings",
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
    "VideoConfig",
    "WhisperSettings",
    "load_video_config",
]
