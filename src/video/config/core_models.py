# src/video/config/core_models.py
"""Core configuration models including VideoConfig, paths, cleanup, and optimization."""

import fnmatch
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

# Platform-specific metadata models (no circular import after extracting LLMSettings)
from src.ai.platform_metadata.models import PlatformMetadataSettings
from src.utils import MAX_FILENAME_LENGTH
from src.video.config.audio_models import (
    AudioProcessingSettings,
    AudioSettings,
    GoogleCloudSTTSettings,
    TTSConfig,
)
from src.video.config.constants import (
    DEFAULT_FALLBACK_FONT,
    DEFAULT_WHISPER_MODEL_DIR,
    FALLBACK_FONT_ALTERNATIVES,
    FONT_FILE_EXTENSIONS,
    FONT_REGULAR_SUFFIXES,
)
from src.video.config.llm_settings import LLMSettings
from src.video.config.subtitle_models import (
    ColorPoolEntry,
    FontPoolEntry,
    PlatformSafeZone,  # re-exported here for backward compat with existing imports
    StylePresetConfig,
    SubtitleEffectsSettings,
    SubtitleSegmentationSettings,
)

__all_reexported__ = ["PlatformSafeZone"]
from src.video.config.visual_models import (
    CTADetectionSettings,
    MediaSettings,
    MediaValidationSettings,
    MergedProfileSettings,
    ProfileInfo,
    StockMediaSettings,
    VideoProcessingSettings,
    VideoProfile,
    VideoSettings,
)

logger = logging.getLogger(__name__)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``.

    Dicts merge key-by-key; non-dict values on either side replace the
    corresponding base entry entirely. Used for nested profile overrides
    (e.g. ``two_part_subtitles``) where a profile should only need to
    specify the fields that differ from the global block.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class DescriptionSettings(BaseModel):
    """Configuration settings for AI-generated video descriptions.

    Controls the generation of social media descriptions using LLM providers.
    Supports both legacy unified mode and new platform-specific metadata generation.
    """

    enabled: bool = Field(True, description="Enable or disable description generation")
    prompt_template_path: str = Field(
        "src/ai/prompts/video_description.md",
        description="Path to prompt template file for description generation",
    )
    target_platforms: list[str] = Field(
        ["tiktok", "youtube", "instagram"],
        description="Target platforms for description optimization",
    )
    max_tokens: int = Field(200, description="Maximum tokens for LLM response")
    min_description_chars: int = Field(
        50, description="Minimum character count for valid descriptions"
    )
    min_description_words: int = Field(
        10, description="Minimum word count for valid descriptions"
    )
    require_hashtags: bool = Field(
        True, description="Whether descriptions must include hashtags"
    )
    require_ad_hashtag: bool = Field(
        True, description="Whether descriptions must include #ad hashtag"
    )

    # Metadata mode: unified (single for all platforms) or optimized (platform-specific)
    metadata_mode: Literal["unified", "optimized"] = Field(
        "unified",
        description=(
            "Metadata mode: 'unified' (single title/description/hashtags for all "
            "platforms) or 'optimized' (platform-specific SEO-tailored metadata)"
        ),
    )

    # Platform-specific metadata generation (optional, new feature)
    target_platform: str = Field(
        "multi",
        description=(
            "Target platform for metadata generation: 'youtube', 'tiktok', "
            "'instagram', or 'multi' for all platforms"
        ),
    )
    platform_metadata: PlatformMetadataSettings | None = Field(
        None,
        description=(
            "Platform-specific metadata settings. If None, uses legacy unified "
            "description mode for backward compatibility."
        ),
    )


class FFmpegSettings(BaseModel):
    executable_path: str | None = Field(None)
    temp_ffmpeg_dir: str = Field("ffmpeg_work")
    intermediate_segment_preset: str = Field("ultrafast")
    final_assembly_timeout_sec: int = Field(600)
    rw_timeout_microseconds: int = Field(30000000)  # 30 seconds for I/O operations
    verification_timeout_sec: int = Field(
        30,
        description="Timeout in seconds for video verification subprocess. "
        "Controls how long to wait for ffprobe verification before failing. "
        "Increase if verifying very large or slow-to-probe videos.",
    )


class AttributionSettings(BaseModel):
    attribution_file_name: str = Field("ATTRIBUTIONS.txt")
    attribution_template: str
    attribution_entry_template: str


class WhisperSettings(BaseModel):
    model_config = {"protected_namespaces": ()}

    enabled: bool = Field(True)
    model_size: str = Field("small")
    model_device: str = Field("cpu")
    model_in_memory: bool = Field(False)
    model_download_root: str = Field("")
    temperature: float = Field(0.0)
    language: str = Field("en")
    beam_size: int = Field(5)
    fp16: bool = Field(False)
    compression_ratio_threshold: float = Field(2.4)
    logprob_threshold: float = Field(-1.0)
    no_speech_threshold: float = Field(0.2)
    condition_on_previous_text: bool = Field(True)
    task: str = Field("transcribe")
    patience: float | None = Field(None)

    # Timeout settings for Whisper processing
    base_timeout_sec: int = Field(120)
    duration_multiplier: float = Field(6.0)  # Increased from 3.0
    max_timeout_sec: int = Field(900)  # Increased from 600
    progress_monitor_interval_sec: int = Field(30)
    enable_resource_monitoring: bool = Field(True)
    enable_resource_cleanup: bool = Field(True)


class ApiSettings(BaseModel):
    """Configuration for API timeouts, retries, and network settings."""

    llm_model_fetch_timeout_sec: int = Field(30)  # Configurable via YAML
    llm_retry_attempts: int = Field(3)  # Configurable via YAML
    llm_retry_min_wait_sec: int = Field(1)  # Configurable via YAML
    llm_retry_max_wait_sec: int = Field(30)  # Configurable via YAML
    llm_retry_multiplier: int = Field(2)  # Configurable via YAML
    stock_media_concurrent_downloads: int = Field(5)
    stock_media_search_multiplier: int = Field(2)
    stock_media_max_per_page: int = Field(80)
    download_timeout_sec: int = Field(30)  # Configurable via YAML
    download_retry_attempts: int = Field(3)  # Configurable via YAML
    download_retry_min_wait_sec: int = Field(1)  # Configurable via YAML
    download_retry_max_wait_sec: int = Field(10)  # Configurable via YAML


class TextProcessingSettings(BaseModel):
    """Configuration for text processing and subtitle generation."""

    script_chars_per_second_estimate: int = Field(15)
    script_min_duration_sec: float = Field(0.05)
    subtitle_text_similarity_min_confidence: float = Field(0.5)
    subtitle_min_segment_duration_sec: float = Field(0.1)
    subtitle_max_segment_duration_sec: float = Field(5.0)
    subtitle_min_words_per_segment: int = Field(3)
    subtitle_max_words_per_segment: int = Field(10)
    subtitle_max_chars_per_line: int = Field(42)
    subtitle_min_segment_gap_sec: float = Field(0.1)


class FilesystemSettings(BaseModel):
    """Configuration for file system operations and supported formats."""

    temp_file_cleanup_delay_sec: int = Field(5)
    file_operation_timeout_sec: int = Field(30)
    max_filename_length: int = Field(MAX_FILENAME_LENGTH)
    supported_image_extensions: list[str] = Field(
        [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    )
    supported_video_extensions: list[str] = Field(
        [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    )
    supported_audio_extensions: list[str] = Field([".wav", ".mp3", ".aac", ".flac"])


class TextRenderingSettings(BaseModel):
    """Configuration for text rendering and character width estimation."""

    # Platform safe zone
    safe_zone: PlatformSafeZone = Field(default_factory=PlatformSafeZone)

    # Character width factors
    narrow_char_width_factor: float = Field(
        0.4, description="Width factor for narrow characters (i, l, etc.)"
    )
    wide_char_width_factor: float = Field(
        1.2, description="Width factor for wide characters (m, w, etc.)"
    )
    space_char_width_factor: float = Field(
        0.3, description="Width factor for space characters"
    )

    # Content-aware positioning
    content_aware_font_offset_multiplier: float = Field(
        5.5,
        description="Font offset multiplier for content-aware subtitle positioning. "
        "Controls how far subtitles are offset from detected content boundaries. "
        "Higher values = more spacing between subtitles and content. "
        "Used in calculate_position() to avoid overlapping visual elements.",
    )


class ScraperTimingSettings(BaseModel):
    """Configuration for scraper delays and timeouts."""

    # Download parameters
    download_timeout_sec: int = Field(30, description="Download timeout in seconds")
    download_chunk_size: int = Field(8192, description="Download chunk size in bytes")
    validation_timeout_sec: int = Field(
        10, description="File validation timeout in seconds"
    )
    max_concurrent_downloads: int = Field(5, description="Maximum concurrent downloads")

    # Retry configuration
    default_max_retries: int = Field(3, description="Default maximum retry attempts")
    base_delay_sec: float = Field(
        1.0, description="Base delay between retries in seconds"
    )
    backoff_factor: float = Field(2.0, description="Exponential backoff factor")
    max_delay_sec: float = Field(
        60.0, description="Maximum delay between retries in seconds"
    )

    # Filename and browser settings
    max_filename_length: int = Field(200, description="Maximum filename length")
    browser_size_percent: float = Field(
        0.8, description="Browser window size as percentage of monitor"
    )

    # Human simulation delays
    human_delay_min_sec: float = Field(
        0.5, description="Minimum human-like delay in seconds"
    )
    human_delay_max_sec: float = Field(
        2.0, description="Maximum human-like delay in seconds"
    )


class LLMValidationSettings(BaseModel):
    """Configuration for LLM response validation."""

    # Retry parameters
    llm_max_retry_attempts: int = Field(
        2, description="Maximum retry attempts for LLM requests"
    )

    # Description validation thresholds
    min_description_chars: int = Field(
        50, description="Minimum character length for generated descriptions"
    )
    min_description_words: int = Field(
        10, description="Minimum word count for generated descriptions"
    )
    description_retry_attempts: int = Field(
        2, description="Maximum retry attempts for incomplete descriptions"
    )


class URLShortenerSettings(BaseModel):
    """Configuration for URL shortening services."""

    enabled: bool = Field(True, description="Enable URL shortening feature")
    provider: str = Field("picsee", description="Primary URL shortening provider")

    # API configuration
    api_timeout_sec: int = Field(30, description="Request timeout in seconds")
    api_max_retries: int = Field(3, description="Maximum retry attempts")
    api_retry_delay_sec: int = Field(2, description="Base retry delay in seconds")
    api_retry_backoff_multiplier: float = Field(
        2.0, description="Exponential backoff multiplier"
    )

    # Picsee-specific configuration
    picsee_api_key_env_var: str = Field(
        "PICSEE_API_KEY", description="Environment variable for Picsee API key"
    )
    picsee_custom_domain: str | None = Field(
        None, description="Optional custom branded short domain (BSD) for Picsee"
    )
    picsee_max_bulk_size: int = Field(
        100, description="Maximum URLs per Picsee bulk request"
    )

    # Integration settings
    shorten_on_scrape: bool = Field(
        True, description="Automatically shorten affiliate links during scraping"
    )
    fallback_to_original: bool = Field(
        True, description="Use original URL if shortening fails"
    )


class DebugSettings(BaseModel):
    """Configuration for debug output and development settings."""

    max_log_line_length: int = Field(200)
    debug_file_retention_days: int = Field(7)
    intermediate_file_cleanup: bool = Field(True)
    cleanup_on_success: bool = Field(False)
    cleanup_on_failure: bool = Field(False)
    cleanup_whisper_files: bool = Field(False)
    operation_timing_threshold_sec: float = Field(5.0)
    memory_usage_warning_mb: int = Field(1000)


class ProductFiles(BaseModel):
    """File names within each product directory"""

    scraped_data: str = Field("data.json")
    script: str = Field("script.txt")
    description: str = Field("description.txt")
    voiceover: str = Field("voiceover.wav")
    subtitles: str = Field("subtitles.srt")
    final_video: str = Field("video_{profile}.mp4")
    attribution: str = Field("attributions.txt")


class ProductTempFiles(BaseModel):
    """Temporary/debug files within product temp directory"""

    pipeline_state: str = Field("pipeline_state.json")
    performance: str = Field("performance.json")
    ffmpeg_log: str = Field("ffmpeg_command.log")
    media_validation_report: str = Field("media_validation_report.json")
    whisper_result_raw: str = Field("whisper_result_raw.json")
    whisper_vs_script: str = Field("whisper_vs_script.txt")
    whisper_word_list: str = Field("whisper_word_list.json")
    gathered_visuals: str = Field("gathered_visuals.json")
    music_choice: str = Field("music_choice.json")
    voiceover_duration: str = Field("voiceover_duration.txt")
    script_prompt: str = Field("script_prompt.txt")


class ProductSubdirs(BaseModel):
    """Subdirectories within each product directory"""

    images: str = Field("images")
    videos: str = Field("videos")
    music: str = Field("music")
    temp: str = Field("temp")


class GlobalDirs(BaseModel):
    """Global directories shared across all products"""

    cache: str = Field("cache")
    logs: str = Field("logs")
    reports: str = Field("reports")
    temp: str = Field("temp")


class OutputStructure(BaseModel):
    """Simplified, product-oriented output structure"""

    product_directory_pattern: str = Field("{product_id}")
    product_files: ProductFiles = Field(default_factory=lambda: ProductFiles())  # type: ignore[call-arg]
    product_temp_files: ProductTempFiles = Field(default_factory=ProductTempFiles)  # type: ignore[arg-type]
    product_subdirs: ProductSubdirs = Field(default_factory=lambda: ProductSubdirs())  # type: ignore[call-arg]
    global_dirs: GlobalDirs = Field(default_factory=lambda: GlobalDirs())  # type: ignore[call-arg]


class CleanupConfig(BaseModel):
    """Cleanup and maintenance settings"""

    remove_temp_on_success: bool = Field(True)
    keep_temp_on_failure: bool = Field(True)
    cache_max_age_hours: int = Field(168)  # 7 days
    debug_file_patterns: list[str] = Field(
        [
            "incomplete_script_*.txt",  # AI model attempt files
            "voiceover_whisper_*.json",  # Whisper debug outputs
            "voiceover_whisper_*.txt",  # Whisper comparison files
            "*_ffmpeg_command.log",  # FFmpeg command logs
        ]
    )


class PathConfig(BaseModel):
    """Path building configuration"""

    use_product_oriented_structure: bool = Field(True)
    cleanup: CleanupConfig = Field(default_factory=lambda: CleanupConfig())  # type: ignore[call-arg]

    # Internal files configuration
    gathered_visuals: str = Field("gathered_visuals.json")
    temp_dir: str = Field("temp")
    music_dir: str = Field("music")


# Removed FilePatterns class - replaced by simplified ProductFiles


class CleanupSettings(BaseModel):
    enabled: bool = Field(True)
    dry_run: bool = Field(False)
    max_age_days: int = Field(7)
    preserve_patterns: list[str] = Field(
        [
            "*.md",
            "*.txt",
            ".gitkeep",
            "cache/**",
            "backup/**",
            # Top-level tracking files. Without these, `make clean-outputs`
            # silently wipes the publish registry, history, and cleanup audit
            # whenever they age past max_age_days.
            "published_products.json",
            "published_products.csv",
            "publish_history.json",
            "cleanup_audit.json",
        ]
    )
    force_cleanup_patterns: list[str] = Field(
        ["*.tmp", "*.temp", "~*", ".DS_Store", "Thumbs.db", "*.log.old"]
    )
    cleanup_empty_dirs: bool = Field(True)
    create_report: bool = Field(True)
    report_file: str = Field("cleanup_report.json")


class OptimizationSettings(BaseModel):
    # Background Processing Configuration
    background_max_concurrent_tasks: int = Field(3)
    background_thread_pool_workers: int = Field(2)
    background_cache_ttl_sec: int = Field(600)
    stock_media_prefetch_priority: int = Field(2)
    tts_warming_priority: int = Field(3)
    background_max_recent_completed: int = Field(5)
    background_cleanup_timeout_sec: float = Field(5.0)
    stock_prefetch_max_images: int = Field(3)
    stock_prefetch_max_videos: int = Field(2)
    stock_prefetch_max_keywords: int = Field(5)
    stock_prefetch_top_keywords: int = Field(2)
    stock_keyword_min_length: int = Field(3)
    stock_max_descriptive_words: int = Field(3)

    # Performance Monitoring Configuration
    performance_history_max_runs: int = Field(100)
    performance_history_cleanup_interval: int = Field(10)
    performance_monitoring_interval_sec: float = Field(0.1)
    memory_mb_conversion_factor: int = Field(1048576)
    performance_report_summary_limit: int = Field(50)
    performance_report_detailed_limit: int = Field(20)
    performance_report_trends_days: int = Field(30)
    performance_report_recent_runs: int = Field(10)
    performance_report_max_runs: int = Field(1000)

    # Connection Pooling Configuration
    connection_pool_total_limit: int = Field(100)
    connection_pool_host_limit: int = Field(20)
    connection_pool_dns_ttl_sec: int = Field(300)
    connection_pool_keepalive_timeout_sec: int = Field(60)
    connection_pool_cleanup_interval_sec: int = Field(300)
    connection_pool_total_timeout_sec: int = Field(300)
    connection_pool_connect_timeout_sec: int = Field(30)
    connection_pool_read_timeout_sec: int = Field(60)
    download_manager_max_concurrent: int = Field(5)
    download_chunk_size_bytes: int = Field(8192)

    # Memory-Mapped I/O Configuration
    mmap_file_size_threshold_bytes: int = Field(1048576)  # 1MB
    mmap_chunk_size_bytes: int = Field(67108864)  # 64MB
    mmap_memory_usage_threshold: float = Field(0.8)
    mmap_fallback_memory_limit_bytes: int = Field(1073741824)  # 1GB

    # Async I/O Configuration
    async_ffmpeg_max_concurrent: int = Field(2)
    async_io_max_concurrent: int = Field(8)
    async_network_max_concurrent: int = Field(4)
    async_default_timeout_sec: int = Field(300)
    async_ffprobe_timeout_sec: int = Field(30)

    # Caching Configuration
    cache_media_metadata_ttl_sec: int = Field(86400)  # 24 hours
    cache_api_response_ttl_sec: int = Field(3600)  # 1 hour
    cache_key_max_length: int = Field(16)


def _default_font_pool() -> list[FontPoolEntry]:
    """Curated bold sans-serif font pool. Serif fonts excluded by design."""
    return [
        FontPoolEntry(
            name="Montserrat",
            file="Montserrat-Bold.ttf",
            ffmpeg_name="Montserrat-Bold",
            system_fallback="Arial",
        ),
        FontPoolEntry(
            name="Poppins",
            file="Poppins-Bold.ttf",
            ffmpeg_name="Poppins-Bold",
            system_fallback="Arial",
        ),
        FontPoolEntry(
            name="Gabarito",
            file="Gabarito-Bold.ttf",
            ffmpeg_name="Gabarito-Bold",
            system_fallback="Arial",
        ),
        FontPoolEntry(
            name="Rubik",
            file="Rubik-Bold.ttf",
            ffmpeg_name="Rubik-Bold",
            system_fallback="Arial",
        ),
    ]


def _default_color_pool() -> list[ColorPoolEntry]:
    """Curated high-contrast color pool. WCAG AA-compliant pairs only."""
    return [
        ColorPoolEntry(
            name="classic",
            display_name="Classic",
            font_color="&H00FFFFFF",
            outline_color="&H00000000",
            description="White on black stroke (21:1 contrast, WCAG AAA)",
        ),
        ColorPoolEntry(
            name="high_contrast",
            display_name="High Contrast",
            font_color="&H0000FFFF",
            outline_color="&H00000000",
            description="Yellow on black stroke - high visibility",
        ),
        ColorPoolEntry(
            name="neon_green",
            display_name="Neon Green",
            font_color="&H004CFF00",
            outline_color="&H00000000",
            description="Bright green on black - high-energy highlight",
        ),
        ColorPoolEntry(
            name="brand_yellow",
            display_name="Brand Yellow",
            font_color="&H0000EBFF",
            outline_color="&H00000000",
            description="Saturated yellow on black - highest-converting per Submagic",
        ),
    ]


class VideoConfig(BaseModel):
    global_output_directory: str = Field("outputs")
    output_structure: OutputStructure = Field(
        default_factory=lambda: OutputStructure()  # type: ignore[call-arg]
    )
    path_config: PathConfig = Field(
        default_factory=lambda: PathConfig()  # type: ignore[call-arg]
    )
    cleanup_settings: CleanupSettings = Field(
        default_factory=lambda: CleanupSettings()  # type: ignore[call-arg]
    )
    pipeline_timeout_sec: int = Field(
        900, description="Total pipeline timeout in seconds (15 minutes default)"
    )
    outro_duration_sec: float = Field(
        1.0,
        description="Duration in seconds after voiceover ends for music fade-out. "
        "This creates a smooth ending where background music fades out gracefully.",
    )
    video_settings: VideoSettings
    media_settings: MediaSettings
    audio_settings: AudioSettings
    tts_config: TTSConfig
    llm_settings: LLMSettings
    description_settings: DescriptionSettings
    stock_media_settings: StockMediaSettings
    ffmpeg_settings: FFmpegSettings
    subtitle_settings: dict[str, Any]  # Now loaded from config/subtitles.yaml
    whisper_settings: WhisperSettings
    google_cloud_stt_settings: GoogleCloudSTTSettings | None = Field(None)
    video_profiles: dict[str, VideoProfile]
    aspect_ratio: dict[str, Any] = Field(
        default_factory=lambda: {"smart_scale_tolerance": 0.10}
    )
    format_normalization: dict[str, Any] = Field(
        default_factory=lambda: {
            "target_fps": 30.0,
            "fps_tolerance": 0.1,
            "default_fps_string": "30/1",
            "target_codec": "h264",
            "target_pixel_format": "yuv420p",
        }
    )

    # New configuration sections for magic numbers
    api_settings: ApiSettings | None = Field(None)
    text_processing: TextProcessingSettings | None = Field(None)
    audio_processing: AudioProcessingSettings | None = Field(None)
    video_processing: VideoProcessingSettings | None = Field(None)
    filesystem: FilesystemSettings | None = Field(None)
    debug_settings: DebugSettings | None = Field(None)
    optimization_settings: OptimizationSettings | None = Field(None)

    # ASS effects and text rendering configuration
    subtitle_effects: SubtitleEffectsSettings | None = Field(None)
    text_rendering: TextRenderingSettings | None = Field(None)
    subtitle_segmentation: SubtitleSegmentationSettings | None = Field(None)
    style_presets: dict[str, StylePresetConfig] = Field(
        default_factory=lambda: {
            "minimal": StylePresetConfig(
                description="Clean, simple styling with no effects",
                font_name="Montserrat",
                outline_thickness=2,
                shadow=False,
            ),
            "modern": StylePresetConfig(
                description="Bold sans-serif with karaoke highlighting",
                font_name="Montserrat",
                outline_thickness=3,
                shadow=True,
                effects=["karaoke"],
            ),
            "bold": StylePresetConfig(
                description="High-impact bold styling with strong outline",
                font_name="Gabarito",
                outline_thickness=4,
                shadow=True,
                effects=["fade"],
            ),
            "animated": StylePresetConfig(
                description="Karaoke with subtitle motion for playful tones",
                font_name="Gabarito",
                outline_thickness=3,
                shadow=True,
                effects=["karaoke"],
            ),
            "random": StylePresetConfig(
                description="Randomized bold sans-serif with per-video effect",
                font_name="Montserrat",
                outline_thickness=3,
                shadow=True,
                effects=["karaoke", "fade", "typewriter"],
            ),
        }
    )
    font_pool: list[FontPoolEntry] = Field(default_factory=_default_font_pool)
    color_pool: list[ColorPoolEntry] = Field(default_factory=_default_color_pool)
    scraper_timing: ScraperTimingSettings | None = Field(None)
    media_validation: MediaValidationSettings | None = Field(None)
    llm_validation: LLMValidationSettings | None = Field(None)
    url_shortener_settings: URLShortenerSettings | None = Field(None)
    cta_detection: CTADetectionSettings | None = Field(None)

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent,
        init=False,
    )
    global_output_root_path: Path = Field(default_factory=Path, init=False)
    video_production_base_runs_path: Path = Field(default_factory=Path, init=False)
    general_video_producer_log_dir_path: Path = Field(default_factory=Path, init=False)
    scraper_data_base_path: Path = Field(default_factory=Path, init=False)
    secrets: dict[str, str] = Field(default_factory=dict, init=False)

    @model_validator(mode="after")
    def derive_and_resolve_paths(self) -> "VideoConfig":
        self.global_output_root_path = self.project_root / self.global_output_directory

        # For backward compatibility, keep some basic paths
        self.video_production_base_runs_path = self.global_output_root_path
        self.general_video_producer_log_dir_path = (
            self.global_output_root_path / self.output_structure.global_dirs.logs
        )
        self.scraper_data_base_path = self.global_output_root_path

        resolved_music_paths = []
        for p_item in self.audio_settings.background_music_paths:
            p_obj = Path(p_item)
            resolved_music_paths.append(
                self.project_root / p_obj if not p_obj.is_absolute() else p_obj
            )
        self.audio_settings.background_music_paths = resolved_music_paths

        llm_template_path_obj = Path(self.llm_settings.prompt_template_path)
        self.llm_settings.prompt_template_path = str(
            (self.project_root / llm_template_path_obj)
            if not llm_template_path_obj.is_absolute()
            else llm_template_path_obj
        )

        font_dir_obj = Path(self.subtitle_settings["font_directory"])
        self.subtitle_settings["font_directory"] = str(
            (self.project_root / font_dir_obj)
            if not font_dir_obj.is_absolute()
            else font_dir_obj
        )
        return self

    def get_profile(self, profile_name: str) -> VideoProfile:
        if profile_name not in self.video_profiles:
            raise KeyError(f"Video profile '{profile_name}' not found.")
        return self.video_profiles[profile_name]

    def get_profile_merged_settings(
        self, profile_name: str, cli_overrides: dict[str, Any] | None = None
    ) -> MergedProfileSettings:
        """Get settings with profile-specific overrides applied.

        Merges global config with profile overrides and CLI overrides.
        Precedence: CLI > Profile > Global.

        Returns typed MergedProfileSettings with Pydantic models.
        """
        profile = self.get_profile(profile_name)

        # --- Video settings: base model + profile overrides ---
        video_overrides = self._collect_overrides(
            profile,
            {
                "image_width_percent": "image_width_percent",
                "image_top_position_percent": "image_top_position_percent",
                "image_vertical_align": "image_vertical_align",
                "preserve_aspect_ratio": "preserve_aspect_ratio",
                "video_assembly_mode": "video_assembly_mode",
                "video_aspect_mode": "video_aspect_mode",
                "video_audio_handling": "video_audio_handling",
                "video_original_volume": "video_original_volume",
                "video_transition_duration": "video_transition_duration",
                "enable_format_normalization": "enable_format_normalization",
                "video_cache_dir": "video_cache_dir",
                "video_top_position_percent": "video_top_position_percent",
                "video_content_height_percent": "video_content_height_percent",
                "video_vertical_align": "video_vertical_align",
            },
        )
        for field_name in (
            "video_top_position_percent",
            "video_content_height_percent",
            "video_vertical_align",
        ):
            if field_name in video_overrides:
                logger.debug(
                    f"[TRACE] Profile '{profile_name}' overrides "
                    f"{field_name}: {video_overrides[field_name]}"
                )
        merged_video = self.video_settings.model_copy(update=video_overrides)

        # --- Subtitle settings: YAML dict -> SubtitleSettings, then deep-merge
        # the profile's nested subtitle_settings PartialSubtitleSettings on top.
        from src.video.config.subtitle_models import SubtitleSettings

        subtitle_data = self._build_subtitle_base()
        if profile.subtitle_positioning:
            subtitle_data.update(profile.subtitle_positioning)
        merged_subtitle = SubtitleSettings.from_legacy_dict(subtitle_data)
        if profile.subtitle_settings is not None:
            merged_subtitle = profile.subtitle_settings.merge_into(merged_subtitle)

        # --- Profile info ---
        profile_info = ProfileInfo(
            name=profile_name,
            description=profile.description,
            use_scraped_images=profile.use_scraped_images,
            use_scraped_videos=profile.use_scraped_videos,
            use_stock_images=profile.use_stock_images,
            use_stock_videos=profile.use_stock_videos,
            stock_image_count=profile.stock_image_count,
            stock_video_count=profile.stock_video_count,
            use_dynamic_image_count=profile.use_dynamic_image_count,
        )

        # --- CLI overrides (highest precedence) ---
        if cli_overrides:
            video_updates: dict[str, Any] = {}
            subtitle_updates: dict[str, Any] = {}
            pycaps_updates: dict[str, Any] = {}
            for key, value in cli_overrides.items():
                parts = key.split(".")
                if len(parts) == 2:
                    section, field = parts
                    if section == "video_settings":
                        video_updates[field] = value
                    elif section == "subtitle_settings":
                        subtitle_updates[field] = value
                elif (
                    len(parts) == 3
                    and parts[0] == "subtitle_settings"
                    and parts[1] == "pycaps"
                ):
                    pycaps_updates[parts[2]] = value
            if video_updates:
                merged_video = merged_video.model_copy(update=video_updates)
            if subtitle_updates:
                merged_subtitle = merged_subtitle.model_copy(update=subtitle_updates)
            if pycaps_updates:
                # Merge into the nested PycapsSettings. Create an empty one
                # with defaults if no YAML/profile value existed yet. All
                # PycapsSettings fields have defaults, so mypy's call-arg
                # warning is a false positive from the missing pydantic
                # plugin.
                from src.video.config.subtitle_models import PycapsSettings

                base = merged_subtitle.pycaps or PycapsSettings()  # type: ignore[call-arg]
                merged_pycaps = base.model_copy(update=pycaps_updates)
                merged_subtitle = merged_subtitle.model_copy(
                    update={"pycaps": merged_pycaps}
                )

        return MergedProfileSettings(
            video_settings=merged_video,
            subtitle_settings=merged_subtitle,
            profile=profile_info,
        )

    @staticmethod
    def _collect_overrides(
        profile: VideoProfile, field_map: dict[str, str]
    ) -> dict[str, Any]:
        """Collect non-None profile fields into an overrides dict."""
        overrides: dict[str, Any] = {}
        for profile_field, target_field in field_map.items():
            value = getattr(profile, profile_field, None)
            if value is not None:
                overrides[target_field] = value
        return overrides

    def _build_subtitle_base(self) -> dict[str, Any]:
        """Build base subtitle settings dict from global YAML config."""
        ss = self.subtitle_settings

        base: dict[str, Any] = {
            "anchor": ss["anchor"],
            "margin": ss["margin"],
            "content_aware": ss["content_aware"],
            "style_preset": ss["style_preset"],
            "font_size_scale": ss["font_size_scale"],
            "horizontal_alignment": ss["horizontal_alignment"],
            "randomize_effects": ss["randomize_effects"],
            "max_line_length": ss["max_line_length"],
            "max_words_per_line": ss["max_words_per_line"],
            "max_subtitle_width_fraction": ss.get("max_subtitle_width_fraction", 0.80),
            "max_subtitle_duration": (
                ss.get("max_subtitle_duration")
                or ss.get("max_duration")
                or ss.get("max_subtitle_duration_sec", 2.5)
            ),
            "min_subtitle_duration": (
                ss.get("min_subtitle_duration")
                or ss.get("min_duration")
                or ss.get("min_subtitle_duration_sec", 0.6)
            ),
            "enabled": ss["enabled"],
            "font_directory": ss["font_directory"],
            "font_size_percent": ss["font_size_percent"],
            "randomize_fonts": (
                ss.get("randomize_fonts") or ss.get("use_random_font", False)
            ),
            "randomize_colors": (
                ss.get("randomize_colors") or ss.get("use_random_colors", False)
            ),
            "available_fonts": ss.get("available_fonts", []),
            "available_color_combinations": ss.get("available_color_combinations", []),
            "temp_subtitle_dir": ss.get("temp_subtitle_dir", "temp"),
            "temp_subtitle_filename": ss.get("temp_subtitle_filename", "captions.srt"),
            "save_srt_with_video": ss.get("save_srt_with_video", True),
            "subtitle_format": ss.get("subtitle_format", "srt"),
            "script_paths": ss.get("script_paths", []),
            # Two-part subtitles nested block (passed through as YAML dict;
            # Pydantic validates the shape when building TwoPartSubtitleSettings).
            "two_part_subtitles": ss.get("two_part_subtitles", {}),
            # Pycaps engine selector + nested sub-settings (YAML layer)
            "subtitle_engine": ss.get("subtitle_engine", "ffmpeg"),
            "pycaps": ss.get("pycaps"),
        }
        # Safe zone only included when the global text_rendering block defined
        # one — otherwise SubtitleSettings uses its own PlatformSafeZone default.
        if self.text_rendering is not None:
            base["safe_zone"] = self.text_rendering.safe_zone
        return base

    def get_product_paths(self, product_id: str, profile_name: str) -> dict[str, Path]:
        """Generate all paths for a product using simplified product-oriented structure.

        Returns flat structure: outputs/{product_id}/
        """
        from src.utils import sanitize_filename

        safe_product_id = sanitize_filename(product_id)
        safe_profile_name = sanitize_filename(profile_name)

        # Product root directory
        product_dir = self.global_output_root_path / safe_product_id

        # Product subdirectories
        images_dir = product_dir / self.output_structure.product_subdirs.images
        videos_dir = product_dir / self.output_structure.product_subdirs.videos
        music_dir = product_dir / self.output_structure.product_subdirs.music
        temp_dir = product_dir / self.output_structure.product_subdirs.temp

        # Product files (in root)
        files = self.output_structure.product_files
        temp_files = self.output_structure.product_temp_files

        return {
            # Directories
            "product_root": product_dir,
            "images_dir": images_dir,
            "videos_dir": videos_dir,
            "music_dir": music_dir,
            "temp_dir": temp_dir,
            # Core production files (in product root)
            "scraped_data": product_dir / files.scraped_data,
            "final_video": product_dir
            / files.final_video.format(
                product_id=product_id, profile=safe_profile_name
            ),
            # Intermediate files (in temp directory)
            "script": temp_dir / files.script,
            "description": temp_dir / files.description,
            "voiceover": temp_dir / files.voiceover,
            "subtitles": temp_dir / self._get_subtitle_filename(files.subtitles),
            "attribution": temp_dir / files.attribution,
            # Debug/temp files (in temp directory)
            "pipeline_state": temp_dir / temp_files.pipeline_state,
            "performance": temp_dir / temp_files.performance,
            "ffmpeg_log": temp_dir / temp_files.ffmpeg_log,
            "media_validation_report": temp_dir
            / f"{product_id}_{temp_files.media_validation_report}",
            "whisper_result_raw": temp_dir
            / f"{product_id}_{temp_files.whisper_result_raw}",
            "whisper_vs_script": temp_dir / temp_files.whisper_vs_script,
            "whisper_word_list": temp_dir
            / f"{product_id}_{temp_files.whisper_word_list}",
            "gathered_visuals": temp_dir / temp_files.gathered_visuals,
            "music_choice": temp_dir / temp_files.music_choice,
            "voiceover_duration": temp_dir / temp_files.voiceover_duration,
            "script_prompt": temp_dir / temp_files.script_prompt,
            # Legacy compatibility
            "project_root": product_dir,
            "working_dir": temp_dir,
            "audio_dir": temp_dir,
            "visual_dir": temp_dir,
            "text_dir": temp_dir,
            "metadata": temp_dir / temp_files.pipeline_state,  # Legacy alias
        }

    def _get_subtitle_filename(self, default_filename: str) -> str:
        """Get subtitle filename with correct extension based on subtitle format."""
        if self.subtitle_settings.get("subtitle_format") == "ass":
            return default_filename.replace(".srt", ".ass")
        return default_filename

    def get_global_paths(self) -> dict[str, Path]:
        """Generate global shared paths."""
        global_dirs = self.output_structure.global_dirs

        return {
            "cache": self.global_output_root_path / global_dirs.cache,
            "logs": self.global_output_root_path / global_dirs.logs,
            "reports": self.global_output_root_path / global_dirs.reports,
            "temp": self.global_output_root_path / global_dirs.temp,
        }

    def get_scraper_data_path(self, product_id: str) -> Path:
        """Get path for scraped product data in simplified structure."""
        from src.utils import sanitize_filename

        safe_product_id = sanitize_filename(product_id)
        product_dir = self.global_output_root_path / safe_product_id

        return product_dir / self.output_structure.product_files.scraped_data

    def get_expected_global_paths(self) -> set[Path]:
        """Generate expected global directory paths."""
        expected = set()
        global_paths = self.get_global_paths()

        for path in global_paths.values():
            expected.add(path)

        return expected

    # Legacy method name for backward compatibility
    def get_video_project_paths(
        self, product_id: str, profile_name: str
    ) -> dict[str, Path]:
        """Legacy method - redirects to get_product_paths for backward compatibility."""
        return self.get_product_paths(product_id, profile_name)

    def cleanup_outputs_directory(self, dry_run: bool | None = None) -> dict[str, Any]:
        """Clean up unexpected files and directories in outputs directory.

        Args:
        ----
            dry_run: Override config dry_run setting if provided


        Returns:
        -------
            Dictionary with cleanup statistics and actions taken

        """
        if not self.cleanup_settings.enabled:
            logger.info("Cleanup is disabled in configuration")
            return {"status": "disabled", "actions": []}

        # Override dry_run if explicitly provided
        is_dry_run = dry_run if dry_run is not None else self.cleanup_settings.dry_run

        logger.info(f"Starting outputs directory cleanup (dry_run={is_dry_run})")

        cleanup_report: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": is_dry_run,
            "config": {
                "max_age_days": self.cleanup_settings.max_age_days,
                "preserve_patterns": self.cleanup_settings.preserve_patterns,
                "force_cleanup_patterns": self.cleanup_settings.force_cleanup_patterns,
            },
            "actions": [],
            "statistics": {
                "files_removed": 0,
                "directories_removed": 0,
                "bytes_freed": 0,
                "errors": 0,
            },
        }

        if not self.global_output_root_path.exists():
            logger.info(
                f"Outputs directory does not exist: {self.global_output_root_path}"
            )
            return cleanup_report

        # Get expected paths
        expected_paths = self.get_expected_global_paths()
        cutoff_date = datetime.now() - timedelta(
            days=self.cleanup_settings.max_age_days
        )

        # Walk through all files and directories
        for item in self.global_output_root_path.rglob("*"):
            try:
                # Skip if path is expected (or parent of expected path)
                if self._is_path_expected(item, expected_paths):
                    continue

                # Check age requirement for files
                if item.is_file():
                    file_age = datetime.fromtimestamp(item.stat().st_mtime)
                    if file_age > cutoff_date and not self._should_force_cleanup(item):
                        continue

                # Check preserve patterns (skip if should preserve)
                if self._should_preserve(item):
                    continue

                # Perform cleanup
                action = self._cleanup_item(item, is_dry_run)
                if action:
                    cleanup_report["actions"].append(action)
                    if action["action"] == "removed_file":
                        cleanup_report["statistics"]["files_removed"] += 1
                        cleanup_report["statistics"]["bytes_freed"] += action.get(
                            "size", 0
                        )
                    elif action["action"] == "removed_directory":
                        cleanup_report["statistics"]["directories_removed"] += 1

            except Exception as e:
                error_msg = f"Error processing {item}: {e}"
                logger.error(error_msg)
                cleanup_report["actions"].append(
                    {
                        "action": "error",
                        "path": str(item),
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                cleanup_report["statistics"]["errors"] += 1

        # Clean up empty directories if configured
        if self.cleanup_settings.cleanup_empty_dirs:
            self._cleanup_empty_directories(cleanup_report, is_dry_run)

        # Save report if configured
        if self.cleanup_settings.create_report and not is_dry_run:
            self._save_cleanup_report(cleanup_report)

        files_removed = cleanup_report["statistics"]["files_removed"]
        dirs_removed = cleanup_report["statistics"]["directories_removed"]
        bytes_freed = cleanup_report["statistics"]["bytes_freed"]
        logger.info(
            f"Cleanup completed: {files_removed} files, "
            f"{dirs_removed} directories removed, {bytes_freed} bytes freed"
        )

        return cleanup_report

    def _is_path_expected(self, path: Path, expected_paths: set[Path]) -> bool:
        """Check if a path is expected based on configured structure."""
        # Check if path itself is expected
        if path in expected_paths:
            return True

        # Check if path is under any expected directory
        for expected in expected_paths:
            try:
                path.relative_to(expected)
                return True
            except ValueError:
                continue

        # Check if path matches expected patterns
        rel_path = path.relative_to(self.global_output_root_path)

        # Videos structure: videos/{product_id}/{profile_name}/...
        if (
            rel_path.parts
            and rel_path.parts[0] == "videos"  # Static videos directory
            and len(rel_path.parts) >= 3
        ):  # Has product_id and profile_name
            return True

        # Scraper structure: data/{platform}/{run_id}/...
        return bool(
            rel_path.parts
            and rel_path.parts[0] == "data"  # Static data directory
            and len(rel_path.parts) >= 3
        )  # Has platform and run_id

    def _should_preserve(self, path: Path) -> bool:
        """Check if path matches preserve patterns."""
        rel_path = path.relative_to(self.global_output_root_path)
        rel_str = str(rel_path)

        for pattern in self.cleanup_settings.preserve_patterns:
            if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(path.name, pattern):
                return True
        return False

    def _should_force_cleanup(self, path: Path) -> bool:
        """Check if path matches force cleanup patterns."""
        rel_path = path.relative_to(self.global_output_root_path)
        rel_str = str(rel_path)

        for pattern in self.cleanup_settings.force_cleanup_patterns:
            if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(path.name, pattern):
                return True
        return False

    def _cleanup_item(self, path: Path, dry_run: bool) -> dict[str, Any] | None:
        """Clean up a single file or directory."""
        action: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "path": str(path.relative_to(self.global_output_root_path)),
        }

        try:
            if path.is_file():
                size = path.stat().st_size
                action.update(
                    {
                        "action": (
                            "removed_file" if not dry_run else "would_remove_file"
                        ),
                        "size": size,
                    }
                )

                if not dry_run:
                    path.unlink()
                    logger.debug(f"Removed file: {path}")
                else:
                    logger.debug(f"Would remove file: {path}")

            elif path.is_dir():
                action_name = (
                    "removed_directory" if not dry_run else "would_remove_directory"
                )
                action.update(
                    {
                        "action": action_name,
                    }
                )

                if not dry_run:
                    shutil.rmtree(path)
                    logger.debug(f"Removed directory: {path}")
                else:
                    logger.debug(f"Would remove directory: {path}")

            return action

        except Exception as e:
            logger.error(f"Failed to remove {path}: {e}")
            action.update(
                {
                    "action": "error",
                    "error": str(e),
                }
            )
            return action

    def _cleanup_empty_directories(self, report: dict[str, Any], dry_run: bool) -> None:
        """Remove empty directories after file cleanup."""
        # Walk from deepest to shallowest to remove nested empty dirs
        for item in sorted(
            self.global_output_root_path.rglob("*"),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            if item.is_dir():
                try:
                    # Check if directory is empty and not an expected base directory
                    if not any(item.iterdir()) and not self._is_expected_base_directory(
                        item
                    ):
                        action_name = (
                            "removed_empty_directory"
                            if not dry_run
                            else "would_remove_empty_directory"
                        )
                        relative_path = str(
                            item.relative_to(self.global_output_root_path)
                        )
                        action = {
                            "action": action_name,
                            "path": relative_path,
                            "timestamp": datetime.now().isoformat(),
                        }

                        if not dry_run:
                            item.rmdir()
                            logger.debug(f"Removed empty directory: {item}")
                            report["statistics"]["directories_removed"] += 1
                        else:
                            logger.debug(f"Would remove empty directory: {item}")

                        report["actions"].append(action)

                except OSError:
                    # Directory not empty or permission error
                    pass

    def _is_expected_base_directory(self, path: Path) -> bool:
        """Check if directory is an expected base directory.

        These directories should not be removed.
        """
        expected_bases = {
            self.global_output_root_path / "videos",  # Static videos directory
            self.global_output_root_path / "data",  # Static scraper data directory
            self.global_output_root_path / self.output_structure.global_dirs.logs,
            self.global_output_root_path / self.output_structure.global_dirs.temp,
            self.global_output_root_path / self.output_structure.global_dirs.cache,
        }
        return path in expected_bases

    def _save_cleanup_report(self, report: dict[str, Any]) -> None:
        """Save cleanup report to file."""
        try:
            report_path = (
                self.global_output_root_path / self.cleanup_settings.report_file
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Cleanup report saved to: {report_path}")

        except Exception as e:
            logger.error(f"Failed to save cleanup report: {e}")


def load_video_config(config_path: Path) -> VideoConfig:
    """Load video configuration from YAML file.

    Args:
    ----
        config_path: Path to the video configuration YAML file

    Returns:
    -------
        VideoConfig: Parsed and validated configuration object

    Raises:
    ------
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails

    """
    logger.info(f"Loading video config from: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Video config file not found: {config_path}")
    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict):
            raise ValueError("Config file is not a valid dictionary.")
        return VideoConfig(**config_data)
    except ValidationError as e:
        logger.error(f"Config validation error: {e}")
        raise ValueError("Config validation failed.") from e
    except Exception as e:
        logger.error(f"Error parsing config data: {e}", exc_info=True)
        raise ValueError("Unexpected error during config parsing.") from e
