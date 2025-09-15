# src/video/video_config.py
import fnmatch
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
)

# Import file handling defaults from utils to avoid circular imports
from src.utils import MAX_FILENAME_LENGTH

logger = logging.getLogger(__name__)

TTS_SPEAKING_RATE_MIN = 0.25
TTS_SPEAKING_RATE_MAX = 4.0
TTS_PITCH_MIN = -20.0
TTS_PITCH_MAX = 20.0
LLM_MAX_TOKENS = 350
LLM_TEMPERATURE = 0.7
LLM_TIMEOUT_SECONDS = 90
FONT_FILE_EXTENSIONS = [".ttf", ".otf"]
FONT_REGULAR_SUFFIXES = ["-regular", "-r"]
DEFAULT_FALLBACK_FONT = "Arial"
FALLBACK_FONT_ALTERNATIVES = ["Montserrat", "Rubik", "Poppins", "Gabarito"]
SRT_TIME_SEPARATOR = " --> "
SRT_BLOCK_SEPARATOR = "\n\n"
SRT_MIN_BLOCK_LINES = 3
SRT_TIME_HOUR_SEPARATOR = ":"
SRT_TIME_MINUTE_SEPARATOR = ":"
SRT_TIME_SECOND_SEPARATOR = ","
SRT_HOURS_IN_SECONDS = 3600
SRT_MINUTES_IN_SECONDS = 60
SRT_MILLISECONDS_DIVISOR = 1000
SRT_LINE_IDENTIFIER = "-->"
SRT_ENCODING = "utf-8"
TEXT_NORMALIZATION_PATTERN = r"[^\w\s]"
TEXT_WHITESPACE_PATTERN = r"\s+"
TEXT_WHITESPACE_REPLACEMENT = " "
ASS_COLOR_PATTERN = r"&H(?:(\w{2}))?(\w{2})(\w{2})(\w{2})"
ASS_DEFAULT_ALPHA = "00"
RGB_HEX_FORMAT = "0x{red}{green}{blue}"
RGB_OPACITY_FORMAT = "{rgb_hex}@{opacity:.2f}"
FULL_OPACITY_THRESHOLD = 0.99
DEFAULT_WHISPER_MODEL_DIR = "~/.cache/whisper_models"
WHISPER_WORD_LEVEL_TIMING_MIN_CONFIDENCE = 0.5
WHISPER_MIN_SEGMENT_DURATION_SEC = 0.1
WHISPER_MAX_SEGMENT_DURATION_SEC = 5.0
WHISPER_MIN_WORDS_PER_SEGMENT = 3
WHISPER_MAX_WORDS_PER_SEGMENT = 10
WHISPER_MAX_CHARS_PER_LINE = 42
WHISPER_MIN_SEGMENT_GAP_SEC = 0.1
ASSEMBLER_DEFAULT_MAX_CHARS_PER_LINE = 20
ASSEMBLER_SUBTITLE_BOX_BORDER_WIDTH = 5
ASSEMBLER_IMAGE_LOOP = 1
ASSEMBLER_PAD_COLOR = "black"

# Freesound API defaults
FREESOUND_DEFAULT_MAX_RESULTS = 15
FREESOUND_DEFAULT_SEARCH_TIMEOUT_SEC = 10
FREESOUND_DEFAULT_DOWNLOAD_TIMEOUT_SEC = 60
FREESOUND_TOKEN_EXPIRY_SEC = 3600
FREESOUND_TOKEN_REFRESH_BUFFER_SEC = 60
FREESOUND_DOWNLOAD_CHUNK_SIZE = 8192 * 4

# LLM API defaults
LLM_MODEL_FETCH_TIMEOUT_SEC = 15
DOWNLOAD_DEFAULT_TIMEOUT_SEC = 30
DOWNLOAD_RETRY_ATTEMPTS = 3
DOWNLOAD_RETRY_MIN_WAIT_SEC = 2
DOWNLOAD_RETRY_MAX_WAIT_SEC = 10
LLM_RETRY_MULTIPLIER = 1
LLM_RETRY_MIN_WAIT_SEC = 2
LLM_RETRY_MAX_WAIT_SEC = 30
LLM_RETRY_ATTEMPTS = 3

# Subtitle positioning fallback
SUBTITLE_FALLBACK_SPACING_PERCENT = 0.02


# Legacy positioning settings - DEPRECATED
# Use UnifiedSubtitleConfig from subtitle_positioning.py instead
class SubtitlePositioningSettings(BaseModel):
    """DEPRECATED: Legacy dynamic positioning settings.

    This class is maintained for backward compatibility only.
    New implementations should use UnifiedSubtitleConfig.
    """

    image_bottom_to_subtitle_top_spacing_percent: float = Field(
        SUBTITLE_FALLBACK_SPACING_PERCENT,
        description="DEPRECATED: Use unified subtitle config instead",
    )
    subtitle_horizontal_margin_percent: float = Field(
        0.05, description="DEPRECATED: Use unified subtitle config instead"
    )
    subtitle_box_bottom_padding_percent: float = Field(
        0.02,
        description="Safety margin at the very bottom of the frame, "
        "as a % of frame height.",
    )


class AbsolutePositioningSettings(BaseModel):
    """DEPRECATED: Legacy absolute positioning settings.

    This class is maintained for backward compatibility only.
    New implementations should use UnifiedSubtitleConfig.
    """

    x_pos: str = Field(
        "(w-tw)/2", description="DEPRECATED: Use unified subtitle config instead"
    )
    y_pos: str = Field(
        "h*0.8", description="DEPRECATED: Use unified subtitle config instead"
    )


class SubtitleSettings(BaseModel):
    enabled: bool = Field(True)
    positioning_mode: str = Field(
        "static",
        description="DEPRECATED: Legacy positioning mode. Use UnifiedSubtitleConfig.",
    )
    font_name: str = Field("Arial")
    font_directory: str = Field(
        "static/fonts", description="Directory containing .ttf/.otf font files."
    )
    font_size_percent: float = Field(0.05)
    font_width_to_height_ratio: float = Field(
        0.5,
        description="Estimated ratio for character width to font size. "
        "Used for text wrapping.",
    )
    font_color: str = Field("&H00FFFFFF")
    outline_color: str = Field("&HFF000000")
    back_color: str = Field("&H99000000")
    alignment: int = Field(
        2,
        description="Legacy alignment for simple mode. "
        "1-3 bottom, 4-6 middle, 7-9 top.",
    )
    margin_v_percent: float = Field(
        0.05,
        description="Legacy vertical margin for simple mode, as a % of frame height.",
    )
    dynamic_positioning: SubtitlePositioningSettings | None = Field(None)
    absolute_positioning: AbsolutePositioningSettings | None = Field(None)
    use_random_font: bool = Field(False)
    use_random_colors: bool = Field(False)
    available_fonts: list[str] = Field(
        ["Montserrat", "Rubik", "Poppins", "Gabarito", "DM Serif Display"]
    )
    available_color_combinations: list[tuple[str, str]] = Field(
        [
            ("&H00FFFFFF", "&HFF000000"),
            ("&H0000FFFF", "&HFF000000"),
            ("&H00CCBF51", "&HFF000000"),
            ("&H0053FB57", "&HFF000000"),
            ("&H00E9967A", "&HFF000000"),
        ]
    )
    temp_subtitle_dir: str = Field("subtitle_processing")
    temp_subtitle_filename: str = Field("captions.srt")
    save_srt_with_video: bool = Field(True)
    subtitle_format: str = Field("srt")
    script_paths: list[str] = Field(["info/script.txt"])

    # ASS-specific settings
    ass_enable_fade: bool = Field(True)
    ass_fade_in_ms: int = Field(500)
    ass_fade_out_ms: int = Field(500)
    ass_enable_karaoke: bool = Field(True)
    ass_karaoke_style: str = Field("sweep")
    ass_enable_colors: bool = Field(True)
    ass_enable_positioning: bool = Field(True)
    ass_enable_transforms: bool = Field(True)
    ass_primary_color: str = Field("&H00FFFFFF")
    ass_secondary_color: str = Field("&H00FF6B35")
    ass_outline_color: str = Field("&H00000000")
    ass_shadow_color: str = Field("&H80000000")

    # Enhanced randomization settings
    ass_randomize_effects: bool = Field(False)
    ass_randomize_fonts: bool = Field(False)
    ass_randomize_colors: bool = Field(False)

    # Layout settings for subtitles
    ass_margin_bottom_percent: float = Field(
        0.25, description="Margin from bottom as percentage of video height"
    )
    ass_closer_to_image: bool = Field(
        False, description="Position subtitles closer to images"
    )
    ass_spacing_reduction_factor: float = Field(
        0.5, description="Factor to reduce spacing when positioning closer to images"
    )

    # Font settings for subtitles
    ass_font_size: int = Field(48)  # Larger default font size
    ass_available_fonts: list[str] = Field(
        [
            "Arial",
            "Helvetica",
            "Impact",
            "Verdana",
            "Tahoma",
            "Comic Sans MS",
            "Times New Roman",
            "Courier New",
            "Georgia",
            "Trebuchet MS",
        ]
    )

    # Color palettes for randomization
    ass_color_palettes: list[dict[str, str]] = Field(
        [
            {
                "primary": "&H00FFFFFF",
                "secondary": "&H00FF6B35",
                "accent": "&H00FFD700",
            },  # White/Orange/Gold
            {
                "primary": "&H0000FFFF",
                "secondary": "&H00FF69B4",
                "accent": "&H0000FF00",
            },  # Cyan/Pink/Green
            {
                "primary": "&H00FFFF00",
                "secondary": "&H00FF0080",
                "accent": "&H008080FF",
            },  # Yellow/Purple/Light Blue
            {
                "primary": "&H0000FF00",
                "secondary": "&H00FFFF00",
                "accent": "&H00FF8C00",
            },  # Green/Yellow/Orange
            {
                "primary": "&H00FF69B4",
                "secondary": "&H009370DB",
                "accent": "&H0020B2AA",
            },  # Pink/Purple/Turquoise
        ]
    )

    # Effect types for randomization
    ass_available_effects: list[str] = Field(
        [
            "fade",
            "slide_in",
            "bounce",
            "glow",
            "shadow_shift",
            "color_wave",
            "scale_pulse",
        ]
    )

    # Animation timing factors for effects (multiplied by segment duration)
    ass_pulse_duration_factor: int = Field(
        500, description="Duration factor for pulse animations in ms"
    )
    ass_wave_duration_factor: int = Field(
        300, description="Duration factor for color wave effects in ms"
    )

    # Position settings
    ass_closer_to_image: bool = Field(True)  # Position closer to images
    ass_spacing_reduction_factor: float = Field(0.3)  # Reduce spacing by 70%
    max_subtitle_duration: float = Field(4.5)
    max_line_length: int = Field(
        38,
        description="Target maximum characters per subtitle line in SRT segmentation.",
    )
    min_subtitle_duration: float = Field(0.4)
    subtitle_split_on_punctuation: bool = Field(True)
    punctuation_marks: list[str] = Field([".", "!", "?", ";", ":", ","])
    subtitle_similarity_threshold: float = Field(0.70)
    subtitle_overlap_threshold: float = Field(65.0)
    word_timestamp_pause_threshold: float = Field(0.4)
    bold: bool = Field(True)
    outline_thickness: int = Field(1)
    shadow: bool = Field(True)
    show_debug_info: bool = False

    # Subtitle extension behavior settings
    enable_subtitle_extension: bool = Field(
        True, description="Whether to extend last subtitle to match voiceover duration"
    )
    subtitle_extension_threshold: float = Field(
        2.0, description="Minimum gap (seconds) required before extending last subtitle"
    )
    max_subtitle_extension: float = Field(
        5.0, description="Maximum seconds to extend a subtitle beyond its natural end"
    )

    # ---- UNIFIED SUBTITLE POSITIONING PARAMETERS ----
    # New unified parameters that replace the legacy positioning system
    anchor: str = Field(
        "bottom",
        description="Anchor: top, center, bottom, above_content, below_content",
    )
    margin: float = Field(
        0.1, description="Margin as fraction of frame height (0.0-0.5)"
    )
    content_aware: bool = Field(
        True, description="Adjust position based on visual content bounds"
    )
    style_preset: str = Field(
        "dynamic", description="Style preset: minimal, modern, dynamic, classic, bold"
    )
    font_size_scale: float = Field(
        1.2, description="Scale factor for font size (0.5-2.0)"
    )
    horizontal_alignment: str = Field(
        "center", description="Text alignment: left, center, right"
    )
    randomize_colors: bool = Field(
        False, description="Use random color combinations from preset palette"
    )
    randomize_effects: bool = Field(
        False, description="Use random animation effects when available"
    )

    @model_validator(mode="after")
    def validate_positioning_mode(self) -> "SubtitleSettings":
        """DEPRECATED: Legacy positioning mode validation.

        This validator is maintained for backward compatibility only.
        New implementations should use UnifiedSubtitleConfig validation.
        """
        if self.positioning_mode not in ["static", "dynamic", "absolute"]:
            # Log deprecation warning instead of raising error
            logger.warning(
                f"DEPRECATED: positioning_mode '{self.positioning_mode}' is "
                "deprecated. Use UnifiedSubtitleConfig from subtitle_positioning.py."
            )
        return self


class VideoSettings(BaseModel):
    resolution: tuple[int, int] = Field(
        ..., description="Video resolution as (width, height)"
    )
    frame_rate: int
    output_codec: str = Field("libx264")
    output_pixel_format: str = Field("yuv420p")
    output_preset: str = Field("medium")
    image_width_percent: float = Field(0.75)
    image_top_position_percent: float = Field(0.15)
    default_image_duration_sec: float = Field(3.0)
    transition_duration_sec: float = Field(0.5)
    total_duration_limit_sec: int = Field(90)
    video_duration_tolerance_sec: float = Field(1.0)
    min_video_file_size_mb: float = Field(0.1)
    inter_product_delay_min_sec: float = Field(1.5)
    inter_product_delay_max_sec: float = Field(4.0)
    min_visual_segment_duration_sec: float = Field(0.1)
    dynamic_image_count_limit: int = Field(25)
    verification_probe_timeout_sec: int = Field(30)
    preserve_aspect_ratio: bool = Field(True)
    default_max_chars_per_line: int = Field(ASSEMBLER_DEFAULT_MAX_CHARS_PER_LINE)
    subtitle_box_border_width: int = Field(ASSEMBLER_SUBTITLE_BOX_BORDER_WIDTH)
    image_loop: int = Field(ASSEMBLER_IMAGE_LOOP)
    pad_color: str = Field(ASSEMBLER_PAD_COLOR)

    @model_validator(mode="after")
    def validate_resolution(self) -> "VideoSettings":
        width, height = self.resolution
        if width <= 0 or height <= 0:
            raise ValueError("Resolution width and height must be positive")
        return self


class MediaSettings(BaseModel):
    stock_media_keywords: list[str]
    stock_video_min_duration_sec: int
    stock_video_max_duration_sec: int
    temp_media_dir: str = Field("downloaded_media_assets")
    product_title_keyword_min_length: int = Field(3)


class AudioSettings(BaseModel):
    music_volume_db: float
    voiceover_volume_db: float
    audio_mix_duration: str = Field("longest")
    background_music_paths: list[Path]
    freesound_api_key_env_var: str
    freesound_client_id_env_var: str = Field("FREESOUND_CLIENT_ID")
    freesound_client_secret_env_var: str = Field("FREESOUND_CLIENT_SECRET")  # noqa: S106
    freesound_refresh_token_env_var: str = Field("FREESOUND_REFRESH_TOKEN")  # noqa: S106
    freesound_sort: str = Field("rating_desc")
    freesound_search_query: str
    freesound_filters: str
    freesound_max_results: int
    freesound_max_search_duration_sec: int = Field(9999)
    freesound_api_timeout_sec: int = Field(FREESOUND_DEFAULT_SEARCH_TIMEOUT_SEC)
    freesound_download_timeout_sec: int = Field(FREESOUND_DEFAULT_DOWNLOAD_TIMEOUT_SEC)
    freesound_token_expiry_sec: int = Field(FREESOUND_TOKEN_EXPIRY_SEC)
    freesound_token_refresh_buffer_sec: int = Field(FREESOUND_TOKEN_REFRESH_BUFFER_SEC)
    freesound_download_chunk_size: int = Field(FREESOUND_DOWNLOAD_CHUNK_SIZE)
    output_audio_codec: str = Field("aac")
    output_audio_bitrate: str = Field("192k")
    music_fade_in_duration: float = Field(2.0)
    music_fade_out_duration: float = Field(3.0)


class GoogleCloudVoiceCriteria(BaseModel):
    language_code: str
    ssml_gender: str | None = Field(None)
    name_contains: str | None = Field(None)


class GoogleCloudTTSSettings(BaseModel):
    model_name: str
    language_code: str
    voice_selection_criteria: list[GoogleCloudVoiceCriteria] = Field(..., min_length=1)
    speaking_rate: float = Field(1.0)
    pitch: float = Field(0.0)
    volume_gain_db: float = Field(0.0)
    debug: bool = Field(False)
    api_timeout_sec: int = Field(60)
    api_max_retries: int = Field(2)
    api_retry_delay_sec: int = Field(5)
    last_word_buffer_sec: float = Field(0.3)

    @model_validator(mode="after")
    def check_audio_config_ranges(self) -> "GoogleCloudTTSSettings":
        if not (TTS_SPEAKING_RATE_MIN <= self.speaking_rate <= TTS_SPEAKING_RATE_MAX):
            logger.warning(f"Google TTS rate {self.speaking_rate} outside range.")
        if not (TTS_PITCH_MIN <= self.pitch <= TTS_PITCH_MAX):
            logger.warning(f"Google TTS pitch {self.pitch} outside range.")
        return self


class CoquiTTSSettings(BaseModel):
    model_name: str
    speaker_name: str | None = Field(None)


class TTSConfig(BaseModel):
    provider_order: list[str] = Field(..., min_length=1)
    google_cloud: GoogleCloudTTSSettings | None = Field(None)
    coqui: CoquiTTSSettings | None = Field(None)

    @model_validator(mode="after")
    def check_provider_settings_exist(self) -> "TTSConfig":
        valid_providers = []
        try:
            from src.video.tts import COQUI_AVAILABLE, GOOGLE_CLOUD_AVAILABLE
        except ImportError:
            GOOGLE_CLOUD_AVAILABLE, COQUI_AVAILABLE = False, False

        for name in self.provider_order:
            if (
                name == "google_cloud" and self.google_cloud and GOOGLE_CLOUD_AVAILABLE
            ) or (name == "coqui" and self.coqui and COQUI_AVAILABLE):
                valid_providers.append(name)
            else:
                logger.warning(
                    f"TTS provider '{name}' skipped (unavailable or config missing)."
                )
        if not valid_providers:
            # In test environments or when no providers are available,
            # allow empty provider list but log warning
            logger.warning(
                "No usable TTS providers configured/available. "
                "This configuration can only be used for testing or components "
                "that don't require TTS."
            )
            self.provider_order = []
        else:
            self.provider_order = valid_providers
        return self


class LLMSettings(BaseModel):
    provider: str
    api_key_env_var: str
    models: list[str] = Field(..., min_length=1)
    prompt_template_path: str
    target_audience: str = Field("General audience")
    base_url: str | None = Field(None)
    auto_select_free_model: bool = Field(True)
    max_tokens: int = Field(LLM_MAX_TOKENS)
    temperature: float = Field(LLM_TEMPERATURE)
    timeout_seconds: int = Field(LLM_TIMEOUT_SECONDS)


class StockMediaSettings(BaseModel):
    pexels_api_key_env_var: str
    source: str = Field("Pexels")


class FFmpegSettings(BaseModel):
    executable_path: str | None = Field(None)
    temp_ffmpeg_dir: str = Field("ffmpeg_work")
    intermediate_segment_preset: str = Field("ultrafast")
    final_assembly_timeout_sec: int = Field(600)
    rw_timeout_microseconds: int = Field(30000000)  # 30 seconds for I/O operations


class AttributionSettings(BaseModel):
    attribution_file_name: str = Field("ATTRIBUTIONS.txt")
    attribution_template: str
    attribution_entry_template: str


class VideoProfile(BaseModel):
    description: str
    use_scraped_images: bool = Field(False)
    use_scraped_videos: bool = Field(False)
    use_stock_images: bool = Field(False)
    use_stock_videos: bool = Field(False)
    stock_image_count: int = Field(0, ge=0)
    stock_video_count: int = Field(0, ge=0)
    use_dynamic_image_count: bool = Field(False)

    # Profile-specific subtitle positioning (optional)
    subtitle_positioning: dict[str, Any] | None = Field(
        None, description="Profile-specific subtitle positioning overrides"
    )


class WhisperSettings(BaseModel):
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
    no_speech_threshold: float = Field(0.4)
    condition_on_previous_text: bool = Field(True)
    task: str = Field("transcribe")
    patience: float | None = Field(None)

    # Timeout settings for Whisper processing
    base_timeout_sec: int = Field(120)
    duration_multiplier: float = Field(3.0)
    max_timeout_sec: int = Field(600)  # 10 minutes
    progress_monitor_interval_sec: int = Field(30)
    enable_resource_monitoring: bool = Field(True)
    enable_resource_cleanup: bool = Field(True)


class GoogleCloudSTTSettings(BaseModel):
    enabled: bool = Field(True)
    language_code: str = Field("en-US")
    encoding: str = Field("LINEAR16")
    sample_rate_hertz: int = Field(24000)
    use_enhanced: bool = Field(True)
    api_timeout_sec: int = Field(120)
    api_max_retries: int = Field(2)
    api_retry_delay_sec: int = Field(10)
    use_speech_adaptation_if_script_provided: bool = Field(True)
    adaptation_boost_value: float = Field(15.0, gt=0, le=20)


class ApiSettings(BaseModel):
    """Configuration for API timeouts, retries, and network settings."""

    llm_model_fetch_timeout_sec: int = Field(LLM_MODEL_FETCH_TIMEOUT_SEC)
    llm_retry_attempts: int = Field(LLM_RETRY_ATTEMPTS)
    llm_retry_min_wait_sec: int = Field(LLM_RETRY_MIN_WAIT_SEC)
    llm_retry_max_wait_sec: int = Field(LLM_RETRY_MAX_WAIT_SEC)
    llm_retry_multiplier: int = Field(LLM_RETRY_MULTIPLIER)
    stock_media_concurrent_downloads: int = Field(5)
    stock_media_search_multiplier: int = Field(2)
    stock_media_max_per_page: int = Field(80)
    default_request_timeout_sec: int = Field(15)
    default_retry_attempts: int = Field(3)
    default_retry_delay_sec: int = Field(5)
    download_timeout_sec: int = Field(DOWNLOAD_DEFAULT_TIMEOUT_SEC)
    download_retry_attempts: int = Field(DOWNLOAD_RETRY_ATTEMPTS)
    download_retry_min_wait_sec: int = Field(DOWNLOAD_RETRY_MIN_WAIT_SEC)
    download_retry_max_wait_sec: int = Field(DOWNLOAD_RETRY_MAX_WAIT_SEC)


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


class AudioProcessingSettings(BaseModel):
    """Configuration for audio processing and TTS settings."""

    coqui_gpu_enabled: bool = Field(False)
    google_tts_audio_encoding: str = Field("LINEAR16")
    min_audio_file_size_bytes: int = Field(100)
    audio_validation_timeout_sec: int = Field(30)


class VideoProcessingSettings(BaseModel):
    """Configuration for video processing and FFmpeg operations."""

    ffmpeg_probe_streams: str = Field("v:0")
    ffmpeg_probe_entries: str = Field("stream=width,height")
    ffmpeg_probe_format: str = Field("csv=s=x:p=0")
    video_stream_check_timeout_sec: int = Field(30)
    min_frame_count: int = Field(1)
    visual_aspect_ratio_tolerance: float = Field(0.01)
    visual_scaling_precision: int = Field(2)


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


class DebugSettings(BaseModel):
    """Configuration for debug output and development settings."""

    max_log_line_length: int = Field(200)
    debug_file_retention_days: int = Field(7)
    intermediate_file_cleanup: bool = Field(True)
    operation_timing_threshold_sec: float = Field(5.0)
    memory_usage_warning_mb: int = Field(1000)


class ProductFiles(BaseModel):
    """File names within each product directory"""

    scraped_data: str = Field("data.json")
    script: str = Field("script.txt")
    voiceover: str = Field("voiceover.wav")
    subtitles: str = Field("subtitles.srt")
    final_video: str = Field("video_{profile}.mp4")
    metadata: str = Field("metadata.json")
    ffmpeg_log: str = Field("ffmpeg_command.log")
    performance: str = Field("performance.json")
    attribution: str = Field("attributions.txt")


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
    product_files: ProductFiles = Field(default_factory=ProductFiles)
    product_subdirs: ProductSubdirs = Field(default_factory=ProductSubdirs)
    global_dirs: GlobalDirs = Field(default_factory=GlobalDirs)


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
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)

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
        ["*.md", "*.txt", ".gitkeep", "cache/**", "backup/**"]
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
    duration_padding_sec: float = Field(
        0.5, description="Duration padding added to prevent audio cutoff in seconds"
    )
    video_settings: VideoSettings
    media_settings: MediaSettings
    audio_settings: AudioSettings
    tts_config: TTSConfig
    llm_settings: LLMSettings
    stock_media_settings: StockMediaSettings
    ffmpeg_settings: FFmpegSettings
    attribution_settings: AttributionSettings
    subtitle_settings: SubtitleSettings
    whisper_settings: WhisperSettings
    google_cloud_stt_settings: GoogleCloudSTTSettings | None = Field(None)
    video_profiles: dict[str, VideoProfile]

    # New configuration sections for magic numbers
    api_settings: ApiSettings | None = Field(None)
    text_processing: TextProcessingSettings | None = Field(None)
    audio_processing: AudioProcessingSettings | None = Field(None)
    video_processing: VideoProcessingSettings | None = Field(None)
    filesystem: FilesystemSettings | None = Field(None)
    debug_settings: DebugSettings | None = Field(None)
    optimization_settings: OptimizationSettings | None = Field(None)

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent,
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

        font_dir_obj = Path(self.subtitle_settings.font_directory)
        self.subtitle_settings.font_directory = str(
            (self.project_root / font_dir_obj)
            if not font_dir_obj.is_absolute()
            else font_dir_obj
        )
        return self

    def get_profile(self, profile_name: str) -> VideoProfile:
        if profile_name not in self.video_profiles:
            raise KeyError(f"Video profile '{profile_name}' not found.")
        return self.video_profiles[profile_name]

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

        return {
            # Directories
            "product_root": product_dir,
            "images_dir": images_dir,
            "videos_dir": videos_dir,
            "music_dir": music_dir,
            "temp_dir": temp_dir,
            # Files (all in product root for flat structure)
            "scraped_data": product_dir / files.scraped_data,
            "script": product_dir / files.script,
            "voiceover": product_dir / files.voiceover,
            "subtitles": product_dir / self._get_subtitle_filename(files.subtitles),
            "final_video": product_dir
            / files.final_video.format(profile=safe_profile_name),
            "metadata": product_dir / files.metadata,
            "ffmpeg_log": product_dir / files.ffmpeg_log,
            "performance": product_dir / files.performance,
            # Legacy compatibility
            "project_root": product_dir,
            "working_dir": temp_dir,
            "audio_dir": temp_dir,
            "visual_dir": temp_dir,
            "text_dir": temp_dir,
            "pipeline_state": product_dir / files.metadata,  # Renamed
            "attribution": product_dir / "attributions.txt",
        }

    def _get_subtitle_filename(self, default_filename: str) -> str:
        """Get subtitle filename with correct extension based on subtitle format."""
        if self.subtitle_settings.subtitle_format == "ass":
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


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "video_producer.yaml"
)
try:
    config = load_video_config(DEFAULT_CONFIG_PATH)
    logger.info("Default video configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load default configuration: {e}")
    config = VideoConfig(
        global_output_directory="outputs",
        output_structure=OutputStructure(),  # type: ignore[call-arg]
        cleanup_settings=CleanupSettings(),  # type: ignore[call-arg]
        pipeline_timeout_sec=900,
        video_settings=VideoSettings(
            resolution=(1080, 1920),
            frame_rate=30,
            output_codec="libx264",
            output_pixel_format="yuv420p",
            output_preset="medium",
            image_width_percent=0.75,
            image_top_position_percent=0.15,
            default_image_duration_sec=3.0,
            transition_duration_sec=0.5,
            total_duration_limit_sec=90,
            video_duration_tolerance_sec=1.0,
            min_video_file_size_mb=0.1,
            inter_product_delay_min_sec=1.5,
            inter_product_delay_max_sec=4.0,
            min_visual_segment_duration_sec=0.1,
            dynamic_image_count_limit=25,
            verification_probe_timeout_sec=30,
            preserve_aspect_ratio=True,
            default_max_chars_per_line=ASSEMBLER_DEFAULT_MAX_CHARS_PER_LINE,
            subtitle_box_border_width=ASSEMBLER_SUBTITLE_BOX_BORDER_WIDTH,
            image_loop=ASSEMBLER_IMAGE_LOOP,
            pad_color=ASSEMBLER_PAD_COLOR,
        ),
        media_settings=MediaSettings(
            stock_media_keywords=["product", "showcase"],
            stock_video_min_duration_sec=5,
            stock_video_max_duration_sec=20,
            temp_media_dir="downloaded_media_assets",
            product_title_keyword_min_length=3,
        ),
        audio_settings=AudioSettings(
            music_volume_db=-20.0,
            voiceover_volume_db=3.0,
            audio_mix_duration="longest",
            background_music_paths=[
                Path("static/background-music-calm-soft-334182.mp3")
            ],
            freesound_api_key_env_var="FREESOUND_API_KEY",
            freesound_client_id_env_var="FREESOUND_CLIENT_ID",
            freesound_client_secret_env_var="FREESOUND_CLIENT_SECRET",  # noqa: S106
            freesound_refresh_token_env_var="FREESOUND_REFRESH_TOKEN",  # noqa: S106
            freesound_sort="rating_desc",
            freesound_search_query="ambient background music",
            freesound_filters="type:wav duration:[10 TO 300]",
            freesound_max_results=10,
            freesound_max_search_duration_sec=9999,
            freesound_api_timeout_sec=30,
            freesound_download_timeout_sec=300,
            freesound_token_expiry_sec=3600,
            freesound_token_refresh_buffer_sec=300,
            freesound_download_chunk_size=8192,
            output_audio_codec="aac",
            output_audio_bitrate="192k",
            music_fade_in_duration=2.0,
            music_fade_out_duration=3.0,
        ),
        tts_config=TTSConfig(
            provider_order=["google_cloud"],
            google_cloud=None,
            coqui=None,
        ),
        llm_settings=LLMSettings(
            provider="openrouter",
            api_key_env_var="OPENROUTER_API_KEY",
            models=["gpt-3.5-turbo"],
            prompt_template_path="prompts/product_script_template.txt",
            target_audience="General audience",
            base_url=None,
            auto_select_free_model=True,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            timeout_seconds=LLM_TIMEOUT_SECONDS,
        ),
        stock_media_settings=StockMediaSettings(
            pexels_api_key_env_var="PEXELS_API_KEY",
            source="Pexels",
        ),
        ffmpeg_settings=FFmpegSettings(
            executable_path=None,
            temp_ffmpeg_dir="ffmpeg_work",
            intermediate_segment_preset="ultrafast",
            final_assembly_timeout_sec=600,
            rw_timeout_microseconds=30000000,
        ),
        api_settings=ApiSettings(
            llm_model_fetch_timeout_sec=15,
            llm_retry_attempts=3,
            llm_retry_min_wait_sec=2,
            llm_retry_max_wait_sec=30,
            llm_retry_multiplier=1,
            stock_media_concurrent_downloads=5,
            stock_media_search_multiplier=2,
            stock_media_max_per_page=80,
            default_request_timeout_sec=15,
            default_retry_attempts=3,
            default_retry_delay_sec=5,
            download_timeout_sec=30,
            download_retry_attempts=3,
            download_retry_min_wait_sec=2,
            download_retry_max_wait_sec=10,
        ),
        text_processing=TextProcessingSettings(
            script_chars_per_second_estimate=15,
            script_min_duration_sec=0.05,
            subtitle_text_similarity_min_confidence=0.5,
            subtitle_min_segment_duration_sec=0.1,
            subtitle_max_segment_duration_sec=5.0,
            subtitle_min_words_per_segment=3,
            subtitle_max_words_per_segment=10,
            subtitle_max_chars_per_line=42,
            subtitle_min_segment_gap_sec=0.1,
        ),
        audio_processing=AudioProcessingSettings(
            coqui_gpu_enabled=False,
            google_tts_audio_encoding="LINEAR16",
            min_audio_file_size_bytes=100,
            audio_validation_timeout_sec=30,
        ),
        video_processing=VideoProcessingSettings(
            ffmpeg_probe_streams="v:0",
            ffmpeg_probe_entries="stream=width,height",
            ffmpeg_probe_format="csv=s=x:p=0",
            video_stream_check_timeout_sec=30,
            min_frame_count=1,
            visual_aspect_ratio_tolerance=0.01,
            visual_scaling_precision=2,
        ),
        filesystem=FilesystemSettings(
            temp_file_cleanup_delay_sec=5,
            file_operation_timeout_sec=30,
            max_filename_length=255,
            supported_image_extensions=[".jpg", ".jpeg", ".png", ".webp", ".bmp"],
            supported_video_extensions=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
            supported_audio_extensions=[".wav", ".mp3", ".aac", ".flac"],
        ),
        debug_settings=DebugSettings(
            max_log_line_length=200,
            debug_file_retention_days=7,
            intermediate_file_cleanup=True,
            operation_timing_threshold_sec=5.0,
            memory_usage_warning_mb=1000,
        ),
        attribution_settings=AttributionSettings(
            attribution_file_name="ATTRIBUTIONS.txt",
            attribution_template="",
            attribution_entry_template="",
        ),
        subtitle_settings=SubtitleSettings(
            enabled=True,
            positioning_mode="static",
            font_name="Arial",
            font_directory="fonts",
            font_size_percent=0.05,
            font_width_to_height_ratio=0.5,
            font_color="&H00FFFFFF",
            outline_color="&HFF000000",
            back_color="&H99000000",
            alignment=2,
            margin_v_percent=0.05,
            dynamic_positioning=None,
            absolute_positioning=None,
            use_random_font=False,
            use_random_colors=False,
            available_fonts=[
                "Montserrat",
                "Rubik",
                "Poppins",
                "Gabarito",
                "DM Serif Display",
            ],
            available_color_combinations=[
                ("&H00FFFFFF", "&HFF000000"),
                ("&H0000FFFF", "&HFF000000"),
                ("&H00CCBF51", "&HFF000000"),
                ("&H0053FB57", "&HFF000000"),
                ("&H00E9967A", "&HFF000000"),
            ],
            temp_subtitle_dir="subtitle_processing",
            temp_subtitle_filename="captions.srt",
            save_srt_with_video=True,
            subtitle_format="srt",
            script_paths=["info/script.txt"],
            max_subtitle_duration=4.5,
            max_line_length=38,
            min_subtitle_duration=0.4,
            subtitle_split_on_punctuation=True,
            punctuation_marks=[".", "!", "?", ";", ":", ","],
            subtitle_similarity_threshold=0.70,
            subtitle_overlap_threshold=65.0,
            word_timestamp_pause_threshold=0.4,
            bold=True,
            outline_thickness=1,
            shadow=True,
            enable_subtitle_extension=True,
            subtitle_extension_threshold=2.0,
            max_subtitle_extension=5.0,
        ),
        whisper_settings=WhisperSettings(
            enabled=True,
            model_size="small",
            model_device="cpu",
            model_in_memory=False,
            model_download_root="",
            temperature=0.0,
            language="en",
            beam_size=5,
            fp16=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.4,
            condition_on_previous_text=True,
            task="transcribe",
            patience=None,
            base_timeout_sec=120,
            duration_multiplier=3.0,
            max_timeout_sec=600,
            progress_monitor_interval_sec=30,
            enable_resource_monitoring=True,
            enable_resource_cleanup=True,
        ),
        google_cloud_stt_settings=GoogleCloudSTTSettings(
            enabled=True,
            language_code="en-US",
            encoding="LINEAR16",
            sample_rate_hertz=24000,
            use_enhanced=True,
            api_timeout_sec=120,
            api_max_retries=2,
            api_retry_delay_sec=10,
            use_speech_adaptation_if_script_provided=True,
            adaptation_boost_value=15.0,
        ),
        video_profiles={},
    )
    logger.warning("Using fallback configuration")
