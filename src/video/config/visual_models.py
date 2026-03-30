# src/video/config/visual_models.py
"""Visual configuration models for video, images, and media settings."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.video.config.constants import (
    ASSEMBLER_IMAGE_LOOP,
    ASSEMBLER_PAD_COLOR,
)


class CTADetectionSettings(BaseModel):
    min_cta_duration: float = Field(
        0.5, description="Minimum total duration (seconds) for detected CTA windows"
    )
    default_cta_duration: float = Field(
        5.0, description="Default CTA window duration (seconds) when detection fails"
    )
    fallback_duration: float = Field(
        9999.0, description="Fallback duration (seconds) when voiceover unavailable"
    )


class VideoSettings(BaseModel):
    resolution: tuple[int, int] = Field(
        ..., description="Video resolution as (width, height)"
    )
    frame_rate: int
    output_codec: str = Field("libx264")
    output_pixel_format: str = Field("yuv420p")
    output_preset: str = Field("medium")
    image_width_percent: float = Field(1.0)
    image_top_position_percent: float = Field(0.0)
    image_vertical_align: Literal["top", "center"] = Field("center")
    default_image_duration_sec: float = Field(3.0)
    transition_duration_sec: float = Field(0.5)
    total_duration_limit_sec: int = Field(90)
    video_duration_tolerance_sec: float = Field(1.0)
    min_video_file_size_mb: float = Field(0.1)
    inter_product_delay_min_sec: float = Field(1.5)
    inter_product_delay_max_sec: float = Field(4.0)
    min_visual_segment_duration_sec: float = Field(0.1)
    dynamic_image_count_limit: int = Field(
        25, description="Maximum images to use in dynamic image count mode"
    )
    verification_probe_timeout_sec: int = Field(30)
    preserve_aspect_ratio: bool = Field(True)
    default_max_chars_per_line: int = Field(20)  # Configurable via YAML
    subtitle_box_border_width: int = Field(5)  # Configurable via YAML
    image_loop: int = Field(ASSEMBLER_IMAGE_LOOP)
    pad_color: str = Field(ASSEMBLER_PAD_COLOR)

    # Media validation requirements (must match scraper config)
    min_total_media: int = Field(3, description="Minimum total media files required")
    min_images_if_no_video: int = Field(5, description="Minimum images when no videos")
    min_images_with_video: int = Field(
        2, description="Minimum images when videos exist"
    )

    # Video assembly configuration fields (Requirement 7, 10)
    video_assembly_mode: Literal[
        "sequential", "single_best", "mixed_media", "video_first_fallback"
    ] = Field(
        "sequential",
        description="Video assembly strategy mode",
    )
    video_aspect_mode: Literal["letterbox", "crop-to-fit", "smart-scale"] = Field(
        "letterbox",
        description="Aspect ratio handling (letterbox/crop-to-fit/smart-scale)",
    )
    video_audio_handling: Literal["remove", "mixed"] = Field(
        "remove",
        description="Audio handling (remove original audio or mix with voiceover)",
    )
    video_original_volume: float = Field(
        -20.0,
        ge=-60.0,
        le=0.0,
        description="Original video volume adjustment in dB (range: -60 to 0)",
    )
    video_transition_duration: float = Field(
        0.5, description="Duration of transitions between video clips in seconds"
    )
    enable_format_normalization: bool = Field(
        True,
        description="Enable video format normalization (H.264, 30fps, yuv420p)",
    )
    video_cache_dir: str = Field(
        "cache/videos", description="Directory for cached normalized videos"
    )

    # Subtitle space calculation
    reserved_space_font_multiplier: float = Field(
        1.3,
        description=(
            "Font height multiplier for calculating subtitle reserved space. "
            "Determines how much vertical space to reserve for subtitles "
            "based on font size. Higher values = more reserved space."
        ),
    )
    base_font_height_percent: float = Field(
        0.05,
        description=(
            "Base font size as percentage of frame height (5% default). "
            "Used for subtitle height estimation in visual_builder.py."
        ),
    )
    fallback_image_top_percent: float = Field(
        0.15,
        description=(
            "Fallback top position for images when video profile expects videos "
            "but only images are available (15% from top)."
        ),
    )
    fallback_image_width_percent: float = Field(
        0.85,
        description=(
            "Fallback width for images when video profile expects videos "
            "but only images are available (85% of frame width)."
        ),
    )
    default_subtitle_reserved_space: float = Field(
        0.15,
        description=(
            "Default subtitle reserved space as fraction of frame height "
            "(0.0-1.0). Fallback value used when subtitle settings are "
            "unavailable. Affects vertical positioning of video content."
        ),
    )

    # Video content positioning defaults
    video_top_position_percent: float = Field(
        0.10,
        description=(
            "Default top position for video content as fraction of frame "
            "height (0.0-1.0). Controls where video content starts vertically "
            "when no profile override exists. "
            "Lower values = content positioned higher on frame."
        ),
    )
    video_content_height_percent: float = Field(
        0.75,
        description=(
            "Default height for video content as fraction of frame height "
            "(0.0-1.0). Determines vertical space allocated to video content. "
            "Larger values = more space for video, less for subtitles/borders."
        ),
    )
    video_vertical_align: str = Field(
        "top",
        description=(
            "Vertical alignment for video content: 'top' uses "
            "video_top_position_percent, 'center' centers in frame."
        ),
    )

    # Video duration constraints
    min_trimmed_video_duration: float = Field(
        0.1,
        description="Minimum duration in seconds for trimmed video clips. "
        "Videos trimmed shorter than this will be rejected. "
        "Prevents extremely short clips that may cause playback issues.",
    )
    min_last_video_duration: float = Field(
        0.5,
        description="Minimum duration in seconds for the last video in a sequence. "
        "Last video must be at least this long after trimming. "
        "Ensures smooth ending without abrupt cuts.",
    )

    # FPS settings for video normalization
    target_fps: float = Field(
        30.0,
        description="Target frame rate for video normalization. "
        "Videos with different FPS will be converted to this rate. "
        "Standard value is 30.0 for consistent playback.",
    )
    fps_tolerance: float = Field(
        0.1,
        description="Acceptable FPS difference threshold for normalization checks. "
        "If actual FPS differs from target by more than this, normalization occurs. "
        "Prevents unnecessary re-encoding for minor FPS variations.",
    )
    default_fps_string: str = Field(
        "30/1",
        description="FFmpeg format string for default frame rate. "
        "Used in FFmpeg filters (format: numerator/denominator). "
        "Must match target_fps value.",
    )

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


class StockMediaSettings(BaseModel):
    pexels_api_key_env_var: str
    source: str = Field("Pexels")


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

    # ---- PER-PROFILE IMAGE SETTINGS ----
    # Image positioning and sizing overrides
    image_width_percent: float | None = Field(
        None, description="Override global image width as percentage of frame (0.0-1.0)"
    )
    image_top_position_percent: float | None = Field(
        None, description="Override global image top position as percentage (0.0-1.0)"
    )
    preserve_aspect_ratio: bool | None = Field(
        None, description="Override global aspect ratio preservation setting"
    )

    # ---- PER-PROFILE VIDEO ASSEMBLY SETTINGS ----
    # Video assembly configuration overrides (Requirement 7, 10)
    video_assembly_mode: (
        Literal["sequential", "single_best", "mixed_media", "video_first_fallback"]
        | None
    ) = Field(None, description="Override video assembly strategy mode")
    video_aspect_mode: Literal["letterbox", "crop-to-fit", "smart-scale"] | None = (
        Field(None, description="Override aspect ratio handling mode")
    )
    video_audio_handling: Literal["remove", "mixed"] | None = Field(
        None, description="Override video audio handling (remove or mix)"
    )
    video_original_volume: float | None = Field(
        None,
        ge=-60.0,
        le=0.0,
        description="Override video audio volume in dB (-60 to 0)",
    )
    video_transition_duration: float | None = Field(
        None, description="Override video transition duration in seconds"
    )
    enable_format_normalization: bool | None = Field(
        None, description="Override format normalization setting"
    )
    video_cache_dir: str | None = Field(
        None, description="Override video cache directory path"
    )

    # ---- PER-PROFILE VIDEO POSITIONING SETTINGS ----
    video_top_position_percent: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Video vertical start position as fraction (0.0-1.0)",
    )
    video_content_height_percent: float | None = Field(
        None, ge=0.0, le=1.0, description="Video height as fraction of frame (0.0-1.0)"
    )
    video_vertical_align: str | None = Field(
        None, description="Video vertical alignment: 'top' or 'center'"
    )

    # ---- PER-PROFILE SUBTITLE SETTINGS ----
    # Complete unified subtitle configuration overrides
    subtitle_anchor: str | None = Field(
        None,
        description=(
            "Override subtitle anchor: top, center, bottom, "
            "above_content, below_content"
        ),
    )
    subtitle_margin: float | None = Field(
        None,
        description="Override subtitle margin as fraction of frame height (0.0-0.5)",
    )
    subtitle_content_aware: bool | None = Field(
        None, description="Override content-aware positioning setting"
    )
    subtitle_style_preset: str | None = Field(
        None,
        description="Override style preset: minimal, modern, bold, random",
    )
    subtitle_font_size_scale: float | None = Field(
        None, description="Override font size scale factor (0.5-2.0)"
    )
    subtitle_horizontal_alignment: str | None = Field(
        None, description="Override text alignment: left, center, right"
    )

    # Platform safe zone overrides (per-profile, fractions of frame)
    subtitle_safe_zone_min_x: float | None = Field(
        None, description="Override left safe zone boundary"
    )
    subtitle_safe_zone_max_x: float | None = Field(
        None, description="Override right safe zone boundary"
    )
    subtitle_safe_zone_min_y: float | None = Field(
        None, description="Override top safe zone boundary"
    )
    subtitle_safe_zone_max_y: float | None = Field(
        None, description="Override bottom safe zone boundary"
    )

    # Advanced subtitle styling overrides
    subtitle_font_name: str | None = Field(
        None, description="Override subtitle font family"
    )
    subtitle_font_color: str | None = Field(
        None, description="Override subtitle text color (ASS format: &H00RRGGBB)"
    )
    subtitle_outline_color: str | None = Field(
        None, description="Override subtitle outline color (ASS format: &H00RRGGBB)"
    )
    subtitle_background_color: str | None = Field(
        None, description="Override subtitle background color (ASS format: &H00RRGGBB)"
    )
    subtitle_randomize_fonts: bool | None = Field(
        None, description="Override font randomization setting"
    )
    subtitle_randomize_colors: bool | None = Field(
        None, description="Override color randomization setting"
    )
    subtitle_randomize_effects: bool | None = Field(
        None, description="Override effect randomization setting"
    )

    # Text formatting overrides
    subtitle_max_line_length: int | None = Field(
        None, description="Override maximum characters per subtitle line"
    )
    subtitle_max_words_per_line: int | None = Field(
        None,
        description=(
            "Override maximum words per subtitle line (0 to disable word-based limit)"
        ),
    )
    subtitle_max_subtitle_width_fraction: float | None = Field(
        None,
        description=(
            "Override max subtitle width as fraction of frame width (0.0-1.0)"
        ),
    )
    subtitle_max_duration: float | None = Field(
        None, description="Override maximum subtitle duration in seconds"
    )
    subtitle_min_duration: float | None = Field(
        None, description="Override minimum subtitle duration in seconds"
    )

    # Manual selection overrides (for testing/debugging)
    subtitle_selected_font: str | None = Field(
        None, description="Override with specific font (bypasses randomization)"
    )
    subtitle_selected_color_pair: str | None = Field(
        None, description="Override with specific color pair name"
    )

    # ---- TWO-PART SUBTITLE SYSTEM ----
    # Per-profile overrides for two-part subtitle system
    two_part_subtitles_enabled: bool | None = Field(
        None, description="Override two-part subtitle system enabled/disabled"
    )
    two_part_subtitles_upper_enabled: bool | None = Field(
        None, description="Override upper subtitle line enabled/disabled"
    )
    two_part_subtitles_upper_source_field: str | None = Field(
        None,
        description=(
            "Override field name to use for upper subtitle "
            "(e.g. 'shortened_affiliate_link')"
        ),
    )
    two_part_subtitles_upper_custom_url: str | None = Field(
        None,
        description=(
            "Override with custom URL to display in upper subtitle "
            "(overrides source_field when set)"
        ),
    )
    two_part_subtitles_upper_anchor: str | None = Field(
        None, description="Override upper subtitle anchor: top, above_content, etc."
    )
    two_part_subtitles_upper_margin: float | None = Field(
        None, description="Override upper subtitle margin as fraction (0.0-0.5)"
    )
    two_part_subtitles_upper_font_size_scale: float | None = Field(
        None, description="Override upper subtitle font size scale (0.5-2.0)"
    )
    two_part_subtitles_upper_style_preset: str | None = Field(
        None, description="Override upper subtitle style preset: minimal, modern, bold"
    )
    two_part_subtitles_upper_use_full_duration: bool | None = Field(
        None, description="Override upper subtitle to display for full video duration"
    )
    two_part_subtitles_upper_randomize_effects: bool | None = Field(
        None, description="Override upper subtitle effect randomization"
    )
    two_part_subtitles_upper_prefix_replace: str | None = Field(
        None, description="Replace URL prefix (e.g., 'https://' → 'Product: ')"
    )
    two_part_subtitles_lower_enabled: bool | None = Field(
        None, description="Override lower subtitle line enabled/disabled"
    )
    two_part_subtitles_lower_anchor: str | None = Field(
        None, description="Override lower subtitle anchor: bottom, below_content, etc."
    )
    two_part_subtitles_lower_margin: float | None = Field(
        None, description="Override lower subtitle margin as fraction (0.0-0.5)"
    )


class MergedSubtitleSettings(BaseModel):
    """Merged subtitle settings (global + profile overrides)."""

    model_config = ConfigDict(extra="allow")

    # Core positioning
    anchor: str = "bottom"
    margin: float = 0.1
    content_aware: bool = True
    style_preset: str = "modern"
    font_size_scale: float = 1.0
    horizontal_alignment: str = "center"

    # Font and color
    font_name: str = "Arial"
    font_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    back_color: str | None = None
    randomize_effects: bool = False

    # Text formatting
    max_line_length: int = 38
    max_words_per_line: int = 3
    max_subtitle_duration: float = 4.5
    min_subtitle_duration: float = 0.4
    max_subtitle_width_fraction: float = 0.67

    # Advanced
    enabled: bool = True
    font_directory: str = "static/fonts"
    font_size_percent: float = 0.05
    font_width_to_height_ratio: float = 0.5

    # Randomization
    randomize_fonts: bool = False
    randomize_colors: bool = False
    available_fonts: list[Any] = Field(default_factory=list)
    available_color_combinations: list[Any] = Field(default_factory=list)

    # Output
    temp_subtitle_dir: str = "temp"
    temp_subtitle_filename: str = "captions.srt"
    save_srt_with_video: bool = True
    subtitle_format: str = "srt"
    script_paths: list[Any] = Field(default_factory=list)
    bold: bool = False
    outline_thickness: int = 2
    shadow: int = 0

    # Manual overrides
    selected_font: str | None = None
    selected_color_pair: str | None = None

    # Two-part subtitle system (flat keys)
    two_part_subtitles_enabled: bool = False
    two_part_subtitles_upper_enabled: bool = True
    two_part_subtitles_upper_source_field: str = "shortened_affiliate_link"
    two_part_subtitles_upper_custom_url: str | None = None
    two_part_subtitles_upper_anchor: str = "above_content"
    two_part_subtitles_upper_margin: float = 0.08
    two_part_subtitles_upper_font_size_scale: float = 0.75
    two_part_subtitles_upper_style_preset: str = "minimal"
    two_part_subtitles_upper_use_full_duration: bool = True
    two_part_subtitles_upper_randomize_effects: bool = False
    two_part_subtitles_upper_prefix_replace: str | None = None
    two_part_subtitles_lower_enabled: bool = True
    two_part_subtitles_lower_anchor: str = "below_content"
    two_part_subtitles_lower_margin: float = 0.05


class ProfileInfo(BaseModel):
    """Profile metadata."""

    name: str
    description: str = ""
    use_scraped_images: bool = True
    use_scraped_videos: bool = False
    use_stock_images: bool = False
    use_stock_videos: bool = False
    stock_image_count: int = 0
    stock_video_count: int = 0
    use_dynamic_image_count: bool = False


class MergedProfileSettings(BaseModel):
    """Typed container for merged profile settings."""

    video_settings: "VideoSettings"
    subtitle_settings: MergedSubtitleSettings
    profile: ProfileInfo


class VideoProcessingSettings(BaseModel):
    """Configuration for video processing and FFmpeg operations."""

    ffmpeg_probe_streams: str = Field("v:0")
    ffmpeg_probe_entries: str = Field("stream=width,height")
    ffmpeg_probe_format: str = Field("csv=s=x:p=0")
    video_stream_check_timeout_sec: int = Field(30)
    min_frame_count: int = Field(1)
    visual_aspect_ratio_tolerance: float = Field(0.01)
    visual_scaling_precision: int = Field(2)


class MediaValidationSettings(BaseModel):
    """Configuration for media validation thresholds."""

    # Image validation parameters
    min_high_res_dimension: int = Field(
        1500, description="Minimum dimension for high-resolution images"
    )
    min_high_res_file_size: int = Field(
        10000, description="Minimum file size for high-resolution images in bytes"
    )
