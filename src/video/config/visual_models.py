# src/video/config/visual_models.py
"""Visual configuration models for video, images, and media settings."""

import logging
import warnings
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.video.config.constants import (
    ASSEMBLER_IMAGE_LOOP,
    ASSEMBLER_PAD_COLOR,
)
from src.video.config.subtitle_models import (
    PartialSubtitleSettings,
    SubtitleSettings,
)

logger = logging.getLogger(__name__)


# Legacy flat field names on VideoProfile (subtitle_*/pycaps_*/two_part_*)
# mapped to their nested PartialSubtitleSettings field path. Used by the
# @model_validator on VideoProfile to migrate profile YAML at load time.
# The values are tuples: (path, value_transform). Path is dotted so
# "pycaps.template_name" lands inside the nested pycaps dict.
_LEGACY_FLAT_TO_NESTED: dict[str, str] = {
    "subtitle_anchor": "anchor",
    "subtitle_margin": "margin",
    "subtitle_content_aware": "content_aware",
    "subtitle_style_preset": "style_preset",
    "subtitle_font_size_scale": "font_size_scale",
    "subtitle_horizontal_alignment": "horizontal_alignment",
    "subtitle_randomize_fonts": "randomize_fonts",
    "subtitle_randomize_colors": "randomize_colors",
    "subtitle_randomize_effects": "randomize_effects",
    "subtitle_max_line_length": "max_line_length",
    "subtitle_max_words_per_line": "max_words_per_line",
    "subtitle_max_subtitle_width_fraction": "max_subtitle_width_fraction",
    "subtitle_max_duration": "max_duration",
    "subtitle_min_duration": "min_duration",
    "subtitle_selected_font": "selected_font",
    "subtitle_selected_color_pair": "selected_color_pair",
    "subtitle_engine": "subtitle_engine",
}

_LEGACY_SAFE_ZONE_FIELDS = (
    "subtitle_safe_zone_min_x",
    "subtitle_safe_zone_max_x",
    "subtitle_safe_zone_min_y",
    "subtitle_safe_zone_max_y",
)

_LEGACY_PYCAPS_FIELDS: dict[str, str] = {
    "pycaps_template": "template_name",
    "pycaps_template_pool": "template_pool",
    "pycaps_renderer": "renderer",
    "pycaps_max_width_ratio": "max_width_ratio",
    "pycaps_vertical_align": "vertical_align",
    "pycaps_vertical_align_offset": "vertical_align_offset",
    "pycaps_fallback_policy": "fallback_policy",
}


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


class DisclosureSettings(BaseModel):
    """On-frame disclosure overlay (FTC `#ad`, Spain `#publi`, etc.).

    Renders a persistent text overlay in a fixed corner of every produced video
    so the disclosure is visible regardless of which subtitle engine ran or
    whether the platform clips part of the frame. Sized relative to the
    subtitle font so it stays smaller than narration captions per FTC guidance
    (50-60% of caption size) but readable on phone screens.
    """

    enabled: bool = Field(True, description="Burn the overlay on every render")
    text: str = Field("#ad", description="Disclosure text. Override per language.")
    position: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = Field(
        "top-right", description="Corner placement within the safe zone"
    )
    size_factor: float = Field(
        0.55,
        ge=0.2,
        le=1.0,
        description="Font size as a fraction of subtitle caption size",
    )
    font_color: str = Field("white", description="FFmpeg color name or hex")
    outline_color: str = Field(
        "black", description="High-contrast outline for readability on any background"
    )
    outline_thickness: int = Field(3, ge=0)
    background_enabled: bool = Field(
        True,
        description="Semi-transparent box behind text for readability on bright frames",
    )
    background_color: str = Field(
        "black@0.5",
        description="FFmpeg color@alpha syntax. 0=transparent, 1=opaque",
    )
    margin_x_percent: float = Field(
        0.04,
        ge=0.0,
        le=0.5,
        description="Horizontal margin from frame edge as fraction of width",
    )
    margin_y_percent: float = Field(
        0.12,
        ge=0.0,
        le=0.5,
        description=(
            "Vertical margin from frame edge as fraction of height. "
            "Default 12% sits below the YouTube Shorts top header (~10%) "
            "and above the TikTok bottom username block (~12%)."
        ),
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
    disclosure_overlay: DisclosureSettings = Field(
        default_factory=DisclosureSettings  # type: ignore[arg-type]
    )

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
    image_vertical_align: Literal["top", "center"] | None = Field(
        None, description="Override global image vertical alignment (top or center)"
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
    # Single nested override block. Replaces the historical 30+ flat
    # subtitle_*/pycaps_*/two_part_subtitles_* fields. Profile YAML written
    # in the legacy flat shape is migrated at load time by the
    # _migrate_legacy_subtitle_keys validator below.
    subtitle_settings: PartialSubtitleSettings | None = Field(
        None,
        description=(
            "Nested partial override for the global SubtitleSettings. Only "
            "non-None fields apply; nested models (pycaps, two_part_subtitles, "
            "safe_zone) deep-merge onto the base. See PartialSubtitleSettings."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_subtitle_keys(cls, data: Any) -> Any:
        """Translate legacy flat subtitle_*/pycaps_*/two_part_subtitles_* keys
        into the nested ``subtitle_settings`` block at load time.

        Kept for one release so external profile YAML doesn't break. Logs
        a DeprecationWarning naming each migrated key. Remove after the
        documented migration window.
        """
        if not isinstance(data, dict):
            return data

        nested = dict(data.get("subtitle_settings") or {})
        legacy_seen: list[str] = []

        for legacy_key, target_field in _LEGACY_FLAT_TO_NESTED.items():
            if legacy_key in data and data[legacy_key] is not None:
                nested.setdefault(target_field, data[legacy_key])
                legacy_seen.append(legacy_key)

        safe_zone_block = dict(nested.get("safe_zone") or {})
        for sz_key in _LEGACY_SAFE_ZONE_FIELDS:
            if sz_key in data and data[sz_key] is not None:
                # subtitle_safe_zone_min_x -> min_x
                short = sz_key.removeprefix("subtitle_safe_zone_")
                safe_zone_block.setdefault(short, data[sz_key])
                legacy_seen.append(sz_key)
        if safe_zone_block:
            nested["safe_zone"] = safe_zone_block

        pycaps_block = dict(nested.get("pycaps") or {})
        for pc_key, target_field in _LEGACY_PYCAPS_FIELDS.items():
            if pc_key in data and data[pc_key] is not None:
                pycaps_block.setdefault(target_field, data[pc_key])
                legacy_seen.append(pc_key)
        if pycaps_block:
            nested["pycaps"] = pycaps_block

        if "two_part_subtitles" in data and isinstance(
            data["two_part_subtitles"], dict
        ):
            existing = nested.get("two_part_subtitles") or {}
            nested["two_part_subtitles"] = {
                **data["two_part_subtitles"],
                **existing,
            }
            legacy_seen.append("two_part_subtitles")

        if legacy_seen:
            warnings.warn(
                "VideoProfile: legacy flat subtitle keys are deprecated; "
                "migrate to a nested 'subtitle_settings' block. Migrated "
                f"this run: {sorted(set(legacy_seen))}",
                DeprecationWarning,
                stacklevel=2,
            )
            for key in set(legacy_seen):
                data.pop(key, None)

        if nested:
            data["subtitle_settings"] = nested

        return data


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
    subtitle_settings: SubtitleSettings
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
