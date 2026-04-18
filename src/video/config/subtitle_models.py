# src/video/config/subtitle_models.py
"""Subtitle configuration models for effects and segmentation."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.video.config.constants import (
    FONT_FILE_EXTENSIONS,
    FONT_REGULAR_SUFFIXES,
)


class SubtitleEffectsSettings(BaseModel):
    """Configuration for ASS subtitle effects and animations."""

    # Karaoke timing parameters
    karaoke_timing_min_ms: int = Field(
        20, description="Minimum karaoke timing per word in milliseconds"
    )
    karaoke_timing_max_ms: int = Field(
        200, description="Maximum karaoke timing per word in milliseconds"
    )

    # Karaoke visual effect parameters
    karaoke_primary_color: str = Field(
        "&H00FFFFFF", description="Primary color (before sweep, ASS format)"
    )
    karaoke_secondary_color: str = Field(
        "&H0000FFFF", description="Secondary color (fill during sweep, ASS format)"
    )
    karaoke_outline_color: str | None = Field(
        None, description="Outline color for karaoke (optional, ASS format)"
    )
    karaoke_use_fill: bool = Field(
        True, description="Use \\kf (fill) instead of \\k (timing only)"
    )

    # Effect duration factors (multiplied by segment duration)
    pulse_duration_factor: int = Field(
        500, description="Duration factor for pulse animations in ms"
    )
    bounce_duration_factor: int = Field(
        300, description="Duration factor for bounce animations in ms"
    )
    glow_duration_factor: int = Field(
        400, description="Duration factor for glow effects in ms"
    )

    # Scale effect parameters
    pulse_scale_max: int = Field(
        110, description="Maximum scale percentage for pulse effect"
    )
    pulse_scale_normal: int = Field(
        100, description="Normal scale percentage for pulse effect"
    )

    # Movement effect parameters
    movement_distance_pixels: int = Field(
        50, description="Vertical movement distance in pixels for movement effect"
    )

    # Rotation bounce parameters
    bounce_rotation_max: int = Field(
        5, description="Maximum rotation degrees for bounce effect"
    )
    bounce_rotation_min: int = Field(
        -5, description="Minimum rotation degrees for bounce effect"
    )
    bounce_rotation_rest: int = Field(
        0, description="Rest rotation degrees for bounce effect"
    )

    # Typewriter effect parameters
    typewriter_char_reveal_max_sec: float = Field(
        0.1, description="Maximum character reveal time for typewriter effect"
    )
    typewriter_min_timing_ms: int = Field(
        50, description="Minimum timing for typewriter effect in ms"
    )

    # Fade effect parameters
    fade_duration_ms: int = Field(
        300, description="Default fade in/out duration in milliseconds"
    )


class SubtitleSegmentationSettings(BaseModel):
    """Configuration for subtitle segmentation and text processing logic."""

    # Fallback duration
    fallback_segment_duration_sec: float = Field(
        2.5, description="Fallback segment duration in seconds"
    )


class StylePresetConfig(BaseModel):
    """Typed configuration for a single subtitle style preset."""

    description: str = ""
    font_name: str = "Montserrat"
    font_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    background_color: str | None = None
    bold: bool = True
    outline_thickness: int = 2
    shadow: bool = True
    effects: list[str] = Field(default_factory=list)
    font_width_to_height_ratio: float = 0.5


class FontPoolEntry(BaseModel):
    """One entry in the curated font randomization pool.

    Pools live in `config/subtitles.yaml` under `font_pool` and feed
    `FontManager`. Each entry maps a display name to a TTF file under
    `subtitle_settings.font_directory` plus a system fallback.
    """

    name: str = Field(..., description="Display name (e.g. 'Montserrat')")
    file: str = Field(
        ..., description="Filename under font_directory (e.g. 'Montserrat-Bold.ttf')"
    )
    ffmpeg_name: str = Field(
        ..., description="FFmpeg/ASS font family identifier (e.g. 'Montserrat-Bold')"
    )
    system_fallback: str = Field(
        "Arial", description="Fallback font family when file is unavailable"
    )


class ColorPoolEntry(BaseModel):
    """One coordinated text + outline color pair in the randomization pool.

    Colors are in ASS hex format `&HAABBGGRR`. See
    `docs/subtitle-best-practices.md` for the contrast research that drives
    pool curation.
    """

    name: str = Field(..., description="Lookup key (e.g. 'classic', 'high_contrast')")
    display_name: str = Field("", description="Human-friendly label used in logs")
    font_color: str = Field(..., description="Text fill in ASS &HAABBGGRR format")
    outline_color: str = Field(
        ..., description="Outline stroke in ASS &HAABBGGRR format"
    )
    description: str = Field("", description="Why this pair is in the pool")


class TwoPartSubtitleUpperLine(BaseModel):
    """Upper line (static product info) of the two-part subtitle system."""

    enabled: bool = True
    source_field: str = Field(
        "shortened_affiliate_link",
        description="ProductData field name whose value is shown on the upper line.",
    )
    custom_url: str | None = Field(
        None,
        description=(
            "Literal URL to display instead of source_field. "
            "Env var SUBTITLE_BUSINESS_URL takes precedence over this."
        ),
    )
    anchor: str = "above_content"
    margin: float = Field(0.08, description="Gap as fraction of frame height (0.0-0.5)")
    font_size_scale: float = Field(
        0.75, description="Scale vs main subtitles (0.5-2.0)"
    )
    style_preset: str = Field(
        "minimal", description="Preset name for upper line styling"
    )
    use_full_duration: bool = Field(
        True,
        description="If true, upper line shows for full video; else only during CTA",
    )
    randomize_effects: bool = False
    prefix_replace: str | None = Field(
        None,
        description='Replace URL scheme prefix (e.g. "https://" → "Product: ")',
    )


class TwoPartSubtitleLowerLine(BaseModel):
    """Lower line (voiceover-synced) of the two-part subtitle system."""

    enabled: bool = True
    anchor: str = "below_content"
    margin: float = Field(0.05, description="Gap as fraction of frame height (0.0-0.5)")


class TwoPartSubtitleSettings(BaseModel):
    """Dual-line subtitle system: upper (static info) + lower (voiceover)."""

    enabled: bool = False
    upper_line: TwoPartSubtitleUpperLine = Field(
        default_factory=lambda: TwoPartSubtitleUpperLine()  # type: ignore[call-arg]
    )
    lower_line: TwoPartSubtitleLowerLine = Field(
        default_factory=lambda: TwoPartSubtitleLowerLine()  # type: ignore[call-arg]
    )


class PycapsSettings(BaseModel):
    """Configuration for the pycaps subtitle rendering engine.

    pycaps (https://github.com/francozanardi/pycaps) burns CSS-styled animated
    captions onto a pre-assembled video. Only consumed when
    ``MergedSubtitleSettings.subtitle_engine == "pycaps"``.

    See ``docs/pycaps-subtitles.md`` for field-by-field guidance and template
    screenshots.
    """

    template_name: str = Field(
        "word-focus",
        description=(
            "Fixed pycaps preset template name. Used when template_pool is empty "
            "or has one entry. Known presets: classic, default, explosive, fast, "
            "hype, line-focus, minimalist, neo-minimal, retro-gaming, vibrant, "
            "word-focus."
        ),
    )
    template_pool: list[str] = Field(
        default_factory=lambda: [
            "word-focus",
            "hype",
            "minimalist",
            "vibrant",
        ],
        description=(
            "Pool for deterministic per-product template selection (md5-keyed "
            "on product_id). Empty list disables selection and always uses "
            "template_name."
        ),
    )
    renderer: Literal["css", "pictex"] = Field(
        "css",
        description=(
            "css = Playwright + Chromium (full CSS fidelity, ~400 MB RAM "
            "per render, ~0.7x realtime). "
            "pictex = browser-free Skia path (fewer CSS features, no Chromium "
            "dep). Default css matches benchmark winner."
        ),
    )
    max_width_ratio: float = Field(
        0.80,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum line width as a fraction of frame width, handed to pycaps "
            "SubtitleLayoutOptions. The actual value is clamped at render time "
            "to the platform safe zone so captions never extend into UI "
            "overlay zones regardless of what's set here."
        ),
    )
    max_number_of_lines: int = Field(
        2,
        ge=1,
        description="Max lines per caption segment, passed to pycaps.",
    )
    vertical_align: Literal["top", "center", "bottom"] = Field(
        "bottom",
        description="Base anchor for caption block. Default 'bottom'.",
    )
    vertical_align_offset: float | None = Field(
        -0.20,
        ge=-1.0,
        le=1.0,
        description=(
            "Vertical offset from the anchor (-1.0..1.0). With bottom anchor "
            "the formula is y = h * (offset + 0.95) - text_height. "
            "-0.20 places captions at ~75% of frame height, matching the "
            "platform safe zone bottom boundary (TikTok overlay starts at 75%). "
            "Set to null to let the pycaps template's own positioning win."
        ),
    )
    fallback_policy: Literal["raise", "fallback_ffmpeg", "warn_and_skip"] = Field(
        "raise",
        description=(
            "'raise' (default) = abort the pipeline if pycaps is unavailable "
            "or fails.  'fallback_ffmpeg' = fall back to the FFmpeg subtitle "
            "engine for this run.  'warn_and_skip' = log a warning and keep "
            "the video without subtitles (not recommended)."
        ),
    )
