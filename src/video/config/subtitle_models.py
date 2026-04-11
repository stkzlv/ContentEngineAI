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

    # Word count thresholds
    min_words_for_sentence_break: int = Field(
        3, description="Minimum words required for sentence break"
    )
    min_words_natural_break: int = Field(
        3, description="Minimum words for natural break"
    )
    min_words_duration_limit: int = Field(
        3, description="Minimum words for duration limit break"
    )

    # Fallback duration
    fallback_segment_duration_sec: float = Field(
        2.5, description="Fallback segment duration in seconds"
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
        0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum line width as a fraction of frame width, handed to pycaps "
            "SubtitleLayoutOptions."
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
        None,
        ge=-1.0,
        le=1.0,
        description=(
            "Manual override for pycaps vertical_align offset (-1.0..1.0). "
            "When None, the burn step derives the offset from VisualBounds so "
            "captions land in the whitespace below the product image."
        ),
    )
    fallback_policy: Literal["warn_and_skip", "raise"] = Field(
        "warn_and_skip",
        description=(
            "'warn_and_skip' = on pycaps failure, log warning and keep the "
            "FFmpeg-assembled video untouched. 'raise' = abort the pipeline."
        ),
    )
