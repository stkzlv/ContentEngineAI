# src/video/config/subtitle_models.py
"""Subtitle configuration models.

Holds the leaf types used across the subtitle pipeline: effect tuning,
segmentation, style presets, font/color pools, two-part / pycaps blocks,
the platform safe zone, and the unified ``SubtitleSettings`` model that
both the config layer and the runtime generator share.
"""

import logging
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.video.config.constants import (
    FONT_FILE_EXTENSIONS,
    FONT_REGULAR_SUFFIXES,
    SAFE_ZONE_MAX_X,
    SAFE_ZONE_MAX_Y,
    SAFE_ZONE_MIN_X,
    SAFE_ZONE_MIN_Y,
)

logger = logging.getLogger(__name__)


class PositionAnchor(str, Enum):
    """Anchor points for subtitle positioning."""

    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"
    ABOVE_CONTENT = "above_content"
    BELOW_CONTENT = "below_content"


class StylePreset(str, Enum):
    """Predefined subtitle style presets."""

    MINIMAL = "minimal"
    MODERN = "modern"
    BOLD = "bold"
    ANIMATED = "animated"
    RANDOM = "random"


class Position(BaseModel):
    """Pixel-fraction position used for manual subtitle placement overrides."""

    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)


class PlatformSafeZone(BaseModel):
    """Safe zone boundaries to avoid platform UI overlays (fractions of frame).

    Default values represent the cross-platform worst case for TikTok,
    YouTube Shorts, and Instagram Reels on a 1080x1920 frame.
    See docs/platform-safe-zones.md for per-platform breakdown.
    """

    min_x: float = Field(
        default=SAFE_ZONE_MIN_X, description="Left boundary (fraction of width)"
    )
    max_x: float = Field(
        default=SAFE_ZONE_MAX_X, description="Right boundary (fraction of width)"
    )
    min_y: float = Field(
        default=SAFE_ZONE_MIN_Y, description="Top boundary (fraction of height)"
    )
    max_y: float = Field(
        default=SAFE_ZONE_MAX_Y, description="Bottom boundary (fraction of height)"
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


# ============================================================================
# Unified subtitle settings model
# ============================================================================
# Single source of truth for both the config-layer merge output and the runtime
# generator input. Replaces the historical pair MergedSubtitleSettings (config
# side, extra="allow") and UnifiedSubtitleConfig (runtime side, strict). The
# round-trip through dict.get(...) between the two models was the hiding place
# for silent-drop bugs like the one tracked in subtitle-config-cleanup.md §3.1.
#
# Canonical names (max_duration, min_duration) — old names like
# max_subtitle_duration are translated by from_legacy_dict() so existing
# YAML keeps loading.

# YAML keys that ``from_legacy_dict`` either renames or drops outright.
# Each entry: legacy key -> canonical key (or None to drop with a warning).
_LEGACY_RENAMES: dict[str, str] = {
    "max_subtitle_duration": "max_duration",
    "min_subtitle_duration": "min_duration",
}

# Keys that have no consumer on SubtitleSettings and used to live on
# MergedSubtitleSettings via extra="allow". Dropped silently.
_LEGACY_DROPS: frozenset[str] = frozenset(
    {
        "available_fonts",
        "available_color_combinations",
    }
)


class SubtitleSettings(BaseModel):
    """Unified subtitle configuration shared by config and runtime layers."""

    model_config = ConfigDict(extra="forbid")

    # ---- Engine + format ----
    enabled: bool = True
    subtitle_engine: Literal["ffmpeg", "pycaps"] = "ffmpeg"
    subtitle_format: Literal["srt", "ass"] = "srt"

    # ---- Positioning ----
    anchor: PositionAnchor = PositionAnchor.BOTTOM
    margin: float = Field(0.1, ge=0.0, le=0.5)
    content_aware: bool = True
    horizontal_alignment: Literal["left", "center", "right"] = "center"
    safe_zone: PlatformSafeZone = Field(default_factory=PlatformSafeZone)
    custom_position: Position | None = None

    # ---- Style ----
    style_preset: StylePreset = StylePreset.MODERN
    font_size_scale: float = Field(1.0, ge=0.5, le=2.0)

    # ---- Text formatting (canonical names) ----
    max_line_length: int = Field(38, ge=1)
    max_words_per_line: int = Field(3, ge=0)
    max_subtitle_width_fraction: float = Field(0.80, ge=0.0, le=1.0)
    max_duration: float = Field(2.5, gt=0)
    min_duration: float = Field(0.6, gt=0)

    # ---- Infrastructure (load-bearing for the rendering pipeline) ----
    font_directory: str = "static/fonts"
    font_size_percent: float = Field(0.075, ge=0.0, le=1.0)

    # ---- Randomization ----
    randomize_fonts: bool = False
    randomize_colors: bool = False
    randomize_effects: bool = False
    selected_font: str | None = None
    selected_color_pair: str | None = None

    # ---- Output ----
    temp_subtitle_dir: str = "temp"
    temp_subtitle_filename: str = "captions.srt"
    save_srt_with_video: bool = True
    script_paths: list[str] = Field(default_factory=list)

    # ---- Quality ----
    subtitle_similarity_threshold: float = Field(0.70, ge=0.0, le=1.0)

    # ---- Whisper timing post-processing (see subtitle_timing_smoother.py) ----
    # Kept as a free-form dict because the smoother reads these as kwargs;
    # the project's smoother module owns the schema, not this model.
    timing_smoothing: dict[str, Any] = Field(default_factory=dict)

    # ---- Nested ----
    pycaps: PycapsSettings | None = None
    two_part_subtitles: TwoPartSubtitleSettings = Field(
        default_factory=TwoPartSubtitleSettings
    )

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any]) -> "SubtitleSettings":
        """Translate a MergedSubtitleSettings dump (or raw merged YAML dict)
        into a SubtitleSettings instance.

        Handles renames from the legacy field names (`max_subtitle_duration`
        → `max_duration`, etc.) and silently drops keys that no longer exist
        on the unified model (the old `extra="allow"` accumulators).
        Unknown keys not in the rename or drop lists raise a ValidationError
        thanks to ``extra="forbid"``.
        """
        translated: dict[str, Any] = {}
        for key, value in data.items():
            # Underscore-prefixed keys are runtime-only side channels
            # (for example "_upper_use_full_duration" threaded through
            # TwoPartSubtitleHandler). They never belong in the typed model.
            if key.startswith("_"):
                continue
            if key in _LEGACY_DROPS:
                continue
            if key in _LEGACY_RENAMES:
                canonical = _LEGACY_RENAMES[key]
                if canonical in translated:
                    # Canonical name already present in the input; legacy
                    # name should not override it.
                    continue
                translated[canonical] = value
            else:
                translated[key] = value
        return cls(**translated)


class PartialSubtitleSettings(BaseModel):
    """All-optional variant of SubtitleSettings used by VideoProfile overrides.

    Every field defaults to ``None``; ``merge_into(base)`` deep-merges only
    the non-None fields onto a base ``SubtitleSettings``. Nested models
    (``pycaps``, ``two_part``, ``safe_zone``) merge recursively so a profile
    can tweak a single nested field without restating the whole block.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    subtitle_engine: Literal["ffmpeg", "pycaps"] | None = None
    subtitle_format: Literal["srt", "ass"] | None = None
    anchor: PositionAnchor | None = None
    margin: float | None = None
    content_aware: bool | None = None
    horizontal_alignment: Literal["left", "center", "right"] | None = None
    safe_zone: dict[str, Any] | None = None
    custom_position: dict[str, Any] | None = None
    style_preset: StylePreset | None = None
    font_size_scale: float | None = None
    max_line_length: int | None = None
    max_words_per_line: int | None = None
    max_subtitle_width_fraction: float | None = None
    max_duration: float | None = None
    min_duration: float | None = None
    font_directory: str | None = None
    font_size_percent: float | None = None
    randomize_fonts: bool | None = None
    randomize_colors: bool | None = None
    randomize_effects: bool | None = None
    selected_font: str | None = None
    selected_color_pair: str | None = None
    temp_subtitle_dir: str | None = None
    temp_subtitle_filename: str | None = None
    save_srt_with_video: bool | None = None
    script_paths: list[str] | None = None
    subtitle_similarity_threshold: float | None = None
    timing_smoothing: dict[str, Any] | None = None
    pycaps: dict[str, Any] | None = None
    two_part_subtitles: dict[str, Any] | None = None

    def merge_into(self, base: SubtitleSettings) -> SubtitleSettings:
        """Return a copy of base with non-None partial fields applied.

        Nested model fields (pycaps, two_part, safe_zone) are dict-typed on
        the partial side; we deep-merge them onto the base nested model and
        re-validate via ``model_copy``.
        """
        updates: dict[str, Any] = {}
        for field_name in self.__class__.model_fields:
            override = getattr(self, field_name)
            if override is None:
                continue
            if field_name == "pycaps":
                base_pycaps = base.pycaps.model_dump() if base.pycaps else {}
                merged = _deep_merge_dicts(base_pycaps, override)
                updates["pycaps"] = PycapsSettings(**merged)
            elif field_name == "two_part_subtitles":
                base_two_part = base.two_part_subtitles.model_dump()
                merged = _deep_merge_dicts(base_two_part, override)
                updates["two_part_subtitles"] = TwoPartSubtitleSettings(**merged)
            elif field_name == "safe_zone":
                base_safe_zone = base.safe_zone.model_dump()
                merged = _deep_merge_dicts(base_safe_zone, override)
                updates["safe_zone"] = PlatformSafeZone(**merged)
            elif field_name == "custom_position":
                updates["custom_position"] = Position(**override)
            else:
                updates[field_name] = override
        return base.model_copy(update=updates)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge: override wins for scalars; dicts merge per-key."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge_dicts(out[key], value)
        else:
            out[key] = value
    return out
