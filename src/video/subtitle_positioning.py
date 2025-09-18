"""Unified subtitle positioning system for ContentEngineAI.

This module provides a simplified, unified approach to subtitle positioning
that replaces the complex multi-mode system with a single flexible configuration.
"""

import contextlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PositionAnchor(str, Enum):
    """Anchor points for subtitle positioning."""

    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"
    ABOVE_CONTENT = "above_content"  # Position above visual content
    BELOW_CONTENT = "below_content"  # Position below visual content


class StylePreset(str, Enum):
    """Predefined subtitle style presets."""

    MINIMAL = "minimal"  # Clean, simple styling
    MODERN = "modern"  # Contemporary look with effects
    RELATIVE = "relative"  # Animated effects
    CLASSIC = "classic"  # Traditional subtitle styling
    BOLD = "bold"  # High contrast, bold styling


@dataclass
class Position:
    """Simple position coordinates."""

    x: float  # Horizontal position (0.0-1.0 as fraction of width)
    y: float  # Vertical position (0.0-1.0 as fraction of height)


@dataclass
class VisualBounds:
    """Bounds of visual content for relative positioning."""

    x: float
    y: float
    width: float
    height: float


class UnifiedSubtitleConfig(BaseModel):
    """Simplified, unified subtitle configuration."""

    # Core positioning
    anchor: PositionAnchor = Field(
        PositionAnchor.BOTTOM,
        description="Where to anchor subtitles relative to frame or content",
    )
    margin: float = Field(
        0.1, description="Margin as fraction of frame height (0.0-0.5)"
    )
    content_aware: bool = Field(
        True, description="Adjust position based on visual content bounds"
    )

    # Visual styling
    style_preset: StylePreset = Field(
        StylePreset.MODERN, description="Predefined style configuration"
    )
    font_size_scale: float = Field(
        1.0, description="Scale factor for font size (0.5-2.0)"
    )

    # Text formatting
    max_line_length: int = Field(38, description="Maximum characters per subtitle line")
    max_duration: float = Field(
        4.5, description="Maximum duration for subtitle segments (seconds)"
    )
    min_duration: float = Field(
        0.4, description="Minimum duration for subtitle segments (seconds)"
    )

    # Randomization (simplified)
    randomize_colors: bool = Field(False, description="Use random color combinations")
    randomize_effects: bool = Field(False, description="Use random animation effects")

    # Advanced positioning (optional fine-tuning)
    custom_position: Position | None = Field(
        None, description="Custom position override (x,y as 0.0-1.0 fractions)"
    )
    horizontal_alignment: str = Field(
        "center", description="Text alignment: left, center, right"
    )


def get_style_config(preset: StylePreset) -> dict[str, Any]:
    """Get style configuration for a given preset.

    Returns a dictionary of style parameters that can be applied to
    subtitle generation for both SRT and ASS formats.
    """
    configs = {
        StylePreset.MINIMAL: {
            "font_name": "Arial",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "background_color": None,
            "bold": False,
            "outline_thickness": 1,
            "shadow": False,
            "effects": [],
            "font_width_to_height_ratio": 0.5,
        },
        StylePreset.MODERN: {
            "font_name": "Montserrat",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "background_color": "&H99000000",  # Semi-transparent
            "bold": True,
            "outline_thickness": 2,
            "shadow": True,
            "effects": ["fade"],
            "font_width_to_height_ratio": 0.5,
        },
        StylePreset.RELATIVE: {
            "font_name": "Impact",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "background_color": "&H99000000",
            "bold": True,
            "outline_thickness": 2,
            "shadow": True,
            "effects": ["fade", "scale_pulse", "karaoke"],
            "font_width_to_height_ratio": 0.5,
        },
        StylePreset.CLASSIC: {
            "font_name": "Times New Roman",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "background_color": "&H80000000",  # More transparent
            "bold": False,
            "outline_thickness": 1,
            "shadow": False,
            "effects": [],
            "font_width_to_height_ratio": 0.5,
        },
        StylePreset.BOLD: {
            "font_name": "Gabarito",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "background_color": "&HCC000000",  # Strong background
            "bold": True,
            "outline_thickness": 3,
            "shadow": True,
            "effects": ["fade"],
            "font_width_to_height_ratio": 0.5,
        },
    }
    result = configs.get(preset)
    if result is not None:
        return result  # type: ignore[return-value]
    else:
        return configs[StylePreset.MODERN]  # type: ignore[return-value]


def calculate_position(
    config: UnifiedSubtitleConfig,
    frame_size: tuple[int, int],
    visual_bounds: VisualBounds | None = None,
) -> Position:
    """Calculate final subtitle position based on configuration.

    Args:
    ----
        config: Unified subtitle configuration
        frame_size: Video frame dimensions (width, height)
        visual_bounds: Optional bounds of visual content for relative positioning

    Returns:
    -------
        Position with x, y coordinates as fractions (0.0-1.0)

    """
    frame_width, frame_height = frame_size

    # Use custom position if specified
    if config.custom_position:
        return config.custom_position

    # Calculate base position from anchor
    if config.anchor == PositionAnchor.TOP:
        base_y = config.margin
    elif config.anchor == PositionAnchor.CENTER:
        base_y = 0.5
    elif config.anchor == PositionAnchor.BOTTOM:
        base_y = 1.0 - config.margin
    elif (
        config.anchor == PositionAnchor.ABOVE_CONTENT
        and config.content_aware
        and visual_bounds
    ):
        base_y = max(0.05, visual_bounds.y - config.margin)
    elif (
        config.anchor == PositionAnchor.BELOW_CONTENT
        and config.content_aware
        and visual_bounds
    ):
        base_y = min(0.95, visual_bounds.y + visual_bounds.height + config.margin)
    else:
        # Default to bottom positioning
        base_y = 1.0 - config.margin

    # Calculate horizontal position based on alignment
    if config.horizontal_alignment == "left":
        base_x = 0.1
    elif config.horizontal_alignment == "right":
        base_x = 0.9
    else:  # center
        base_x = 0.5

    return Position(x=base_x, y=base_y)


def get_font_size(
    config: UnifiedSubtitleConfig, frame_height: int, base_size_percent: float = 0.04
) -> int:
    """Calculate font size based on configuration and frame size.

    Args:
    ----
        config: Unified subtitle configuration
        frame_height: Height of video frame in pixels
        base_size_percent: Base font size as percentage of frame height

    Returns:
    -------
        Font size in pixels

    """
    base_size = int(frame_height * base_size_percent)
    scaled_size = int(base_size * config.font_size_scale)

    # Ensure reasonable bounds
    return max(16, min(100, scaled_size))


def convert_legacy_config(legacy_settings: dict[str, Any]) -> UnifiedSubtitleConfig:
    """Convert legacy multi-mode configuration to unified format.

    MIGRATION HELPER: This function provides backward compatibility by converting
    legacy positioning_mode configurations to the new unified system.

    Legacy modes converted:
    - "absolute" -> bottom anchor with fixed positioning
    - "relative" -> below_content anchor with content-aware positioning
    - "absolute" -> bottom anchor with custom position (if specified)

    New implementations should use UnifiedSubtitleConfig directly instead
    of relying on this conversion function.
    """
    # Extract positioning mode
    positioning_mode = legacy_settings.get("positioning_mode", "absolute")
    custom_pos = None  # Initialize here to avoid UnboundLocalError

    # Map legacy modes to new anchor system
    if positioning_mode == "relative":
        anchor = PositionAnchor.BELOW_CONTENT
        content_aware = True
    elif positioning_mode == "absolute":
        anchor = PositionAnchor.BOTTOM
        content_aware = False
        # Try to extract custom position from absolute settings
        if "absolute_positioning" in legacy_settings:
            legacy_settings["absolute_positioning"]
            # This would need FFmpeg expression parsing - simplified for now
            custom_pos = Position(x=0.5, y=0.8)  # Default approximation
    else:  # absolute
        anchor = PositionAnchor.BOTTOM
        content_aware = False

    # Check if using new unified parameters directly
    if "anchor" in legacy_settings:
        try:
            anchor = PositionAnchor(legacy_settings["anchor"])
            content_aware = legacy_settings.get("content_aware", True)
        except ValueError:
            pass  # Fall back to legacy conversion

    # Determine style preset from legacy settings
    style_preset = StylePreset.MODERN  # Default
    if "style_preset" in legacy_settings:
        with contextlib.suppress(ValueError):
            style_preset = StylePreset(legacy_settings["style_preset"])
    else:
        # Analyze legacy settings
        if legacy_settings.get("subtitle_format") == "ass":
            if legacy_settings.get("ass_enable_transforms", False):
                style_preset = StylePreset.RELATIVE
            elif legacy_settings.get("bold", False):
                style_preset = StylePreset.BOLD
        else:
            if legacy_settings.get("bold", False):
                style_preset = StylePreset.BOLD
            else:
                style_preset = StylePreset.CLASSIC

    return UnifiedSubtitleConfig(
        anchor=anchor,
        content_aware=content_aware,
        style_preset=style_preset,
        margin=legacy_settings.get("margin", 0.1),
        font_size_scale=legacy_settings.get("font_size_scale", 1.0),
        max_line_length=legacy_settings.get("max_line_length", 38),
        max_duration=legacy_settings.get("max_duration", 4.5),
        min_duration=legacy_settings.get("min_duration", 0.4),
        randomize_colors=legacy_settings.get("randomize_colors", False),
        randomize_effects=legacy_settings.get("randomize_effects", False),
        custom_position=custom_pos,
        horizontal_alignment=legacy_settings.get("horizontal_alignment", "center"),
    )
