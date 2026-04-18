"""Unified subtitle positioning system for ContentEngineAI.

This module provides a simplified, unified approach to subtitle positioning
that replaces the complex multi-mode system with a single flexible configuration.
"""

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.video.config.constants import (
    SUBTITLE_BASE_FONT_SIZE_PERCENT,
    SUBTITLE_CENTER_POSITION_FRACTION,
    SUBTITLE_MAX_FONT_SIZE,
    SUBTITLE_MIN_FONT_SIZE,
)

# Re-exports of leaf types that moved into subtitle_models. Kept here so
# existing imports (`from src.video.subtitle_positioning import PositionAnchor`)
# keep working without rippling through the codebase.
from src.video.config.subtitle_models import (  # noqa: F401
    PlatformSafeZone,
    Position,
    PositionAnchor,
    StylePreset,
    SubtitleSettings,
)

logger = logging.getLogger(__name__)


@dataclass
class VisualBounds:
    """Bounds of visual content for relative positioning."""

    x: float
    y: float
    width: float
    height: float


def clamp_to_safe_zone(
    x: int,
    y: int,
    frame_width: int,
    frame_height: int,
    safe_zone: PlatformSafeZone | None = None,
) -> tuple[int, int]:
    """Clamp pixel coordinates to platform safe zone boundaries."""
    sz = safe_zone or PlatformSafeZone()
    clamped_x = max(int(frame_width * sz.min_x), min(x, int(frame_width * sz.max_x)))
    clamped_y = max(int(frame_height * sz.min_y), min(y, int(frame_height * sz.max_y)))
    return clamped_x, clamped_y


def get_style_config(
    preset: StylePreset,
    config: SubtitleSettings | None = None,
    product_id: str | None = None,
    video_config: Any = None,
) -> dict[str, Any]:
    """Get style configuration for a given preset with optional randomization.

    Args:
    ----
        preset: Base style preset to use
        config: Unified subtitle configuration with randomization settings
        product_id: Product identifier for consistent randomization seeding
        video_config: VideoConfig with validated style_presets (preferred path)

    Returns:
    -------
        Dictionary of style parameters for subtitle generation

    """
    preset_key = preset.value if isinstance(preset, StylePreset) else preset

    base_config: dict[str, Any] | None = None

    # Preferred path: typed config, no YAML re-read, no CWD dependency.
    if video_config is not None:
        presets = getattr(video_config, "style_presets", None)
        if isinstance(presets, dict):
            preset_obj = presets.get(preset_key) or presets.get("modern")
            if preset_obj is not None and hasattr(preset_obj, "model_dump"):
                base_config = preset_obj.model_dump(exclude={"description"})

    # Legacy path: re-read YAML for callers that don't have VideoConfig yet.
    if base_config is None:
        from pathlib import Path

        import yaml

        subtitles_config_path = Path("config/subtitles.yaml")
        style_presets: dict[str, Any] = {}
        try:
            if subtitles_config_path.exists():
                with open(subtitles_config_path, encoding="utf-8") as f:
                    subtitles_data = yaml.safe_load(f)
                    style_presets = subtitles_data.get("style_presets", {})
        except Exception as e:
            logger.warning("Could not load style presets from config: %s", e)

        if preset_key in style_presets:
            base_config = style_presets[preset_key].copy()
            base_config.pop("description", None)
        elif "modern" in style_presets:
            base_config = style_presets["modern"].copy()
            base_config.pop("description", None)

    # Absolute last resort: inline modern defaults.
    if base_config is None:
        base_config = {
            "font_name": "Montserrat",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "background_color": None,
            "bold": True,
            "outline_thickness": 3,
            "shadow": True,
            "effects": ["karaoke"],
            "font_width_to_height_ratio": 0.5,
        }

    # Handle RANDOM preset - force randomization and select one random effect
    if preset == StylePreset.RANDOM and product_id:
        import random

        # Seed random generator with product_id for consistent selection per video
        random.seed(hash(product_id + "random_preset"))

        # Select one random effect from all available effects
        all_effects = base_config["effects"]
        if all_effects:
            selected_effect = random.choice(all_effects)  # noqa: S311
            base_config["effects"] = [selected_effect]
            logger.debug(f"RANDOM preset selected effect: {selected_effect}")

        # Force randomization for fonts, colors, and effects
        if config:
            config.randomize_fonts = True
            config.randomize_colors = True
            config.randomize_effects = True

    # Apply randomization if enabled and product_id provided
    if config and product_id:
        logger.debug(
            f"Randomization check - config: {config is not None}, "
            f"product_id: {product_id}, "
            f"randomize_fonts: {config.randomize_fonts if config else 'N/A'}"
        )
        # Import here to avoid circular imports
        try:
            from src.video.font_color_manager import RandomizationEngine

            randomizer = RandomizationEngine(video_config=video_config)
            logger.debug(
                f"RandomizationEngine imported successfully, calling "
                f"generate_randomized_style with fonts={config.randomize_fonts}, "
                f"colors={config.randomize_colors}"
            )

            # Apply font/color randomization
            randomized_style = randomizer.generate_randomized_style(
                product_id=product_id,
                enable_font_randomization=config.randomize_fonts,
                enable_color_randomization=config.randomize_colors,
                base_style=base_config,
            )
            logger.debug(f"Randomization result: {randomized_style}")

            # Merge randomized settings
            base_config.update(randomized_style)

            # Apply manual overrides if provided
            if config.selected_font:
                base_config["font_name"] = config.selected_font
                logger.debug(f"Font manually overridden to: {config.selected_font}")

            if config.selected_color_pair:
                logger.debug(
                    f"Color pair manually overridden to: "
                    f"{config.selected_color_pair}"
                )

        except ImportError as e:
            logger.warning(f"Could not import RandomizationEngine: {e}")

    return dict(base_config)


def calculate_position(
    config: SubtitleSettings,
    frame_size: tuple[int, int],
    visual_bounds: VisualBounds | None = None,
    safe_zone: PlatformSafeZone | None = None,
) -> Position:
    """Calculate final subtitle position based on configuration.

    Args:
    ----
        config: Unified subtitle configuration
        frame_size: Video frame dimensions (width, height)
        visual_bounds: Optional bounds of visual content for relative positioning
        safe_zone: Platform safe zone boundaries for UI avoidance

    Returns:
    -------
        Position with x, y coordinates as fractions (0.0-1.0)

    """
    sz = safe_zone or PlatformSafeZone()
    frame_width, frame_height = frame_size

    # Clamp custom position to safe zone
    if config.custom_position:
        return Position(
            x=max(sz.min_x, min(config.custom_position.x, sz.max_x)),
            y=max(sz.min_y, min(config.custom_position.y, sz.max_y)),
        )

    # Calculate base Y from anchor
    if config.anchor == PositionAnchor.TOP:
        base_y = max(sz.min_y, config.margin)
    elif config.anchor == PositionAnchor.CENTER:
        base_y = SUBTITLE_CENTER_POSITION_FRACTION
    elif config.anchor == PositionAnchor.BOTTOM:
        base_y = min(sz.max_y, 1.0 - config.margin)
    elif config.anchor == PositionAnchor.ABOVE_CONTENT:
        if config.content_aware and visual_bounds:
            base_y = visual_bounds.y - config.margin
            logger.debug(
                "ABOVE_CONTENT: visual_bounds.y=%.4f, margin=%.4f, "
                "base_y=%.4f (Y=%dpx)",
                visual_bounds.y,
                config.margin,
                base_y,
                int(base_y * frame_height),
            )
        else:
            base_y = config.margin
            logger.debug(
                "ABOVE_CONTENT fallback: content_aware=%s, "
                "visual_bounds=%s, base_y=%.4f",
                config.content_aware,
                visual_bounds,
                base_y,
            )
        base_y = max(sz.min_y, base_y)
    elif config.anchor == PositionAnchor.BELOW_CONTENT:
        if config.content_aware and visual_bounds:
            base_y = visual_bounds.y + visual_bounds.height + config.margin
        else:
            base_y = 1.0 - config.margin
        base_y = min(sz.max_y, base_y)

    # Clamp Y to safe zone
    base_y = max(sz.min_y, min(base_y, sz.max_y))

    # Calculate horizontal position from alignment, clamped to safe zone
    if config.horizontal_alignment == "left":
        base_x = sz.min_x
    elif config.horizontal_alignment == "right":
        base_x = sz.max_x
    else:  # center
        base_x = SUBTITLE_CENTER_POSITION_FRACTION
    base_x = max(sz.min_x, min(base_x, sz.max_x))

    return Position(x=base_x, y=base_y)


def get_font_size(
    config: SubtitleSettings,
    frame_height: int,
    base_size_percent: float | None = None,
) -> int:
    """Calculate font size based on configuration and frame size.

    Args:
    ----
        config: Unified subtitle configuration
        frame_height: Height of video frame in pixels
        base_size_percent: Base font size as percentage of frame height (optional)

    Returns:
    -------
        Font size in pixels

    """
    if base_size_percent is None:
        base_size_percent = SUBTITLE_BASE_FONT_SIZE_PERCENT
    min_font = SUBTITLE_MIN_FONT_SIZE
    max_font = SUBTITLE_MAX_FONT_SIZE

    base_size = int(frame_height * base_size_percent)
    scaled_size = int(base_size * config.font_size_scale)

    # Ensure reasonable bounds from config
    return max(min_font, min(max_font, scaled_size))
