"""Unified Assembler Integration for Simplified Subtitle Positioning.

This module provides integration between the unified subtitle positioning system
and the video assembler, eliminating the complex multi-mode positioning logic.
"""

import logging
from pathlib import Path
from typing import Any

from src.video.result_types import SubtitleResult
from src.video.subtitle_positioning import (
    VisualBounds,
    calculate_position,
    convert_legacy_config,
)
from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

logger = logging.getLogger(__name__)


class UnifiedAssemblerIntegration:
    """Simplified integration between assembler and subtitle positioning."""

    def __init__(self, config: Any):
        """Initialize with video configuration."""
        self.config = config
        self.frame_size = config.video_settings.resolution

        # Convert legacy subtitle settings to unified config
        legacy_settings = config.subtitle_settings.__dict__
        self.unified_config = convert_legacy_config(legacy_settings)

    def generate_positioned_subtitles(
        self,
        timings: list[dict[str, Any]] | None,
        script_text: str | None,
        voiceover_duration: float | None,
        visual_geometries: list[Any] | None,
        output_path: Path,
        debug_mode: bool = False,
    ) -> SubtitleResult:
        """Generate subtitles with optimal positioning based on visual content.

        This replaces the complex multi-mode positioning system with a single,
        content-aware approach that automatically adjusts to visual layout.

        Args:
        ----
            timings: Word timing data from STT (if available)
            script_text: Fallback script text
            voiceover_duration: Total audio duration
            visual_geometries: Visual element positions and sizes
            output_path: Where to save the subtitle file
            debug_mode: Enable debug output

        Returns:
        -------
            GenerationResult with success status and generated file path

        """
        try:
            # Extract visual bounds from geometries if available
            visual_bounds = self._calculate_content_bounds(visual_geometries)

            # Initialize unified generator
            generator = UnifiedSubtitleGenerator(self.unified_config, self.frame_size)

            # Determine output format
            format_type = getattr(
                self.config.subtitle_settings, "subtitle_format", "srt"
            )

            # Try timing-based generation first
            if timings:
                result = generator.generate_from_timings(
                    timings=timings,
                    output_path=output_path,
                    format_type=format_type,
                    voiceover_duration=voiceover_duration,
                    visual_bounds=visual_bounds,
                    debug_mode=debug_mode,
                )

                if result.success:
                    logger.info(
                        f"Generated positioned subtitles from timings: "
                        f"{result.segments_created} segments, bounds={visual_bounds}"
                    )
                    return result

            # Fallback to script-based generation
            if script_text and voiceover_duration:
                result = generator.generate_from_script(
                    script_text=script_text,
                    duration=voiceover_duration,
                    output_path=output_path,
                    format_type=format_type,
                    visual_bounds=visual_bounds,
                    debug_mode=debug_mode,
                )

                if result.success:
                    logger.info(
                        f"Generated positioned subtitles from script: "
                        f"{result.segments_created} segments, bounds={visual_bounds}"
                    )
                    return result

            # Complete failure
            result = SubtitleResult(
                success=False,
                path=None,
                format=format_type,
            )
            result.add_error("No valid input data for subtitle generation")
            return result

        except Exception as e:
            logger.error(
                f"Error in unified subtitle positioning: {e}", exc_info=debug_mode
            )
            result = SubtitleResult(
                success=False,
                path=None,
                format=getattr(self.config.subtitle_settings, "subtitle_format", "srt"),
            )
            result.add_error(f"Positioning integration error: {str(e)}")
            return result

    def _calculate_content_bounds(
        self, visual_geometries: list[Any] | None
    ) -> VisualBounds | None:
        """Calculate the bounds of visual content for relative positioning.

        This consolidates all visual elements into a single bounding box
        that can be used for content-aware subtitle positioning.
        """
        if not visual_geometries:
            return None

        try:
            # Find the overall bounds of all visual elements
            min_x = float("inf")
            min_y = float("inf")
            max_x = float("-inf")
            max_y = float("-inf")

            for geometry in visual_geometries:
                if hasattr(geometry, "rendered_x"):
                    # Handle VisualGeometry objects
                    x = geometry.rendered_x
                    y = geometry.rendered_y
                    w = geometry.rendered_w
                    h = geometry.rendered_h
                elif isinstance(geometry, dict):
                    # Handle dictionary format
                    x = geometry.get("x", 0)
                    y = geometry.get("y", 0)
                    w = geometry.get("width", 0)
                    h = geometry.get("height", 0)
                else:
                    continue

                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

            if min_x == float("inf"):
                return None

            # Convert to normalized coordinates (0.0-1.0)
            frame_width, frame_height = self.frame_size

            return VisualBounds(
                x=min_x / frame_width,
                y=min_y / frame_height,
                width=(max_x - min_x) / frame_width,
                height=(max_y - min_y) / frame_height,
            )

        except Exception as e:
            logger.warning(f"Failed to calculate visual bounds: {e}")
            return None

    def get_positioning_preview(
        self,
        visual_geometries: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Get a preview of subtitle positioning for debugging.

        Returns positioning information that can be used for validation
        and debugging of the subtitle layout.
        """
        visual_bounds = self._calculate_content_bounds(visual_geometries)
        position = calculate_position(
            self.unified_config, self.frame_size, visual_bounds
        )

        return {
            "unified_config": {
                "anchor": self.unified_config.anchor,
                "content_aware": self.unified_config.content_aware,
                "style_preset": self.unified_config.style_preset,
                "margin": self.unified_config.margin,
            },
            "visual_bounds": {
                "x": visual_bounds.x if visual_bounds else None,
                "y": visual_bounds.y if visual_bounds else None,
                "width": visual_bounds.width if visual_bounds else None,
                "height": visual_bounds.height if visual_bounds else None,
            }
            if visual_bounds
            else None,
            "calculated_position": {
                "x": position.x,
                "y": position.y,
                "x_pixels": int(position.x * self.frame_size[0]),
                "y_pixels": int(position.y * self.frame_size[1]),
            },
            "frame_size": self.frame_size,
        }


def replace_complex_positioning(
    assembler_instance: Any,
    subtitle_path: Path,
    timings: list[dict[str, Any]] | None = None,
    script_text: str | None = None,
    voiceover_duration: float | None = None,
    visual_geometries: list[Any] | None = None,
    debug_mode: bool = False,
) -> SubtitleResult:
    """Replace complex positioning logic with unified approach.

    This function can be used as a drop-in replacement for the existing
    complex positioning logic in the assembler.

    Args:
    ----
        assembler_instance: The VideoAssembler instance
        subtitle_path: Path where subtitles should be generated
        timings: Optional word timing data
        script_text: Optional script text for fallback
        voiceover_duration: Duration of audio track
        visual_geometries: Visual element positioning data
        debug_mode: Enable debug output

    Returns:
    -------
        GenerationResult indicating success/failure and file path

    """
    integration = UnifiedAssemblerIntegration(assembler_instance.config)

    return integration.generate_positioned_subtitles(
        timings=timings,
        script_text=script_text,
        voiceover_duration=voiceover_duration,
        visual_geometries=visual_geometries,
        output_path=subtitle_path,
        debug_mode=debug_mode,
    )
