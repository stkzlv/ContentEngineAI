"""Visual filter chain construction.

This module provides utilities for building FFmpeg visual filter chains with
aspect ratio handling, positioning, and transitions.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.video.assembler.media_inspector import MediaInspector
from src.video.assembler.video_strategies import VideoStrategyFactory

if TYPE_CHECKING:
    from src.video.config.visual_models import MergedProfileSettings
from src.video.config import VideoConfig

logger = logging.getLogger(__name__)


@dataclass
class VisualGeometry:
    """Position and dimensions of visual element in video.

    This class stores the coordinates and size of a visual element after it
    has been positioned and scaled within the output video frame.

    Attributes
    ----------
        rendered_x: X-coordinate of the top-left corner
        rendered_y: Y-coordinate of the top-left corner
        rendered_w: Width of the rendered visual
        rendered_h: Height of the rendered visual

    """

    rendered_x: int
    rendered_y: int
    rendered_w: int
    rendered_h: int


class VisualFilterBuilder:
    """Build FFmpeg visual filter chains with aspect ratio handling."""

    def __init__(
        self,
        media_inspector: MediaInspector,
        config: VideoConfig,
        strategy_factory: VideoStrategyFactory | None,
        profile_settings: "MergedProfileSettings | None",
        debug_mode: bool = False,
        normalize_video_callback: (Callable[[Path], Awaitable[Path]] | None) = None,
    ):
        """Initialize VisualFilterBuilder.

        Args:
        ----
            media_inspector: MediaInspector instance for media operations
            config: VideoConfig containing settings
            strategy_factory: VideoStrategyFactory for video mode strategies
            profile_settings: Merged profile settings (may be None initially)
            debug_mode: Enable debug logging
            normalize_video_callback: Async callback for video format normalization

        """
        self.inspector = media_inspector
        self.config = config
        self.strategy_factory = strategy_factory
        self.profile_settings = profile_settings
        self.debug_mode = debug_mode
        self.normalize_video_callback = normalize_video_callback

    def _get_effective_subtitle_settings(self) -> dict[str, Any]:
        """Get effective subtitle settings with profile overrides applied."""
        if self.profile_settings:
            return self.profile_settings.subtitle_settings.model_dump()
        return dict(self.config.subtitle_settings)

    def apply_aspect_ratio_mode(
        self,
        input_label: str,
        aspect_mode: str,
        target_width: int,
        target_height: int,
        video_width: int,
        video_height: int,
        output_label: str | None = None,
        video_top_percent: float | None = None,
        target_content_height: int | None = None,
    ) -> tuple[str, str, VisualGeometry | None]:
        """Apply aspect ratio transformation based on configured mode.

        Generates FFmpeg filter strings for aspect ratio handling:
        - letterbox: Maintain aspect ratio with black padding (centered)
        - crop-to-fit: Scale to fill frame and crop edges (centered)
        - smart-scale: Auto-select based on aspect ratio similarity

        Args:
        ----
            input_label: FFmpeg input label (e.g., "[v0]")
            aspect_mode: Mode ("letterbox", "crop-to-fit", "smart-scale")
            target_width: Target output width in pixels
            target_height: Target output height in pixels (full frame)
            video_width: Source video width in pixels
            video_height: Source video height in pixels
            output_label: Optional output label override
            video_top_percent: Optional vertical position override (0.0-1.0)
            target_content_height: Optional content height limit

        Returns:
        -------
            Tuple of (filter_string, output_label, geometry)
            geometry contains actual rendered position and dimensions

        """
        # Calculate aspect ratios
        target_aspect = target_width / target_height
        video_aspect = video_width / video_height

        # Smart-scale: auto-select mode based on aspect ratio similarity
        if aspect_mode == "smart-scale":
            aspect_diff = abs(target_aspect - video_aspect) / target_aspect
            aspect_tolerance = self.config.aspect_ratio.get(
                "smart_scale_tolerance", 0.10
            )
            aspect_mode = (
                "crop-to-fit" if aspect_diff <= aspect_tolerance else "letterbox"
            )

        # Use provided output_label or generate one from input_label
        if output_label is None:
            output_label = f"{input_label}_scaled"

        # Letterbox mode: scale with aspect ratio, add black padding
        geometry: VisualGeometry | None = None
        if aspect_mode == "letterbox":
            # Determine scaling target height
            if target_content_height is not None:
                scale_height = target_content_height
            else:
                scale_height = target_height

            # Calculate vertical position
            if video_top_percent is not None:
                y_offset = int(target_height * video_top_percent)
                pad_y = str(y_offset)
            else:
                pad_y = "(oh-ih)/2"

            filter_string = (
                f"{input_label}scale={target_width}:{scale_height}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:"
                f"(ow-iw)/2:{pad_y}:black"
            )

            # Compute actual scaled dimensions (force_original_aspect_ratio=decrease)
            # Scale to fit within target_width x scale_height while maintaining aspect
            scale_by_width = target_width / video_width
            scale_by_height = scale_height / video_height
            scale_factor = min(scale_by_width, scale_by_height)
            actual_w = int(video_width * scale_factor)
            actual_h = int(video_height * scale_factor)

            # Compute actual position within padded frame
            actual_x = (target_width - actual_w) // 2
            if video_top_percent is not None:
                # Content area starts at y_offset, video centered within scale_height
                content_offset = (scale_height - actual_h) // 2
                actual_y = y_offset + content_offset
            else:
                # Centered in full frame
                actual_y = (target_height - actual_h) // 2

            geometry = VisualGeometry(
                rendered_x=actual_x,
                rendered_y=actual_y,
                rendered_w=actual_w,
                rendered_h=actual_h,
            )

            # Debug logging
            if target_content_height is not None:
                logger.debug(
                    f"[LETTERBOX] Constrained video: scale to "
                    f"{target_width}x{scale_height}, "
                    f"pad to {target_width}x{target_height} at Y={pad_y}"
                )
                logger.debug(
                    f"[LETTERBOX] Actual geometry: {actual_w}x{actual_h} "
                    f"at ({actual_x}, {actual_y})"
                )

        # Crop-to-fit mode: scale to fill, crop excess
        elif aspect_mode == "crop-to-fit":
            filter_string = (
                f"{input_label}scale={target_width}:{target_height}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={target_width}:{target_height}"
            )

        else:
            # Invalid mode - fallback to letterbox with warning
            logger.warning(
                f"Invalid aspect_mode '{aspect_mode}', falling back to letterbox"
            )
            filter_string = (
                f"{input_label}scale={target_width}:{target_height}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:"
                f"(ow-iw)/2:(oh-ih)/2:black{output_label}"
            )

        return filter_string, output_label, geometry

    def _calculate_subtitle_reserved_space(self, height: int) -> int:
        """Calculate subtitle space reservation to prevent overlap.

        Args:
        ----
            height: Frame height in pixels

        Returns:
        -------
            Reserved space in pixels for subtitles

        """
        subtitle_reserved_space = 0
        try:
            # Check if subtitles are enabled at the profile level
            if self.profile_settings:
                subtitle_settings_dict = self.profile_settings.get(
                    "subtitle_settings", {}
                )
                subtitle_enabled = (
                    subtitle_settings_dict.get("enabled", False)
                    if subtitle_settings_dict
                    else False
                )
            else:
                subtitle_enabled = False

            if subtitle_enabled:
                subtitle_settings = self._get_effective_subtitle_settings()
                # Estimate subtitle height based on font size and margins
                font_size_scale = subtitle_settings.get("font_size_scale", 1.0)
                base_pct = self.config.video_settings.base_font_height_percent
                base_font_height = height * base_pct
                font_height = base_font_height * font_size_scale

                margin = subtitle_settings.get("margin", 0.05)
                margin_pixels = height * margin

                # Reserve space for font + outline + margin + buffer
                multiplier = self.config.video_settings.reserved_space_font_multiplier
                subtitle_reserved_space = int(font_height * multiplier + margin_pixels)
        except Exception as e:
            # If we can't get subtitle settings, use conservative default
            default_space = self.config.video_settings.default_subtitle_reserved_space
            logger.debug(
                f"Could not calculate subtitle space from settings ({e}), "
                f"using default {default_space * 100}% reservation"
            )
            subtitle_reserved_space = int(height * default_space)

        return (
            subtitle_reserved_space
            if subtitle_reserved_space > 0
            else int(height * default_space)
        )

    async def build_visual_chain(
        self,
        visual_inputs: list[Path],
        total_video_duration: float,
        is_relative_mode: bool,
        video_settings_dict: dict,
    ) -> tuple[
        list[str],
        list[str],
        list[tuple[Path, float, bool]],
        str,
        list[VisualGeometry],
        bool,
    ]:
        """Build visual filter chain and return geometry.

        Args:
        ----
            visual_inputs: List of visual input file paths
            total_video_duration: Target video duration in seconds
            is_relative_mode: Whether to use relative positioning
            video_settings_dict: Video settings from profile

        Returns:
        -------
            Tuple of (filter_parts, input_cmd_parts, timed_visuals,
                     final_video_stream_label, geometries,
                     image_positioning_overridden)

        """
        video_settings = self.config.video_settings
        video_files = [path for path in visual_inputs if self.inspector.is_video(path)]
        image_files = [
            path for path in visual_inputs if not self.inspector.is_video(path)
        ]

        # Apply format normalization to videos when enabled
        if video_files and video_settings.enable_format_normalization:
            if self.debug_mode:
                logger.debug(
                    f"Normalizing {len(video_files)} videos to H.264/30fps/yuv420p"
                )
            if self.normalize_video_callback:
                video_files = list(
                    await asyncio.gather(
                        *[self.normalize_video_callback(vf) for vf in video_files]
                    )
                )

        # Assemble videos using configured mode
        timed_visuals: list[tuple[Path, float, bool]] = []
        mode_info = ""

        if video_files and self.strategy_factory:
            # Call video assembly mode dispatcher
            strategy = self.strategy_factory.get_strategy(
                video_settings.video_assembly_mode
            )
            timed_visuals, mode_info = await strategy.assemble(
                video_files, image_files, total_video_duration
            )
        elif image_files:
            # Backward compatibility: image-only behavior
            num_visuals_total = len(image_files)
            if num_visuals_total > 1:
                num_transitions = num_visuals_total - 1
                transition_duration = video_settings.transition_duration_sec
                total_gross_image_duration = total_video_duration + (
                    num_transitions * transition_duration
                )
                if total_gross_image_duration > 0:
                    image_segment_duration = total_gross_image_duration / len(
                        image_files
                    )
                    if (
                        image_segment_duration
                        < video_settings.min_visual_segment_duration_sec
                    ):
                        image_segment_duration = (
                            video_settings.min_visual_segment_duration_sec
                        )
                    for path in image_files:
                        timed_visuals.append((path, image_segment_duration, False))
            elif num_visuals_total == 1:
                timed_visuals.append((image_files[0], total_video_duration, False))
            mode_info = f"image_only ({len(image_files)} images)"

        if not timed_visuals:
            raise ValueError("No visual media could be prepared for the timeline.")

        if self.debug_mode and mode_info:
            logger.debug(f"Visual assembly mode: {mode_info}")

        # Detect no-video scenario
        has_any_videos = any(is_video for _, _, is_video in timed_visuals)
        image_positioning_overridden = False
        if not has_any_videos and self.profile_settings:
            vs_model = self.profile_settings.video_settings
            has_video_positioning = (
                vs_model.video_top_position_percent
                != self.config.video_settings.video_top_position_percent
                or vs_model.video_content_height_percent
                != self.config.video_settings.video_content_height_percent
            )

            if has_video_positioning:
                # Override with image-optimized positioning from config
                vs = self.config.video_settings
                fallback_top = vs.fallback_image_top_percent
                fallback_width = vs.fallback_image_width_percent
                video_settings_dict["image_top_position_percent"] = fallback_top
                video_settings_dict["image_width_percent"] = fallback_width
                image_positioning_overridden = True
                logger.info(
                    "No videos detected in video-centric profile - "
                    "applying image-optimized positioning "
                    f"(top={fallback_top:.0%}, width={fallback_width:.0%})"
                )

        # Build filter chain
        input_cmd_parts: list[str] = []
        filter_parts: list[str] = []
        stream_labels: list[str] = []
        geometries: list[VisualGeometry] = []
        width, height = video_settings.resolution
        pix_fmt = video_settings.output_pixel_format

        all_visuals_dims = await asyncio.gather(
            *[self.inspector.get_media_dimensions(p) for p, _, _ in timed_visuals]
        )

        uniform_height = -1
        if not is_relative_mode:
            scaled_heights = []
            for orig_w, orig_h in all_visuals_dims:
                if orig_w > 0 and orig_h > 0:
                    scaled_h = int(
                        (width * video_settings_dict["image_width_percent"])
                        * (orig_h / orig_w)
                    )
                    scaled_heights.append(scaled_h)
            if scaled_heights:
                uniform_height = min(scaled_heights)

        for i, (path, duration, is_video_item) in enumerate(timed_visuals):
            if is_video_item:
                input_cmd_parts.extend(["-i", str(path)])
            else:
                input_cmd_parts.extend(
                    [
                        "-loop",
                        str(video_settings.image_loop),
                        "-framerate",
                        str(video_settings.frame_rate),
                        "-t",
                        str(duration),
                        "-i",
                        str(path),
                    ]
                )

            proc_label = f"[v_proc_{i}]"
            orig_w, orig_h = all_visuals_dims[i]

            # Apply aspect ratio handling for videos
            if is_video_item:
                # Get video positioning from profile or global settings
                vs = (
                    self.profile_settings.video_settings
                    if self.profile_settings
                    else self.config.video_settings
                )
                video_top_percent = vs.video_top_position_percent
                video_height_percent = vs.video_content_height_percent
                video_valign = vs.video_vertical_align
                logger.debug(
                    f"[VIDEO POS] top={video_top_percent:.2%}, "
                    f"height={video_height_percent:.2%}, align={video_valign}"
                )

                target_content_height = int(height * video_height_percent)

                # When centering, pass video_top_percent=None so
                # apply_aspect_ratio_mode uses FFmpeg's (oh-ih)/2 expression
                effective_top = None if video_valign == "center" else video_top_percent

                aspect_filter, aspect_label, actual_geom = self.apply_aspect_ratio_mode(
                    f"[{i}:v]",
                    video_settings.video_aspect_mode,
                    width,
                    height,
                    orig_w,
                    orig_h,
                    output_label=f"[v{i}_scaled]",
                    video_top_percent=effective_top,
                    target_content_height=target_content_height,
                )

                vf_string = (
                    f"{aspect_filter}{aspect_label};"
                    f"{aspect_label}format={pix_fmt}[v_temp_{i}];"
                    f"[v_temp_{i}]trim=duration={duration},"
                    f"setpts=PTS-STARTPTS{proc_label}"
                )

                # Use actual geometry from apply_aspect_ratio_mode if available
                if actual_geom:
                    logger.debug(
                        f"Video {i}: Actual geometry "
                        f"y={actual_geom.rendered_y}px, "
                        f"height={actual_geom.rendered_h}px"
                    )
                    geometries.append(actual_geom)
                else:
                    # Fallback: compute from config (for non-letterbox modes)
                    video_height_pixels = int(height * video_height_percent)
                    if video_valign == "center":
                        video_top_pixels = (height - video_height_pixels) // 2
                    else:
                        video_top_pixels = int(height * video_top_percent)
                    logger.debug(
                        f"Video {i}: Config-based geometry "
                        f"y={video_top_pixels}px, height={video_height_pixels}px"
                    )
                    geometries.append(
                        VisualGeometry(
                            rendered_x=0,
                            rendered_y=video_top_pixels,
                            rendered_w=width,
                            rendered_h=video_height_pixels,
                        )
                    )
            else:
                # Image handling logic
                scaled_w_base = int(width * video_settings_dict["image_width_percent"])
                scaled_w, scaled_h = 0, 0
                vertical_align = video_settings_dict.get(
                    "image_vertical_align", "center"
                )

                # Calculate subtitle space reservation
                subtitle_reserved_space = self._calculate_subtitle_reserved_space(
                    height
                )

                # For centering, we calculate Y after knowing scaled_h
                # For top alignment, use the configured top position
                top_offset = video_settings_dict["image_top_position_percent"] * height
                max_available_height = height - top_offset - subtitle_reserved_space
                logger.debug(
                    f"Image {i}: Reserved {subtitle_reserved_space}px for "
                    f"subtitles, max available height: {max_available_height}px "
                    f"(frame: {height}px, align: {vertical_align})"
                )

                if not is_relative_mode and uniform_height > 0:
                    scaled_h = uniform_height
                    scaled_w = (
                        int(scaled_h * (orig_w / orig_h))
                        if orig_h > 0
                        else scaled_w_base
                    )
                    vf_scale = f"scale={scaled_w}:{scaled_h}"
                else:
                    scaled_w = scaled_w_base
                    scaled_h = int(scaled_w * (orig_h / orig_w)) if orig_w > 0 else -1

                    # Ensure scaled height doesn't exceed available space
                    if scaled_h > max_available_height:
                        scaled_h = int(max_available_height)
                        scaled_w = (
                            int(scaled_h * (orig_w / orig_h))
                            if orig_h > 0
                            else scaled_w_base
                        )

                    vf_scale = f"scale={scaled_w}:{scaled_h}"

                # Calculate Y position based on alignment
                if vertical_align == "center":
                    # Center image vertically in frame
                    target_y_pos = (height - scaled_h) / 2
                else:
                    # Use configured top offset
                    target_y_pos = top_offset

                geometries.append(
                    VisualGeometry(
                        rendered_x=int((width - scaled_w) / 2),
                        rendered_y=int(target_y_pos),
                        rendered_w=scaled_w,
                        rendered_h=scaled_h,
                    )
                )

                vf_string = (
                    f"[{i}:v]{vf_scale},setsar=1,"
                    f"pad={width}:{height}:(ow-iw)/2:{int(target_y_pos)}:"
                    f"color={video_settings.pad_color},"
                    f"format={pix_fmt}[v_temp_{i}];"
                    f"[v_temp_{i}]trim=duration={duration},"
                    f"setpts=PTS-STARTPTS{proc_label}"
                )

            filter_parts.append(vf_string)
            stream_labels.append(proc_label)

        # Apply transitions
        if len(stream_labels) > 1:
            transition_duration = video_settings.transition_duration_sec
            current_stream = stream_labels[0]
            current_offset = timed_visuals[0][1] - transition_duration
            for i in range(1, len(stream_labels)):
                next_stream = stream_labels[i]
                output_stream_label = f"[v_chain_{i}]"
                filter_parts.append(
                    f"{current_stream}{next_stream}xfade=transition=fade:"
                    f"duration={transition_duration}:"
                    f"offset={current_offset:.4f}{output_stream_label}"
                )
                current_stream = output_stream_label
                if i < len(timed_visuals) - 1:
                    current_offset += timed_visuals[i][1] - transition_duration
            final_video_stream_label = current_stream
        else:
            final_video_stream_label = stream_labels[0]

        return (
            filter_parts,
            input_cmd_parts,
            timed_visuals,
            final_video_stream_label,
            geometries,
            image_positioning_overridden,
        )
