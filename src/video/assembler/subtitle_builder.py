"""Subtitle graph construction for FFmpeg.

This module provides utilities for building FFmpeg subtitle filter graphs
with content-aware positioning, dual-line support, and ASS file generation.
"""

import logging
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from src.video.assembler.subtitle_utils import SubtitleParser, SubtitleStyler
from src.video.assembler.visual_builder import VisualGeometry
from src.video.config import VideoConfig

if TYPE_CHECKING:
    from src.video.config.visual_models import MergedProfileSettings

logger = logging.getLogger(__name__)


class SubtitleGraphBuilder:
    """Build FFmpeg subtitle filter graphs with positioning."""

    def __init__(
        self,
        config: VideoConfig,
        profile_settings: "MergedProfileSettings | None",
        product_id: str,
        debug_mode: bool = False,
    ):
        """Initialize SubtitleGraphBuilder.

        Args:
        ----
            config: VideoConfig containing settings
            profile_settings: Merged profile settings
            product_id: Product identifier for style randomization
            debug_mode: Enable debug logging

        """
        self.config = config
        self.profile_settings = profile_settings
        self.product_id = product_id
        self.debug_mode = debug_mode
        self.parser = SubtitleParser()
        self.styler = SubtitleStyler()

    def _get_effective_subtitle_settings(self) -> dict:
        """Get merged subtitle settings from profile and config."""
        if self.profile_settings:
            return self.profile_settings.subtitle_settings.model_dump()
        return dict(self.config.subtitle_settings)

    def _resolve_font_path(self, font_name: str) -> Path | None:
        """Resolve font name to path using SubtitleStyler."""
        # font_directory is in subtitle_settings dict, not video_settings
        font_dir = Path(self.config.subtitle_settings["font_directory"])
        return self.styler.resolve_font_path(font_name, font_dir)

    async def build_subtitle_graph(
        self,
        visual_chain_result: tuple,
        subtitle_path: Path | None,
        temp_sub_dir: Path,
    ) -> tuple[list[str], list[str]]:
        """Build video processing graph with subtitle overlay.

        Args:
        ----
            visual_chain_result: Result from VisualFilterBuilder.build_visual_chain()
            subtitle_path: Path to subtitle file (SRT or ASS)
            temp_sub_dir: Temporary directory for processing

        Returns:
        -------
            Tuple of (video_filters, input_cmd_parts)

        """
        (
            video_filters,
            input_cmd_parts,
            timed_visuals,
            final_visual_stream,
            geometries,
            image_positioning_overridden,
        ) = visual_chain_result

        settings_dict = self._get_effective_subtitle_settings()

        from src.video.subtitle_positioning import UnifiedSubtitleConfig

        try:
            unified_config = UnifiedSubtitleConfig(**settings_dict)
            use_content_aware = unified_config.content_aware
        except Exception as e:
            logger.warning(f"Failed to parse subtitle settings, using fallback: {e}")
            use_content_aware = settings_dict.get("content_aware", True)

        subtitles_enabled = settings_dict.get("enabled", True)

        if not subtitle_path or not subtitles_enabled:
            video_filters.append(f"{final_visual_stream}copy[v_out]")
            return video_filters, input_cmd_parts

        # Handle ASS files
        if subtitle_path.suffix.lower() == ".ass":
            if use_content_aware and geometries:
                if self.debug_mode:
                    logger.debug("Regenerating ASS with content-aware positioning")

                content_aware_ass_path = await self._create_content_aware_ass_file(
                    subtitle_path,
                    geometries,
                    timed_visuals,
                    temp_sub_dir,
                    image_positioning_overridden,
                )
                if content_aware_ass_path:
                    ass_path = content_aware_ass_path.as_posix().replace(":", r"\:")
                else:
                    ass_path = subtitle_path.as_posix().replace(":", r"\:")
            else:
                ass_path = subtitle_path.as_posix().replace(":", r"\:")

            video_filters.append(f"{final_visual_stream}ass='{ass_path}'[v_out]")
            return video_filters, input_cmd_parts

        # Handle SRT files with drawtext
        sub_entries = self.parser.parse_srt(subtitle_path)
        current_video_stream = final_visual_stream

        segment_end_times = self._calculate_segment_times(timed_visuals)

        # Get style configuration
        from src.video.subtitle_positioning import get_style_config

        try:
            unified_config = UnifiedSubtitleConfig(**settings_dict)
            style_config = get_style_config(
                preset=unified_config.style_preset,
                config=unified_config,
                product_id=self.product_id,
            )
            font_name = style_config.get("font_name", "Arial")
            font_color = style_config.get("font_color", "&H00FFFFFF")
            outline_color = style_config.get("outline_color", "&H00000000")
        except Exception as e:
            logger.warning(f"Failed to get style config, using fallback: {e}")
            font_name = settings_dict.get("font_name", "Arial")
            font_color = settings_dict.get("font_color", "&H00FFFFFF")
            outline_color = settings_dict.get("outline_color", "&H00000000")

        font_path = self._resolve_font_path(font_name)
        if not font_path:
            logger.warning(f"Could not resolve font path for '{font_name}'")
            video_filters.append(f"{final_visual_stream}copy[v_out]")
            return video_filters, input_cmd_parts

        drawtext_count = 0
        for sub in sub_entries:
            sub_start, sub_end = sub.start, sub.end
            for i, end_time in enumerate(segment_end_times):
                start_time = segment_end_times[i - 1] if i > 0 else 0
                overlap_start = max(sub_start, start_time)
                overlap_end = min(sub_end, end_time)

                if overlap_start < overlap_end:
                    geom = geometries[i]

                    # Calculate text wrapping
                    font_size_pixels = self.config.video_settings.resolution[
                        1
                    ] * settings_dict.get("font_size_percent", 0.04)
                    avg_char_width = font_size_pixels * settings_dict.get(
                        "font_width_to_height_ratio", 0.5
                    )
                    max_chars = (
                        int(geom.rendered_w / avg_char_width)
                        if avg_char_width > 0
                        else self.config.video_settings.default_max_chars_per_line
                    )

                    wrapper = textwrap.TextWrapper(
                        width=max_chars,
                        break_long_words=True,
                        replace_whitespace=False,
                    )
                    wrapped_text = "\n".join(wrapper.wrap(sub.text))

                    sub_text_file = temp_sub_dir / f"sub_text_{drawtext_count}.txt"
                    sub_text_file.write_text(wrapped_text, encoding="utf-8")

                    # Calculate position
                    position = self._calculate_subtitle_position(
                        unified_config, geom, settings_dict
                    )

                    x_pos_expr = f"w*{position.x} - text_w/2"
                    y_pos_expr = f"h*{position.y}"

                    back_color = settings_dict.get("back_color", "&H80000000")
                    back_color_ffmpeg = self.styler.convert_ass_color_to_ffmpeg(
                        back_color
                    )

                    output_stream = f"[v_sub_{drawtext_count + 1}]"
                    font_path_escaped = font_path.as_posix().replace(":", r"\:")
                    sub_text_escaped = sub_text_file.as_posix().replace(":", r"\:")

                    drawtext_filter = (
                        f"{current_video_stream}drawtext="
                        f"fontfile='{font_path_escaped}':"
                        f"textfile='{sub_text_escaped}':"
                        f"fontsize={font_size_pixels}:"
                        f"fontcolor='"
                        f"{self.styler.convert_ass_color_to_ffmpeg(font_color)}':"
                        f"borderw={settings_dict.get('outline_thickness', 2)}:"
                        f"bordercolor='"
                        f"{self.styler.convert_ass_color_to_ffmpeg(outline_color)}':"
                        f"box=1:boxcolor='{back_color_ffmpeg}'"
                        f":boxborderw="
                        f"{self.config.video_settings.subtitle_box_border_width}:"
                        f"x='{x_pos_expr}':y='{y_pos_expr}':"
                        f"enable='between(t,{overlap_start},{overlap_end})'"
                        f"{output_stream}"
                    )
                    video_filters.append(drawtext_filter)
                    current_video_stream = output_stream
                    drawtext_count += 1

        video_filters.append(f"{current_video_stream}copy[v_out]")
        return video_filters, input_cmd_parts

    async def build_dual_subtitle_graph(
        self,
        visual_chain_result: tuple,
        subtitle_lower_path: Path | None,
        subtitle_upper_path: Path,
        temp_sub_dir: Path,
    ) -> tuple[list[str], list[str]]:
        """Build video graph with dual independent subtitle lines.

        Args:
        ----
            visual_chain_result: Result from VisualFilterBuilder.build_visual_chain()
            subtitle_lower_path: Path to lower subtitle file (voiceover)
            subtitle_upper_path: Path to upper subtitle file (product info)
            temp_sub_dir: Temporary directory for processing

        Returns:
        -------
            Tuple of (video_filters, input_cmd_parts)

        """
        (
            video_filters,
            input_cmd_parts,
            timed_visuals,
            final_visual_stream,
            geometries,
            image_positioning_overridden,
        ) = visual_chain_result

        settings_dict = self._get_effective_subtitle_settings()

        from src.video.subtitle_positioning import UnifiedSubtitleConfig

        try:
            unified_config = UnifiedSubtitleConfig(**settings_dict)
            use_content_aware = unified_config.content_aware
        except Exception as e:
            logger.warning(f"Failed to parse subtitle settings, using fallback: {e}")
            use_content_aware = settings_dict.get("content_aware", True)

        current_stream = final_visual_stream

        # Apply upper subtitle (static product info)
        if subtitle_upper_path.suffix.lower() == ".ass":
            if use_content_aware and geometries:
                content_aware_upper = await self._create_content_aware_upper_ass_file(
                    subtitle_upper_path,
                    geometries,
                    timed_visuals,
                    temp_sub_dir,
                )
                if content_aware_upper:
                    subtitle_upper_path = content_aware_upper
            ass_path_upper = subtitle_upper_path.as_posix().replace(":", r"\:")
            video_filters.append(f"{current_stream}ass='{ass_path_upper}'[v_upper]")
            current_stream = "[v_upper]"
        else:
            # SRT format for upper line
            sub_entries_upper = self.parser.parse_srt(subtitle_upper_path)
            if sub_entries_upper:
                upper_text = sub_entries_upper[0].text

                two_part_config = settings_dict.get("two_part_subtitles", {})
                upper_config = two_part_config.get("upper_line", {})

                from src.video.subtitle_positioning import get_style_config

                style_preset = upper_config.get("style_preset", "minimal")
                upper_settings = settings_dict.copy()
                upper_settings["style_preset"] = style_preset
                upper_settings["anchor"] = upper_config.get("anchor", "above_content")
                upper_settings["margin"] = upper_config.get("margin", 0.04)
                upper_settings["font_size_scale"] = upper_config.get(
                    "font_size_scale", 0.7
                )

                try:
                    upper_unified = UnifiedSubtitleConfig(**upper_settings)
                    style_config = get_style_config(
                        preset=style_preset,
                        config=upper_unified,
                        product_id=self.product_id,
                    )
                    font_name = style_config.get("font_name", "Arial")
                    font_color = style_config.get("font_color", "&H00FFFFFF")
                    outline_color = style_config.get("outline_color", "&H00000000")
                except Exception as e:
                    logger.warning(f"Failed to get upper line style config: {e}")
                    font_name = "Arial"
                    font_color = "&H00FFFFFF"
                    outline_color = "&H00000000"

                font_path = self._resolve_font_path(font_name)
                if font_path:
                    upper_text_file = temp_sub_dir / "upper_subtitle.txt"
                    upper_text_file.write_text(upper_text, encoding="utf-8")

                    # Calculate position for upper line with configured bounds priority
                    from src.video.subtitle_positioning import (
                        VisualBounds,
                        calculate_position,
                    )

                    geom = geometries[0] if geometries else None
                    visual_bounds = None

                    if upper_unified.content_aware:
                        frame_width, frame_height = (
                            self.config.video_settings.resolution
                        )

                        # Prefer actual geometry over config - it reflects real
                        # letterboxing/scaling
                        if geom and geom.rendered_h > 0:
                            visual_bounds = VisualBounds(
                                x=geom.rendered_x / frame_width,
                                y=geom.rendered_y / frame_height,
                                width=geom.rendered_w / frame_width,
                                height=geom.rendered_h / frame_height,
                            )
                            if self.debug_mode:
                                logger.debug(
                                    f"Upper subtitle using actual geometry: "
                                    f"y={geom.rendered_y / frame_height:.2%}, "
                                    f"height={geom.rendered_h / frame_height:.2%}"
                                )
                        else:
                            # Fall back to configured video positioning
                            if self.profile_settings:
                                vs = self.profile_settings.video_settings
                                video_top_percent = vs.video_top_position_percent
                                video_height_percent = vs.video_content_height_percent

                                if (
                                    video_top_percent is not None
                                    and video_height_percent is not None
                                ):
                                    visual_bounds = VisualBounds(
                                        x=0.0,
                                        y=video_top_percent,
                                        width=1.0,
                                        height=video_height_percent,
                                    )
                                    if self.debug_mode:
                                        logger.debug(
                                            f"Upper subtitle using configured bounds: "
                                            f"y={video_top_percent:.2%}, "
                                            f"height={video_height_percent:.2%}"
                                        )

                    position = calculate_position(
                        upper_unified,
                        self.config.video_settings.resolution,
                        visual_bounds,
                    )

                    base_font_size = self.config.video_settings.resolution[
                        1
                    ] * settings_dict.get("font_size_percent", 0.04)
                    upper_font_size = base_font_size * upper_settings.get(
                        "font_size_scale", 0.7
                    )

                    x_pos_expr = f"w*{position.x} - text_w/2"
                    y_pos_expr = f"h*{position.y}"

                    drawtext_filter = (
                        f"{current_stream}drawtext="
                        f"fontfile='{font_path.as_posix().replace(':', r'\\:')}':"
                        f"textfile='{upper_text_file.as_posix().replace(':', r'\\:')}':"
                        f"fontsize={upper_font_size}:"
                        f"fontcolor='"
                        f"{self.styler.convert_ass_color_to_ffmpeg(font_color)}':"
                        f"borderw={upper_settings.get('outline_thickness', 1)}:"
                        f"bordercolor='"
                        f"{self.styler.convert_ass_color_to_ffmpeg(outline_color)}':"
                        f"x='{x_pos_expr}':y='{y_pos_expr}'"
                        f"[v_upper]"
                    )
                    video_filters.append(drawtext_filter)
                    current_stream = "[v_upper]"

        # Apply lower subtitle (timed voiceover)
        if subtitle_lower_path and subtitle_lower_path.exists():
            if subtitle_lower_path.suffix.lower() == ".ass":
                if use_content_aware and geometries:
                    content_aware_ass = await self._create_content_aware_ass_file(
                        subtitle_lower_path,
                        geometries,
                        timed_visuals,
                        temp_sub_dir,
                        image_positioning_overridden,
                    )
                    if content_aware_ass:
                        ass_path = content_aware_ass.as_posix().replace(":", r"\:")
                    else:
                        ass_path = subtitle_lower_path.as_posix().replace(":", r"\:")
                else:
                    ass_path = subtitle_lower_path.as_posix().replace(":", r"\:")

                video_filters.append(f"{current_stream}ass='{ass_path}'[v_out]")
            else:
                # SRT format for lower line - use timed drawtext approach
                sub_entries_lower = self.parser.parse_srt(subtitle_lower_path)

                segment_end_times = self._calculate_segment_times(timed_visuals)

                # Get lower line styling
                two_part_config = settings_dict.get("two_part_subtitles", {})
                lower_config = two_part_config.get("lower_line", {})
                lower_settings = settings_dict.copy()
                lower_settings["anchor"] = lower_config.get("anchor", "below_content")
                lower_settings["margin"] = lower_config.get("margin", 0.04)

                # Get style configuration for lower line
                from src.video.subtitle_positioning import get_style_config

                try:
                    lower_unified = UnifiedSubtitleConfig(**lower_settings)
                    style_config = get_style_config(
                        preset=lower_unified.style_preset,
                        config=lower_unified,
                        product_id=self.product_id,
                    )
                    font_name = style_config.get("font_name", "Arial")
                    font_color = style_config.get("font_color", "&H00FFFFFF")
                    outline_color = style_config.get("outline_color", "&H00000000")
                except Exception as e:
                    logger.warning(f"Failed to get lower line style config: {e}")
                    font_name = lower_settings.get("font_name", "Arial")
                    font_color = lower_settings.get("font_color", "&H00FFFFFF")
                    outline_color = lower_settings.get("outline_color", "&H00000000")

                font_path = self._resolve_font_path(font_name)
                if font_path:
                    drawtext_count = 0
                    for sub in sub_entries_lower:
                        sub_start, sub_end = sub.start, sub.end
                        for i, end_time in enumerate(segment_end_times):
                            start_time = segment_end_times[i - 1] if i > 0 else 0
                            overlap_start = max(sub_start, start_time)
                            overlap_end = min(sub_end, end_time)

                            if overlap_start < overlap_end:
                                geom = geometries[i]

                                # Calculate text wrapping
                                font_size_pixels = (
                                    self.config.video_settings.resolution[1]
                                    * lower_settings.get("font_size_percent", 0.04)
                                )
                                avg_char_width = font_size_pixels * lower_settings.get(
                                    "font_width_to_height_ratio", 0.5
                                )
                                max_chars = (
                                    int(geom.rendered_w / avg_char_width)
                                    if avg_char_width > 0
                                    else (
                                        self.config.video_settings.default_max_chars_per_line
                                    )
                                )

                                wrapper = textwrap.TextWrapper(
                                    width=max_chars,
                                    break_long_words=True,
                                    replace_whitespace=False,
                                )
                                wrapped_text = "\n".join(wrapper.wrap(sub.text))

                                sub_text_file = (
                                    temp_sub_dir / f"lower_text_{drawtext_count}.txt"
                                )
                                sub_text_file.write_text(wrapped_text, encoding="utf-8")

                                # Calculate position for lower line with
                                # configured bounds
                                from src.video.subtitle_positioning import (
                                    VisualBounds,
                                    calculate_position,
                                )

                                visual_bounds = None
                                if lower_unified.content_aware:
                                    (
                                        frame_width,
                                        frame_height,
                                    ) = self.config.video_settings.resolution

                                    # Prefer actual geometry over config - reflects
                                    # real letterboxing
                                    if geom and geom.rendered_h > 0:
                                        visual_bounds = VisualBounds(
                                            x=geom.rendered_x / frame_width,
                                            y=geom.rendered_y / frame_height,
                                            width=geom.rendered_w / frame_width,
                                            height=geom.rendered_h / frame_height,
                                        )
                                    else:
                                        # Fall back to configured video positioning
                                        if self.profile_settings:
                                            vs = self.profile_settings.video_settings
                                            video_top_percent = (
                                                vs.video_top_position_percent
                                            )
                                            video_height_percent = (
                                                vs.video_content_height_percent
                                            )

                                            if (
                                                video_top_percent is not None
                                                and video_height_percent is not None
                                            ):
                                                visual_bounds = VisualBounds(
                                                    x=0.0,
                                                    y=video_top_percent,
                                                    width=1.0,
                                                    height=video_height_percent,
                                                )

                                position = calculate_position(
                                    lower_unified,
                                    self.config.video_settings.resolution,
                                    visual_bounds,
                                )

                                x_pos_expr = f"w*{position.x} - text_w/2"
                                y_pos_expr = f"h*{position.y}"

                                output_stream = f"[v_lower_{drawtext_count + 1}]"
                                font_esc = font_path.as_posix().replace(":", r"\\:")
                                text_esc = sub_text_file.as_posix().replace(":", r"\\:")
                                fc = self.styler.convert_ass_color_to_ffmpeg(font_color)
                                bc = self.styler.convert_ass_color_to_ffmpeg(
                                    outline_color
                                )
                                back_col = self.styler.convert_ass_color_to_ffmpeg(
                                    lower_settings.get("back_color", "&H80000000")
                                )
                                drawtext_filter = (
                                    f"{current_stream}drawtext="
                                    f"fontfile='{font_esc}':"
                                    f"textfile='{text_esc}':"
                                    f"fontsize={font_size_pixels}:"
                                    f"fontcolor='{fc}':"
                                    f"borderw="
                                    f"{lower_settings.get('outline_thickness', 2)}:"
                                    f"bordercolor='{bc}':"
                                    f"box=1:boxcolor='{back_col}':"
                                    f"boxborderw="
                                    f"{self.config.video_settings.subtitle_box_border_width}:"
                                    f"x='{x_pos_expr}':y='{y_pos_expr}':"
                                    f"enable='between(t,{overlap_start},{overlap_end})'"
                                    f"{output_stream}"
                                )
                                video_filters.append(drawtext_filter)
                                current_stream = output_stream
                                drawtext_count += 1

                    video_filters.append(f"{current_stream}copy[v_out]")
                else:
                    video_filters.append(f"{current_stream}copy[v_out]")
        else:
            video_filters.append(f"{current_stream}copy[v_out]")

        return video_filters, input_cmd_parts

    async def _create_content_aware_ass_file(
        self,
        original_ass_path: Path,
        geometries: list[VisualGeometry],
        timed_visuals: list[tuple[Path, float, bool]],
        temp_dir: Path,
        image_positioning_overridden: bool = False,
    ) -> Path | None:
        """Create ASS file with content-aware positioning.

        Args:
        ----
            original_ass_path: Path to the original ASS file
            geometries: List of visual geometries for each segment
            timed_visuals: List of visual timeline data
            temp_dir: Temporary directory for output
            image_positioning_overridden: Whether image positioning was overridden

        Returns:
        -------
            Path to the new content-aware ASS file, or None if failed

        """
        try:
            logger.info(
                "Creating content-aware ASS file with image-relative positioning"
            )

            with open(original_ass_path, encoding="utf-8") as f:
                original_content = f.read()

            lines = original_content.strip().split("\n")
            header_lines = []
            events_lines = []
            in_events = False

            for line in lines:
                if line.strip().startswith("[Events]"):
                    in_events = True
                    header_lines.append(line)
                elif in_events and line.strip().startswith("Dialogue:"):
                    events_lines.append(line)
                elif (
                    in_events
                    and line.strip()
                    and not line.strip().startswith("Dialogue:")
                ):
                    header_lines.append(line)
                else:
                    header_lines.append(line)

            if not events_lines:
                logger.warning("No dialogue events found in ASS file")
                return None

            segment_end_times = self._calculate_segment_times(timed_visuals)

            settings_dict = self._get_effective_subtitle_settings()

            from src.video.subtitle_positioning import (
                PositionAnchor,
                UnifiedSubtitleConfig,
            )

            try:
                unified_config = UnifiedSubtitleConfig(**settings_dict)
            except Exception as e:
                logger.warning(f"Failed to parse unified subtitle config: {e}")
                return original_ass_path

            if (
                not unified_config.content_aware
                or unified_config.anchor != PositionAnchor.BELOW_CONTENT
            ):
                return original_ass_path

            content_aware_events = []
            frame_height = self.config.video_settings.resolution[1]

            for event_line in events_lines:
                parts = event_line.split(",", 9)
                if len(parts) < 10:
                    content_aware_events.append(event_line)
                    continue

                start_time = self.parser.parse_ass_time(parts[1])

                segment_idx = 0
                for i, end_time in enumerate(segment_end_times):
                    if start_time <= end_time:
                        segment_idx = i
                        break

                if segment_idx < len(geometries):
                    geom = geometries[segment_idx]

                    # Calculate content bottom - prefer actual geometry over config
                    # Geometry contains accurate letterbox/scale positioning
                    if geom.rendered_h > 0:
                        content_bottom = geom.rendered_y + geom.rendered_h
                    else:
                        content_bottom = self._get_content_bottom(geom, frame_height)

                    spacing_px = unified_config.margin * frame_height

                    from src.video.subtitle_positioning import get_font_size

                    font_size = get_font_size(unified_config, frame_height)

                    # Calculate subtitle Y position
                    is_portrait = (
                        self.config.video_settings.resolution[1]
                        > self.config.video_settings.resolution[0]
                    )

                    if is_portrait:
                        subtitle_y = int(content_bottom + spacing_px)
                    else:
                        font_offset_multiplier = self._get_font_offset_multiplier()
                        font_offset = font_size * font_offset_multiplier
                        subtitle_y = int(content_bottom + spacing_px - font_offset)

                    max_safe_y = self._get_max_safe_y()
                    max_y = int(frame_height * max_safe_y)
                    subtitle_y = min(subtitle_y, max_y)

                    # Update text with positioning
                    text_content = parts[9]
                    text_content = re.sub(r"\\pos\([^)]+\)", "", text_content)

                    subtitle_x = geom.rendered_x + geom.rendered_w // 2

                    if text_content.startswith("{") and "}" in text_content:
                        effect_end = text_content.find("}") + 1
                        effect_content = text_content[1 : effect_end - 1]
                        after_effects = text_content[effect_end:]

                        if r"\move(" in effect_content:
                            positioned_text = text_content
                        else:
                            positioned_text = (
                                f"{{\\pos({subtitle_x},{subtitle_y})"
                                f"{effect_content}}}{after_effects}"
                            )
                    else:
                        positioned_text = (
                            f"{{\\pos({subtitle_x},{subtitle_y})}}{text_content}"
                        )

                    new_parts = parts[:9] + [positioned_text]
                    content_aware_events.append(",".join(new_parts))
                else:
                    content_aware_events.append(event_line)

            output_dir = original_ass_path.parent
            content_aware_ass_path = output_dir / "subtitles_content_aware.ass"

            with open(content_aware_ass_path, "w", encoding="utf-8") as f:
                for line in header_lines:
                    f.write(line + "\n")
                for event_line in content_aware_events:
                    f.write(event_line + "\n")

            logger.info(f"Created content-aware ASS file: {content_aware_ass_path}")
            return content_aware_ass_path

        except Exception as e:
            logger.error(f"Failed to create content-aware ASS file: {e}")
            return None

    async def _create_content_aware_upper_ass_file(
        self,
        original_ass_path: Path,
        geometries: list[VisualGeometry],
        timed_visuals: list[tuple[Path, float, bool]],
        temp_dir: Path,
    ) -> Path | None:
        """Create ASS file with per-segment positioning for upper subtitle.

        The upper subtitle (CTA/URL) may span the entire video, but each visual
        segment has different geometry (video vs images). This splits long
        dialogue lines into per-segment sub-dialogues positioned just above
        each segment's actual content.
        """
        try:
            logger.info(
                "Creating content-aware upper ASS file with per-segment positioning"
            )

            with open(original_ass_path, encoding="utf-8") as f:
                original_content = f.read()

            lines = original_content.strip().split("\n")
            header_lines = []
            events_lines = []
            in_events = False

            for line in lines:
                if line.strip().startswith("[Events]"):
                    in_events = True
                    header_lines.append(line)
                elif in_events and line.strip().startswith("Dialogue:"):
                    events_lines.append(line)
                elif (
                    in_events
                    and line.strip()
                    and not line.strip().startswith("Dialogue:")
                ):
                    header_lines.append(line)
                else:
                    header_lines.append(line)

            if not events_lines:
                logger.warning("No dialogue events found in upper ASS file")
                return None

            segment_end_times = self._calculate_segment_times(timed_visuals)
            if not segment_end_times:
                return None

            # Build segment start times from end times
            segment_start_times = [0.0] + segment_end_times[:-1]

            frame_width, frame_height = self.config.video_settings.resolution

            # Get upper line margin from settings
            settings_dict = self._get_effective_subtitle_settings()
            upper_margin = settings_dict.get(
                "two_part_subtitles_upper_margin", 0.04
            )
            spacing_px = upper_margin * frame_height

            # Parse actual font size from ASS Style line (much more accurate
            # than recalculating from font_size_percent which is a legacy value)
            font_size = 53.0  # safe default
            for line in header_lines:
                if line.strip().startswith("Style:"):
                    style_parts = line.split(",")
                    if len(style_parts) > 2:
                        try:
                            font_size = float(style_parts[2])
                        except ValueError:
                            pass
                    break

            min_safe_y = (
                self.config.text_rendering.min_safe_y_position
                if self.config.text_rendering
                else 0.05
            )
            min_y = int(frame_height * min_safe_y)

            content_aware_events = []

            for event_line in events_lines:
                parts = event_line.split(",", 9)
                if len(parts) < 10:
                    content_aware_events.append(event_line)
                    continue

                ev_start = self.parser.parse_ass_time(parts[1])
                ev_end = self.parser.parse_ass_time(parts[2])
                text_content = parts[9]

                # Strip existing \pos() tags
                clean_text = re.sub(r"\\pos\([^)]+\)", "", text_content)

                # Find all segments this dialogue overlaps
                for seg_idx, (seg_start, seg_end) in enumerate(
                    zip(segment_start_times, segment_end_times)
                ):
                    if ev_end <= seg_start or ev_start >= seg_end:
                        continue  # no overlap

                    if seg_idx >= len(geometries):
                        continue

                    geom = geometries[seg_idx]

                    # Clip to segment boundaries
                    clip_start = max(ev_start, seg_start)
                    clip_end = min(ev_end, seg_end)

                    # Calculate Y just above content
                    if geom.rendered_h > 0:
                        content_top = geom.rendered_y
                    else:
                        # Fallback to config
                        content_top = int(frame_height * 0.34)

                    # Cap spacing so the text stays near the content edge.
                    # The configured margin (8% of frame) is too large for
                    # letterboxed content with big black bars. Limit to 1.5x
                    # font size for a proportional gap.
                    effective_gap = min(spacing_px, font_size * 1.5)
                    # Alignment 5 = center, so \pos y is text center.
                    # Place text bottom at: content_top - effective_gap
                    subtitle_y = int(content_top - effective_gap - font_size / 2)
                    subtitle_y = max(subtitle_y, min_y)

                    # Center X on content
                    subtitle_x = geom.rendered_x + geom.rendered_w // 2

                    # Format times back to ASS
                    start_str = self._format_ass_time_str(clip_start)
                    end_str = self._format_ass_time_str(clip_end)

                    # Rebuild text with new position
                    if clean_text.startswith("{") and "}" in clean_text:
                        effect_end = clean_text.find("}") + 1
                        effect_content = clean_text[1 : effect_end - 1]
                        after_effects = clean_text[effect_end:]
                        positioned_text = (
                            f"{{\\pos({subtitle_x},{subtitle_y})"
                            f"{effect_content}}}{after_effects}"
                        )
                    else:
                        positioned_text = (
                            f"{{\\pos({subtitle_x},{subtitle_y})}}{clean_text}"
                        )

                    new_parts = parts[:1] + [start_str, end_str] + parts[3:9] + [positioned_text]
                    content_aware_events.append(",".join(new_parts))

                    if self.debug_mode:
                        logger.debug(
                            "Upper subtitle seg %d: y=%d (content_top=%d), "
                            "time=%.2f-%.2fs",
                            seg_idx, subtitle_y, content_top,
                            clip_start, clip_end,
                        )

            if not content_aware_events:
                logger.warning("No content-aware upper events generated")
                return None

            output_path = temp_dir / "subtitles_upper_content_aware.ass"
            with open(output_path, "w", encoding="utf-8") as f:
                for line in header_lines:
                    f.write(line + "\n")
                for event_line in content_aware_events:
                    f.write(event_line + "\n")

            logger.info("Created content-aware upper ASS: %s", output_path.name)
            return output_path

        except Exception as e:
            logger.error("Failed to create content-aware upper ASS: %s", e)
            return None

    @staticmethod
    def _format_ass_time_str(seconds: float) -> str:
        """Format seconds to ASS time (H:MM:SS.CC)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        cs = int((s % 1) * 100)
        return f" {h}:{m:02d}:{int(s):02d}.{cs:02d}"

    def _calculate_segment_times(
        self, timed_visuals: list[tuple[Path, float, bool]]
    ) -> list[float]:
        """Calculate segment end times from timed visuals."""
        segment_end_times = []
        cumulative_time = 0.0
        transition_duration = self.config.video_settings.transition_duration_sec

        for i, (_, duration, _) in enumerate(timed_visuals):
            effective_duration = duration - (transition_duration if i > 0 else 0)
            cumulative_time += effective_duration
            segment_end_times.append(cumulative_time)

        return segment_end_times

    def _calculate_subtitle_position(
        self, unified_config, geom: VisualGeometry | None, settings_dict: dict
    ):
        """Calculate subtitle position using unified positioning system."""
        from src.video.subtitle_positioning import (
            Position,
            VisualBounds,
            calculate_position,
        )

        visual_bounds = None
        if unified_config.content_aware and geom:
            frame_width, frame_height = self.config.video_settings.resolution

            if (
                geom.rendered_x >= 0
                and geom.rendered_y >= 0
                and geom.rendered_w > 0
                and geom.rendered_h > 0
                and geom.rendered_x + geom.rendered_w <= frame_width
                and geom.rendered_y + geom.rendered_h <= frame_height
            ):
                visual_bounds = VisualBounds(
                    x=geom.rendered_x / frame_width,
                    y=geom.rendered_y / frame_height,
                    width=geom.rendered_w / frame_width,
                    height=geom.rendered_h / frame_height,
                )

        try:
            position = calculate_position(
                unified_config,
                self.config.video_settings.resolution,
                visual_bounds,
            )
        except Exception as e:
            logger.warning(f"Position calculation failed, using fallback: {e}")
            position = Position(x=0.5, y=0.8)

        return position

    def _get_content_bottom(self, geom: VisualGeometry, frame_height: int) -> int:
        """Get content bottom position from geometry or settings."""
        if self.profile_settings:
            vs = self.profile_settings.video_settings
            video_top = vs.video_top_position_percent
            video_height = vs.video_content_height_percent

            if video_top is not None and video_height is not None:
                return int(frame_height * (video_top + video_height))

        return geom.rendered_y + geom.rendered_h

    def _get_font_offset_multiplier(self) -> float:
        """Get font offset multiplier from config."""
        if self.config.text_rendering:
            return self.config.text_rendering.content_aware_font_offset_multiplier
        return 5.5  # Default fallback

    def _get_max_safe_y(self) -> float:
        """Get max safe Y position from config."""
        if self.config.text_rendering:
            return self.config.text_rendering.max_safe_y_position
        return 0.95  # Default fallback
