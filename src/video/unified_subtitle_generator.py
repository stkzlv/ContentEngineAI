"""Unified Subtitle Generator for ContentEngineAI.

This module consolidates all subtitle generation functionality into a single,
streamlined interface that replaces the complex multi-function approach with
a unified API for both SRT and ASS formats.
"""

import logging
import random
import re
from pathlib import Path
from typing import Any

import pysrt

from src.utils import ensure_dirs_exist
from src.video.result_types import SubtitleResult
from src.video.subtitle_positioning import (
    Position,
    UnifiedSubtitleConfig,
    VisualBounds,
    calculate_position,
    get_font_size,
    get_style_config,
)
from src.video.video_config import config

logger = logging.getLogger(__name__)


# Use standardized result type from result_types module


class UnifiedSubtitleGenerator:
    """Unified subtitle generator supporting both SRT and ASS formats."""

    def __init__(
        self,
        config: UnifiedSubtitleConfig,
        frame_size: tuple[int, int],
        product_id: str | None = None,
    ):
        """Initialize the unified subtitle generator.

        Args:
        ----
            config: Unified subtitle configuration
            frame_size: Video frame dimensions (width, height)
            product_id: Product identifier for randomization seeding

        """
        self.config = config
        self.frame_size = frame_size
        self.product_id = product_id
        self.style_config = get_style_config(
            config.style_preset, config=config, product_id=product_id
        )
        # Pre-select colors once per producer run to ensure consistency
        self._selected_colors = self._select_colors()
        # Pre-select effects once per video to ensure consistency
        self._selected_effects = self._select_effects()

    def estimate_text_width_pixels(self, text: str, font_size: int) -> int:
        """Estimate the pixel width of text based on font size and character count.

        Args:
        ----
            text: The text to measure
            font_size: Font size in pixels

        Returns:
        -------
            Estimated width in pixels

        """
        # Use the font width to height ratio from config
        # Most fonts have a ratio around 0.5-0.6 (characters are roughly half tall)
        font_width_ratio = self.style_config["font_width_to_height_ratio"]

        # Calculate average character width for the font size
        avg_char_width = font_size * font_width_ratio

        # Apply character-specific adjustments for better accuracy
        # Get character width factors from configuration
        text_rendering = config.text_rendering
        narrow_factor = (
            text_rendering.narrow_char_width_factor if text_rendering else 0.4
        )
        wide_factor = text_rendering.wide_char_width_factor if text_rendering else 1.2
        space_factor = text_rendering.space_char_width_factor if text_rendering else 0.3

        total_width = 0
        for char in text:
            if char in "iIjl|':;,.'`":  # Narrow characters
                char_width = avg_char_width * narrow_factor
            elif char in "mwMWAGOQ":  # Wide characters
                char_width = avg_char_width * wide_factor
            elif char == " ":  # Space
                char_width = avg_char_width * space_factor
            else:  # Regular characters
                char_width = avg_char_width
            total_width += char_width

        return int(total_width)

    def fits_within_image_width(
        self,
        text: str,
        font_size: int,
        image_width: int,
        max_width_percent: float | None = None,
    ) -> bool:
        """Check if text fits within the specified image width constraint.

        Args:
        ----
            text: The text to check
            font_size: Font size in pixels
            image_width: Width of the image in pixels
            max_width_percent: Maximum percentage of image width to use (default 95%)

        Returns:
        -------
            True if text fits within constraints, False otherwise

        """
        # Get max width percent from configuration if not provided
        if max_width_percent is None:
            text_rendering = config.text_rendering
            max_width_percent = (
                text_rendering.max_text_width_percent if text_rendering else 0.95
            )

        text_width = self.estimate_text_width_pixels(text, font_size)
        max_allowed_width = image_width * max_width_percent
        return text_width <= max_allowed_width

    def generate_from_timings(
        self,
        timings: list[dict[str, Any]],
        output_path: Path,
        format_type: str = "srt",
        voiceover_duration: float | None = None,
        visual_bounds: VisualBounds | None = None,
        debug_mode: bool = False,
    ) -> SubtitleResult:
        """Generate subtitles from word timing data.

        Args:
        ----
            timings: Word timing data with 'word', 'start_time', 'end_time'
            output_path: Where to save the subtitle file
            format_type: Output format ('srt' or 'ass')
            voiceover_duration: Total audio duration for validation
            visual_bounds: Bounds of visual content for positioning
            debug_mode: Enable debug output

        Returns:
        -------
            GenerationResult with success status and details

        """
        if not timings:
            result = SubtitleResult(
                success=False,
                path=None,
                format=format_type,
            )
            result.add_error("No timing data provided")
            return result

        try:
            # Clean and validate timings
            cleaned_timings = self._clean_timings(timings)
            if not cleaned_timings:
                result = SubtitleResult(
                    success=False,
                    path=None,
                    format=format_type,
                )
                result.add_error("No valid timings after cleaning")
                return result

            # Create subtitle segments
            segments = self._create_segments(
                cleaned_timings, voiceover_duration, visual_bounds, debug_mode
            )

            if not segments:
                result = SubtitleResult(
                    success=False,
                    path=None,
                    format=format_type,
                )
                result.add_error("No segments created from timings")
                return result

            # Generate output based on format
            if format_type.lower() == "ass":
                result_path = self._generate_ass(
                    segments, output_path, visual_bounds, debug_mode
                )
            else:
                result_path = self._generate_srt(segments, output_path, debug_mode)

            if result_path and result_path.exists():
                result = SubtitleResult(
                    success=True,
                    path=result_path,
                    format=format_type,
                    segments_created=len(segments),
                    generation_method="timing_based",
                )
            else:
                result = SubtitleResult(
                    success=False,
                    path=None,
                    format=format_type,
                    errors=[f"Failed to generate {format_type.upper()} file"],
                )

            return result

        except Exception as e:
            logger.error(f"Error generating subtitles: {e}", exc_info=debug_mode)
            result = SubtitleResult(
                success=False,
                path=None,
                format=format_type,
                errors=[f"Generation error: {str(e)}"],
            )
            return result

    def generate_from_script(
        self,
        script_text: str,
        duration: float,
        output_path: Path,
        format_type: str = "srt",
        visual_bounds: VisualBounds | None = None,
        debug_mode: bool = False,
    ) -> SubtitleResult:
        """Generate subtitles from script text with estimated timing.

        Args:
        ----
            script_text: Text to convert to subtitles
            duration: Total duration to spread text across
            output_path: Where to save the subtitle file
            format_type: Output format ('srt' or 'ass')
            visual_bounds: Bounds of visual content for positioning
            debug_mode: Enable debug output

        Returns:
        -------
            GenerationResult with success status and details

        """
        if not script_text or not script_text.strip():
            SubtitleResult(
                success=False,
                path=None,
                format=format_type,
                errors=["No script text provided"],
            )

        if duration <= 0:
            SubtitleResult(
                success=False,
                path=None,
                format=format_type,
                errors=[f"Invalid duration: {duration}"],
            )

        try:
            # Create segments from script with estimated timing
            segments = self._create_script_segments(
                script_text, duration, visual_bounds
            )

            if not segments:
                SubtitleResult(
                    success=False,
                    path=None,
                    format=format_type,
                    errors=["Could not create segments from script"],
                )

            # Generate output based on format
            if format_type.lower() == "ass":
                result_path = self._generate_ass(
                    segments, output_path, visual_bounds, debug_mode
                )
            else:
                result_path = self._generate_srt(segments, output_path, debug_mode)

            if result_path and result_path.exists():
                return SubtitleResult(
                    success=True,
                    path=result_path,
                    format=format_type,
                    segments_created=len(segments),
                    generation_method="script_based",
                    timing_source="estimated",
                )
            else:
                return SubtitleResult(
                    success=False,
                    path=None,
                    format=format_type,
                    errors=[
                        f"Failed to generate {format_type.upper()} file from script"
                    ],
                )

        except Exception as e:
            logger.error(
                f"Error generating subtitles from script: {e}", exc_info=debug_mode
            )
            return SubtitleResult(
                success=False,
                path=None,
                format=format_type,
                errors=[f"Script generation error: {str(e)}"],
            )

    def _clean_timings(self, timings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean and validate timing data."""
        cleaned = []
        word_pattern = re.compile(r"^\W+|\W+$")

        for entry in timings:
            word = entry.get("word")
            start = entry.get("start_time")
            end = entry.get("end_time")

            if not isinstance(word, str) or not word.strip():
                continue
            if start is None or end is None:
                continue

            try:
                start_f = float(start)
                end_f = float(end)

                if start_f < 0 or end_f <= start_f:
                    continue

                cleaned.append(
                    {
                        "word": word,
                        "clean_word": word_pattern.sub("", word),
                        "start_time": start_f,
                        "end_time": end_f,
                    }
                )

            except (ValueError, TypeError):
                continue

        return cleaned

    def _create_segments(
        self,
        cleaned_timings: list[dict[str, Any]],
        voiceover_duration: float | None,
        visual_bounds: VisualBounds | None = None,
        debug_mode: bool = False,
    ) -> list[dict[str, Any]]:
        """Create subtitle segments from cleaned timing data."""
        segments = []
        current_words: list[dict[str, Any]] = []
        current_text = ""
        current_start = None

        for _i, word_data in enumerate(cleaned_timings):
            word = word_data["word"]
            start_time = word_data["start_time"]
            end_time = word_data["end_time"]

            if not current_words:
                current_start = start_time

            # Check if we should break the segment
            should_break = False

            if current_words:
                potential_text = f"{current_text} {word}".strip()
                current_duration = end_time - current_start if current_start else 0

                # Break conditions
                if (
                    len(potential_text) > self.config.max_line_length
                    or current_duration > self.config.max_duration
                    or (word.endswith((".", "!", "?")) and len(current_words) >= 3)
                ):
                    should_break = True

            if should_break:
                # Create segment from current words
                if current_words and current_text:
                    segment_end = current_words[-1]["end_time"]
                    segment_duration = segment_end - current_start

                    # Apply minimum duration
                    if (
                        segment_duration < self.config.min_duration
                        and current_start is not None
                    ):
                        segment_end = current_start + self.config.min_duration

                    # Respect voiceover duration
                    if voiceover_duration:
                        segment_end = min(segment_end, voiceover_duration)

                    if segment_end > current_start:
                        segments.append(
                            {
                                "text": current_text,
                                "start": current_start,
                                "end": segment_end,
                            }
                        )

                # Start new segment
                current_words = [word_data]
                current_text = word
                current_start = start_time
            else:
                # Add to current segment
                current_words.append(word_data)
                current_text = (
                    f"{current_text} {word}".strip() if current_text else word
                )

        # Handle final segment
        if current_words and current_text:
            segment_end = current_words[-1]["end_time"]
            segment_duration = segment_end - current_start if current_start else 0

            if (
                segment_duration < self.config.min_duration
                and current_start is not None
            ):
                segment_end = current_start + self.config.min_duration

            if voiceover_duration:
                segment_end = min(segment_end, voiceover_duration)

            if segment_end > current_start:
                segments.append(
                    {
                        "text": current_text,
                        "start": current_start,
                        "end": segment_end,
                    }
                )

        return segments

    def _create_script_segments(
        self,
        script_text: str,
        duration: float,
        visual_bounds: VisualBounds | None = None,
    ) -> list[dict[str, Any]]:
        """Create subtitle segments from script text with estimated timing."""
        # Clean the script and split into words
        words = script_text.strip().split()
        if not words:
            return []

        segments = []
        current_time = 0.0

        # Get speaking rate from configuration with fallback
        subtitle_segmentation = config.subtitle_segmentation
        fallback_rate = (
            subtitle_segmentation.fallback_segment_duration_sec
            if subtitle_segmentation
            else 2.5
        )

        speaking_rate = (
            self.config.speaking_rate_words_per_sec
            if hasattr(self.config, "speaking_rate_words_per_sec")
            else fallback_rate
        )
        estimated_duration = len(words) / speaking_rate

        # Adjust speaking rate if needed to fit actual duration
        if estimated_duration > 0:
            speaking_rate = len(words) / duration

        current_segment_words: list[str] = []
        current_segment_text = ""

        for _i, word in enumerate(words):
            # Check if we should create a segment
            should_break = False

            # Potential new text if we add this word
            potential_text = (
                f"{current_segment_text} {word}".strip()
                if current_segment_text
                else word
            )

            # Break conditions:
            # 1. Line length limit (character count and pixel width)
            if len(potential_text) > self.config.max_line_length:
                should_break = True
            elif visual_bounds is not None and visual_bounds.width > 0:
                # Check pixel-based width constraint
                font_size = get_font_size(self.config, self.frame_size[1])
                if not self.fits_within_image_width(
                    potential_text, font_size, int(visual_bounds.width)
                ):
                    should_break = True

            # 2. Natural sentence breaks
            elif word.endswith((".", "!", "?")) and len(current_segment_words) >= 3:
                # Add this word to current segment before breaking
                current_segment_words.append(word)
                current_segment_text = potential_text
                should_break = True

            # 3. Duration limit (but allow at least 3 words)
            elif len(current_segment_words) >= 3:
                estimated_word_duration = len(current_segment_words) / speaking_rate
                if estimated_word_duration >= self.config.max_duration:
                    should_break = True

            if should_break and current_segment_words:
                # Calculate segment duration based on word count
                word_duration = len(current_segment_words) / speaking_rate
                word_duration = max(self.config.min_duration, word_duration)
                word_duration = min(self.config.max_duration, word_duration)

                # Ensure we don't exceed total duration
                if current_time + word_duration > duration:
                    word_duration = duration - current_time

                if word_duration > 0:
                    segments.append(
                        {
                            "text": current_segment_text,
                            "start": current_time,
                            "end": current_time + word_duration,
                        }
                    )
                    current_time += word_duration

                # Start new segment
                if not word.endswith((".", "!", "?")):  # Not just added
                    current_segment_words = [word]
                    current_segment_text = word
                else:
                    current_segment_words = []
                    current_segment_text = ""
            else:
                # Add word to current segment
                current_segment_words.append(word)
                current_segment_text = potential_text

        # Handle final segment
        if current_segment_words and current_segment_text:
            remaining_time = duration - current_time
            final_duration = max(
                self.config.min_duration, min(remaining_time, self.config.max_duration)
            )

            if final_duration > 0:
                segments.append(
                    {
                        "text": current_segment_text,
                        "start": current_time,
                        "end": current_time + final_duration,
                    }
                )

        return segments

    def _generate_srt(
        self,
        segments: list[dict[str, Any]],
        output_path: Path,
        debug_mode: bool = False,
    ) -> Path | None:
        """Generate SRT file from segments using pysrt."""
        try:
            ensure_dirs_exist(output_path.parent)

            # Create SubRipFile
            subs = pysrt.SubRipFile()

            for i, segment in enumerate(segments, 1):
                start_ms = int(segment["start"] * 1000)
                end_ms = int(segment["end"] * 1000)

                if start_ms >= end_ms:
                    continue

                item = pysrt.SubRipItem(
                    index=i,
                    start=pysrt.SubRipTime(milliseconds=start_ms),
                    end=pysrt.SubRipTime(milliseconds=end_ms),
                    text=segment["text"].strip(),
                )
                subs.append(item)

            if not subs:
                logger.warning("No valid SRT items created")
                return None

            # Save file
            subs.save(str(output_path), encoding="utf-8")

            if debug_mode:
                logger.debug(f"Generated SRT with {len(subs)} segments: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate SRT: {e}")
            return None

    def _generate_ass(
        self,
        segments: list[dict[str, Any]],
        output_path: Path,
        visual_bounds: VisualBounds | None = None,
        debug_mode: bool = False,
    ) -> Path | None:
        """Generate ASS file from segments with positioning."""
        try:
            ensure_dirs_exist(output_path.parent)

            # Calculate position
            position = calculate_position(self.config, self.frame_size, visual_bounds)
            font_size = get_font_size(self.config, self.frame_size[1])

            # Randomize colors if enabled
            colors = self._get_colors()

            # Create ASS content
            ass_lines = self._create_ass_header(font_size, colors)

            # Add dialogue lines
            for segment in segments:
                dialogue_line = self._create_dialogue_line(
                    segment, position, colors, debug_mode
                )
                if dialogue_line:
                    ass_lines.append(dialogue_line)

            # Write file
            output_path.write_text("\n".join(ass_lines), encoding="utf-8")

            if debug_mode:
                logger.debug(
                    f"Generated ASS with {len(segments)} segments: {output_path}"
                )

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate ASS: {e}")
            return None

    def _select_colors(self) -> dict[str, str]:
        """Select colors once during initialization for consistent per-run coloring."""
        colors = {
            "primary": self.style_config["font_color"],
            "outline": self.style_config["outline_color"],
        }

        if self.config.randomize_colors:
            # Simple color randomization - selected once per producer run
            color_options = [
                {"primary": "&H00FFFFFF", "outline": "&H00000000"},  # White/Black
                {"primary": "&H0000FFFF", "outline": "&H00000000"},  # Yellow/Black
                {"primary": "&H00FF00FF", "outline": "&H00000000"},  # Magenta/Black
                {"primary": "&H00FFFF00", "outline": "&H00000000"},  # Cyan/Black
            ]
            colors.update(random.choice(color_options))  # noqa: S311

        return colors

    def _select_effects(self) -> dict[str, Any]:
        """Select effects once for consistent per-video effects."""
        selected_effects = {
            "scale_pulse": False,
            "rotation_bounce": False,
            "glow": False,
            "typewriter": False,
            "karaoke": False,
            "movement": False,
        }

        if self.config.randomize_effects and len(self.style_config["effects"]) > 0:
            # Seed random generator with product_id for consistent selection per video
            if self.product_id:
                random.seed(hash(self.product_id + "effects"))

            # Choose 1 effect for this video
            available_effects = [
                effect
                for effect in selected_effects
                if effect in self.style_config["effects"]
            ]

            if available_effects:
                num_effects = 1
                chosen_effects = random.sample(available_effects, num_effects)

                for effect in chosen_effects:
                    selected_effects[effect] = True

                logger.debug(f"Selected effects for video: {chosen_effects}")

        return selected_effects

    def _get_colors(self) -> dict[str, str]:
        """Get the pre-selected color configuration."""
        return self._selected_colors

    def _create_ass_header(self, font_size: int, colors: dict[str, str]) -> list[str]:
        """Create ASS file header sections."""
        width, height = self.frame_size
        font_name = self.style_config["font_name"]

        return [
            "[Script Info]",
            "Title: Generated Subtitles",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 1",
            "",
            "[V4+ Styles]",
            (
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, "
                "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
            ),
            (
                f"Style: Default,{font_name},{font_size},{colors['primary']},"
                f"{colors['primary']},{colors['outline']},&H80000000,"
                f"{-1 if self.style_config['bold'] else 0},0,1,"
                f"{self.style_config['outline_thickness']},"
                f"{1 if self.style_config['shadow'] else 0},5,10,10,0,1"
            ),
            "",
            "[Events]",
            (
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
                "MarginV, Effect, Text"
            ),
        ]

    def _create_karaoke_effects(self, text: str, segment_duration: float) -> str:
        """Create karaoke-style word-by-word highlighting effects.

        Args:
        ----
            text: The text to add karaoke effects to
            segment_duration: Duration of the segment in seconds

        Returns:
        -------
            Text with karaoke timing tags properly formatted for ASS

        """
        words = text.split()
        if len(words) <= 1:
            return text

        # Calculate time per word in centiseconds (ASS karaoke uses centiseconds)
        time_per_word = int((segment_duration * 100) / len(words))

        # Get karaoke timing limits from configuration
        subtitle_effects = config.subtitle_effects
        min_timing = subtitle_effects.karaoke_timing_min_ms if subtitle_effects else 20
        max_timing = subtitle_effects.karaoke_timing_max_ms if subtitle_effects else 200

        time_per_word = max(min_timing, min(time_per_word, max_timing))

        # Create karaoke text with \k tags properly enclosed in braces
        karaoke_parts = []
        for i, word in enumerate(words):
            if i == 0:
                karaoke_parts.append(f"{{\\k{time_per_word}}}{word}")
            else:
                karaoke_parts.append(f" {{\\k{time_per_word}}}{word}")

        return "".join(karaoke_parts)

    def _create_dialogue_line(
        self,
        segment: dict[str, Any],
        position: Position,
        colors: dict[str, str],
        debug_mode: bool = False,
    ) -> str | None:
        """Create ASS dialogue line with positioning and effects."""
        try:
            start_time = self._format_ass_time(segment["start"])
            end_time = self._format_ass_time(segment["end"])
            text = segment["text"]

            # Calculate pixel coordinates
            pos_x = int(position.x * self.frame_size[0])
            pos_y = int(position.y * self.frame_size[1])

            # Apply effects if enabled
            effects = []
            segment_duration = segment["end"] - segment["start"]

            # Basic fade effect
            if "fade" in self.style_config["effects"]:
                subtitle_effects = config.subtitle_effects
                fade_duration = (
                    subtitle_effects.fade_duration_ms if subtitle_effects else 300
                )
                effects.append(f"\\fad({fade_duration},{fade_duration})")

            # Apply pre-selected effects consistently across all segments
            final_text = text
            movement_effect = ""

            if self.config.randomize_effects:
                # Scale pulse effect
                if self._selected_effects.get("scale_pulse", False):
                    subtitle_effects = config.subtitle_effects
                    pulse_factor = (
                        subtitle_effects.pulse_duration_factor
                        if subtitle_effects
                        else 500
                    )
                    pulse_scale_max = (
                        subtitle_effects.pulse_scale_max if subtitle_effects else 110
                    )
                    pulse_scale_normal = (
                        subtitle_effects.pulse_scale_normal if subtitle_effects else 100
                    )

                    pulse_duration = min(int(segment_duration * pulse_factor), 1000)
                    effects.append(
                        f"\\t(0,{pulse_duration},\\fscx{pulse_scale_max}\\fscy{pulse_scale_max})"
                        f"\\t({pulse_duration},{pulse_duration * 2},"
                        f"\\fscx{pulse_scale_normal}\\fscy{pulse_scale_normal})"
                    )

                # Rotation bounce effect
                if self._selected_effects.get("rotation_bounce", False):
                    subtitle_effects = config.subtitle_effects
                    bounce_factor = (
                        subtitle_effects.bounce_duration_factor
                        if subtitle_effects
                        else 300
                    )
                    bounce_max = (
                        subtitle_effects.bounce_rotation_max if subtitle_effects else 5
                    )
                    bounce_min = (
                        subtitle_effects.bounce_rotation_min if subtitle_effects else -5
                    )
                    bounce_rest = (
                        subtitle_effects.bounce_rotation_rest if subtitle_effects else 0
                    )

                    bounce_duration = int(segment_duration * bounce_factor)
                    effects.append(
                        f"\\t(0,{bounce_duration},\\frz{bounce_max})"
                        f"\\t({bounce_duration},{bounce_duration * 2},"
                        f"\\frz{bounce_min})"
                        f"\\t({bounce_duration * 2},{bounce_duration * 3},"
                        f"\\frz{bounce_rest})"
                    )

                # Glow effect with color transition
                if self._selected_effects.get("glow", False):
                    subtitle_effects = config.subtitle_effects
                    glow_factor = (
                        subtitle_effects.glow_duration_factor
                        if subtitle_effects
                        else 400
                    )

                    glow_duration = int(segment_duration * glow_factor)
                    glow_end = glow_duration * 2
                    effects.append(
                        f"\\t(0,{glow_duration},\\3c&H00FFFF&)"
                        f"\\t({glow_duration},{glow_end},\\3c{colors['outline']})"
                    )

                # Typewriter reveal effect
                if self._selected_effects.get("typewriter", False):
                    subtitle_effects = config.subtitle_effects
                    max_reveal_time = (
                        subtitle_effects.typewriter_char_reveal_max_sec
                        if subtitle_effects
                        else 0.1
                    )
                    min_timing = (
                        subtitle_effects.typewriter_min_timing_ms
                        if subtitle_effects
                        else 50
                    )

                    char_reveal_time = (
                        min(segment_duration / len(text), max_reveal_time) * 1000
                    )
                    if char_reveal_time > min_timing:  # Only apply if reasonable timing
                        reveal_duration = int(char_reveal_time * len(text))
                        effects.append(
                            f"\\t(0,{reveal_duration},\\alpha&HFF&)"
                            f"\\t({reveal_duration},0,\\alpha&H00&)"
                        )

                # Karaoke effects
                if self._selected_effects.get("karaoke", False):
                    final_text = self._create_karaoke_effects(text, segment_duration)

                # Movement effect (subtle floating)
                if self._selected_effects.get("movement", False):
                    subtitle_effects = config.subtitle_effects
                    move_distance = (
                        subtitle_effects.movement_distance_pixels
                        if subtitle_effects
                        else 10
                    )

                    move_duration = int(segment_duration * 1000)
                    start_y = pos_y
                    end_y = pos_y - move_distance
                    movement_effect = (
                        f"\\move({pos_x},{start_y},{pos_x},{end_y},0,{move_duration})"
                    )

            # Set positioning effect
            if movement_effect:
                pos_effect = movement_effect
            else:
                pos_effect = f"\\pos({pos_x},{pos_y})"

            # Combine effects
            effect_str = "".join(effects)

            # Create dialogue line
            dialogue = (
                f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,"
                f"{{{pos_effect}{effect_str}}}{final_text}"
            )

            return dialogue

        except Exception as e:
            if debug_mode:
                logger.warning(f"Failed to create dialogue line: {e}")
            return None

    def _format_ass_time(self, seconds: float) -> str:
        """Format time for ASS format (H:MM:SS.CC) - single digit hours per spec."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        centiseconds = int((secs % 1) * 100)
        secs = int(secs)

        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
