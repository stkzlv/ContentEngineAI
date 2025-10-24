"""Video Assembler Module

This module is responsible for combining all the elements of a video (visuals,
audio, subtitles) into a final rendered output using FFmpeg. It handles complex
media processing operations including video compositing, audio mixing, and
subtitle rendering.
"""

import asyncio
import json
import logging
import mimetypes
import re
import secrets
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import ensure_dirs_exist
from src.utils.async_io import (
    async_get_media_duration,
    async_run_ffmpeg,
    ffmpeg_semaphore,
)
from src.utils.caching import cache_media_metadata, get_cached_media_metadata
from src.video.video_config import (
    DEFAULT_FALLBACK_FONT,
    FALLBACK_FONT_ALTERNATIVES,
    FONT_FILE_EXTENSIONS,
    FONT_REGULAR_SUFFIXES,
    SRT_BLOCK_SEPARATOR,
    SRT_ENCODING,
    SRT_HOURS_IN_SECONDS,
    SRT_MILLISECONDS_DIVISOR,
    SRT_MIN_BLOCK_LINES,
    SRT_MINUTES_IN_SECONDS,
    SRT_TIME_HOUR_SEPARATOR,
    SRT_TIME_SECOND_SEPARATOR,
    SRT_TIME_SEPARATOR,
    VideoConfig,
)

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class VisualGeometry:
    """Represents the position and dimensions of a visual element in the video.

    This class stores the coordinates and size of a visual element (image or video)
    after it has been positioned and scaled within the output video frame.

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


@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry with timing and text.

    This class stores the start and end times for a subtitle segment along with
    the text to be displayed during that time interval.

    Attributes
    ----------
        start: Start time in seconds
        end: End time in seconds
        text: Subtitle text to display

    """

    start: float
    end: float
    text: str


class VideoAssembler:
    """Assembles final videos from various media components using FFmpeg.

    This class is responsible for combining visual media (images/videos),
    audio (voiceover and background music), and subtitles into a cohesive
    final video. It handles all FFmpeg command generation and execution.

    The assembler manages complex operations like:
    - Scaling and positioning visuals
    - Creating transitions between media elements
    - Mixing multiple audio tracks
    - Rendering subtitles with styling
    - Applying filters and effects
    """

    def __init__(self, config: VideoConfig, debug_mode: bool = False):
        """Initialize the video assembler with configuration.

        Args:
        ----
            config: Video configuration containing FFmpeg settings, output specs,
                   subtitle styling, and other assembly parameters
            debug_mode: Enable debug logging for assembly operations

        """
        self.config = config
        self.debug_mode = debug_mode
        self.ffmpeg_path = config.ffmpeg_settings.executable_path or "ffmpeg"
        self.ffprobe_path = (
            config.ffmpeg_settings.executable_path or "ffmpeg"
        ).replace("ffmpeg", "ffprobe")

        # Profile-specific settings (applied when using set_profile_settings)
        self.profile_settings: dict[str, Any] | None = None

        # Product identifier for randomization seeding
        self.product_id: str | None = None

    def set_profile_settings(
        self, profile_name: str, cli_overrides: dict[str, Any] | None = None
    ) -> None:
        """Apply profile-specific settings to override global configuration.

        This method retrieves and applies profile-merged settings for image
        positioning/sizing and subtitle styling based on the specified profile.

        Args:
        ----
            profile_name: Name of the video profile to apply settings for
            cli_overrides: Optional CLI overrides to apply with highest precedence

        """
        self.profile_settings = self.config.get_profile_merged_settings(
            profile_name, cli_overrides
        )

        if self.debug_mode:
            logger.debug(f"Applied profile settings for '{profile_name}'")
            logger.debug(
                f"Image width percent: "
                f"{self.profile_settings['video_settings']['image_width_percent']}"
            )
            logger.debug(
                f"Image top position: "
                f"{self.profile_settings['video_settings']['image_top_position_percent']}"
            )
            logger.debug(
                f"Subtitle anchor: "
                f"{self.profile_settings['subtitle_settings']['anchor']}"
            )
            logger.debug(
                f"Subtitle style preset: "
                f"{self.profile_settings['subtitle_settings']['style_preset']}"
            )

    def set_product_id(self, product_id: str) -> None:
        """Set the product identifier for randomization seeding.

        Args:
        ----
            product_id: Product identifier (e.g., ASIN or sanitized product name)

        """
        self.product_id = product_id
        if self.debug_mode:
            logger.debug(f"Set product_id for randomization: {product_id}")

    def _get_effective_video_settings(self) -> dict[str, Any]:
        """Get effective video settings with profile overrides applied."""
        if self.profile_settings:
            return self.profile_settings["video_settings"]  # type: ignore[no-any-return]
        # Fallback to global config if no profile settings
        # Use model_dump() for Pydantic v2 compatibility instead of __dict__
        return self.config.video_settings.model_dump()

    def _get_effective_subtitle_settings(self) -> dict[str, Any]:
        """Get effective subtitle settings with profile overrides applied."""
        if self.profile_settings:
            # Use the subtitle_settings from profile_settings directly
            settings = self.profile_settings["subtitle_settings"]

            # Validate using UnifiedSubtitleConfig if needed
            from src.video.subtitle_positioning import UnifiedSubtitleConfig

            try:
                # Attempt to validate the settings through Pydantic
                validated_config = UnifiedSubtitleConfig(**settings)
                return validated_config.model_dump()
            except Exception as e:
                logger.warning(
                    f"Subtitle settings validation failed, using raw settings: {e}"
                )
                return settings  # type: ignore[no-any-return]

        # Fallback to global config if no profile settings
        # subtitle_settings is dict[str, Any], return as-is
        return self.config.subtitle_settings

    def _is_video(self, path: Path) -> bool:
        """Determine if a file is a video based on its MIME type.

        Args:
        ----
            path: Path to the media file

        Returns:
        -------
            True if the file is a video, False otherwise

        """
        content_type, _ = mimetypes.guess_type(path)
        return content_type is not None and content_type.startswith("video")

    async def _get_media_dimensions(self, file_path: Path) -> tuple[int, int]:
        """Extract the width and height of a media file using FFprobe.

        This method uses FFprobe to analyze the media file and extract its
        dimensions. It works for both images and videos.

        Args:
        ----
            file_path: Path to the media file

        Returns:
        -------
            Tuple of (width, height) in pixels

        Raises:
        ------
            ValueError: If the dimensions cannot be extracted

        """
        # Use configurable FFprobe parameters
        streams = (
            self.config.video_processing.ffmpeg_probe_streams
            if hasattr(self.config, "video_processing") and self.config.video_processing
            else "v:0"
        )
        entries = (
            self.config.video_processing.ffmpeg_probe_entries
            if hasattr(self.config, "video_processing") and self.config.video_processing
            else "stream=width,height"
        )
        format_spec = (
            self.config.video_processing.ffmpeg_probe_format
            if hasattr(self.config, "video_processing") and self.config.video_processing
            else "csv=s=x:p=0"
        )

        cmd = [
            self.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            streams,
            "-show_entries",
            entries,
            "-of",
            format_spec,
            str(file_path),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    f"ffprobe failed to get dimensions for {file_path.name}: "
                    f"{stderr.decode()}"
                )
                return 0, 0
            w_str, h_str = stdout.decode().strip().split("x")
            return int(w_str), int(h_str)
        except Exception as e:
            logger.error(f"Error getting dimensions for {file_path.name}: {e}")
            return 0, 0

    async def _get_media_duration(self, file_path: Path) -> float:
        # Check cache first
        cached_metadata = get_cached_media_metadata(file_path)
        if cached_metadata and "duration" in cached_metadata:
            duration_value: float = cached_metadata["duration"]
            return duration_value

        # Get duration and cache it
        duration = await async_get_media_duration(
            file_path,
            self.ffprobe_path,
            timeout_sec=self.config.video_settings.verification_probe_timeout_sec,
        )

        # Cache the metadata
        if duration > 0:
            cache_media_metadata(file_path, {"duration": duration})

        return duration

    def _parse_srt(self, subtitle_path: Path) -> list[SubtitleEntry]:
        content = subtitle_path.read_text(encoding=SRT_ENCODING)
        entries = []
        for block in content.strip().split(SRT_BLOCK_SEPARATOR):
            lines = block.split("\n")
            if len(lines) < SRT_MIN_BLOCK_LINES:
                continue
            try:
                time_line = lines[1]
                text = "\n".join(lines[2:])

                def parse_time(ts: str) -> float:
                    h, m, s_ms = ts.split(SRT_TIME_HOUR_SEPARATOR)
                    s, ms = s_ms.split(SRT_TIME_SECOND_SEPARATOR)
                    return (
                        int(h) * SRT_HOURS_IN_SECONDS
                        + int(m) * SRT_MINUTES_IN_SECONDS
                        + int(s)
                        + int(ms) / SRT_MILLISECONDS_DIVISOR
                    )

                start_str, end_str = time_line.split(SRT_TIME_SEPARATOR)
                entries.append(
                    SubtitleEntry(
                        start=parse_time(start_str), end=parse_time(end_str), text=text
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse SRT block: {e}")
        return entries

    def _resolve_font_path(self, font_name: str) -> Path | None:
        """Resolve a font name to an actual font file path.

        This method searches the configured font directory for a font file that matches
        the requested font name. It normalizes both the requested font name and the
        font file names to improve matching accuracy.

        If the requested font cannot be found, it will try to use a fallback font
        in the following order:
        1. The default fallback font (DEFAULT_FALLBACK_FONT)
        2. Any of the alternative fallback fonts (FALLBACK_FONT_ALTERNATIVES)
        3. The first valid font file found in the directory

        Args:
        ----
            font_name: Name of the font to resolve (e.g., "Arial", "DM Serif Display")

        Returns:
        -------
            Path to the font file if found, None if no usable font could be found

        """
        font_dir = Path(self.config.subtitle_settings["font_directory"])
        if not font_dir.is_dir():
            logger.warning(f"Font directory does not exist: {font_dir}")
            return None

        # Get all valid font files in the directory
        font_files = [
            f
            for f in font_dir.iterdir()
            if f.is_file() and f.suffix.lower() in FONT_FILE_EXTENSIONS
        ]

        if not font_files:
            logger.error(f"No valid font files found in directory: {font_dir}")
            return None

        # Normalize font name by removing spaces and converting to lowercase
        normalized_font_name = font_name.lower().replace(" ", "").replace("-", "")

        # First attempt: Try to find the exact requested font
        for file_path in font_files:
            # Normalize file stem by removing common suffixes and converting to
            # lowercase
            file_stem = file_path.stem.lower()
            normalized_file_stem = file_stem

            # Remove common suffixes like -Regular, -R
            for suffix in FONT_REGULAR_SUFFIXES:
                normalized_file_stem = normalized_file_stem.replace(suffix, "")

            # Remove spaces and hyphens for comparison
            normalized_file_stem = normalized_file_stem.replace(" ", "").replace(
                "-", ""
            )

            logger.debug(
                f"Checking font file: {file_path.name} "
                f"(normalized: '{normalized_file_stem}')"
            )

            if (
                normalized_file_stem == normalized_font_name
                or normalized_file_stem.startswith(normalized_font_name)
            ):
                logger.info(f"Resolved font '{font_name}' to path: {file_path}")
                return file_path

        # Second attempt: Try to find the default fallback font
        logger.warning(
            f"Could not find font '{font_name}', trying fallback: "
            f"{DEFAULT_FALLBACK_FONT}"
        )
        fallback_path = self._resolve_font_path(DEFAULT_FALLBACK_FONT)
        if fallback_path:
            logger.info(
                f"Using fallback font: {DEFAULT_FALLBACK_FONT} -> {fallback_path}"
            )
            return fallback_path

        # Third attempt: Try alternative fallback fonts
        for alt_font in FALLBACK_FONT_ALTERNATIVES:
            logger.warning(f"Trying alternative fallback font: {alt_font}")
            alt_path = self._resolve_font_path(alt_font)
            if alt_path:
                logger.info(
                    f"Using alternative fallback font: {alt_font} -> {alt_path}"
                )
                return alt_path

        # Last resort: Use the first valid font file
        logger.warning(
            f"No fallback fonts found, using first available font: {font_files[0].name}"
        )
        return font_files[0]

    def _convert_ass_color_to_ffmpeg(self, ass_color: str) -> str:
        match = re.match(
            r"&H(?:(\w{2}))?(\w{2})(\w{2})(\w{2})", ass_color, re.IGNORECASE
        )
        if not match:
            return ass_color
        alpha_ass, blue, green, red = match.groups()
        if alpha_ass is None:
            alpha_ass = "00"
        rgb_hex = f"0x{red}{green}{blue}"
        opacity = 1.0 - (int(alpha_ass, 16) / 255.0)
        # Consider opacity >= 0.99 as fully opaque (threshold to avoid tiny decimals)
        opacity_threshold = 0.99
        if opacity >= opacity_threshold:
            return rgb_hex
        else:
            return f"{rgb_hex}@{opacity:.2f}"

    def _normalize_text_for_verification(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[^\w\s]", "", text).lower()
        return re.sub(r"\s+", " ", text).strip()

    def verify_video(
        self,
        video_path: Path,
        expected_duration: float,
        should_have_subtitles: bool,
        script: str | None = None,
        subtitle_path: Path | None = None,
    ) -> dict[str, Any]:
        if not video_path.exists():
            return {
                "success": False,
                "message": f"Verification failed: Video file not found at {video_path}",
            }
        issues, warnings, details = [], [], {}
        try:
            cmd_probe = [
                self.ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=codec_type,width,height",
                "-of",
                "json",
                str(video_path),
            ]
            result = subprocess.run(
                cmd_probe,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.config.video_settings.verification_probe_timeout_sec,
            )
            video_info = json.loads(result.stdout)
            details["probe_info"] = video_info

            actual_duration = float(video_info.get("format", {}).get("duration", 0.0))
            if (
                abs(actual_duration - expected_duration)
                > self.config.video_settings.video_duration_tolerance_sec
            ):
                warnings.append(
                    f"Duration mismatch: expected {expected_duration:.2f}s, "
                    f"got {actual_duration:.2f}s"
                )
            if not any(
                s.get("codec_type") == "video" for s in video_info.get("streams", [])
            ):
                issues.append("Video stream missing.")
            if not any(
                s.get("codec_type") == "audio" for s in video_info.get("streams", [])
            ):
                warnings.append("Audio stream missing.")

            if (
                should_have_subtitles
                and subtitle_path
                and subtitle_path.exists()
                and script
            ):
                from difflib import SequenceMatcher

                srt_text = " ".join(
                    line
                    for line in subtitle_path.read_text(encoding="utf-8").splitlines()
                    if not (line.strip().isdigit() or "-->" in line)
                )
                similarity = SequenceMatcher(
                    None,
                    self._normalize_text_for_verification(script),
                    self._normalize_text_for_verification(srt_text),
                ).ratio()
                details["subtitle_content_similarity"] = similarity
                threshold = self.config.subtitle_settings[
                    "subtitle_similarity_threshold"
                ]
                if similarity < threshold:
                    warnings.append(
                        f"Subtitle content similarity to script is low "
                        f"({similarity:.2%})"
                    )
        except Exception as e:
            return {
                "success": False,
                "message": f"An unexpected error occurred during verification: {e}",
            }
        message = "Video verified successfully."
        if warnings:
            message += f" Warnings: {'; '.join(warnings)}"
        if issues:
            return {
                "success": False,
                "message": f"Issues: {'; '.join(issues)}. {message}",
                "details": details,
            }
        return {"success": True, "message": message, "details": details}

    def _prepare_audio_inputs(
        self,
        input_cmd_parts: list[str],
        voiceover_audio_path: Path | None,
        music_track_path: Path | None,
        visual_input_count: int,
    ) -> tuple[int | None, int | None]:
        """Add audio inputs to FFmpeg command and return their indices.

        Args:
        ----
            input_cmd_parts: List of FFmpeg input command parts to extend
            voiceover_audio_path: Path to voiceover audio file
            music_track_path: Path to background music file
            visual_input_count: Number of visual inputs (for index calculation)

        Returns:
        -------
            Tuple of (voiceover_input_idx, music_input_idx)

        """
        audio_input_idx_start = visual_input_count
        voiceover_input_idx, music_input_idx = None, None

        if voiceover_audio_path:
            input_cmd_parts.extend(["-i", str(voiceover_audio_path)])
            voiceover_input_idx = audio_input_idx_start
            audio_input_idx_start += 1

        if music_track_path:
            input_cmd_parts.extend(["-i", str(music_track_path)])
            music_input_idx = audio_input_idx_start

        return voiceover_input_idx, music_input_idx

    def _build_audio_filters(
        self,
        voiceover_input_idx: int | None,
        music_input_idx: int | None,
        total_video_duration: float,
    ) -> tuple[list[str], str]:
        """Build audio processing filters for FFmpeg.

        Args:
        ----
            voiceover_input_idx: Index of voiceover input in FFmpeg command
            music_input_idx: Index of music input in FFmpeg command
            total_video_duration: Target video duration for fade calculations

        Returns:
        -------
            Tuple of (audio_filters, final_audio_label)

        """
        audio_settings = self.config.audio_settings
        audio_filters = []
        audio_to_mix = []

        if voiceover_input_idx is not None:
            proc_label = "[a_voice_proc]"
            audio_filters.append(
                f"[{voiceover_input_idx}:a]volume={audio_settings.voiceover_volume_db}dB{proc_label}"
            )
            audio_to_mix.append(proc_label)

        if music_input_idx is not None:
            music_label, proc_label = f"[{music_input_idx}:a]", "[a_music_proc]"
            fade_out_start = max(
                0, total_video_duration - audio_settings.music_fade_out_duration
            )
            audio_filters.append(
                f"{music_label}volume={audio_settings.music_volume_db}dB,"
                f"afade=t=in:st=0:d={audio_settings.music_fade_in_duration},"
                f"afade=t=out:st={fade_out_start:.3f}:d={audio_settings.music_fade_out_duration}"
                f"{proc_label}"
            )
            audio_to_mix.append(proc_label)

        final_audio_label = ""
        if len(audio_to_mix) > 1:
            final_audio_label = "[a_mixed]"
            audio_filters.append(
                f"{''.join(audio_to_mix)}amix=inputs={len(audio_to_mix)}:"
                f"duration={audio_settings.audio_mix_duration}{final_audio_label}"
            )
        elif len(audio_to_mix) == 1:
            final_audio_label = audio_to_mix[0]

        return audio_filters, final_audio_label

    def _build_ffmpeg_command(
        self,
        input_cmd_parts: list[str],
        video_filters: list[str],
        audio_filters: list[str],
        final_audio_label: str,
        total_video_duration: float,
        output_path: Path,
    ) -> list[str]:
        """Build the complete FFmpeg command.

        Args:
        ----
            input_cmd_parts: Input command parts
            video_filters: Video processing filters
            audio_filters: Audio processing filters
            final_audio_label: Label for final audio stream
            total_video_duration: Target video duration
            output_path: Output file path

        Returns:
        -------
            Complete FFmpeg command as list of strings

        """
        video_settings = self.config.video_settings
        audio_settings = self.config.audio_settings

        all_filters = video_filters + audio_filters
        final_cmd = [
            self.ffmpeg_path,
            "-y",
            "-rw_timeout",
            str(self.config.ffmpeg_settings.rw_timeout_microseconds),
        ] + input_cmd_parts

        final_cmd.extend(["-filter_complex", ";".join(all_filters)])
        final_cmd.extend(["-map", "[v_out]"])

        if final_audio_label:
            final_cmd.extend(
                [
                    "-map",
                    final_audio_label,
                    "-c:a",
                    audio_settings.output_audio_codec,
                    "-b:a",
                    audio_settings.output_audio_bitrate,
                ]
            )

        final_cmd.extend(
            [
                "-c:v",
                video_settings.output_codec,
                "-preset",
                video_settings.output_preset,
                "-pix_fmt",
                video_settings.output_pixel_format,
                "-r",
                str(video_settings.frame_rate),
                "-t",
                str(total_video_duration),
                str(output_path),
            ]
        )

        return final_cmd

    def _should_create_ffmpeg_logs(self) -> bool:
        """Determine if FFmpeg command logs should be created.

        Returns
        -------
            True if logs should be created, False otherwise

        Notes
        -----
            Defaults to True if debug_settings is not configured or if any error occurs.
            This ensures FFmpeg commands are logged for debugging by default.

        """
        try:
            if hasattr(self.config, "debug_settings") and self.config.debug_settings:
                # Get the setting value, defaulting to True
                create_logs = getattr(
                    self.config.debug_settings, "create_ffmpeg_command_logs", True
                )
                return bool(create_logs)
            # Default to True when debug_settings is not configured
            return True
        except Exception as e:
            # Log the exception and default to True for safety
            logger.debug(f"Error checking FFmpeg log setting, defaulting to True: {e}")
            return True

    async def assemble_video(
        self,
        visual_inputs: list[Path],
        voiceover_audio_path: Path | None,
        music_track_path: Path | None,
        output_path: Path,
        subtitle_path: Path | None,
        total_video_duration: float,
        temp_dir: Path,
        debug_mode: bool = False,
        subtitle_upper_path: Path | None = None,
    ) -> Path | None:
        """Assemble final video from visual inputs, audio, and subtitles.

        Args:
        ----
            visual_inputs: List of visual input file paths
            voiceover_audio_path: Optional voiceover audio file
            music_track_path: Optional background music file
            output_path: Output video file path
            subtitle_path: Optional subtitle file path (lower line for two-part mode)
            total_video_duration: Target video duration in seconds
            temp_dir: Temporary directory for processing
            debug_mode: Enable debug output
            subtitle_upper_path: Optional upper subtitle file path (two-part mode only)

        Returns:
        -------
            Path to assembled video file or None if failed

        """
        logger.info(
            f"Starting single-pass video assembly for '{output_path.name}'. "
            f"Target Duration: {total_video_duration:.2f}s"
        )
        if not visual_inputs:
            logger.error("No visual inputs provided for video assembly.")
            return None

        with tempfile.TemporaryDirectory() as temp_sub_dir:
            # Check if dual subtitle mode is enabled
            if subtitle_upper_path and subtitle_upper_path.exists():
                logger.info("Two-part subtitle mode: rendering dual subtitle lines")
                # Build video processing chain with dual subtitles
                video_filters, input_cmd_parts = await self._build_dual_subtitle_graph(
                    visual_inputs,
                    total_video_duration,
                    subtitle_path,
                    subtitle_upper_path,
                    Path(temp_sub_dir),
                )
            else:
                # Build video processing chain with single subtitle line
                video_filters, input_cmd_parts = await self._build_subtitle_graph(
                    visual_inputs,
                    total_video_duration,
                    subtitle_path,
                    Path(temp_sub_dir),
                )

            # Add audio inputs to command
            voiceover_input_idx, music_input_idx = self._prepare_audio_inputs(
                input_cmd_parts,
                voiceover_audio_path,
                music_track_path,
                len(visual_inputs),
            )

            # Build audio processing filters
            audio_filters, final_audio_label = self._build_audio_filters(
                voiceover_input_idx,
                music_input_idx,
                total_video_duration,
            )

            # Build complete FFmpeg command
            final_cmd = self._build_ffmpeg_command(
                input_cmd_parts,
                video_filters,
                audio_filters,
                final_audio_label,
                total_video_duration,
                output_path,
            )

            ensure_dirs_exist(output_path)

            # Determine log file path
            command_log_path = (
                temp_dir / f"{output_path.stem}_ffmpeg_command.log"
                if self._should_create_ffmpeg_logs()
                else None
            )

            # Execute FFmpeg with concurrency control
            success, stdout, stderr = await ffmpeg_semaphore.run_with_limit(
                async_run_ffmpeg(
                    final_cmd,
                    timeout_sec=self.config.ffmpeg_settings.final_assembly_timeout_sec,
                    log_path=command_log_path,
                )
            )

            if success:
                logger.info(f"Successfully assembled video: {output_path}")
                return output_path
            else:
                logger.error(f"FFmpeg failed. Stderr: {stderr}")
                return None

    async def _build_visual_chain(
        self,
        visual_inputs: list[Path],
        total_video_duration: float,
        is_relative_mode: bool,
    ) -> tuple[
        list[str], list[str], list[tuple[Path, float, bool]], str, list[VisualGeometry]
    ]:
        # Get effective video settings with profile/CLI overrides applied
        video_settings_dict = self._get_effective_video_settings()
        video_settings = self.config.video_settings
        video_files = [path for path in visual_inputs if self._is_video(path)]
        image_files = [path for path in visual_inputs if not self._is_video(path)]
        video_durations = await asyncio.gather(
            *[self._get_media_duration(p) for p in video_files]
        )
        total_video_clip_duration = sum(video_durations)

        timed_visuals: list[tuple[Path, float, bool]] = []
        for path, duration in zip(video_files, video_durations, strict=False):
            timed_visuals.append((path, duration, True))

        if image_files:
            num_visuals_total = len(visual_inputs)
            if num_visuals_total > 1:
                num_transitions = num_visuals_total - 1
                transition_duration = video_settings.transition_duration_sec
                total_gross_image_duration = (
                    total_video_duration
                    - total_video_clip_duration
                    + (num_transitions * transition_duration)
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

        if not timed_visuals:
            raise ValueError("No visual media could be prepared for the timeline.")

        input_cmd_parts: list[str] = []
        filter_parts: list[str] = []
        stream_labels: list[str] = []
        geometries: list[VisualGeometry] = []
        width, height = video_settings.resolution
        pix_fmt = video_settings.output_pixel_format
        all_visuals_dims = await asyncio.gather(
            *[self._get_media_dimensions(p) for p, _, _ in timed_visuals]
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

        for i, (path, duration, is_video) in enumerate(timed_visuals):
            if is_video:
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
            scaled_w_base = int(width * video_settings_dict["image_width_percent"])
            orig_w, orig_h = all_visuals_dims[i]

            scaled_w, scaled_h = 0, 0
            target_y_pos = video_settings_dict["image_top_position_percent"] * height
            max_available_height = height - target_y_pos

            if not is_relative_mode and uniform_height > 0:
                scaled_h = uniform_height
                scaled_w = (
                    int(scaled_h * (orig_w / orig_h)) if orig_h > 0 else scaled_w_base
                )
                vf_scale = f"scale={scaled_w}:{scaled_h}"
            else:
                scaled_w = scaled_w_base
                scaled_h = int(scaled_w * (orig_h / orig_w)) if orig_w > 0 else -1

                # Ensure scaled height doesn't exceed available space in frame
                if scaled_h > max_available_height:
                    scaled_h = int(max_available_height)
                    scaled_w = (
                        int(scaled_h * (orig_w / orig_h))
                        if orig_h > 0
                        else scaled_w_base
                    )

                vf_scale = f"scale={scaled_w}:{scaled_h}"

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
                f"pad={width}:{height}:(ow-iw)/2:{target_y_pos}:color={video_settings.pad_color},"
                f"format={pix_fmt}[v_temp_{i}];"
                f"[v_temp_{i}]trim=duration={duration},setpts=PTS-STARTPTS{proc_label}"
            )
            filter_parts.append(vf_string)
            stream_labels.append(proc_label)

        if len(stream_labels) > 1:
            transition_duration = video_settings.transition_duration_sec
            current_stream = stream_labels[0]
            current_offset = timed_visuals[0][1] - transition_duration
            for i in range(1, len(stream_labels)):
                next_stream = stream_labels[i]
                output_stream_label = f"[v_chain_{i}]"
                filter_parts.append(
                    f"{current_stream}{next_stream}xfade=transition=fade"
                    f":duration={transition_duration}:offset={current_offset:.4f}{output_stream_label}"
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
        )

    async def _build_subtitle_graph(
        self,
        visual_inputs: list[Path],
        total_video_duration: float,
        subtitle_path: Path | None,
        temp_sub_dir: Path,
    ) -> tuple[list[str], list[str]]:
        settings_dict = self._get_effective_subtitle_settings()

        # Import and use UnifiedSubtitleConfig for proper validation
        from src.video.subtitle_positioning import UnifiedSubtitleConfig

        try:
            unified_config = UnifiedSubtitleConfig(**settings_dict)
            use_content_aware = unified_config.content_aware
        except Exception as e:
            logger.warning(f"Failed to parse subtitle settings, using fallback: {e}")
            use_content_aware = settings_dict.get("content_aware", True)

        (
            video_filters,
            input_cmd_parts,
            timed_visuals,
            final_visual_stream,
            geometries,
        ) = await self._build_visual_chain(
            visual_inputs, total_video_duration, use_content_aware
        )

        # Check if subtitles are enabled (from settings_dict, not unified_config)
        subtitles_enabled = settings_dict.get("enabled", True)

        if not subtitle_path or not subtitles_enabled:
            video_filters.append(f"{final_visual_stream}copy[v_out]")
            return video_filters, input_cmd_parts

        # Check if this is an ASS file or SRT file
        if subtitle_path.suffix.lower() == ".ass":
            # For content-aware positioning, regenerate ASS file with visual bounds
            if use_content_aware and geometries:
                if self.debug_mode:
                    logger.debug(
                        "Regenerating ASS file with visual bounds for content-aware "
                        "positioning"
                    )
                    logger.debug(f"Visual geometries available: {len(geometries)}")

                content_aware_ass_path = await self._create_content_aware_ass_file(
                    subtitle_path, geometries, timed_visuals, temp_sub_dir
                )
                if content_aware_ass_path:
                    ass_path = content_aware_ass_path.as_posix().replace(":", r"\:")
                    if self.debug_mode:
                        logger.debug(
                            f"Using content-aware ASS file: {content_aware_ass_path}"
                        )
                else:
                    logger.warning(
                        "Failed to create content-aware ASS file, "
                        "falling back to original"
                    )
                    ass_path = subtitle_path.as_posix().replace(":", r"\:")
            else:
                # Use original ASS file positioning
                ass_path = subtitle_path.as_posix().replace(":", r"\:")

            video_filters.append(f"{final_visual_stream}ass='{ass_path}'[v_out]")
            return video_filters, input_cmd_parts

        # For SRT files, use the existing drawtext-based approach
        sub_entries = self._parse_srt(subtitle_path)
        current_video_stream = final_visual_stream

        segment_end_times = []
        cumulative_time = 0.0
        transition_duration = self.config.video_settings.transition_duration_sec
        for i, (_, duration, _) in enumerate(timed_visuals):
            effective_duration = duration - (transition_duration if i > 0 else 0)
            cumulative_time += effective_duration
            segment_end_times.append(cumulative_time)

        # Get style configuration with randomization support
        from src.video.subtitle_positioning import StylePreset, get_style_config

        try:
            # Get the unified config for style processing
            unified_config = UnifiedSubtitleConfig(**settings_dict)
            if self.debug_mode:
                logger.debug(
                    f"UnifiedSubtitleConfig randomization flags - "
                    f"fonts: {unified_config.randomize_fonts}, "
                    f"colors: {unified_config.randomize_colors}"
                )

            # Determine style preset
            style_preset = unified_config.style_preset

            # Get style configuration with randomization
            style_config = get_style_config(
                preset=style_preset, config=unified_config, product_id=self.product_id
            )
            if self.debug_mode:
                logger.debug(
                    f"Style config result - font: {style_config.get('font_name')}, "
                    f"color: {style_config.get('font_color')}"
                )

            # Extract styling parameters
            font_name = style_config.get("font_name", "Arial")
            font_color = style_config.get("font_color", "&H00FFFFFF")
            outline_color = style_config.get("outline_color", "&H00000000")

        except Exception as e:
            logger.warning(f"Failed to get style config, using fallback: {e}")
            # Fallback to legacy behavior
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

                    font_size_pixels = self.config.video_settings.resolution[
                        1
                    ] * settings_dict.get("font_size_percent", 0.04)
                    avg_char_width = font_size_pixels * settings_dict.get(
                        "font_width_to_height_ratio", 0.5
                    )
                    max_chars_per_line = (
                        int(geom.rendered_w / avg_char_width)
                        if avg_char_width > 0
                        else self.config.video_settings.default_max_chars_per_line
                    )

                    wrapper = textwrap.TextWrapper(
                        width=max_chars_per_line,
                        break_long_words=True,
                        replace_whitespace=False,
                    )
                    wrapped_text = "\n".join(wrapper.wrap(sub.text))

                    sub_text_file = temp_sub_dir / f"sub_text_{drawtext_count}.txt"
                    sub_text_file.write_text(wrapped_text, encoding="utf-8")

                    # Use unified positioning system for consistent results
                    from src.video.subtitle_positioning import (
                        VisualBounds,
                        calculate_position,
                        create_unified_config_from_settings,
                    )

                    # Create unified config from settings
                    unified_config = create_unified_config_from_settings(settings_dict)

                    # Create visual bounds with error handling
                    visual_bounds = None
                    try:
                        if unified_config.content_aware and geom:
                            # Validate geometry dimensions
                            (
                                frame_width,
                                frame_height,
                            ) = self.config.video_settings.resolution
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
                            else:
                                logger.warning(
                                    f"Invalid visual geometry for segment "
                                    f"{drawtext_count}: "
                                    f"x={geom.rendered_x}, y={geom.rendered_y}, "
                                    f"w={geom.rendered_w}, h={geom.rendered_h}, "
                                    f"frame={frame_width}x{frame_height}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create visual bounds for segment "
                            f"{drawtext_count}: {e}"
                        )
                        visual_bounds = None

                    # Debug visual bounds
                    if visual_bounds and self.debug_mode:
                        logger.debug(
                            f"Visual bounds for segment {drawtext_count}: "
                            f"x={visual_bounds.x:.3f}, y={visual_bounds.y:.3f}, "
                            f"w={visual_bounds.width:.3f}, h={visual_bounds.height:.3f}"
                        )
                        logger.debug(
                            f"Geometry pixels: x={geom.rendered_x}, "
                            f"y={geom.rendered_y}, "
                            f"w={geom.rendered_w}, h={geom.rendered_h}"
                        )

                    # Calculate position using unified system with fallback
                    try:
                        position = calculate_position(
                            unified_config,
                            self.config.video_settings.resolution,
                            visual_bounds,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Position calculation failed for segment "
                            f"{drawtext_count}, using fallback: {e}"
                        )
                        # Fallback to bottom positioning using config values
                        from pathlib import Path

                        import yaml

                        from src.video.subtitle_positioning import Position

                        center_x = 0.5  # Default
                        fallback_y = 0.8  # Default
                        config_path = Path("config/subtitles.yaml")
                        if config_path.exists():
                            with open(config_path, encoding="utf-8") as f:
                                data = yaml.safe_load(f)
                                text_rendering = data.get("text_rendering", {})
                                center_x = text_rendering.get(
                                    "center_position_fraction", 0.5
                                )
                                # Use 80% down as fallback when no visual bounds
                                fallback_y = 0.8

                        position = Position(x=center_x, y=fallback_y)

                    # Debug positioning
                    if self.debug_mode:
                        pixel_x = int(
                            position.x * self.config.video_settings.resolution[0]
                        )
                        pixel_y = int(
                            position.y * self.config.video_settings.resolution[1]
                        )
                        logger.debug(
                            f"Calculated position for segment {drawtext_count}: "
                            f"({position.x:.3f}, {position.y:.3f}) = "
                            f"({pixel_x}, {pixel_y}) pixels"
                        )

                    # Convert to FFmpeg expressions
                    x_pos_expr = f"w*{position.x} - text_w/2"
                    y_pos_expr = f"h*{position.y}"

                    output_stream = f"[v_sub_{drawtext_count+1}]"
                    font_path_escaped = font_path.as_posix().replace(":", r"\:")
                    sub_text_escaped = sub_text_file.as_posix().replace(":", r"\:")
                    drawtext_filter = (
                        f"{current_video_stream}drawtext="
                        f"fontfile='{font_path_escaped}':"
                        f"textfile='{sub_text_escaped}':"
                        f"fontsize={font_size_pixels}:"
                        f"fontcolor='{self._convert_ass_color_to_ffmpeg(font_color)}':"
                        f"borderw={settings_dict.get('outline_thickness', 2)}:"
                        f"bordercolor='{self._convert_ass_color_to_ffmpeg(outline_color)}':"
                        f"box=1:boxcolor='"
                        f"{self._convert_ass_color_to_ffmpeg(settings_dict.get('back_color', '&H80000000'))}"  # noqa: E501
                        f"':boxborderw={self.config.video_settings.subtitle_box_border_width}:"
                        f"x='{x_pos_expr}':y='{y_pos_expr}':"
                        f"enable='between(t,{overlap_start},{overlap_end})'"
                        f"{output_stream}"
                    )
                    video_filters.append(drawtext_filter)
                    current_video_stream = output_stream
                    drawtext_count += 1

        video_filters.append(f"{current_video_stream}copy[v_out]")
        return video_filters, input_cmd_parts

    async def _build_dual_subtitle_graph(
        self,
        visual_inputs: list[Path],
        total_video_duration: float,
        subtitle_lower_path: Path | None,
        subtitle_upper_path: Path,
        temp_sub_dir: Path,
    ) -> tuple[list[str], list[str]]:
        """Build video processing graph with dual independent subtitle lines.

        Args:
        ----
            visual_inputs: List of visual input file paths
            total_video_duration: Target video duration in seconds
            subtitle_lower_path: Path to lower subtitle file (voiceover subtitles)
            subtitle_upper_path: Path to upper subtitle file (product info)
            temp_sub_dir: Temporary directory for processing

        Returns:
        -------
            Tuple of (video_filters, input_cmd_parts)

        """
        settings_dict = self._get_effective_subtitle_settings()

        # Import and use UnifiedSubtitleConfig for proper validation
        from src.video.subtitle_positioning import UnifiedSubtitleConfig

        try:
            unified_config = UnifiedSubtitleConfig(**settings_dict)
            use_content_aware = unified_config.content_aware
        except Exception as e:
            logger.warning(f"Failed to parse subtitle settings, using fallback: {e}")
            use_content_aware = settings_dict.get("content_aware", True)

        # Build visual chain
        (
            video_filters,
            input_cmd_parts,
            timed_visuals,
            final_visual_stream,
            geometries,
        ) = await self._build_visual_chain(
            visual_inputs, total_video_duration, use_content_aware
        )

        current_stream = final_visual_stream

        # First, apply upper subtitle (static product info)
        if subtitle_upper_path.suffix.lower() == ".ass":
            # For ASS format, apply using ass filter
            ass_path_upper = subtitle_upper_path.as_posix().replace(":", r"\:")
            video_filters.append(f"{current_stream}ass='{ass_path_upper}'[v_upper]")
            current_stream = "[v_upper]"
        else:
            # For SRT format upper line, use drawtext (static display)
            # Parse SRT to get the text (should be single static entry)
            sub_entries_upper = self._parse_srt(subtitle_upper_path)
            if sub_entries_upper:
                upper_text = sub_entries_upper[0].text  # Get first entry text

                # Get upper line styling from two_part config
                two_part_config = settings_dict.get("two_part_subtitles", {})
                upper_config = two_part_config.get("upper_line", {})

                # Get style configuration
                from src.video.subtitle_positioning import get_style_config

                # Use upper line's style preset if specified
                style_preset = upper_config.get("style_preset", "minimal")
                upper_settings = settings_dict.copy()
                upper_settings["style_preset"] = style_preset
                upper_settings["anchor"] = upper_config.get("anchor", "above_content")
                upper_settings["margin"] = upper_config.get("margin", 0.03)
                upper_settings["font_size_scale"] = upper_config.get(
                    "font_size_scale", 0.8
                )

                try:
                    upper_unified_config = UnifiedSubtitleConfig(**upper_settings)
                    style_config = get_style_config(
                        preset=style_preset,
                        config=upper_unified_config,
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
                    # Create temp text file for upper line
                    upper_text_file = temp_sub_dir / "upper_subtitle.txt"
                    upper_text_file.write_text(upper_text, encoding="utf-8")

                    # Calculate position for upper line
                    from src.video.subtitle_positioning import (
                        VisualBounds,
                        calculate_position,
                    )

                    # Use first geometry for positioning reference
                    geom = geometries[0] if geometries else None
                    visual_bounds = None

                    if upper_unified_config.content_aware and geom:
                        frame_width, frame_height = (
                            self.config.video_settings.resolution
                        )
                        visual_bounds = VisualBounds(
                            x=geom.rendered_x / frame_width,
                            y=geom.rendered_y / frame_height,
                            width=geom.rendered_w / frame_width,
                            height=geom.rendered_h / frame_height,
                        )

                    position = calculate_position(
                        upper_unified_config,
                        self.config.video_settings.resolution,
                        visual_bounds,
                    )

                    # Font size with scale factor
                    base_font_size = self.config.video_settings.resolution[
                        1
                    ] * settings_dict.get("font_size_percent", 0.04)
                    upper_font_size = base_font_size * upper_settings.get(
                        "font_size_scale", 0.8
                    )

                    # Convert to FFmpeg expressions
                    x_pos_expr = f"w*{position.x} - text_w/2"
                    y_pos_expr = f"h*{position.y}"

                    # Apply static upper subtitle (visible throughout video)
                    drawtext_filter_upper = (
                        f"{current_stream}drawtext="
                        f"fontfile='{font_path.as_posix().replace(':', r'\:')}':"
                        f"textfile='{upper_text_file.as_posix().replace(':', r'\:')}':"
                        f"fontsize={upper_font_size}:"
                        f"fontcolor='{self._convert_ass_color_to_ffmpeg(font_color)}':"
                        f"borderw={upper_settings.get('outline_thickness', 1)}:"
                        f"bordercolor='{self._convert_ass_color_to_ffmpeg(outline_color)}':"
                        f"x='{x_pos_expr}':y='{y_pos_expr}'"
                        f"[v_upper]"
                    )
                    video_filters.append(drawtext_filter_upper)
                    current_stream = "[v_upper]"

        # Second, apply lower subtitle (timed voiceover subtitles) if provided
        if subtitle_lower_path and subtitle_lower_path.exists():
            if subtitle_lower_path.suffix.lower() == ".ass":
                # For content-aware positioning, regenerate ASS file with visual bounds
                if use_content_aware and geometries:
                    content_aware_ass_path = await self._create_content_aware_ass_file(
                        subtitle_lower_path, geometries, timed_visuals, temp_sub_dir
                    )
                    if content_aware_ass_path:
                        ass_path_lower = content_aware_ass_path.as_posix().replace(
                            ":", r"\:"
                        )
                    else:
                        ass_path_lower = subtitle_lower_path.as_posix().replace(
                            ":", r"\:"
                        )
                else:
                    ass_path_lower = subtitle_lower_path.as_posix().replace(":", r"\:")

                video_filters.append(f"{current_stream}ass='{ass_path_lower}'[v_out]")
            else:
                # For SRT lower line, use the standard drawtext approach
                # from _build_subtitle_graph
                # This is timed subtitle generation - reuse logic but
                # start from current_stream
                sub_entries_lower = self._parse_srt(subtitle_lower_path)

                segment_end_times = []
                cumulative_time = 0.0
                transition_duration = self.config.video_settings.transition_duration_sec
                for i, (_, duration, _) in enumerate(timed_visuals):
                    effective_duration = duration - (
                        transition_duration if i > 0 else 0
                    )
                    cumulative_time += effective_duration
                    segment_end_times.append(cumulative_time)

                # Get lower line styling (uses standard subtitle settings by default)
                two_part_config = settings_dict.get("two_part_subtitles", {})
                lower_config = two_part_config.get("lower_line", {})
                lower_settings = settings_dict.copy()
                lower_settings["anchor"] = lower_config.get("anchor", "below_content")
                lower_settings["margin"] = lower_config.get("margin", 0.05)

                # Get style configuration for lower line
                from src.video.subtitle_positioning import get_style_config

                try:
                    lower_unified_config = UnifiedSubtitleConfig(**lower_settings)
                    style_config = get_style_config(
                        preset=lower_unified_config.style_preset,
                        config=lower_unified_config,
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

                                font_size_pixels = (
                                    self.config.video_settings.resolution[1]
                                    * lower_settings.get("font_size_percent", 0.04)
                                )
                                avg_char_width = font_size_pixels * lower_settings.get(
                                    "font_width_to_height_ratio", 0.5
                                )
                                max_chars_per_line = (
                                    int(geom.rendered_w / avg_char_width)
                                    if avg_char_width > 0
                                    else (
                                        self.config.video_settings.default_max_chars_per_line
                                    )
                                )

                                wrapper = textwrap.TextWrapper(
                                    width=max_chars_per_line,
                                    break_long_words=True,
                                    replace_whitespace=False,
                                )
                                wrapped_text = "\n".join(wrapper.wrap(sub.text))

                                sub_text_file = (
                                    temp_sub_dir / f"lower_text_{drawtext_count}.txt"
                                )
                                sub_text_file.write_text(wrapped_text, encoding="utf-8")

                                # Calculate position for lower line
                                from src.video.subtitle_positioning import (
                                    VisualBounds,
                                    calculate_position,
                                )

                                visual_bounds = None
                                if lower_unified_config.content_aware and geom:
                                    (
                                        frame_width,
                                        frame_height,
                                    ) = self.config.video_settings.resolution
                                    visual_bounds = VisualBounds(
                                        x=geom.rendered_x / frame_width,
                                        y=geom.rendered_y / frame_height,
                                        width=geom.rendered_w / frame_width,
                                        height=geom.rendered_h / frame_height,
                                    )

                                position = calculate_position(
                                    lower_unified_config,
                                    self.config.video_settings.resolution,
                                    visual_bounds,
                                )

                                # Convert to FFmpeg expressions
                                x_pos_expr = f"w*{position.x} - text_w/2"
                                y_pos_expr = f"h*{position.y}"

                                output_stream = f"[v_lower_{drawtext_count+1}]"
                                font_path_escaped = font_path.as_posix().replace(
                                    ":", r"\:"
                                )
                                sub_text_escaped = sub_text_file.as_posix().replace(
                                    ":", r"\:"
                                )
                                drawtext_filter = (
                                    f"{current_stream}drawtext="
                                    f"fontfile='{font_path_escaped}':"
                                    f"textfile='{sub_text_escaped}':"
                                    f"fontsize={font_size_pixels}:"
                                    f"fontcolor='{self._convert_ass_color_to_ffmpeg(font_color)}':"
                                    f"borderw="
                                    f"{lower_settings.get('outline_thickness', 2)}:"
                                    f"bordercolor='{self._convert_ass_color_to_ffmpeg(outline_color)}':"
                                    f"box=1:boxcolor='"
                                    f"{self._convert_ass_color_to_ffmpeg(lower_settings.get('back_color', '&H80000000'))}"  # noqa: E501
                                    f"':boxborderw="
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
            # Only upper subtitle, no lower subtitle
            video_filters.append(f"{current_stream}copy[v_out]")

        return video_filters, input_cmd_parts

    async def _create_content_aware_ass_file(
        self,
        original_ass_path: Path,
        geometries: list[VisualGeometry],
        timed_visuals: list[tuple[Path, float, bool]],
        temp_dir: Path,
    ) -> Path | None:
        """Create a new ASS file with content-aware positioning based on image geometry.

        Args:
        ----
            original_ass_path: Path to the original ASS file
            geometries: List of visual geometries for each timeline segment
            timed_visuals: List of visual timeline data
            temp_dir: Temporary directory for content-aware ASS file

        Returns:
        -------
            Path to the new content-aware ASS file, or None if generation fails

        """
        try:
            logger.info(
                "Creating content-aware ASS file with image-relative positioning"
            )

            # Read original ASS file
            with open(original_ass_path, encoding="utf-8") as f:
                original_content = f.read()

            # Parse ASS content
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
                    header_lines.append(line)  # Format line in Events section
                else:
                    header_lines.append(line)

            if not events_lines:
                logger.warning("No dialogue events found in ASS file")
                return None

            # Calculate timeline segment boundaries
            segment_end_times = []
            cumulative_time = 0.0
            transition_duration = self.config.video_settings.transition_duration_sec

            for i, (_, duration, _) in enumerate(timed_visuals):
                effective_duration = duration - (transition_duration if i > 0 else 0)
                cumulative_time += effective_duration
                segment_end_times.append(cumulative_time)

            # Process each dialogue line for content-aware positioning
            content_aware_events = []
            settings_dict = self._get_effective_subtitle_settings()

            # Import and use UnifiedSubtitleConfig for proper validation
            from src.video.subtitle_positioning import (
                PositionAnchor,
                UnifiedSubtitleConfig,
            )

            try:
                unified_config = UnifiedSubtitleConfig(**settings_dict)
            except Exception as e:
                logger.warning(f"Failed to parse unified subtitle config: {e}")
                return original_ass_path

            # Check if content-aware positioning is enabled and anchor is below_content
            if (
                not unified_config.content_aware
                or unified_config.anchor != PositionAnchor.BELOW_CONTENT
            ):
                logger.warning(
                    f"Content-aware positioning not enabled or "
                    f"anchor not below_content "
                    f"(content_aware={unified_config.content_aware}, "
                    f"anchor={unified_config.anchor}), using original file"
                )
                return original_ass_path

            for event_line in events_lines:
                # Parse ASS dialogue line
                parts = event_line.split(",", 9)  # Split into 10 parts max
                if len(parts) < 10:
                    content_aware_events.append(event_line)  # Keep malformed as-is
                    continue

                # Extract timing
                start_time = self._parse_ass_time(parts[1])
                self._parse_ass_time(parts[2])

                # Find which visual segment this subtitle belongs to
                segment_idx = 0
                for i, segment_end_time in enumerate(segment_end_times):
                    if start_time <= segment_end_time:
                        segment_idx = i
                        break

                # Calculate content-aware position based on image geometry
                if segment_idx < len(geometries):
                    geom = geometries[segment_idx]

                    # Calculate subtitle position relative to image using unified config
                    frame_height = self.config.video_settings.resolution[1]
                    image_bottom = geom.rendered_y + geom.rendered_h

                    # Use margin from unified config
                    # (margin is as fraction of frame height)
                    spacing_px = unified_config.margin * frame_height

                    logger.debug(
                        f"Content-aware positioning: image_bottom={image_bottom}px, "
                        f"margin={unified_config.margin}, spacing={spacing_px}px"
                    )

                    subtitle_y = int(image_bottom + spacing_px)

                    # Ensure subtitle doesn't go off-screen
                    # (leave room for subtitle height)
                    # Get font size from unified config
                    from src.video.subtitle_positioning import get_font_size

                    font_size = get_font_size(unified_config, frame_height)

                    # Allow subtitles to go up to max safe position from config
                    from pathlib import Path

                    import yaml

                    max_safe_y = 0.95  # Default
                    config_path = Path("config/subtitles.yaml")
                    if config_path.exists():
                        with open(config_path, encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                            text_rendering = data.get("text_rendering", {})
                            max_safe_y = text_rendering.get("max_safe_y_position", 0.95)

                    max_y = int(frame_height * max_safe_y)
                    subtitle_y = min(subtitle_y, max_y)

                    logger.debug(
                        f"Subtitle positioned at y={subtitle_y} "
                        f"(image_bottom={image_bottom}, spacing={spacing_px}px, "
                        f"font_size={font_size}px)"
                    )

                    # Create positioning override using ASS \pos tag
                    text_content = parts[9]  # Original text with any existing effects

                    # Remove any existing \pos tags
                    import re

                    text_content = re.sub(r"\\pos\([^)]+\)", "", text_content)

                    # Add new positioning - place \pos tag at START of effect block
                    # for ASS compatibility
                    subtitle_x = (
                        geom.rendered_x + geom.rendered_w // 2
                    )  # Center horizontally on image

                    # Check if there are existing effect tags at the start
                    if text_content.startswith("{") and "}" in text_content:
                        # Find the end of the first effect block
                        effect_end = text_content.find("}") + 1
                        # Extract content without braces
                        effect_content = text_content[1 : effect_end - 1]
                        after_effects = text_content[effect_end:]

                        # Check if there's already a \move tag - if so, don't add \pos
                        # because \pos overrides \move in ASS rendering
                        if r"\move(" in effect_content:
                            # Keep existing effects including \move, don't add \pos
                            positioned_text = text_content
                        else:
                            # Place positioning at the start of the effect block
                            # for better ASS compatibility
                            positioned_text = (
                                f"{{\\pos({subtitle_x},{subtitle_y}){effect_content}}}"
                                f"{after_effects}"
                            )
                    else:
                        # No existing effects, add positioning normally
                        positioned_text = (
                            f"{{\\pos({subtitle_x},{subtitle_y})}}{text_content}"
                        )

                    # Reconstruct dialogue line with new positioning
                    new_parts = parts[:9] + [positioned_text]
                    content_aware_events.append(",".join(new_parts))

                    logger.debug(
                        f"Segment {segment_idx}: Positioned subtitle at "
                        f"({subtitle_x}, {subtitle_y}) "
                        f"for image at ({geom.rendered_x}, {geom.rendered_y}, "
                        f"{geom.rendered_w}x{geom.rendered_h})"
                    )
                else:
                    # Fallback: use original positioning
                    content_aware_events.append(event_line)

            # Write content-aware ASS file to permanent output directory instead of temp
            # directory
            # Use the output directory (parent of the subtitle file) for permanent
            # storage
            output_dir = original_ass_path.parent
            content_aware_ass_path = output_dir / "subtitles_content_aware.ass"

            with open(content_aware_ass_path, "w", encoding="utf-8") as f:
                # Write header
                for line in header_lines:
                    f.write(line + "\n")

                # Write content-aware events
                for event_line in content_aware_events:
                    f.write(event_line + "\n")

            logger.info(f"Created content-aware ASS file: {content_aware_ass_path}")
            return content_aware_ass_path

        except Exception as e:
            logger.error(f"Failed to create content-aware ASS file: {e}")
            return None

    def _parse_ass_time(self, time_str: str) -> float:
        """Parse ASS time format (H:MM:SS.CC) to seconds."""
        try:
            # Format: H:MM:SS.CC (e.g., "0:00:01.72")
            parts = time_str.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds_parts = parts[2].split(".")
            seconds = int(seconds_parts[0])
            centiseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0

            return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse ASS time '{time_str}': {e}")
            return 0.0
