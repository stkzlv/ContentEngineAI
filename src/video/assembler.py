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
        font_dir = Path(self.config.subtitle_settings.font_directory)
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
        if opacity >= 0.99:
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
                if (
                    similarity
                    < self.config.subtitle_settings.subtitle_similarity_threshold
                ):
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

        """
        try:
            return (
                self.config.debug_settings.get("create_ffmpeg_command_logs", True)
                if hasattr(self.config, "debug_settings") and self.config.debug_settings
                else True
            )
        except Exception:
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
    ) -> Path | None:
        """Assemble final video from visual inputs, audio, and subtitles.

        Args:
        ----
            visual_inputs: List of visual input file paths
            voiceover_audio_path: Optional voiceover audio file
            music_track_path: Optional background music file
            output_path: Output video file path
            subtitle_path: Optional subtitle file path
            total_video_duration: Target video duration in seconds
            temp_dir: Temporary directory for processing
            debug_mode: Enable debug output

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
            # Build video processing chain with subtitles
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
                output_path.parent / f"{output_path.stem}_ffmpeg_command.log"
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
        is_dynamic_mode: bool,
    ) -> tuple[
        list[str], list[str], list[tuple[Path, float, bool]], str, list[VisualGeometry]
    ]:
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
        if not is_dynamic_mode:
            scaled_heights = []
            for orig_w, orig_h in all_visuals_dims:
                if orig_w > 0 and orig_h > 0:
                    scaled_h = int(
                        (width * video_settings.image_width_percent) * (orig_h / orig_w)
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
            scaled_w_base = int(width * video_settings.image_width_percent)
            orig_w, orig_h = all_visuals_dims[i]

            scaled_w, scaled_h = 0, 0
            if not is_dynamic_mode and uniform_height > 0:
                scaled_h = uniform_height
                scaled_w = (
                    int(scaled_h * (orig_w / orig_h)) if orig_h > 0 else scaled_w_base
                )
                vf_scale = f"scale={scaled_w}:{scaled_h}"
            else:
                scaled_w = scaled_w_base
                scaled_h = int(scaled_w * (orig_h / orig_w)) if orig_w > 0 else -1
                vf_scale = f"scale={scaled_w}:{scaled_h}"

            target_y_pos = video_settings.image_top_position_percent * height

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
        settings = self.config.subtitle_settings
        # Use unified positioning system - no need for mode-specific logic
        use_content_aware = getattr(settings, "content_aware", True)

        (
            video_filters,
            input_cmd_parts,
            timed_visuals,
            final_visual_stream,
            geometries,
        ) = await self._build_visual_chain(
            visual_inputs, total_video_duration, use_content_aware
        )

        if not subtitle_path or not settings.enabled:
            video_filters.append(f"{final_visual_stream}copy[v_out]")
            return video_filters, input_cmd_parts

        # Check if this is an ASS file or SRT file
        if subtitle_path.suffix.lower() == ".ass":
            # For content-aware positioning, regenerate ASS file with visual bounds
            if use_content_aware and geometries:
                if self.debug_mode:
                    logger.debug(
                        "Regenerating ASS file with visual bounds for content-aware positioning"
                    )
                    logger.debug(f"Visual geometries available: {len(geometries)}")

                dynamic_ass_path = await self._create_dynamic_ass_file(
                    subtitle_path, geometries, timed_visuals, temp_sub_dir
                )
                if dynamic_ass_path:
                    ass_path = dynamic_ass_path.as_posix().replace(":", r"\:")
                    if self.debug_mode:
                        logger.debug(f"Using dynamic ASS file: {dynamic_ass_path}")
                else:
                    logger.warning(
                        "Failed to create dynamic ASS file, falling back to original"
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

        font_name = settings.font_name
        if settings.use_random_font and settings.available_fonts:
            font_name = secrets.choice(settings.available_fonts)
        font_path = self._resolve_font_path(font_name)
        if not font_path:
            logger.warning(f"Could not resolve font path for '{font_name}'")
            video_filters.append(f"{final_visual_stream}copy[v_out]")
            return video_filters, input_cmd_parts

        font_color, outline_color = settings.font_color, settings.outline_color
        if settings.use_random_colors and settings.available_color_combinations:
            font_color, outline_color = secrets.choice(
                settings.available_color_combinations
            )

        drawtext_count = 0
        for sub in sub_entries:
            sub_start, sub_end = sub.start, sub.end
            for i, end_time in enumerate(segment_end_times):
                start_time = segment_end_times[i - 1] if i > 0 else 0
                overlap_start = max(sub_start, start_time)
                overlap_end = min(sub_end, end_time)

                if overlap_start < overlap_end:
                    geom = geometries[i]

                    font_size_pixels = (
                        self.config.video_settings.resolution[1]
                        * settings.font_size_percent
                    )
                    avg_char_width = (
                        font_size_pixels * settings.font_width_to_height_ratio
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
                        convert_legacy_config,
                    )

                    # Convert legacy settings to unified config
                    unified_config = convert_legacy_config(settings.__dict__)

                    # Create visual bounds for content-aware positioning
                    visual_bounds = (
                        VisualBounds(
                            x=geom.rendered_x
                            / self.config.video_settings.resolution[0],
                            y=geom.rendered_y
                            / self.config.video_settings.resolution[1],
                            width=geom.rendered_w
                            / self.config.video_settings.resolution[0],
                            height=geom.rendered_h
                            / self.config.video_settings.resolution[1],
                        )
                        if unified_config.content_aware
                        else None
                    )

                    # Debug visual bounds
                    if visual_bounds and self.debug_mode:
                        logger.debug(
                            f"Visual bounds for segment {drawtext_count}: x={visual_bounds.x:.3f}, y={visual_bounds.y:.3f}, w={visual_bounds.width:.3f}, h={visual_bounds.height:.3f}"
                        )
                        logger.debug(
                            f"Geometry pixels: x={geom.rendered_x}, y={geom.rendered_y}, w={geom.rendered_w}, h={geom.rendered_h}"
                        )

                    # Calculate position using unified system
                    position = calculate_position(
                        unified_config,
                        self.config.video_settings.resolution,
                        visual_bounds,
                    )

                    # Debug positioning
                    if self.debug_mode:
                        pixel_x = int(
                            position.x * self.config.video_settings.resolution[0]
                        )
                        pixel_y = int(
                            position.y * self.config.video_settings.resolution[1]
                        )
                        logger.debug(
                            f"Calculated position for segment {drawtext_count}: ({position.x:.3f}, {position.y:.3f}) = ({pixel_x}, {pixel_y}) pixels"
                        )

                    # Convert to FFmpeg expressions
                    x_pos_expr = f"w*{position.x} - text_w/2"
                    y_pos_expr = f"h*{position.y}"

                    output_stream = f"[v_sub_{drawtext_count+1}]"
                    drawtext_filter = (
                        f"{current_video_stream}drawtext="
                        f"fontfile='{font_path.as_posix().replace(':', r'\:')}':"
                        f"textfile='{sub_text_file.as_posix().replace(':', r'\:')}':"
                        f"fontsize={font_size_pixels}:"
                        f"fontcolor='{self._convert_ass_color_to_ffmpeg(font_color)}':"
                        f"borderw={settings.outline_thickness}:"
                        f"bordercolor='{self._convert_ass_color_to_ffmpeg(outline_color)}':"
                        f"box=1:boxcolor='{self._convert_ass_color_to_ffmpeg(settings.back_color)}':boxborderw={self.config.video_settings.subtitle_box_border_width}:"
                        f"x='{x_pos_expr}':y='{y_pos_expr}':"
                        f"enable='between(t,{overlap_start},{overlap_end})'"
                        f"{output_stream}"
                    )
                    video_filters.append(drawtext_filter)
                    current_video_stream = output_stream
                    drawtext_count += 1

        video_filters.append(f"{current_video_stream}copy[v_out]")
        return video_filters, input_cmd_parts

    async def _create_dynamic_ass_file(
        self,
        original_ass_path: Path,
        geometries: list[VisualGeometry],
        timed_visuals: list[tuple[Path, float, bool]],
        temp_dir: Path,
    ) -> Path | None:
        """Create a new ASS file with dynamic positioning based on image geometry.

        Args:
        ----
            original_ass_path: Path to the original ASS file
            geometries: List of visual geometries for each timeline segment
            timed_visuals: List of visual timeline data
            temp_dir: Temporary directory for dynamic ASS file

        Returns:
        -------
            Path to the new dynamic ASS file, or None if generation fails

        """
        try:
            logger.info("Creating dynamic ASS file with image-relative positioning")

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

            # Process each dialogue line for dynamic positioning
            dynamic_events = []
            settings = self.config.subtitle_settings
            dyn_settings = settings.dynamic_positioning

            for event_line in events_lines:
                # Parse ASS dialogue line
                parts = event_line.split(",", 9)  # Split into 10 parts max
                if len(parts) < 10:
                    dynamic_events.append(event_line)  # Keep malformed lines as-is
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

                # Calculate dynamic position based on image geometry
                if segment_idx < len(geometries):
                    geom = geometries[segment_idx]

                    # Calculate subtitle position relative to image
                    image_bottom = geom.rendered_y + geom.rendered_h
                    spacing = (
                        dyn_settings.image_bottom_to_subtitle_top_spacing_percent
                        * self.config.video_settings.resolution[1]
                    )

                    # Apply closer positioning if enabled
                    subtitle_settings = self.config.subtitle_settings
                    if getattr(subtitle_settings, "ass_closer_to_image", True):
                        reduction_factor = getattr(
                            subtitle_settings, "ass_spacing_reduction_factor", 0.5
                        )
                        spacing *= reduction_factor
                        logger.debug(
                            f"Applied spacing reduction factor {reduction_factor}: {spacing}px"
                        )

                    subtitle_y = int(image_bottom + spacing)

                    # Ensure subtitle doesn't go off-screen (leave room for subtitle height)
                    # Estimate subtitle height based on font size
                    font_size = getattr(subtitle_settings, "ass_font_size", 48)
                    int(font_size * 1.5)  # More accurate height estimate
                    max_y = int(
                        self.config.video_settings.resolution[1] * 0.90
                    )  # Allow subtitles to go lower
                    subtitle_y = min(subtitle_y, max_y)

                    logger.debug(
                        f"Subtitle positioned at y={subtitle_y} (image_bottom={image_bottom}, spacing={spacing}px, font_size={font_size}px)"
                    )

                    # Create positioning override using ASS \pos tag
                    text_content = parts[9]  # Original text with any existing effects

                    # Remove any existing \pos tags
                    import re

                    text_content = re.sub(r"\\pos\([^)]+\)", "", text_content)

                    # Add new positioning - place \pos tag at START of effect block for ASS compatibility
                    subtitle_x = (
                        geom.rendered_x + geom.rendered_w // 2
                    )  # Center horizontally on image

                    # Check if there are existing effect tags at the start
                    if text_content.startswith("{") and "}" in text_content:
                        # Find the end of the first effect block
                        effect_end = text_content.find("}") + 1
                        effect_content = text_content[
                            1 : effect_end - 1
                        ]  # Extract content without braces
                        after_effects = text_content[effect_end:]
                        # Place positioning at the start of the effect block for better ASS compatibility
                        positioned_text = f"{{\\pos({subtitle_x},{subtitle_y}){effect_content}}}{after_effects}"
                    else:
                        # No existing effects, add positioning normally
                        positioned_text = (
                            f"{{\\pos({subtitle_x},{subtitle_y})}}{text_content}"
                        )

                    # Reconstruct dialogue line with new positioning
                    new_parts = parts[:9] + [positioned_text]
                    dynamic_events.append(",".join(new_parts))

                    logger.debug(
                        f"Segment {segment_idx}: Positioned subtitle at ({subtitle_x}, {subtitle_y}) "
                        f"for image at ({geom.rendered_x}, {geom.rendered_y}, {geom.rendered_w}x{geom.rendered_h})"
                    )
                else:
                    # Fallback: use original positioning
                    dynamic_events.append(event_line)

            # Write dynamic ASS file to permanent output directory instead of temp directory
            # Use the output directory (parent of the subtitle file) for permanent storage
            output_dir = original_ass_path.parent
            dynamic_ass_path = output_dir / "subtitles_dynamic.ass"

            with open(dynamic_ass_path, "w", encoding="utf-8") as f:
                # Write header
                for line in header_lines:
                    f.write(line + "\n")

                # Write dynamic events
                for event_line in dynamic_events:
                    f.write(event_line + "\n")

            logger.info(f"Created dynamic ASS file: {dynamic_ass_path}")
            return dynamic_ass_path

        except Exception as e:
            logger.error(f"Failed to create dynamic ASS file: {e}")
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
