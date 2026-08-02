"""Slim VideoAssembler orchestrator via composition.

This module contains the slim VideoAssembler class that coordinates
specialized builders for video assembly operations. The class has been
reduced from 3,311 lines to ~500 lines by delegating to:

- MediaInspector: media file inspection
- SubtitleParser/SubtitleStyler: subtitle utilities
- AudioFilterBuilder: audio filter chains
- VideoStrategyFactory: video mode strategies
- VisualFilterBuilder: visual filter chains
- SubtitleGraphBuilder: subtitle positioning
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.utils import ensure_dirs_exist
from src.utils.async_io import (
    async_run_ffmpeg,
    ffmpeg_semaphore,
)
from src.video.assembler.audio_builder import AudioFilterBuilder
from src.video.assembler.media_inspector import MediaInspector
from src.video.assembler.overlay_builder import (
    apply_disclosure_overlay,
    apply_hook_overlay,
    resolve_hook_line,
)
from src.video.assembler.subtitle_builder import SubtitleGraphBuilder
from src.video.assembler.subtitle_utils import SubtitleStyler
from src.video.assembler.video_strategies import VideoStrategyFactory
from src.video.assembler.visual_builder import VisualFilterBuilder
from src.video.config import VideoConfig
from src.video.config.visual_models import MergedProfileSettings

logger = logging.getLogger(__name__)


class VideoAssembler:
    """Assembles final videos from various media components using FFmpeg.

    This class is responsible for combining visual media (images/videos),
    audio (voiceover and background music), and subtitles into a cohesive
    final video. It coordinates specialized builder classes for the actual
    filter chain construction.

    The assembler manages complex operations like:
    - Scaling and positioning visuals (via VisualFilterBuilder)
    - Creating transitions between media elements
    - Mixing multiple audio tracks (via AudioFilterBuilder)
    - Rendering subtitles with styling (via SubtitleGraphBuilder)
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
        self.profile_settings: MergedProfileSettings | None = None

        # Product identifier for randomization seeding
        self.product_id: str | None = None

        # Initialize standalone utilities (no profile dependency)
        self.media_inspector = MediaInspector()
        self.subtitle_styler = SubtitleStyler()
        self.audio_builder = AudioFilterBuilder(config)

        # Builders requiring profile settings (lazy init)
        self.visual_builder: VisualFilterBuilder | None = None
        self.subtitle_builder: SubtitleGraphBuilder | None = None
        self.strategy_factory: VideoStrategyFactory | None = None

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
                f"{self.profile_settings.video_settings.image_width_percent}"
            )
            logger.debug(
                f"Image top position: "
                f"{self.profile_settings.video_settings.image_top_position_percent}"
            )
            logger.debug(
                f"Subtitle anchor: " f"{self.profile_settings.subtitle_settings.anchor}"
            )
            logger.debug(
                f"Subtitle style preset: "
                f"{self.profile_settings.subtitle_settings.style_preset}"
            )

        # Initialize builders that depend on profile settings
        self._init_profile_dependent_builders()

    def set_product_id(self, product_id: str) -> None:
        """Set the product identifier for randomization seeding.

        Args:
        ----
            product_id: Product identifier (e.g., ASIN or sanitized product name)

        """
        self.product_id = product_id
        if self.debug_mode:
            logger.debug(f"Set product_id for randomization: {product_id}")

        # Re-initialize strategy factory with product_id
        if self.profile_settings:
            self._init_profile_dependent_builders()

    def _init_profile_dependent_builders(self) -> None:
        """Initialize builders that depend on profile settings."""
        if not self.profile_settings:
            return

        # Initialize strategy factory if product_id is available
        if self.product_id:
            self.strategy_factory = VideoStrategyFactory(
                self.media_inspector,
                self.config,
                self.product_id,
            )

        # Initialize visual builder with video normalization callback
        self.visual_builder = VisualFilterBuilder(
            self.media_inspector,
            self.config,
            self.strategy_factory,
            self.profile_settings,
            self.debug_mode,
            normalize_video_callback=self._normalize_video_format,
        )

        # Initialize subtitle builder
        self.subtitle_builder = SubtitleGraphBuilder(
            self.config,
            self.profile_settings,
            self.product_id or "",
            self.debug_mode,
        )

    def _get_effective_video_settings(self) -> dict[str, Any]:
        """Get effective video settings with profile overrides applied."""
        if self.profile_settings:
            return self.profile_settings.video_settings.model_dump()
        return self.config.video_settings.model_dump()

    def _get_effective_subtitle_settings(self) -> dict[str, Any]:
        """Get effective subtitle settings with profile overrides applied."""
        if self.profile_settings:
            return self.profile_settings.subtitle_settings.model_dump()
        return self.config.subtitle_settings

    def verify_video(
        self,
        video_path: Path,
        expected_duration: float,
        should_have_subtitles: bool,
        script: str | None = None,
        subtitle_path: Path | None = None,
    ) -> dict[str, Any]:
        """Verify assembled video meets quality requirements.

        Args:
        ----
            video_path: Path to video file to verify
            expected_duration: Expected video duration in seconds
            should_have_subtitles: Whether subtitles are expected
            script: Optional script text for subtitle verification
            subtitle_path: Optional subtitle file for content verification

        Returns:
        -------
            Dict with success status, message, and details

        """
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
                    self.subtitle_styler.normalize_text_for_verification(script),
                    self.subtitle_styler.normalize_text_for_verification(srt_text),
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

    async def _normalize_video_format(
        self, video_path: Path, cache_dir: Path | None = None
    ) -> Path:
        """Normalize video format to H.264/30fps/yuv420p with caching.

        Probes the video to detect codec, frame rate, and pixel format.
        If the video doesn't match the target format (H.264, 30fps, yuv420p),
        it transcodes to the normalized format and caches the result.

        Args:
        ----
            video_path: Path to the video file to normalize
            cache_dir: Directory for cached normalized videos (uses config if None)

        Returns:
        -------
            Path to the normalized video (original if already correct,
            cached if transcoded)

        """
        if cache_dir is None:
            video_settings = self.config.video_settings
            cache_dir = Path(video_settings.video_cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                self.ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,r_frame_rate,pix_fmt",
                "-of",
                "json",
                str(video_path),
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.warning(
                    f"FFprobe failed for {video_path.name}, "
                    f"using original: {stderr.decode()}"
                )
                return video_path

            probe_data = json.loads(stdout.decode())
            if not probe_data.get("streams"):
                logger.warning(
                    f"No video stream found in {video_path.name}, using original"
                )
                return video_path

            stream = probe_data["streams"][0]
            codec = stream.get("codec_name", "")
            pix_fmt = stream.get("pix_fmt", "")

            format_norm = self.config.format_normalization
            target_fps = format_norm.get(
                "target_fps", self.config.video_settings.target_fps
            )
            fps_tolerance = format_norm.get(
                "fps_tolerance", self.config.video_settings.fps_tolerance
            )
            default_fps_string = format_norm.get(
                "default_fps_string", self.config.video_settings.default_fps_string
            )
            target_codec = format_norm.get("target_codec", "h264")
            target_pixel_format = format_norm.get("target_pixel_format", "yuv420p")

            fps_str = stream.get("r_frame_rate", default_fps_string)

            try:
                num, den = map(int, fps_str.split("/"))
                fps = num / den if den != 0 else target_fps
            except (ValueError, ZeroDivisionError):
                fps = target_fps

            is_h264 = codec == target_codec
            is_30fps = abs(fps - target_fps) < fps_tolerance
            is_yuv420p = pix_fmt == target_pixel_format

            if is_h264 and is_30fps and is_yuv420p:
                if self.debug_mode:
                    logger.debug(
                        f"Video {video_path.name} already H.264/30fps/yuv420p, "
                        "skipping transcode"
                    )
                return video_path

            cache_filename = f"{video_path.stem}_normalized.mp4"
            cache_path = cache_dir / cache_filename

            if cache_path.exists():
                if self.debug_mode:
                    logger.debug(f"Using cached normalized video: {cache_path.name}")
                return cache_path

            if self.debug_mode:
                logger.debug(
                    f"Transcoding {video_path.name} to "
                    f"{target_codec}/{target_fps}fps/{target_pixel_format} "
                    f"(current: {codec}/{fps:.1f}fps/{pix_fmt})"
                )

            transcode_cmd = [
                self.ffmpeg_path,
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-r",
                str(int(target_fps)),
                "-pix_fmt",
                target_pixel_format,
                "-c:a",
                "copy",
                "-y",
                str(cache_path),
            ]

            transcode_proc = await asyncio.create_subprocess_exec(
                *transcode_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, transcode_stderr = await transcode_proc.communicate()

            if transcode_proc.returncode != 0:
                logger.error(
                    f"Transcode failed for {video_path.name}: "
                    f"{transcode_stderr.decode()}, using original"
                )
                return video_path

            if self.debug_mode:
                logger.debug(f"Transcode complete: {cache_path.name}")

            return cache_path

        except Exception as e:
            logger.error(
                f"Error normalizing video format for {video_path.name}: {e}, "
                "using original"
            )
            return video_path

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
            if hasattr(self.config, "debug_settings") and self.config.debug_settings:
                create_logs = getattr(
                    self.config.debug_settings, "create_ffmpeg_command_logs", True
                )
                return bool(create_logs)
            return True
        except Exception as e:
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
        hook_text: str | None = None,
        hook_headline: str | None = None,
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
            hook_text: Spoken script text used to source the hook overlay's
                first sentence when no authored headline is available. None /
                empty (with no headline) skips the overlay even when
                hook_overlay.enabled is True. See overlay_builder.resolve_hook_line.
            hook_headline: Authored short hook headline, distinct from the spoken
                script. When present it is used verbatim as the overlay text
                (preferred over hook_text) so the hook is not a copy of the first
                caption line. See overlay_builder.resolve_hook_line.

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

        # Ensure builders are initialized
        if not self.subtitle_builder or not self.visual_builder:
            logger.error("Builders not initialized. Call set_profile_settings first.")
            return None

        with tempfile.TemporaryDirectory() as temp_sub_dir:
            # Check if dual subtitle mode is enabled
            upper_exists = (
                subtitle_upper_path.exists() if subtitle_upper_path else False
            )
            logger.debug(
                f"Checking dual subtitle mode: "
                f"subtitle_upper_path={subtitle_upper_path}, "
                f"exists={upper_exists}"
            )

            # Build visual chain first (needed for subtitle positioning)
            video_settings_dict = self._get_effective_video_settings()
            is_relative_mode = video_settings_dict.get("subtitle_relative_mode", True)

            visual_chain_result = await self.visual_builder.build_visual_chain(
                visual_inputs,
                total_video_duration,
                is_relative_mode,
                video_settings_dict,
            )

            if subtitle_upper_path and subtitle_upper_path.exists():
                logger.info("Two-part subtitle mode: rendering dual subtitle lines")
                (
                    video_filters,
                    input_cmd_parts,
                ) = await self.subtitle_builder.build_dual_subtitle_graph(
                    visual_chain_result,
                    subtitle_path,
                    subtitle_upper_path,
                    Path(temp_sub_dir),
                )
            else:
                (
                    video_filters,
                    input_cmd_parts,
                ) = await self.subtitle_builder.build_subtitle_graph(
                    visual_chain_result,
                    subtitle_path,
                    Path(temp_sub_dir),
                )

            # Burn the persistent disclosure overlay (#ad / #publi etc.) as the
            # final video filter. Required by FTC's two-punch guidance: the
            # caption text disclosure (Phase 0.2) is not enough on its own.
            frame_height = self.config.video_settings.resolution[1]
            base_font_pct = self.config.video_settings.base_font_height_percent
            subtitle_font_size_pixels = max(8, int(round(frame_height * base_font_pct)))

            # Phase 1.2c / Issue #102: burn the hook overlay BEFORE the
            # disclosure so the disclosure stays on top in the z-order.
            # The first sentence of the script is the hook text; an empty
            # hook_text (no script available) makes the overlay a no-op.
            hook_settings = self.config.video_settings.hook_overlay
            if hook_settings.enabled:
                hook_line = resolve_hook_line(
                    hook_headline, hook_text, hook_settings.max_words
                )
                if hook_line:
                    video_filters = apply_hook_overlay(
                        video_filters,
                        hook_settings,
                        hook_line,
                        subtitle_font_size_pixels,
                        self.config.video_settings.resolution[0],
                        temp_dir,
                    )

            disclosure = self.config.video_settings.disclosure_overlay
            video_filters = apply_disclosure_overlay(
                video_filters, disclosure, subtitle_font_size_pixels
            )

            # Add audio inputs to command
            num_visual_inputs = input_cmd_parts.count("-i")
            voiceover_input_idx, music_input_idx = (
                self.audio_builder.prepare_audio_inputs(
                    input_cmd_parts,
                    voiceover_audio_path,
                    music_track_path,
                    num_visual_inputs,
                )
            )

            # Build audio processing filters
            audio_filters, final_audio_label = self.audio_builder.build_audio_filters(
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

            command_log_path = (
                temp_dir / f"{output_path.stem}_ffmpeg_command.log"
                if self._should_create_ffmpeg_logs()
                else None
            )

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
