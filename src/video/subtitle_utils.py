"""Subtitle utilities for post-processing and manipulation.

This module provides utility functions for working with SRT subtitle files,
including timing adjustments, validation, and format conversion using the
pysrt library for robust subtitle handling.
"""

import logging
from pathlib import Path
from typing import Any

import pysrt

from src.utils import ensure_dirs_exist
from src.video.result_types import SubtitleResult
from src.video.stt_functions import (
    GOOGLE_CLOUD_STT_AVAILABLE,
    WHISPER_AVAILABLE,
    generate_subtitles_with_whisper,
    transcribe_with_google_cloud_stt,
)
from src.video.subtitle_positioning import create_unified_config_from_settings
from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator
from src.video.video_config import GoogleCloudSTTSettings, WhisperSettings

logger = logging.getLogger(__name__)


def adjust_subtitle_timing(
    srt_path: Path,
    time_offset_ms: int,
    output_path: Path | None = None,
) -> Path | None:
    """Adjust subtitle timing by shifting all subtitles by a specified offset.

    Args:
    ----
        srt_path: Path to the existing SRT file
        time_offset_ms: Time offset in milliseconds (positive = delay,
            negative = advance)
        output_path: Output path (defaults to input path if None)

    Returns:
    -------
        Path to the adjusted SRT file if successful, None otherwise

    """
    if not srt_path.exists():
        logger.error(f"SRT file not found: {srt_path}")
        return None

    output_path = output_path or srt_path

    try:
        # Load existing SRT file
        subs = pysrt.open(str(srt_path), encoding="utf-8")

        # Shift timing
        subs.shift(milliseconds=time_offset_ms)  # type: ignore[attr-defined]

        # Save adjusted file
        ensure_dirs_exist(output_path.parent)
        subs.save(str(output_path), encoding="utf-8")

        logger.info(
            f"Subtitle timing adjusted by {time_offset_ms}ms: "
            f"{srt_path} -> {output_path}"
        )
        return output_path

    except Exception as e:
        logger.error(f"Failed to adjust subtitle timing: {e}", exc_info=True)
        return None


def slice_subtitles(
    srt_path: Path,
    start_time_ms: int,
    end_time_ms: int,
    output_path: Path,
) -> Path | None:
    """Extract a portion of subtitles between specified time ranges.

    Args:
    ----
        srt_path: Path to the existing SRT file
        start_time_ms: Start time in milliseconds
        end_time_ms: End time in milliseconds
        output_path: Output path for the sliced subtitles

    Returns:
    -------
        Path to the sliced SRT file if successful, None otherwise

    """
    if not srt_path.exists():
        logger.error(f"SRT file not found: {srt_path}")
        return None

    try:
        # Load existing SRT file
        subs = pysrt.open(str(srt_path), encoding="utf-8")

        # Create time objects for slicing
        start_time = pysrt.SubRipTime(milliseconds=start_time_ms)
        end_time = pysrt.SubRipTime(milliseconds=end_time_ms)

        # Slice the subtitles (pysrt slice parameters work differently)
        sliced_subs = subs.slice(starts_after=start_time, ends_before=end_time)  # type: ignore[attr-defined]

        # If no results, try a broader search
        if not sliced_subs:
            # Try getting overlapping subtitles instead
            sliced_subs = pysrt.SubRipFile()
            for sub in subs:
                if sub.start is None or sub.end is None:
                    continue
                sub_start_ms = sub.start.ordinal  # type: ignore[attr-defined]
                sub_end_ms = sub.end.ordinal  # type: ignore[attr-defined]
                # Include if subtitle overlaps with time range
                if sub_start_ms < end_time_ms and sub_end_ms > start_time_ms:
                    sliced_subs.append(sub)

        if not sliced_subs:
            logger.warning(
                f"No subtitles found in time range {start_time_ms}ms - {end_time_ms}ms"
            )
            return None

        # Adjust timing to start from 0
        sliced_subs.shift(milliseconds=-start_time_ms)

        # Save sliced file
        ensure_dirs_exist(output_path.parent)
        sliced_subs.save(str(output_path), encoding="utf-8")

        logger.info(
            f"Subtitles sliced ({start_time_ms}ms - {end_time_ms}ms): "
            f"{srt_path} -> {output_path} ({len(sliced_subs)} segments)"
        )
        return output_path

    except Exception as e:
        logger.error(f"Failed to slice subtitles: {e}", exc_info=True)
        return None


def get_subtitle_info(srt_path: Path) -> dict[str, Any] | None:
    """Get information about an SRT file.

    Args:
    ----
        srt_path: Path to the SRT file

    Returns:
    -------
        Dictionary with subtitle information or None if failed

    """
    if not srt_path.exists():
        logger.error(f"SRT file not found: {srt_path}")
        return None

    try:
        subs = pysrt.open(str(srt_path), encoding="utf-8")

        if not subs:
            return {
                "file_path": str(srt_path),
                "segment_count": 0,
                "duration_ms": 0,
                "is_valid": False,
            }

        # Calculate total duration
        total_duration_ms = (
            int(subs[-1].end.ordinal) if subs and subs[-1].end is not None else 0  # type: ignore[attr-defined]
        )

        return {
            "file_path": str(srt_path),
            "segment_count": len(subs),
            "duration_ms": total_duration_ms,
            "is_valid": True,
            "first_segment_start": (
                int(subs[0].start.ordinal)  # type: ignore[attr-defined]
                if subs and subs[0].start is not None
                else None
            ),
            "last_segment_end": (
                int(subs[-1].end.ordinal) if subs and subs[-1].end is not None else None  # type: ignore[attr-defined]
            ),
        }

    except Exception as e:
        logger.error(f"Failed to get subtitle info: {e}")
        return None


def convert_timestamps_to_seconds(srt_path: Path, output_path: Path) -> Path | None:
    """Convert SRT timestamps to show time in seconds for debugging.

    Args:
    ----
        srt_path: Path to the input SRT file
        output_path: Path for the output file with timestamps in seconds

    Returns:
    -------
        Path to the converted file if successful, None otherwise

    """
    if not srt_path.exists():
        logger.error(f"SRT file not found: {srt_path}")
        return None

    try:
        subs = pysrt.open(str(srt_path), encoding="utf-8")

        ensure_dirs_exist(output_path.parent)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, sub in enumerate(subs, 1):
                if sub.start is None or sub.end is None:
                    continue
                start_seconds = sub.start.ordinal / 1000  # type: ignore[attr-defined]
                end_seconds = sub.end.ordinal / 1000  # type: ignore[attr-defined]

                f.write(f"{i}\n")
                f.write(f"{start_seconds:.3f}s --> {end_seconds:.3f}s\n")
                f.write(f"{sub.text}\n\n")

        logger.info(f"SRT converted to seconds format: {srt_path} -> {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to convert timestamps to seconds: {e}")
        return None


# ============================================================================
# UNIFIED SUBTITLE GENERATION INTERFACE
# ============================================================================


def create_static_upper_subtitle(
    text: str,
    output_path: Path,
    subtitle_settings: dict[str, Any],
    video_config: Any = None,
    format_type: str = "ass",
    product_id: str | None = None,
    voiceover_duration: float | None = None,
    visual_bounds: Any | None = None,
) -> Path | None:
    """Generate static subtitle file for upper line (product URL/info).

    Creates a subtitle file with a single static entry that displays
    throughout the entire video. Used for product URLs or other persistent info.

    Args:
    ----
        text: Static text to display (e.g., shortened product URL)
        output_path: Output path for subtitle file
        subtitle_settings: Subtitle settings dict (includes two_part_subtitles config)
        video_config: Video configuration for frame size
        format_type: Subtitle format ("ass" or "srt")
        product_id: Product ID for randomization (if applicable)
        voiceover_duration: Duration of voiceover (for full-duration display)
        visual_bounds: Visual bounds for content-aware positioning

    Returns:
    -------
        Path to generated subtitle file or None if failed

    """
    try:
        # Extract upper line configuration (flat keys from profile settings)
        use_full_duration = subtitle_settings.get(
            "two_part_subtitles_upper_use_full_duration", True
        )
        randomize_effects = subtitle_settings.get(
            "two_part_subtitles_upper_randomize_effects", False
        )

        # Create unified config for upper line using profile settings
        # Profile settings use flat keys like two_part_subtitles_upper_anchor
        upper_subtitle_settings = subtitle_settings.copy()

        # Use flat profile settings if available, otherwise fallback to nested config
        upper_subtitle_settings["anchor"] = subtitle_settings.get(
            "two_part_subtitles_upper_anchor", "above_content"
        )
        upper_subtitle_settings["margin"] = subtitle_settings.get(
            "two_part_subtitles_upper_margin", 0.03
        )
        upper_subtitle_settings["font_size_scale"] = subtitle_settings.get(
            "two_part_subtitles_upper_font_size_scale", 0.8
        )
        upper_subtitle_settings["style_preset"] = subtitle_settings.get(
            "two_part_subtitles_upper_style_preset", "minimal"
        )
        upper_subtitle_settings["randomize_effects"] = randomize_effects

        # Create unified configuration from settings
        unified_config = create_unified_config_from_settings(upper_subtitle_settings)

        # Get frame size from video config
        frame_size = (1080, 1920)  # Default
        if video_config and hasattr(video_config, "video_settings"):
            frame_size = video_config.video_settings.resolution

        # Initialize unified generator
        generator = UnifiedSubtitleGenerator(unified_config, frame_size, product_id)

        # Determine end time based on use_full_duration setting
        if use_full_duration and voiceover_duration:
            end_time = voiceover_duration
            logger.info(
                f"Upper subtitle set to full video duration: {end_time:.2f}s "
                f"(use_full_duration=True)"
            )
        else:
            end_time = 9999.0  # Large duration for static display
            logger.info("Upper subtitle using default large duration (9999s)")

        # For static subtitles, bypass the normal segment creation
        # and directly create a single ASS dialogue line
        if format_type == "ass":
            # Generate ASS file directly with a single static dialogue
            from src.video.subtitle_positioning import calculate_position, get_font_size

            position = calculate_position(
                unified_config,
                frame_size,
                visual_bounds  # Pass visual bounds for content-aware positioning
            )
            font_size = get_font_size(unified_config, frame_size[1])

            # Calculate pixel coordinates
            pos_x = int(position.x * frame_size[0])
            pos_y = int(position.y * frame_size[1])

            # Get colors from generator
            colors = generator._get_colors()

            # Format times
            start_time_str = generator._format_ass_time(0.0)
            end_time_str = generator._format_ass_time(end_time)

            # Create ASS content
            ass_lines = generator._create_ass_header(font_size, colors)

            # Add single static dialogue line
            dialogue = (
                f"Dialogue: 0,{start_time_str},{end_time_str},Default,,0,0,0,,"
                f"{{\\pos({pos_x},{pos_y})}}{text}"
            )
            ass_lines.append(dialogue)

            # Write file
            ensure_dirs_exist(output_path.parent)
            output_path.write_text("\n".join(ass_lines), encoding="utf-8")

            result = SubtitleResult(
                success=True,
                path=output_path,
                format="ass",
                segments_created=1,
                generation_method="static",
            )
        else:
            # For SRT, use the normal timing-based generation
            static_timing = [
                {
                    "word": text,
                    "start_time": 0.0,
                    "end_time": end_time,
                }
            ]

            result = generator.generate_from_timings(
                timings=static_timing,
                output_path=output_path,
                format_type=format_type,
                voiceover_duration=voiceover_duration,
                debug_mode=False,
            )

        if result.success and result.path and result.path.exists():
            logger.info(
                f"Successfully generated static upper subtitle ({format_type.upper()}): "
                f"{result.path} (randomize_effects={randomize_effects})"
            )
            return result.path
        else:
            logger.error(
                f"Failed to generate static upper subtitle: "
                f"{result.errors if result.errors else 'Unknown error'}"
            )
            return None

    except Exception as e:
        logger.error(f"Static upper subtitle generation failed: {e}", exc_info=True)
        return None


async def create_unified_subtitles(
    audio_path: Path,
    output_srt_path: Path,
    subtitle_settings: dict[str, Any],
    whisper_stt_settings: WhisperSettings | None,
    google_stt_settings: GoogleCloudSTTSettings | None,
    secrets: dict[str, str],
    script: str | None,
    voiceover_duration: float | None,
    debug_mode: bool = False,
    video_config: Any = None,
    temp_dir: Path | None = None,
    product_id: str | None = None,
) -> Path | None:
    """Generate subtitles using the unified system with STT integration.

    This function replaces the legacy create_final_subtitles() while maintaining
    the same interface and STT capabilities but using the modern
    UnifiedSubtitleGenerator
    which fixes the multi-color karaoke issue.

    Args:
    ----
        audio_path: Path to audio file for transcription
        output_srt_path: Output path for subtitle file
        subtitle_settings: Subtitle generation settings dict
        whisper_stt_settings: Whisper STT configuration
        google_stt_settings: Google Cloud STT configuration
        secrets: Dictionary with API keys and credentials
        script: Optional script text for fallback timing
        voiceover_duration: Duration of voiceover audio
        debug_mode: Enable debug output
        video_config: Video configuration for frame size
        temp_dir: Optional temp directory for debug files (defaults to output parent)
        product_id: Product ID for randomization (if applicable)

    Returns:
    -------
        Path to generated subtitle file or None if failed

    """
    # Determine output format and path
    subtitle_format = subtitle_settings.get("subtitle_format", "srt")
    if subtitle_format == "ass":
        output_path = output_srt_path.with_suffix(".ass")
        format_type = "ass"
        logger.info(
            f"Generating ASS subtitles: {audio_path.name} -> {output_path.name}"
        )
    else:
        output_path = output_srt_path
        format_type = "srt"
        logger.info(
            f"Generating SRT subtitles: {audio_path.name} -> {output_path.name}"
        )

    # Create unified configuration from subtitle settings dict
    unified_config = create_unified_config_from_settings(subtitle_settings)

    # Get frame size from video config
    frame_size = (1080, 1920)  # Default
    if video_config and hasattr(video_config, "video_settings"):
        frame_size = video_config.video_settings.resolution

    # Initialize unified generator (fixes karaoke color issue)
    generator = UnifiedSubtitleGenerator(unified_config, frame_size, product_id)

    # Try to get STT timings (Whisper first, then Google Cloud STT)
    stt_timings = None

    # Try Whisper STT first
    if whisper_stt_settings and whisper_stt_settings.enabled and WHISPER_AVAILABLE:
        logger.info("Using Whisper for STT and word timings.")
        try:
            stt_timings = await generate_subtitles_with_whisper(
                audio_path,
                temp_dir or output_path.parent,
                whisper_stt_settings,
                script,
                debug_mode,
            )
            if stt_timings:
                logger.info(
                    f"Whisper STT successful, got {len(stt_timings)} word timings."
                )
            else:
                logger.warning("Whisper STT did not return usable word timings")
        except Exception as e:
            logger.error(f"Whisper STT failed: {e}", exc_info=debug_mode)
    elif whisper_stt_settings and whisper_stt_settings.enabled:
        logger.warning("Whisper STT configured but library not available")
    else:
        logger.info("Whisper STT not configured or not enabled.")

    # Try Google Cloud STT as fallback
    if (
        not stt_timings
        and google_stt_settings
        and google_stt_settings.enabled
        and GOOGLE_CLOUD_STT_AVAILABLE
    ):
        creds_path = secrets.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and Path(creds_path).is_file():
            logger.info(
                "Using Google Cloud STT for word timings "
                "(Whisper fallback/alternative)."
            )
            try:
                stt_timings = await transcribe_with_google_cloud_stt(
                    audio_path, google_stt_settings, secrets, script, debug_mode
                )
                if stt_timings:
                    logger.info(
                        f"Google Cloud STT successful, got "
                        f"{len(stt_timings)} word timings."
                    )
                else:
                    logger.warning(
                        "Google Cloud STT did not return usable word timings"
                    )
            except Exception as e:
                logger.error(f"Google Cloud STT failed: {e}", exc_info=debug_mode)
        else:
            logger.warning(
                "Google Cloud STT configured but "
                "GOOGLE_APPLICATION_CREDENTIALS invalid/not found."
            )
    elif google_stt_settings and google_stt_settings.enabled:
        logger.warning("Google Cloud STT configured but library not available.")

    # Generate subtitles using unified system
    try:
        if stt_timings:
            # Use STT timing data for precise subtitles
            logger.info(
                f"Generating {format_type.upper()} from "
                f"{len(stt_timings)} word timings."
            )
            result = generator.generate_from_timings(
                timings=stt_timings,
                output_path=output_path,
                format_type=format_type,
                voiceover_duration=voiceover_duration,
                debug_mode=debug_mode,
            )
        elif script and voiceover_duration:
            # Fallback to script-based timing estimation
            logger.info(
                f"Generating {format_type.upper()} from script "
                f"with estimated timing."
            )
            result = generator.generate_from_script(
                script_text=script,
                duration=voiceover_duration,
                output_path=output_path,
                format_type=format_type,
                debug_mode=debug_mode,
            )
        else:
            logger.error(
                "No STT timings available and insufficient "
                "script/duration for fallback."
            )
            return None

        if result.success and result.path and result.path.exists():
            logger.info(
                f"Successfully generated {format_type.upper()} subtitles: "
                f"{result.path}"
            )
            return result.path
        else:
            logger.error(
                f"Failed to generate subtitles: "
                f"{result.errors if result.errors else 'Unknown error'}"
            )
            return None

    except Exception as e:
        logger.error(f"Unified subtitle generation failed: {e}", exc_info=debug_mode)
        return None
