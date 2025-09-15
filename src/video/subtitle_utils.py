"""Subtitle utilities for post-processing and manipulation.

This module provides utility functions for working with SRT subtitle files,
including timing adjustments, validation, and format conversion using the
pysrt library for robust subtitle handling.
"""

import logging
from pathlib import Path
from typing import Any

import pysrt  # type: ignore[import-untyped]

from src.utils import ensure_dirs_exist
from src.video.stt_functions import (
    GOOGLE_CLOUD_STT_AVAILABLE,
    WHISPER_AVAILABLE,
    generate_subtitles_with_whisper,
    transcribe_with_google_cloud_stt,
)
from src.video.subtitle_positioning import convert_legacy_config
from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator
from src.video.video_config import (
    GoogleCloudSTTSettings,
    SubtitleSettings,
    WhisperSettings,
)

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
        subs.shift(milliseconds=time_offset_ms)

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
        sliced_subs = subs.slice(starts_after=start_time, ends_before=end_time)

        # If no results, try a broader search
        if not sliced_subs:
            # Try getting overlapping subtitles instead
            sliced_subs = pysrt.SubRipFile()
            for sub in subs:
                sub_start_ms = sub.start.ordinal
                sub_end_ms = sub.end.ordinal
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
        total_duration_ms = int(subs[-1].end.ordinal) if subs else 0

        return {
            "file_path": str(srt_path),
            "segment_count": len(subs),
            "duration_ms": total_duration_ms,
            "is_valid": True,
            "first_segment_start": int(subs[0].start.ordinal) if subs else None,
            "last_segment_end": int(subs[-1].end.ordinal) if subs else None,
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
                start_seconds = sub.start.ordinal / 1000
                end_seconds = sub.end.ordinal / 1000

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


async def create_unified_subtitles(
    audio_path: Path,
    output_srt_path: Path,
    subtitle_settings: SubtitleSettings,
    whisper_stt_settings: WhisperSettings | None,
    google_stt_settings: GoogleCloudSTTSettings | None,
    secrets: dict[str, str],
    script: str | None,
    voiceover_duration: float | None,
    debug_mode: bool = False,
    video_config: Any = None,
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
        subtitle_settings: Subtitle generation settings
        whisper_stt_settings: Whisper STT configuration
        google_stt_settings: Google Cloud STT configuration
        secrets: Dictionary with API keys and credentials
        script: Optional script text for fallback timing
        voiceover_duration: Duration of voiceover audio
        debug_mode: Enable debug output
        video_config: Video configuration for frame size

    Returns:
    -------
        Path to generated subtitle file or None if failed

    """
    # Determine output format and path
    if subtitle_settings.subtitle_format == "ass":
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

    # Create unified configuration from legacy settings
    unified_config = convert_legacy_config(subtitle_settings.__dict__)

    # Get frame size from video config
    frame_size = (1080, 1920)  # Default
    if video_config and hasattr(video_config, "video_settings"):
        frame_size = video_config.video_settings.resolution

    # Initialize unified generator (fixes karaoke color issue)
    generator = UnifiedSubtitleGenerator(unified_config, frame_size)

    # Try to get STT timings (Whisper first, then Google Cloud STT)
    stt_timings = None

    # Try Whisper STT first
    if whisper_stt_settings and whisper_stt_settings.enabled and WHISPER_AVAILABLE:
        logger.info("Using Whisper for STT and word timings.")
        try:
            stt_timings = await generate_subtitles_with_whisper(
                audio_path,
                output_path.parent,
                subtitle_settings,
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
