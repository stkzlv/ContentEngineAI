"""Subtitle validation utilities for ContentEngineAI video module.

This module provides centralized validation functions for SRT and ASS subtitle files,
consolidating previously duplicated validation logic from subtitle_generator.py and
subtitle_utils.py.

Functions:
    validate_srt_file: Validate SRT file format and content
    validate_subtitle_segments: Validate timing data structure
    validate_ass_file: Validate ASS file format (future implementation)
"""

import logging
from pathlib import Path
from typing import Any

import pysrt

logger = logging.getLogger(__name__)


def validate_srt_file(srt_path: Path, debug_mode: bool = False) -> bool:
    """Validate that an SRT file can be loaded and has valid content.

    This function consolidates SRT validation logic previously duplicated
    in subtitle_generator.py and subtitle_utils.py.

    Args:
    ----
        srt_path: Path to the SRT file to validate
        debug_mode: Whether to output detailed debug information

    Returns:
    -------
        True if the SRT file is valid, False otherwise

    """
    if not srt_path.exists():
        logger.error(f"SRT file not found: {srt_path}")
        return False

    try:
        subs = pysrt.open(str(srt_path), encoding="utf-8")

        if not subs:
            logger.warning(f"SRT file is empty: {srt_path}")
            return False

        # Check for basic validity
        valid_segments = 0
        for i, sub in enumerate(subs):
            if not sub.text.strip():
                logger.warning(f"Empty subtitle text at index {i}")
                continue
            if sub.start >= sub.end:
                logger.warning(f"Invalid timing at index {i}: {sub.start} >= {sub.end}")
                return False
            valid_segments += 1

        if valid_segments == 0:
            logger.warning(f"No valid subtitle segments found in: {srt_path}")
            return False

        if debug_mode:
            logger.debug(
                f"SRT validation passed: {valid_segments} valid segments "
                f"out of {len(subs)} total"
            )
        return True

    except Exception as e:
        logger.error(f"SRT validation failed: {e}")
        return False


def validate_subtitle_segments(
    segments: list[dict[str, Any]], debug_mode: bool = False
) -> bool:
    """Validate subtitle segment timing data structure.

    Args:
    ----
        segments: List of subtitle segments with timing information
        debug_mode: Whether to output detailed debug information

    Returns:
    -------
        True if all segments are valid, False otherwise

    """
    if not segments:
        logger.warning("No subtitle segments provided for validation")
        return False

    valid_segments = 0
    for i, segment in enumerate(segments):
        # Check required fields
        if "text" not in segment or "start" not in segment or "end" not in segment:
            logger.warning(
                f"Missing required fields at segment {i}: "
                f"required ['text', 'start', 'end'], got {list(segment.keys())}"
            )
            continue

        # Check text content
        if not segment["text"] or not segment["text"].strip():
            logger.warning(f"Empty text in segment {i}")
            continue

        # Check timing validity
        start_time = segment["start"]
        end_time = segment["end"]

        if not isinstance(start_time, int | float) or start_time < 0:
            logger.warning(f"Invalid start time in segment {i}: {start_time}")
            return False

        if not isinstance(end_time, int | float) or end_time < 0:
            logger.warning(f"Invalid end time in segment {i}: {end_time}")
            return False

        if start_time >= end_time:
            logger.warning(
                f"Invalid timing in segment {i}: start {start_time} >= end {end_time}"
            )
            return False

        valid_segments += 1

    if valid_segments == 0:
        logger.warning("No valid subtitle segments found")
        return False

    if debug_mode:
        logger.debug(
            f"Segment validation passed: {valid_segments} valid segments "
            f"out of {len(segments)} total"
        )

    return True


def validate_ass_file(ass_path: Path, debug_mode: bool = False) -> bool:
    """Validate that an ASS file has proper structure and content.

    Args:
    ----
        ass_path: Path to the ASS file to validate
        debug_mode: Whether to output detailed debug information

    Returns:
    -------
        True if the ASS file is valid, False otherwise

    """
    if not ass_path.exists():
        logger.error(f"ASS file not found: {ass_path}")
        return False

    try:
        content = ass_path.read_text(encoding="utf-8")

        if not content.strip():
            logger.warning(f"ASS file is empty: {ass_path}")
            return False

        # Check for essential ASS sections
        required_sections = ["[V4+ Styles]", "[Events]"]
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)

        if missing_sections:
            logger.warning(f"ASS file missing required sections: {missing_sections}")
            return False

        # Count dialogue lines
        dialogue_lines = content.count("Dialogue:")
        if dialogue_lines == 0:
            logger.warning(f"No dialogue lines found in ASS file: {ass_path}")
            return False

        if debug_mode:
            logger.debug(
                f"ASS validation passed: {dialogue_lines} dialogue lines found"
            )

        return True

    except Exception as e:
        logger.error(f"ASS validation failed: {e}")
        return False
