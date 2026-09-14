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
        logger.error("SRT file not found: %s", srt_path)
        return False

    try:
        subs = pysrt.open(str(srt_path), encoding="utf-8")

        if not subs:
            logger.warning("SRT file is empty: %s", srt_path)
            return False

        # Check for basic validity
        valid_segments = 0
        for i, sub in enumerate(subs):
            if not sub.text.strip():
                logger.warning("Empty subtitle text at index %s", i)
                continue
            if sub.start is None or sub.end is None or sub.start >= sub.end:
                logger.warning(
                    "Invalid timing at index %s: %s >= %s", i, sub.start, sub.end
                )
                return False
            valid_segments += 1

        if valid_segments == 0:
            logger.warning("No valid subtitle segments found in: %s", srt_path)
            return False

        if debug_mode:
            logger.debug(
                "SRT validation passed: %s valid segments out of %s total",
                valid_segments,
                len(subs),
            )
        return True

    except Exception as e:
        logger.error("SRT validation failed: %s", e)
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
                "Missing required fields at segment %s: required ['text', 'start', "
                "'end'], got %s",
                i,
                list(segment.keys()),
            )
            continue

        # Check text content
        if not segment["text"] or not segment["text"].strip():
            logger.warning("Empty text in segment %s", i)
            continue

        # Check timing validity
        start_time = segment["start"]
        end_time = segment["end"]

        if not isinstance(start_time, int | float) or start_time < 0:
            logger.warning("Invalid start time in segment %s: %s", i, start_time)
            return False

        if not isinstance(end_time, int | float) or end_time < 0:
            logger.warning("Invalid end time in segment %s: %s", i, end_time)
            return False

        if start_time >= end_time:
            logger.warning(
                "Invalid timing in segment %s: start %s >= end %s",
                i,
                start_time,
                end_time,
            )
            return False

        valid_segments += 1

    if valid_segments == 0:
        logger.warning("No valid subtitle segments found")
        return False

    if debug_mode:
        logger.debug(
            "Segment validation passed: %s valid segments out of %s total",
            valid_segments,
            len(segments),
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
        logger.error("ASS file not found: %s", ass_path)
        return False

    try:
        content = ass_path.read_text(encoding="utf-8")

        if not content.strip():
            logger.warning("ASS file is empty: %s", ass_path)
            return False

        # Check for essential ASS sections
        required_sections = ["[V4+ Styles]", "[Events]"]
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)

        if missing_sections:
            logger.warning("ASS file missing required sections: %s", missing_sections)
            return False

        # Count dialogue lines
        dialogue_lines = content.count("Dialogue:")
        if dialogue_lines == 0:
            logger.warning("No dialogue lines found in ASS file: %s", ass_path)
            return False

        if debug_mode:
            logger.debug(
                "ASS validation passed: %s dialogue lines found", dialogue_lines
            )

        return True

    except Exception as e:
        logger.error("ASS validation failed: %s", e)
        return False
