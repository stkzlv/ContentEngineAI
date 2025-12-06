"""Subtitle parsing and styling utilities.

This module provides standalone utilities for parsing subtitle files (SRT, ASS)
and styling utilities for font resolution and color conversion. These utilities
have zero dependencies on other assembler components.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.video.config import (
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
)

logger = logging.getLogger(__name__)


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


class SubtitleParser:
    """SRT/ASS subtitle parsing utilities."""

    @staticmethod
    def parse_srt(subtitle_path: Path) -> list[SubtitleEntry]:
        """Parse SRT file into SubtitleEntry list.

        Args:
        ----
            subtitle_path: Path to the SRT subtitle file

        Returns:
        -------
            List of SubtitleEntry objects

        """
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

    @staticmethod
    def parse_ass_time(time_str: str) -> float:
        """Parse ASS time format (H:MM:SS.CC) to seconds.

        Args:
        ----
            time_str: Time string in ASS format (e.g., "0:00:01.72")

        Returns:
        -------
            Time in seconds

        """
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


class SubtitleStyler:
    """Font and color utilities for subtitle styling."""

    @staticmethod
    def resolve_font_path(font_name: str, font_directory: Path) -> Path | None:
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
            font_directory: Directory containing font files

        Returns:
        -------
            Path to the font file if found, None if no usable font could be found

        """
        if not font_directory.is_dir():
            logger.warning(f"Font directory does not exist: {font_directory}")
            return None

        # Get all valid font files in the directory
        font_files = [
            f
            for f in font_directory.iterdir()
            if f.is_file() and f.suffix.lower() in FONT_FILE_EXTENSIONS
        ]

        if not font_files:
            logger.error(f"No valid font files found in directory: {font_directory}")
            return None

        # Normalize font name by removing spaces and converting to lowercase
        normalized_font_name = font_name.lower().replace(" ", "").replace("-", "")

        # First attempt: Try to find the exact requested font
        for file_path in font_files:
            # Normalize file stem by removing common suffixes and lowercase
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
            f"Could not find font '{font_name}', "
            f"trying fallback: {DEFAULT_FALLBACK_FONT}"
        )
        fallback_path = SubtitleStyler.resolve_font_path(
            DEFAULT_FALLBACK_FONT, font_directory
        )
        if fallback_path:
            logger.info(
                f"Using fallback font: {DEFAULT_FALLBACK_FONT} -> {fallback_path}"
            )
            return fallback_path

        # Third attempt: Try alternative fallback fonts
        for alt_font in FALLBACK_FONT_ALTERNATIVES:
            logger.warning(f"Trying alternative fallback font: {alt_font}")
            alt_path = SubtitleStyler.resolve_font_path(alt_font, font_directory)
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

    @staticmethod
    def convert_ass_color_to_ffmpeg(ass_color: str) -> str:
        """Convert ASS color format to FFmpeg format.

        ASS uses BGR color format with optional alpha: &H[AA]BBGGRR
        FFmpeg uses RGB hex with optional opacity: 0xRRGGBB[@opacity]

        Args:
        ----
            ass_color: Color in ASS format (e.g., "&H00FFFFFF")

        Returns:
        -------
            Color in FFmpeg format (e.g., "0xFFFFFF" or "0xFFFFFF@0.50")

        """
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

    @staticmethod
    def normalize_text_for_verification(text: str) -> str:
        """Normalize subtitle text for comparison.

        Removes punctuation and extra whitespace to make text comparison
        more robust.

        Args:
        ----
            text: Text to normalize

        Returns:
        -------
            Normalized text (lowercase, no punctuation, single spaces)

        """
        if not text:
            return ""
        text = re.sub(r"[^\w\s]", "", text).lower()
        return re.sub(r"\s+", " ", text).strip()
