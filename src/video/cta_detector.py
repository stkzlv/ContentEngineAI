"""CTA (Call-to-Action) detection for subtitle timing synchronization.

This module provides keyword-based detection of CTA moments in video scripts
to enable synchronized display of promotional URLs and links during relevant
voiceover segments.
"""

import logging
import re
from typing import Any

from src.video.config import config

logger = logging.getLogger(__name__)


def detect_cta_timing_windows(
    subtitle_segments: list[dict[str, Any]],
    cta_keywords: list[str] | None = None,
    case_sensitive: bool | None = None,
    merge_gap_threshold: float | None = None,
) -> list[tuple[float, float]]:
    """Detect timing windows where CTA phrases are spoken.

    Args:
    ----
        subtitle_segments: List of subtitle segments with 'text', 'start_time',
            'end_time'
        cta_keywords: List of keywords to detect (uses config defaults if None)
        case_sensitive: Whether keyword matching should be case-sensitive
            (uses config default if None)
        merge_gap_threshold: Maximum gap between segments to merge in seconds
            (uses config default if None)

    Returns:
    -------
        List of (start_time, end_time) tuples in seconds where CTA is detected

    """
    # Load settings from config
    cta_config = config.cta_detection if hasattr(config, "cta_detection") else None

    if cta_keywords is None:
        if cta_config and hasattr(cta_config, "keywords"):
            cta_keywords = cta_config.keywords
        else:
            # Configuration not available - see config/video_production.yaml
            cta_keywords = []

    if case_sensitive is None:
        if cta_config and hasattr(cta_config, "case_sensitive"):
            case_sensitive = cta_config.case_sensitive
        else:
            # Configuration not available - see config/video_production.yaml
            case_sensitive = False

    if merge_gap_threshold is None:
        if cta_config and hasattr(cta_config, "merge_gap_threshold"):
            merge_gap_threshold = cta_config.merge_gap_threshold
        else:
            # Configuration not available - see config/video_production.yaml
            merge_gap_threshold = 0.5

    cta_windows = []

    for segment in subtitle_segments:
        text = segment.get("text", "")
        start_time = segment.get("start_time", 0.0)
        end_time = segment.get("end_time", 0.0)

        if not text or start_time is None or end_time is None:
            continue

        # Check if any CTA keyword is present in the segment
        if contains_cta_keyword(text, cta_keywords, case_sensitive):
            cta_windows.append((float(start_time), float(end_time)))
            logger.debug(f"CTA detected at {start_time:.2f}s-{end_time:.2f}s: '{text}'")

    # Merge all windows into a single continuous period from first to last CTA
    merged_windows = merge_timing_windows(cta_windows, gap_threshold=None)

    logger.info(f"Detected {len(merged_windows)} CTA timing window(s)")
    return merged_windows


def contains_cta_keyword(
    text: str,
    keywords: list[str],
    case_sensitive: bool = False,
) -> bool:
    """Check if text contains any CTA keyword.

    Args:
    ----
        text: Text to check
        keywords: List of keywords to search for
        case_sensitive: Whether matching should be case-sensitive

    Returns:
    -------
        True if any keyword is found, False otherwise

    """
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]

    # Use word boundary matching to avoid partial matches
    # e.g., "like" should match "I like this" but not "likelihood"
    for keyword in keywords:
        # Escape special regex characters in keyword
        escaped_keyword = re.escape(keyword)
        # Create word boundary pattern
        pattern = r"\b" + escaped_keyword + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE if not case_sensitive else 0):
            return True

    return False


def merge_timing_windows(
    windows: list[tuple[float, float]],
    gap_threshold: float | None = 0.5,
) -> list[tuple[float, float]]:
    """Merge adjacent or overlapping timing windows.

    Args:
    ----
        windows: List of (start_time, end_time) tuples
        gap_threshold: Maximum gap between windows to merge (seconds)
            If None, merges all windows into a single continuous window
            from first to last

    Returns:
    -------
        List of merged timing windows

    """
    if not windows:
        return []

    # Sort windows by start time
    sorted_windows = sorted(windows, key=lambda w: w[0])

    # If gap_threshold is None, merge all into one continuous window
    if gap_threshold is None:
        first_start = sorted_windows[0][0]
        last_end = max(end for _, end in sorted_windows)
        return [(first_start, last_end)]

    # gap_threshold must be float here (not None) due to early return above
    threshold: float = gap_threshold  # type narrowing for MyPy
    merged = [sorted_windows[0]]

    for current_start, current_end in sorted_windows[1:]:
        last_start, last_end = merged[-1]

        # Check if current window overlaps or is close to last window
        if current_start <= last_end + threshold:
            # Merge by extending the last window
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # Add as separate window
            merged.append((current_start, current_end))

    return merged


def is_within_timing_windows(
    time_point: float,
    windows: list[tuple[float, float]],
) -> bool:
    """Check if a time point falls within any timing window.

    Args:
    ----
        time_point: Time in seconds to check
        windows: List of (start_time, end_time) tuples

    Returns:
    -------
        True if time_point is within any window, False otherwise

    """
    return any(start <= time_point <= end for start, end in windows)


def filter_segments_by_timing_windows(
    segments: list[dict[str, Any]],
    windows: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Filter subtitle segments to only include those within timing windows.

    Args:
    ----
        segments: List of subtitle segments with 'start_time', 'end_time'
        windows: List of (start_time, end_time) tuples to filter by

    Returns:
    -------
        Filtered list of segments that overlap with timing windows

    """
    if not windows:
        return []

    filtered = []

    for segment in segments:
        start_time = segment.get("start_time", 0.0)
        end_time = segment.get("end_time", 0.0)

        # Check if segment overlaps with any window
        for window_start, window_end in windows:
            # Segment overlaps if starts before window ends and ends after
            # window starts
            if start_time < window_end and end_time > window_start:
                # Clip segment timing to window boundaries
                clipped_segment = segment.copy()
                clipped_segment["start_time"] = max(start_time, window_start)
                clipped_segment["end_time"] = min(end_time, window_end)
                filtered.append(clipped_segment)
                break  # Only add each segment once

    return filtered
