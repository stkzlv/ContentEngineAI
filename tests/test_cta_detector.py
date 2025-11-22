"""Tests for CTA detection functionality."""

import pytest

from src.video.cta_detector import (
    contains_cta_keyword,
    detect_cta_timing_windows,
    filter_segments_by_timing_windows,
    is_within_timing_windows,
    merge_timing_windows,
)


@pytest.mark.unit
class TestCTAKeywordDetection:
    """Test CTA keyword detection."""

    def test_contains_cta_keyword_basic(self):
        """Test basic CTA keyword detection."""
        assert contains_cta_keyword("Check out the link in my bio", ["link", "bio"])
        assert contains_cta_keyword("Visit our website today", ["visit"])
        assert contains_cta_keyword(
            "Don't forget to follow and subscribe", ["follow", "subscribe"]
        )

    def test_contains_cta_keyword_case_insensitive(self):
        """Test case-insensitive keyword matching."""
        assert contains_cta_keyword("LINK in bio", ["link"])
        assert contains_cta_keyword("Visit NOW", ["visit"], case_sensitive=False)

    def test_contains_cta_keyword_word_boundary(self):
        """Test word boundary matching to avoid partial matches."""
        # "like" should match "I like this"
        assert contains_cta_keyword("I like this product", ["like"])
        # "like" should NOT match "likelihood"
        assert not contains_cta_keyword("The likelihood is high", ["like"])

    def test_contains_cta_keyword_no_match(self):
        """Test when no CTA keywords are present."""
        assert not contains_cta_keyword("This is a great product", ["link", "bio"])


@pytest.mark.unit
class TestTimingWindowMerge:
    """Test timing window merging."""

    def test_merge_non_overlapping_windows(self):
        """Test merging of non-overlapping windows."""
        windows = [(0.0, 2.0), (5.0, 7.0), (10.0, 12.0)]
        merged = merge_timing_windows(windows)
        assert merged == [(0.0, 2.0), (5.0, 7.0), (10.0, 12.0)]

    def test_merge_overlapping_windows(self):
        """Test merging of overlapping windows."""
        windows = [(0.0, 2.0), (1.5, 3.5), (3.0, 5.0)]
        merged = merge_timing_windows(windows)
        assert merged == [(0.0, 5.0)]

    def test_merge_adjacent_windows(self):
        """Test merging of adjacent windows with gap threshold."""
        windows = [(0.0, 2.0), (2.3, 4.0), (4.2, 6.0)]
        merged = merge_timing_windows(windows, gap_threshold=0.5)
        assert merged == [(0.0, 6.0)]

    def test_merge_empty_windows(self):
        """Test merging of empty window list."""
        merged = merge_timing_windows([])
        assert merged == []

    def test_merge_all_windows_into_continuous(self):
        """Test merging all windows into single continuous window."""
        windows = [(0.0, 2.0), (5.0, 7.0), (10.0, 12.0), (15.0, 18.0)]
        # With gap_threshold=None, merge all into one continuous window
        merged = merge_timing_windows(windows, gap_threshold=None)
        assert len(merged) == 1
        assert merged[0] == (0.0, 18.0)

    def test_merge_continuous_single_window(self):
        """Test continuous merge with single window."""
        windows = [(5.0, 10.0)]
        merged = merge_timing_windows(windows, gap_threshold=None)
        assert len(merged) == 1
        assert merged[0] == (5.0, 10.0)


@pytest.mark.unit
class TestCTATimingDetection:
    """Test CTA timing window detection."""

    def test_detect_single_cta_segment(self):
        """Test detection of single CTA segment."""
        segments = [
            {"text": "This is a great product", "start_time": 0.0, "end_time": 2.0},
            {"text": "Check the link in bio", "start_time": 2.0, "end_time": 4.0},
            {"text": "Thank you for watching", "start_time": 4.0, "end_time": 6.0},
        ]
        keywords = ["link", "bio", "check out", "visit"]
        windows = detect_cta_timing_windows(segments, cta_keywords=keywords)
        assert len(windows) == 1
        assert windows[0] == (2.0, 4.0)

    def test_detect_multiple_cta_segments(self):
        """Test detection of multiple CTA segments merged continuously."""
        segments = [
            {"text": "Welcome to our video", "start_time": 0.0, "end_time": 2.0},
            {"text": "Visit our website", "start_time": 2.0, "end_time": 4.0},
            {"text": "Here are the features", "start_time": 4.0, "end_time": 8.0},
            {"text": "Follow and subscribe", "start_time": 8.0, "end_time": 10.0},
        ]
        keywords = ["visit", "follow", "subscribe"]
        windows = detect_cta_timing_windows(segments, cta_keywords=keywords)
        # All CTA segments are merged into one continuous window
        assert len(windows) == 1
        assert windows[0] == (2.0, 10.0)

    def test_detect_adjacent_cta_segments_merged(self):
        """Test adjacent CTA segments are merged into continuous window."""
        segments = [
            {"text": "Check out the link", "start_time": 0.0, "end_time": 2.0},
            {"text": "Visit our bio", "start_time": 2.1, "end_time": 4.0},
        ]
        keywords = ["check out", "link", "visit", "bio"]
        windows = detect_cta_timing_windows(segments, cta_keywords=keywords)
        # Should be merged into one continuous window (first to last)
        assert len(windows) == 1
        assert windows[0] == (0.0, 4.0)

    def test_detect_multiple_cta_segments_merged_continuously(self):
        """Test multiple CTA segments are merged into single continuous window."""
        segments = [
            {"text": "Welcome", "start_time": 0.0, "end_time": 2.0},
            {"text": "Follow me", "start_time": 2.0, "end_time": 4.0},
            {"text": "Regular content", "start_time": 4.0, "end_time": 8.0},
            {"text": "Like this video", "start_time": 8.0, "end_time": 10.0},
            {"text": "More content", "start_time": 10.0, "end_time": 15.0},
            {"text": "Visit the link in bio", "start_time": 15.0, "end_time": 18.0},
        ]
        keywords = ["follow", "like", "visit", "link", "bio"]
        windows = detect_cta_timing_windows(segments, cta_keywords=keywords)
        # Should be merged into one continuous window from first CTA to last CTA
        assert len(windows) == 1
        assert windows[0] == (2.0, 18.0)

    def test_detect_no_cta_segments(self):
        """Test when no CTA segments are present."""
        segments = [
            {"text": "This is a product review", "start_time": 0.0, "end_time": 2.0},
            {"text": "It has great features", "start_time": 2.0, "end_time": 4.0},
        ]
        windows = detect_cta_timing_windows(segments)
        assert len(windows) == 0


@pytest.mark.unit
class TestTimingWindowOperations:
    """Test timing window utility operations."""

    def test_is_within_timing_windows(self):
        """Test checking if time point is within windows."""
        windows = [(2.0, 4.0), (8.0, 10.0)]
        assert is_within_timing_windows(3.0, windows)
        assert is_within_timing_windows(9.0, windows)
        assert not is_within_timing_windows(5.0, windows)
        assert not is_within_timing_windows(1.0, windows)

    def test_filter_segments_by_timing_windows(self):
        """Test filtering segments by timing windows."""
        segments = [
            {"text": "Segment 1", "start_time": 0.0, "end_time": 2.0},
            {"text": "Segment 2", "start_time": 2.0, "end_time": 4.0},
            {"text": "Segment 3", "start_time": 4.0, "end_time": 6.0},
            {"text": "Segment 4", "start_time": 6.0, "end_time": 8.0},
        ]
        windows = [(1.5, 4.5)]
        filtered = filter_segments_by_timing_windows(segments, windows)
        # Should include segments 1, 2, and 3 (partial or full overlap)
        assert len(filtered) == 3
        assert filtered[0]["text"] == "Segment 1"
        assert filtered[1]["text"] == "Segment 2"
        assert filtered[2]["text"] == "Segment 3"

    def test_filter_segments_clips_to_window_boundaries(self):
        """Test that filtered segments are clipped to window boundaries."""
        segments = [
            {"text": "Long segment", "start_time": 0.0, "end_time": 10.0},
        ]
        windows = [(2.0, 5.0)]
        filtered = filter_segments_by_timing_windows(segments, windows)
        assert len(filtered) == 1
        # Segment should be clipped to window boundaries
        assert filtered[0]["start_time"] == 2.0
        assert filtered[0]["end_time"] == 5.0
