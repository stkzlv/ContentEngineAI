"""Unit tests for profile selection utilities.

Tests validate deterministic profile selection, configuration precedence,
usage tracking, and validation logic for profile randomization.
All tests use mocked VideoConfig to ensure isolation.
"""

from unittest.mock import Mock

import pytest

from src.video.producer.utils import (
    ProfileUsageTracker,
    load_profile_pool,
    select_profile_for_product,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_config():
    """Create mock VideoConfig instance with test profiles."""
    config = Mock()
    config.video_profiles = {
        "slideshow_images1": Mock(),
        "video_sequential": Mock(),
        "mixed_media": Mock(),
        "slideshow_images2": Mock(),
    }
    return config


class TestSelectProfileForProduct:
    """Test deterministic profile selection for products."""

    def test_deterministic_selection_same_product_id(self, mock_config):
        """Test that same product ID always gets same profile."""
        product_id = "B0BTYCRJSS"
        profile_pool = ["slideshow_images1", "video_sequential", "mixed_media"]

        # Call multiple times with same product ID
        results = [
            select_profile_for_product(product_id, profile_pool, mock_config)
            for _ in range(10)
        ]

        # All results should be identical
        assert len(set(results)) == 1, "Same product ID should always get same profile"
        assert results[0] in profile_pool

    def test_deterministic_selection_different_product_ids(self, mock_config):
        """Test that different product IDs can get different profiles."""
        profile_pool = ["slideshow_images1", "video_sequential", "mixed_media"]
        product_ids = [f"B{i:09d}" for i in range(100)]

        # Get profiles for many different products
        results = [
            select_profile_for_product(pid, profile_pool, mock_config)
            for pid in product_ids
        ]

        # Should see distribution across profiles (not all the same)
        unique_profiles = set(results)
        assert (
            len(unique_profiles) > 1
        ), "Different products should get different profiles"

    def test_single_profile_pool(self, mock_config):
        """Test selection with single profile in pool."""
        product_id = "B0BTYCRJSS"
        profile_pool = ["slideshow_images1"]

        result = select_profile_for_product(product_id, profile_pool, mock_config)

        assert result == "slideshow_images1"

    def test_empty_pool_raises_error(self, mock_config):
        """Test that empty pool raises ValueError."""
        product_id = "B0BTYCRJSS"
        profile_pool: list[str] = []

        with pytest.raises(ValueError, match="Profile pool cannot be empty"):
            select_profile_for_product(product_id, profile_pool, mock_config)

    def test_invalid_profile_in_pool_raises_error(self, mock_config):
        """Test that invalid profile in pool raises ValueError with helpful message."""
        product_id = "B0BTYCRJSS"
        profile_pool = ["slideshow_images1", "invalid_profile", "video_sequential"]

        with pytest.raises(ValueError) as exc_info:
            select_profile_for_product(product_id, profile_pool, mock_config)

        error_msg = str(exc_info.value)
        assert "invalid_profile" in error_msg
        assert "Available profiles:" in error_msg

    def test_all_invalid_profiles_raises_error(self, mock_config):
        """Test that pool with all invalid profiles raises ValueError."""
        product_id = "B0BTYCRJSS"
        profile_pool = ["nonexistent1", "nonexistent2"]

        with pytest.raises(ValueError) as exc_info:
            select_profile_for_product(product_id, profile_pool, mock_config)

        error_msg = str(exc_info.value)
        assert "nonexistent1" in error_msg
        assert "nonexistent2" in error_msg

    def test_reproducible_across_runs(self, mock_config):
        """Test that selection is reproducible across different test runs."""
        product_id = "B0BTYCRJSS"
        profile_pool = ["slideshow_images1", "video_sequential", "mixed_media"]

        # First "run"
        first_result = select_profile_for_product(product_id, profile_pool, mock_config)

        # Second "run" (simulating different process)
        second_result = select_profile_for_product(
            product_id, profile_pool, mock_config
        )

        assert first_result == second_result, "Profile selection should be reproducible"


class TestLoadProfilePool:
    """Test profile pool loading with precedence."""

    def test_cli_overrides_yaml(self, mock_config):
        """Test that CLI pool takes precedence over YAML pool."""
        cli_pool = ["slideshow_images1", "video_sequential"]
        yaml_pool = ["mixed_media", "slideshow_images2"]

        result = load_profile_pool(cli_pool, yaml_pool, mock_config)

        assert result == cli_pool

    def test_yaml_used_when_no_cli(self, mock_config):
        """Test that YAML pool is used when CLI pool is None."""
        yaml_pool = ["mixed_media", "slideshow_images2"]

        result = load_profile_pool(None, yaml_pool, mock_config)

        assert result == yaml_pool

    def test_all_profiles_when_empty_yaml(self, mock_config):
        """Test that empty YAML list defaults to all available profiles."""
        yaml_pool: list[str] = []

        result = load_profile_pool(None, yaml_pool, mock_config)

        assert set(result) == set(mock_config.video_profiles.keys())

    def test_all_profiles_when_none_yaml(self, mock_config):
        """Test that None YAML defaults to all available profiles."""
        result = load_profile_pool(None, None, mock_config)

        assert set(result) == set(mock_config.video_profiles.keys())

    def test_base_excluded_from_all_profiles_fallback(self, mock_config):
        """Base is the inheritance template, never picked by random selection."""
        mock_config.video_profiles["base"] = Mock()

        result = load_profile_pool(None, None, mock_config)

        assert "base" not in result
        assert "slideshow_images1" in result

    def test_cli_empty_list_overrides_yaml(self, mock_config):
        """Test empty CLI list is respected (returns empty, triggers all later)."""
        cli_pool: list[str] = []
        yaml_pool = ["slideshow_images1"]

        result = load_profile_pool(cli_pool, yaml_pool, mock_config)

        # Empty CLI list is respected - returns empty pool
        # (Caller will handle empty pool by using all profiles)
        assert result == []

    def test_invalid_profile_in_cli_raises_error(self, mock_config):
        """Test that invalid profile in CLI pool raises ValueError."""
        cli_pool = ["slideshow_images1", "invalid_profile"]

        with pytest.raises(ValueError) as exc_info:
            load_profile_pool(cli_pool, None, mock_config)

        error_msg = str(exc_info.value)
        assert "invalid_profile" in error_msg
        assert "Available profiles:" in error_msg

    def test_invalid_profile_in_yaml_raises_error(self, mock_config):
        """Test that invalid profile in YAML pool raises ValueError."""
        yaml_pool = ["slideshow_images1", "nonexistent"]

        with pytest.raises(ValueError) as exc_info:
            load_profile_pool(None, yaml_pool, mock_config)

        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg

    def test_precedence_with_all_sources(self, mock_config):
        """Test precedence when all sources are provided."""
        cli_pool = ["slideshow_images1"]
        yaml_pool = ["video_sequential"]

        result = load_profile_pool(cli_pool, yaml_pool, mock_config)

        # CLI should win
        assert result == cli_pool


class TestProfileUsageTracker:
    """Test profile usage tracking and statistics."""

    def test_initial_state_empty(self):
        """Test that tracker starts with empty counts."""
        tracker = ProfileUsageTracker()

        assert tracker.get_counts() == {}
        assert tracker.format_summary() == "No profile usage recorded"

    def test_record_single_usage(self):
        """Test recording single profile usage."""
        tracker = ProfileUsageTracker()

        tracker.record_usage("slideshow_images1")

        assert tracker.get_counts() == {"slideshow_images1": 1}

    def test_record_multiple_usages_same_profile(self):
        """Test recording multiple usages of same profile."""
        tracker = ProfileUsageTracker()

        tracker.record_usage("slideshow_images1")
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("slideshow_images1")

        assert tracker.get_counts() == {"slideshow_images1": 3}

    def test_record_multiple_profiles(self):
        """Test recording usages across multiple profiles."""
        tracker = ProfileUsageTracker()

        tracker.record_usage("slideshow_images1")
        tracker.record_usage("video_sequential")
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("mixed_media")
        tracker.record_usage("slideshow_images1")

        counts = tracker.get_counts()
        assert counts == {
            "slideshow_images1": 3,
            "video_sequential": 1,
            "mixed_media": 1,
        }

    def test_get_counts_returns_copy(self):
        """Test that get_counts returns a copy (not internal state)."""
        tracker = ProfileUsageTracker()
        tracker.record_usage("slideshow_images1")

        counts = tracker.get_counts()
        counts["slideshow_images1"] = 999
        counts["new_profile"] = 1

        # Original tracker should be unchanged
        assert tracker.get_counts() == {"slideshow_images1": 1}

    def test_format_summary_single_profile(self):
        """Test formatted summary with single profile."""
        tracker = ProfileUsageTracker()
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("slideshow_images1")

        summary = tracker.format_summary()

        assert "Profile Distribution:" in summary
        assert "slideshow_images1: 2 (100.0%)" in summary

    def test_format_summary_multiple_profiles(self):
        """Test formatted summary with multiple profiles."""
        tracker = ProfileUsageTracker()
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("video_sequential")

        summary = tracker.format_summary()

        assert "Profile Distribution:" in summary
        assert "slideshow_images1: 3 (75.0%)" in summary
        assert "video_sequential: 1 (25.0%)" in summary

    def test_format_summary_sorted_by_count(self):
        """Test that summary is sorted by usage count (descending)."""
        tracker = ProfileUsageTracker()
        tracker.record_usage("mixed_media")
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("video_sequential")
        tracker.record_usage("video_sequential")

        summary = tracker.format_summary()
        lines = summary.split("\n")

        # Should be sorted: slideshow_images1 (3), video_sequential (2), mixed_media (1)
        assert "slideshow_images1" in lines[1]
        assert "video_sequential" in lines[2]
        assert "mixed_media" in lines[3]

    def test_format_summary_percentage_calculation(self):
        """Test that percentages are calculated correctly."""
        tracker = ProfileUsageTracker()

        # Record 10 usages with known distribution
        for _ in range(7):
            tracker.record_usage("slideshow_images1")
        for _ in range(2):
            tracker.record_usage("video_sequential")
        tracker.record_usage("mixed_media")

        summary = tracker.format_summary()

        # 7/10 = 70%, 2/10 = 20%, 1/10 = 10%
        assert "70.0%" in summary
        assert "20.0%" in summary
        assert "10.0%" in summary
