"""Integration tests for batch profile randomization.

These tests validate the complete batch profile randomization workflow including:
- End-to-end batch processing with random profile selection
- CLI vs YAML configuration precedence for profile pools
- Profile usage distribution in summary output
- Product discovery and deterministic selection

Tests use mocked video creation to avoid actual processing while testing
the complete batch orchestration flow.

Run with: pytest tests/video/producer/test_batch_profile_integration.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from src.video.config import VideoConfig
from src.video.producer.cli import discover_products_for_batch
from src.video.producer.utils import (
    ProfileUsageTracker,
    load_profile_pool,
    select_profile_for_product,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def temp_outputs_dir():
    """Create temporary outputs directory with test products."""
    with tempfile.TemporaryDirectory(prefix="test_outputs_") as temp_dir:
        outputs_path = Path(temp_dir)

        # Create test product directories with data.json files
        for product_id in ["B0TEST001A", "B0TEST002B", "B0TEST003C"]:
            product_dir = outputs_path / product_id
            product_dir.mkdir()

            data_file = product_dir / "data.json"
            data_file.write_text(
                f'{{"asin": "{product_id}", "title": "Test Product {product_id}", '
                f'"price": "$29.99", "url": "https://amazon.com/dp/{product_id}", '
                f'"platform": "amazon", '
                f'"images": ["img1.jpg", "img2.jpg"], "videos": ["vid1.mp4"]}}'
            )

        yield outputs_path


@pytest.fixture
def temp_config_file():
    """Create temporary YAML config file for testing."""
    with tempfile.TemporaryDirectory(prefix="test_config_") as temp_dir:
        config_path = Path(temp_dir) / "video_production.yaml"

        config_data = {
            "batch": {"profile_pool": ["slideshow_images1", "video_sequential"]}
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        yield config_path


@pytest.fixture
def mock_video_config():
    """Create mock VideoConfig with test profiles."""
    config = Mock(spec=VideoConfig)
    config.video_profiles = {
        "slideshow_images1": Mock(),
        "video_sequential": Mock(),
        "mixed_media": Mock(),
        "slideshow_images2": Mock(),
    }
    config.batch = {"profile_pool": ["slideshow_images1", "video_sequential"]}
    return config


class TestProfilePoolPrecedence:
    """Test configuration precedence for profile pools."""

    def test_cli_pool_overrides_yaml(self, mock_video_config):
        """Test that CLI --profile-pool overrides YAML configuration."""
        cli_pool = ["mixed_media", "slideshow_images2"]
        yaml_pool = ["slideshow_images1", "video_sequential"]

        result = load_profile_pool(cli_pool, yaml_pool, mock_video_config)

        assert result == cli_pool

    def test_yaml_pool_used_when_no_cli(self, mock_video_config):
        """Test that YAML pool is used when no CLI override."""
        yaml_pool = ["slideshow_images1", "video_sequential"]

        result = load_profile_pool(None, yaml_pool, mock_video_config)

        assert result == yaml_pool

    def test_all_profiles_when_neither_specified(self, mock_video_config):
        """Test that all profiles are used when neither CLI nor YAML specified."""
        result = load_profile_pool(None, None, mock_video_config)

        assert set(result) == set(mock_video_config.video_profiles.keys())


class TestProfileSelectionDeterminism:
    """Test deterministic profile selection across batch."""

    def test_same_product_gets_same_profile_across_runs(self, mock_video_config):
        """Test that the same product ID consistently gets the same profile."""
        product_id = "B0TEST001A"
        profile_pool = ["slideshow_images1", "video_sequential", "mixed_media"]

        # Run selection 5 times
        results = [
            select_profile_for_product(product_id, profile_pool, mock_video_config)
            for _ in range(5)
        ]

        # All results should be identical
        assert len(set(results)) == 1

    def test_different_products_can_get_different_profiles(self, mock_video_config):
        """Test that different products can get different profiles."""
        profile_pool = ["slideshow_images1", "video_sequential", "mixed_media"]
        product_ids = [f"B0TEST{i:03d}A" for i in range(20)]

        results = [
            select_profile_for_product(pid, profile_pool, mock_video_config)
            for pid in product_ids
        ]

        # Should see distribution across profiles
        unique_profiles = set(results)
        assert len(unique_profiles) > 1


class TestProfileUsageTracking:
    """Test profile usage statistics tracking."""

    def test_usage_tracker_records_selections(self):
        """Test that ProfileUsageTracker correctly records profile usage."""
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

    def test_usage_summary_format(self):
        """Test that usage summary is formatted correctly."""
        tracker = ProfileUsageTracker()

        tracker.record_usage("slideshow_images1")
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("slideshow_images1")
        tracker.record_usage("video_sequential")

        summary = tracker.format_summary()

        assert "Profile Distribution:" in summary
        assert "slideshow_images1: 3 (75.0%)" in summary
        assert "video_sequential: 1 (25.0%)" in summary

    def test_summary_sorted_by_count(self):
        """Test that summary is sorted by usage count descending."""
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


class TestProductDiscovery:
    """Test product discovery for batch processing."""

    def test_discover_products_finds_valid_products(self, temp_outputs_dir):
        """Test that product discovery finds all valid products."""
        products = discover_products_for_batch(temp_outputs_dir)

        assert len(products) == 3

        product_ids = {product.asin for _, product in products}
        assert product_ids == {"B0TEST001A", "B0TEST002B", "B0TEST003C"}

    def test_discover_products_skips_global_dirs(self, temp_outputs_dir):
        """Test that global directories are skipped during discovery."""
        # Create global directories
        (temp_outputs_dir / "cache").mkdir()
        (temp_outputs_dir / "logs").mkdir()
        (temp_outputs_dir / "coverage").mkdir()

        products = discover_products_for_batch(temp_outputs_dir)

        # Should still only find the 3 test products
        assert len(products) == 3

    def test_discover_products_skips_invalid_json(self, temp_outputs_dir):
        """Test that products with invalid data.json are skipped."""
        # Create product with invalid JSON
        invalid_dir = temp_outputs_dir / "B0INVALID1"
        invalid_dir.mkdir()
        (invalid_dir / "data.json").write_text("invalid json content {")

        products = discover_products_for_batch(temp_outputs_dir)

        # Should skip the invalid product
        assert len(products) == 3
        product_ids = {product.asin for _, product in products}
        assert "B0INVALID1" not in product_ids


class TestBatchProfileRandomizationEndToEnd:
    """Integration tests for complete batch flow with profile randomization."""

    def test_batch_with_random_profiles(self, temp_outputs_dir, mock_video_config):
        """Test end-to-end batch processing with random profile selection."""
        # Simulate batch loop logic (simplified)
        profile_pool = ["slideshow_images1", "video_sequential", "mixed_media"]
        tracker = ProfileUsageTracker()
        products = discover_products_for_batch(temp_outputs_dir)

        for _product_dir, product in products:
            selected_profile = select_profile_for_product(
                product.asin, profile_pool, mock_video_config
            )
            tracker.record_usage(selected_profile)

        # Verify tracking worked
        counts = tracker.get_counts()
        total_selections = sum(counts.values())
        assert total_selections == 3

        # Verify all selected profiles were from the pool
        for profile in counts:
            assert profile in profile_pool

    def test_deterministic_selection_across_batch_run(
        self, temp_outputs_dir, mock_video_config
    ):
        """Test that profile selection is deterministic for each product."""
        profile_pool = ["slideshow_images1", "video_sequential"]
        products = discover_products_for_batch(temp_outputs_dir)

        # First run
        first_run_selections = {}
        for _product_dir, product in products:
            selected = select_profile_for_product(
                product.asin, profile_pool, mock_video_config
            )
            first_run_selections[product.asin] = selected

        # Second run
        second_run_selections = {}
        for _product_dir, product in products:
            selected = select_profile_for_product(
                product.asin, profile_pool, mock_video_config
            )
            second_run_selections[product.asin] = selected

        # Selections should be identical across runs
        assert first_run_selections == second_run_selections


class TestValidationErrors:
    """Test validation error handling."""

    def test_invalid_profile_in_pool_raises_error(self, mock_video_config):
        """Test that invalid profiles in pool are caught before processing."""
        invalid_pool = ["slideshow_images1", "nonexistent_profile"]

        with pytest.raises(ValueError) as exc_info:
            load_profile_pool(invalid_pool, None, mock_video_config)

        error_msg = str(exc_info.value)
        assert "nonexistent_profile" in error_msg
        assert "Available profiles:" in error_msg

    def test_empty_pool_raises_error(self, mock_video_config):
        """Test that empty profile pool is caught during selection."""
        product_id = "B0TEST001A"
        empty_pool: list[str] = []

        with pytest.raises(ValueError, match="Profile pool cannot be empty"):
            select_profile_for_product(product_id, empty_pool, mock_video_config)

    def test_validation_shows_available_profiles(self, mock_video_config):
        """Test that validation errors show available profiles."""
        invalid_pool = ["invalid1", "invalid2"]

        with pytest.raises(ValueError) as exc_info:
            load_profile_pool(invalid_pool, None, mock_video_config)

        error_msg = str(exc_info.value)
        # Should list available profiles
        assert "slideshow_images1" in error_msg
        assert "video_sequential" in error_msg
        assert "mixed_media" in error_msg
