"""Tests for enhanced scraper configuration system.

Tests the updated configuration with new browser timeouts, media settings,
and comprehensive documentation added during configuration optimization.
"""

import pytest
import yaml
from pathlib import Path

from src.scraper.amazon.config import (
    get_default_search_parameters,
    get_filename_pattern,
    get_output_path,
)


@pytest.mark.unit
class TestEnhancedScraperConfig:
    """Test enhanced scraper configuration features."""

    @pytest.fixture
    def config_data(self):
        """Load scraper config YAML directly."""
        config_path = Path("config/scraper.yaml")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_browser_timeout_configuration(self, config_data):
        """Test browser timeout configuration values."""
        # Check browser timeout settings
        browser_config = config_data.get("global_settings", {}).get("browser_config", {})

        assert "page_load_timeout_ms" in browser_config
        assert "script_execution_timeout_ms" in browser_config
        assert "element_selection_timeout" in browser_config

        # Verify default values
        assert browser_config["page_load_timeout_ms"] == 60000  # 60 seconds
        assert browser_config["script_execution_timeout_ms"] == 30000  # 30 seconds
        assert browser_config["element_selection_timeout"] == 10

    def test_image_config_max_images(self, config_data):
        """Test max images per product configuration."""
        

        image_config = config_data.get("global_settings", {}).get("image_config", {})

        assert "max_images_per_product" in image_config
        assert image_config["max_images_per_product"] == 10

    def test_media_config_js_context_chars(self, config_data):
        """Test JavaScript context character limit configuration."""
        

        media_config = config_data.get("global_settings", {}).get("media_config", {})

        assert "js_context_chars" in media_config
        assert media_config["js_context_chars"] == 500

    def test_media_config_valid_http_codes(self, config_data):
        """Test valid HTTP status codes configuration."""
        

        media_config = config_data.get("global_settings", {}).get("media_config", {})

        assert "valid_http_status_codes" in media_config
        assert media_config["valid_http_status_codes"] == [200, 206]

    def test_validation_config_media_timeout(self, config_data):
        """Test media validation timeout configuration."""
        

        validation_config = config_data.get("global_settings", {}).get(
            "validation_config", {}
        )

        assert "media_validation_timeout" in validation_config
        assert validation_config["media_validation_timeout"] == 30

    def test_validation_config_report_top_issues(self, config_data):
        """Test validation report issue limit configuration."""
        

        validation_config = config_data.get("global_settings", {}).get(
            "validation_config", {}
        )

        assert "validation_report_top_issues" in validation_config
        assert validation_config["validation_report_top_issues"] == 10

    def test_browser_config_selector_limits(self, config_data):
        """Test browser selector attempt limits."""
        

        browser_config = config_data.get("global_settings", {}).get("browser_config", {})

        assert "max_title_selector_attempts" in browser_config
        assert browser_config["max_title_selector_attempts"] == 10

        assert "search_result_wait_reduced" in browser_config
        assert browser_config["search_result_wait_reduced"] == 5

    def test_media_config_min_file_size_absolute(self, config_data):
        """Test absolute minimum file size configuration."""
        

        media_config = config_data.get("global_settings", {}).get("media_config", {})

        assert "min_file_size_absolute" in media_config
        assert media_config["min_file_size_absolute"] == 1000  # 1KB

    def test_retry_config_comprehensive(self, config_data):
        """Test comprehensive retry configuration."""
        

        retry_config = config_data.get("global_settings", {}).get("retry_config", {})

        # Check all retry parameters
        assert retry_config["default_max_retries"] == 3
        assert retry_config["base_delay"] == 1.0
        assert retry_config["max_delay"] == 60.0
        assert retry_config["backoff_factor"] == 2.0
        assert retry_config["use_jitter"] is True
        assert retry_config["jitter_factor"] == 0.5

    def test_rate_limiting_video_validation_delay(self, config_data):
        """Test video validation delay range configuration."""
        

        rate_limiting = config_data.get("global_settings", {}).get("rate_limiting", {})

        assert "video_validation_delay" in rate_limiting
        delay_range = rate_limiting["video_validation_delay"]
        assert isinstance(delay_range, list)
        assert len(delay_range) == 2
        assert delay_range[0] == 0.5
        assert delay_range[1] == 1.5

    def test_debug_config_string_truncation(self, config_data):
        """Test debug output string truncation configuration."""
        

        debug_config = config_data.get("global_settings", {}).get("debug_config", {})

        assert debug_config["title_preview_length"] == 50
        assert debug_config["url_preview_length"] == 100
        assert debug_config["result_preview_length"] == 100

    def test_output_path_configuration(self, config_data):
        """Test output path generation with configuration."""
        # Test base path
        base_path = get_output_path("base")
        assert base_path is not None
        assert isinstance(base_path, str)

        # Test botasaurus path
        botasaurus_path = get_output_path("botasaurus")
        assert botasaurus_path is not None
        assert isinstance(botasaurus_path, str)

    def test_filename_pattern_configuration(self, config_data):
        """Test filename pattern generation with configuration."""
        # Test product file pattern
        product_file = get_filename_pattern(
            "product", keyword="test", ext="json"
        )
        assert "test" in product_file
        assert product_file.endswith("json")

        # Test image file pattern
        image_file = get_filename_pattern(
            "image", asin="B0TEST123", index=1, ext="jpg"
        )
        assert "B0TEST123" in image_file
        assert "1" in image_file
        assert image_file.endswith("jpg")

        # Test video file pattern
        video_file = get_filename_pattern(
            "video", asin="B0TEST123", index=2, ext="mp4"
        )
        assert "B0TEST123" in video_file
        assert "2" in video_file
        assert video_file.endswith("mp4")

    def test_default_search_parameters(self, config_data):
        """Test default search parameters retrieval."""
        params = get_default_search_parameters()

        assert params is not None
        assert hasattr(params, "min_price")
        assert hasattr(params, "max_price")
        assert hasattr(params, "prime_only")
        assert hasattr(params, "sort_order")

    def test_config_fallback_handling(self, config_data):
        """Test configuration fallback mechanisms."""
        # Test output path with invalid type
        fallback_path = get_output_path("invalid_type")
        assert fallback_path is not None
        assert isinstance(fallback_path, str)

        # Test filename pattern with missing parameters
        fallback_filename = get_filename_pattern("unknown_type")
        assert fallback_filename is not None
        assert isinstance(fallback_filename, str)


@pytest.mark.unit
class TestConfigurationDocumentation:
    """Test that configuration is well-documented."""

    @pytest.fixture
    def config_data(self):
        """Load scraper config YAML directly."""
        config_path = Path("config/scraper.yaml")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_browser_config_has_all_documented_settings(self, config_data):
        """Test that all documented browser settings are present."""
        

        browser_config = config_data.get("global_settings", {}).get("browser_config", {})

        # Settings mentioned in documentation
        documented_settings = [
            "debug_window_width",
            "debug_window_height",
            "fallback_window_position",
            "search_result_timeout",
            "max_products_per_search",
            "page_load_timeout_ms",
            "script_execution_timeout_ms",
            "element_selection_timeout",
            "max_title_selector_attempts",
            "search_result_wait_reduced",
        ]

        for setting in documented_settings:
            assert setting in browser_config, f"Missing setting: {setting}"

    def test_media_config_comprehensive_settings(self, config_data):
        """Test that media config has comprehensive settings."""
        

        media_config = config_data.get("global_settings", {}).get("media_config", {})

        comprehensive_settings = [
            "default_image_extension",
            "amazon_media_domains",
            "amazon_high_res_suffix",
            "high_res_upgrade_dimension",
            "js_context_chars",
            "valid_http_status_codes",
            "min_file_size_absolute",
        ]

        for setting in comprehensive_settings:
            assert setting in media_config, f"Missing media setting: {setting}"

    def test_validation_config_complete(self, config_data):
        """Test that validation config is complete."""
        

        validation_config = config_data.get("global_settings", {}).get(
            "validation_config", {}
        )

        required_settings = [
            "min_images_required",
            "min_videos_required",
            "min_total_media_files",
            "media_validation_timeout",
            "validation_report_top_issues",
        ]

        for setting in required_settings:
            assert (
                setting in validation_config
            ), f"Missing validation setting: {setting}"


@pytest.mark.unit
class TestConfigurationValues:
    """Test configuration value types and ranges."""

    @pytest.fixture
    def config_data(self):
        """Load scraper config YAML directly."""
        config_path = Path("config/scraper.yaml")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_timeout_values_are_positive(self, config_data):
        """Test that all timeout values are positive numbers."""
        

        # Browser timeouts
        browser_config = config_data.get("global_settings", {}).get("browser_config", {})
        assert browser_config["page_load_timeout_ms"] > 0
        assert browser_config["script_execution_timeout_ms"] > 0
        assert browser_config["element_selection_timeout"] > 0

        # System timeouts
        system_timeouts = config_data.get("global_settings", {}).get(
            "system_timeouts", {}
        )
        assert system_timeouts["system_command_timeout"] > 0
        assert system_timeouts["head_request_timeout"] > 0

        # Download timeouts
        download_config = config_data.get("global_settings", {}).get(
            "download_config", {}
        )
        assert download_config["download_timeout"] > 0

    def test_dimension_thresholds_are_reasonable(self, config_data):
        """Test that dimension thresholds are reasonable values."""
        

        image_config = config_data.get("global_settings", {}).get("image_config", {})

        # Image dimensions should be reasonable (HD and above)
        assert image_config["min_high_res_dimension"] >= 1000
        assert image_config["very_high_res_dimension"] >= 1500

        # Video dimensions
        video_config = config_data.get("global_settings", {}).get("video_config", {})
        assert video_config["min_dimension"] >= 480  # At least SD quality

    def test_retry_config_exponential_backoff_valid(self, config_data):
        """Test that retry configuration supports valid exponential backoff."""
        

        retry_config = config_data.get("global_settings", {}).get("retry_config", {})

        # Backoff factor should be > 1 for exponential growth
        assert retry_config["backoff_factor"] > 1.0

        # Base delay should be less than max delay
        assert retry_config["base_delay"] < retry_config["max_delay"]

        # Jitter factor should be between 0 and 1
        assert 0 < retry_config["jitter_factor"] <= 1.0

    def test_validation_requirements_are_logical(self, config_data):
        """Test that validation requirements are logically consistent."""
        

        validation_config = config_data.get("global_settings", {}).get(
            "validation_config", {}
        )

        # At least some media should be required
        min_images = validation_config["min_images_required"]
        min_videos = validation_config["min_videos_required"]
        min_total = validation_config["min_total_media_files"]

        # Total should be at least as much as individual requirements
        assert min_total >= min_images
        assert min_total >= min_videos
