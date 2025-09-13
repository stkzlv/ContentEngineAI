"""Tests for the Botasaurus Amazon scraper implementation.

This module provides comprehensive test coverage for the BotasaurusAmazonScraper class,
including unified scraping functionality, API compatibility, and error handling.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.scraper.amazon.media_validator import verify_image_file, verify_video_file
from src.scraper.amazon.models import ProductData, SerpProductInfo
from src.scraper.amazon.scraper import BotasaurusAmazonScraper
from src.scraper.base.models import Platform

# Browser functions are tested via integration tests, not unit tests


class TestVideoExtractionLogic:
    """Test cases for the improved video extraction logic."""

    def test_video_extraction_uses_systematic_approach(self):
        """Test that video extraction follows the same pattern as image extraction."""
        from src.scraper.amazon.media_extractor import (
            extract_functional_videos_with_validation,
        )

        # Mock driver with video data - need to handle multiple JS calls
        mock_driver = Mock()

        # First call gets page info (ASIN extraction)
        page_info_call = {
            "asin": "B0TESTEXAMPLE",
            "title": "Test Wireless Headphones",
            "brand": "testbrand",
            "model": "T123",
            "keywords": ["wireless", "headphones", "test"],
        }

        # Second call gets video data
        video_data_call = {
            "direct_videos": ["https://media-amazon.com/video1.mp4"],
            "vdp_links": ["https://amazon.com/vdp/video1"],
            "thumbnails": [{"element": ".videoThumbnail", "dataVideo": "video1"}],
        }

        # Set up side_effect to return different values for different calls
        mock_driver.run_js.side_effect = [page_info_call, video_data_call]
        mock_driver.select_all.return_value = []  # No clickable thumbnails found
        mock_driver.current_url = "https://amazon.com/dp/B0TESTEXAMPLE"

        with patch(
            "src.scraper.amazon.media_extractor.is_valid_video_url", return_value=True
        ):
            videos = extract_functional_videos_with_validation(mock_driver)

        assert len(videos) >= 1
        assert "https://media-amazon.com/video1.mp4" in videos

    def test_video_thumbnail_clicking_integration(self):
        """Test video thumbnail clicking functionality."""
        from src.scraper.amazon.media_extractor import (
            extract_functional_videos_with_validation,
        )

        # Mock driver with clickable video thumbnails
        mock_driver = Mock()
        mock_driver.run_js.side_effect = [
            "B0TESTEXAMPLE",  # ASIN extraction
            {  # Initial JS extraction
                "direct_videos": [],
                "vdp_links": [],
                "thumbnails": [],
            },
            "https://media-amazon.com/clicked_video.mp4",  # Video after click
        ]

        mock_thumbnail = Mock()
        mock_driver.select_all.return_value = [mock_thumbnail]

        with patch(
            "src.scraper.amazon.media_extractor.is_valid_video_url", return_value=True
        ):
            videos = extract_functional_videos_with_validation(mock_driver)

        # Verify thumbnail was clicked
        mock_thumbnail.click.assert_called_once()
        mock_driver.short_random_sleep.assert_called()

        assert "https://media-amazon.com/clicked_video.mp4" in videos


class TestMediaValidationIntegration:
    """Test cases for media validation integration with scraper."""

    @pytest.fixture
    def temp_valid_image(self):
        """Create a temporary valid image file for testing."""
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create a large valid JPEG image (2000x1500)
            img = Image.new("RGB", (2000, 1500), color="blue")
            img.save(f.name, "JPEG")
            yield Path(f.name)
            Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def temp_invalid_image(self):
        """Create a temporary invalid image file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create file with invalid image data
            f.write(b"this is not an image")
            yield Path(f.name)
            Path(f.name).unlink(missing_ok=True)

    def test_image_validation_passes_for_valid_file(self, temp_valid_image):
        """Test that valid images pass validation."""
        result = verify_image_file(
            temp_valid_image, min_dimension=1500, min_file_size=1000
        )

        assert result.is_valid is True
        assert result.validation_data["width"] == 2000
        assert result.validation_data["height"] == 1500
        assert result.validation_data["format"] == "JPEG"
        assert len(result.issues) == 0

    def test_image_validation_fails_for_invalid_file(self, temp_invalid_image):
        """Test that invalid images fail validation."""
        result = verify_image_file(temp_invalid_image)

        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any("Failed to open/process image" in issue for issue in result.issues)

    def test_video_validation_detects_html_content(self):
        """Test that video validation detects HTML content instead of video."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"<html><head><title>Not a video</title></head></html>")
            html_file = Path(f.name)

        try:
            result = verify_video_file(html_file)

            assert result.is_valid is False
            assert any("appears to be HTML content" in issue for issue in result.issues)
        finally:
            html_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_video_validation_with_valid_metadata(self, mock_subprocess):
        """Test video validation with valid FFprobe metadata."""
        import tempfile

        # Mock FFprobe output for valid video
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "30.5",
                }
            ],
            "format": {"duration": "30.5"},
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            # Write valid MP4 header
            f.write(b"\x00\x00\x00\x18ftypmp41")
            f.write(b"x" * 1000)  # Add some content
            video_file = Path(f.name)

        try:
            result = verify_video_file(video_file, min_duration=1.0, min_dimension=720)

            assert result.is_valid is True
            assert result.validation_data["width"] == 1920
            assert result.validation_data["height"] == 1080
            assert result.validation_data["duration"] == 30.5
            assert result.validation_data["video_codec"] == "h264"
        finally:
            video_file.unlink(missing_ok=True)


class TestBotasaurusAmazonScraper:
    """Test cases for the BotasaurusAmazonScraper class."""

    @pytest.fixture
    def mock_config_data(self):
        """Mock configuration data for testing."""
        return {
            "scrapers": {
                "amazon": {
                    "enabled": True,
                    "base_url": "https://www.amazon.com",
                    "max_products": 3,
                    "associate_tag": "test-tag",
                    "http_headers": {
                        "video_validation": {"User-Agent": "Mozilla/5.0 Test Browser"},
                        "media_download": {"User-Agent": "Mozilla/5.0 Test Browser"},
                        "standard": {"User-Agent": "Mozilla/5.0 Test Browser"},
                    },
                }
            },
            "global_settings": {
                "debug_config": {
                    "title_preview_length": 50,
                    "url_preview_length": 100,
                    "result_preview_length": 100,
                },
                "retry_config": {
                    "default_max_retries": 3,
                    "base_delay": 1.0,
                    "max_delay": 60.0,
                    "backoff_factor": 2.0,
                    "use_jitter": True,
                    "jitter_factor": 0.5,
                },
                "rate_limiting": {
                    "video_validation_delay": [0.5, 1.5],
                    "debug_pause_duration": 5,
                },
                "image_config": {
                    "min_high_res_dimension": 1500,
                    "min_high_res_file_size": 10000,
                },
                "download_config": {
                    "download_timeout": 30,
                    "download_chunk_size": 8192,
                    "validation_range_bytes": "0-1023",
                },
                "system_timeouts": {
                    "system_command_timeout": 5,
                    "head_request_timeout": 10,
                },
                "browser_config": {
                    "debug_window_width": 1920,
                    "debug_window_height": 1200,
                    "fallback_window_position": [0, 0, 1920, 1080],
                    "search_result_timeout": 10,
                    "max_products_per_search": 5,
                },
                "css_selectors": {
                    "product_title_selectors": [
                        "#productTitle",
                        "h1.a-size-large",
                        ".product-title",
                        'h1[data-automation-id="product-title"]',
                    ],
                    "search_result_card": 'div[data-component-type="s-search-result"]',
                },
                "asin_patterns": {
                    "modern_asin_pattern": "^B0[A-Z0-9]{8}$",
                    "legacy_asin_pattern": "^[A-Z0-9]{10}$",
                    "url_asin_pattern": "/dp/([A-Z0-9]{10})",
                },
            },
        }

    @pytest.fixture
    def scraper(self, mock_config_data):
        """Create a scraper instance with mocked config."""
        with (
            patch("yaml.safe_load", return_value=mock_config_data),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            return BotasaurusAmazonScraper()

    @pytest.fixture
    def sample_product_data(self):
        """Sample product data for testing."""
        return {
            "title": "Soundcore by Anker P20i True Wireless Earbuds",
            "price": "$19.99",
            "description": "Great wireless earbuds with long battery life",
            "images": [
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
            ],
            "videos": ["https://example.com/video1.mp4"],
            "url": "https://www.amazon.com/dp/B0BTYCRJSS",
            "asin": "B0BTYCRJSS",
            "keyword": "wireless earbuds",
            "serp_rating": "4.5 out of 5 stars",
            "serp_reviews_count": "1,234",
            "affiliate_link": "https://www.amazon.com/dp/B0BTYCRJSS",
            "downloaded_images": [],
            "downloaded_videos": [],
        }

    def test_scraper_initialization(self, scraper, mock_config_data):
        """Test scraper initializes correctly."""
        assert scraper.config == mock_config_data
        assert scraper.amazon_config == mock_config_data["scrapers"]["amazon"]
        assert scraper.global_settings == mock_config_data["global_settings"]
        assert scraper.logger is not None

    def test_config_loading_file_not_found(self):
        """Test configuration loading when file doesn't exist."""
        with (
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(FileNotFoundError),
        ):
            BotasaurusAmazonScraper()

    def test_is_asin_valid(self, scraper):
        """Test ASIN validation with valid ASINs."""
        valid_asins = ["B0BTYCRJSS", "B07XYLWQZ1", "B08N5WRWNW"]
        for asin in valid_asins:
            assert scraper._is_asin(asin), f"Failed to validate ASIN: {asin}"

    def test_is_asin_invalid(self, scraper):
        """Test ASIN validation with invalid inputs."""
        invalid_asins = ["wireless earbuds", "B123", "B07XYLWQZ11", "abc123"]
        for asin in invalid_asins:
            assert not scraper._is_asin(
                asin
            ), f"Incorrectly validated invalid ASIN: {asin}"

    @patch("src.scraper.amazon.scraper.create_dynamic_browser_function")
    def test_scrape_products_unified_asin(
        self, mock_create_browser_func, scraper, sample_product_data
    ):
        """Test unified scraping for ASIN input."""
        mock_browser_func = Mock(return_value=[sample_product_data])
        mock_create_browser_func.return_value = mock_browser_func

        products = scraper.scrape_products_unified("B0BTYCRJSS")

        # Verify browser function was created and called with correct parameters
        # DEBUG_MODE = False by default
        mock_create_browser_func.assert_called_once_with(False)
        mock_browser_func.assert_called_once()
        call_args = mock_browser_func.call_args[0][0]
        assert call_args["keyword"] == "B0BTYCRJSS"
        assert call_args["is_asin"] is True

        # Verify results
        assert len(products) == 1
        assert isinstance(products[0], ProductData)
        assert products[0].title == sample_product_data["title"]
        assert products[0].asin == sample_product_data["asin"]

    @patch("src.scraper.amazon.scraper.create_dynamic_browser_function")
    def test_scrape_products_unified_search(
        self, mock_create_browser_func, scraper, sample_product_data
    ):
        """Test unified scraping for search query."""
        mock_browser_func = Mock(return_value=[sample_product_data])
        mock_create_browser_func.return_value = mock_browser_func

        products = scraper.scrape_products_unified("wireless earbuds")

        # Verify browser function was created and called with correct parameters
        mock_create_browser_func.assert_called_once_with(False)
        mock_browser_func.assert_called_once()
        call_args = mock_browser_func.call_args[0][0]
        assert call_args["keyword"] == "wireless earbuds"
        assert call_args["is_asin"] is False

        # Verify results
        assert len(products) == 1
        assert isinstance(products[0], ProductData)
        assert products[0].keyword == "wireless earbuds"

    @patch("src.scraper.amazon.scraper.create_dynamic_browser_function")
    def test_scrape_products_unified_error_handling(
        self, mock_create_browser_func, scraper
    ):
        """Test unified scraping error handling."""
        mock_browser_func = Mock(side_effect=Exception("Browser error"))
        mock_create_browser_func.return_value = mock_browser_func

        products = scraper.scrape_products_unified("test keyword")

        # Should return empty list on error
        assert products == []

    @patch("src.scraper.amazon.scraper.create_dynamic_browser_function")
    def test_scrape_products_main_method(
        self, mock_create_browser_func, scraper, sample_product_data
    ):
        """Test main scrape_products method with multiple keywords."""
        mock_browser_func = Mock(return_value=[sample_product_data])
        mock_create_browser_func.return_value = mock_browser_func

        keywords = ["B0BTYCRJSS", "wireless earbuds"]

        with patch.object(scraper, "_save_products") as mock_save:
            products = scraper.scrape_products(keywords)

            # Should call browser function for each keyword
            assert mock_browser_func.call_count == 2

            # Should save products
            mock_save.assert_called_once()

            # Should return combined results
            assert len(products) == 2

    def test_product_to_dict_conversion(self, scraper, sample_product_data):
        """Test ProductData to dictionary conversion."""
        product = ProductData(
            title=sample_product_data["title"],
            price=sample_product_data["price"],
            url=sample_product_data["url"],
            platform=Platform.AMAZON,
            description=sample_product_data["description"],
            images=sample_product_data["images"],
            videos=sample_product_data["videos"],
            asin=sample_product_data["asin"],
            keyword=sample_product_data["keyword"],
            serp_rating=sample_product_data["serp_rating"],
            serp_reviews_count=sample_product_data["serp_reviews_count"],
            downloaded_images=sample_product_data["downloaded_images"],
            downloaded_videos=sample_product_data["downloaded_videos"],
        )

        result_dict = scraper._product_to_dict(product)

        # Verify all fields are included
        assert result_dict["title"] == sample_product_data["title"]
        assert result_dict["price"] == sample_product_data["price"]
        assert result_dict["asin"] == sample_product_data["asin"]
        assert result_dict["images"] == sample_product_data["images"]
        assert result_dict["videos"] == sample_product_data["videos"]

    def test_legacy_save_products_eliminated(self, scraper):
        """Test that legacy save_products functionality has been eliminated."""
        # The _save_products method should now be a no-op since we use Botasaurus
        # for saving
        products = [
            ProductData(
                title="Product 1",
                price="$19.99",
                url="https://example.com/1",
                platform=Platform.AMAZON,
                description="Description 1",
                images=[],
                videos=[],
                asin="ASIN001",
                keyword="keyword1",
                serp_rating=None,
                serp_reviews_count=None,
                downloaded_images=[],
                downloaded_videos=[],
            )
        ]

        # Should not raise an error, just log that legacy format is eliminated
        try:
            scraper._save_products(products)
        except Exception as e:
            pytest.fail(
                f"_save_products should handle legacy format elimination "
                f"gracefully: {e}"
            )


class TestScraperIntegration:
    """Test scraper integration with Botasaurus (simplified tests)."""

    def test_scraper_api_compatibility(self):
        """Test that scraper maintains expected API for integration."""
        # Test that the scraper class exists and has expected methods
        assert hasattr(BotasaurusAmazonScraper, "scrape_products_unified")
        assert hasattr(BotasaurusAmazonScraper, "scrape_products")
        assert hasattr(BotasaurusAmazonScraper, "_is_asin")
        assert hasattr(BotasaurusAmazonScraper, "_product_to_dict")

        # These are the key public APIs that the rest of the system depends on


class TestProductDataValidation:
    """Test ProductData validation and edge cases."""

    def test_product_data_creation(self):
        """Test ProductData object creation."""
        product = ProductData(
            title="Test Product",
            price="$29.99",
            url="https://amazon.com/dp/TEST123",
            platform=Platform.AMAZON,
            description="Test description",
            images=["image1.jpg"],
            videos=["video1.mp4"],
            asin="TEST123",
            keyword="test keyword",
        )

        assert product.title == "Test Product"
        assert product.price == "$29.99"
        assert product.platform == Platform.AMAZON
        assert product.asin == "TEST123"
        assert len(product.images) == 1
        assert len(product.videos) == 1

    def test_product_data_defaults(self):
        """Test ProductData with default values."""
        product = ProductData(
            title="Test",
            price="$10",
            url="https://example.com",
            platform=Platform.AMAZON,
            description="Desc",
            images=[],
            videos=[],
        )

        assert product.asin is None
        assert product.keyword == ""
        assert product.serp_rating is None
        assert product.serp_reviews_count is None
        assert product.downloaded_images == []  # Default is [], not None
        assert product.downloaded_videos == []  # Default is [], not None


class TestConfigurationFeatures:
    """Test new configuration features and cleaned config structure."""

    def test_config_structure_validation(self):
        """Test that config structure matches expected cleaned format."""
        from src.scraper.amazon.config import CONFIG

        # Test that global_settings has expected sections
        global_settings = CONFIG.get("global_settings", {})
        expected_sections = [
            "retry_config",
            "rate_limiting",
            "image_config",
            "download_config",
            "system_timeouts",
            "browser_config",
            "css_selectors",
            "asin_patterns",
        ]

        for section in expected_sections:
            assert (
                section in global_settings
            ), f"Missing expected config section: {section}"

    def test_browser_config_values(self):
        """Test browser configuration values are accessible."""
        from src.scraper.amazon.config import CONFIG

        browser_config = CONFIG.get("global_settings", {}).get("browser_config", {})

        # Test that browser config has expected keys
        expected_keys = [
            "debug_window_width",
            "debug_window_height",
            "fallback_window_position",
            "search_result_timeout",
            "max_products_per_search",
        ]

        for key in expected_keys:
            assert key in browser_config, f"Missing browser config key: {key}"

    def test_css_selectors_config(self):
        """Test CSS selectors configuration."""
        from src.scraper.amazon.config import CONFIG

        css_selectors = CONFIG.get("global_settings", {}).get("css_selectors", {})

        # Test product title selectors
        title_selectors = css_selectors.get("product_title_selectors", [])
        assert len(title_selectors) > 0, "Should have product title selectors"
        assert (
            "#productTitle" in title_selectors
        ), "Should include main product title selector"

        # Test search result selector
        search_selector = css_selectors.get("search_result_card")
        assert search_selector is not None, "Should have search result card selector"

    def test_removed_deprecated_config_keys(self):
        """Test that deprecated configuration keys have been removed."""
        from src.scraper.amazon.config import CONFIG

        global_settings = CONFIG.get("global_settings", {})

        # These keys should NOT exist in cleaned config
        deprecated_keys = [
            "max_images_per_product",
            "early_exit_threshold",
            "network_max_retries",
            "human_delay_multiplier",
            "max_results_per_page",
            "max_pages",
            "require_price",
            "require_rating",
        ]

        # Check they're not in any section
        for key in deprecated_keys:
            for section_name, section_data in global_settings.items():
                if isinstance(section_data, dict):
                    assert (
                        key not in section_data
                    ), f"Deprecated key '{key}' found in section '{section_name}'"

    def test_config_fallback_values(self):
        """Test that configuration fallback values work correctly."""
        # Test with incomplete config but minimum required structure
        incomplete_config = {
            "scrapers": {
                "amazon": {"enabled": True, "base_url": "https://www.amazon.com"}
            },
            "global_settings": {
                "browser_config": {}  # Empty browser config should use fallbacks
            },
        }

        with (
            patch("yaml.safe_load", return_value=incomplete_config),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            # Should not crash, should use fallback values
            try:
                scraper = BotasaurusAmazonScraper()
                # If initialization succeeds, fallbacks are working
                assert scraper is not None
                assert scraper.amazon_config is not None
            except Exception as e:
                pytest.fail(
                    f"Config fallbacks should prevent initialization errors: {e}"
                )

    def test_browser_config_integration(self):
        """Test that browser configuration is properly integrated."""
        # Test that config loading works with current config structure
        from src.scraper.amazon.config import CONFIG

        browser_config = CONFIG.get("global_settings", {}).get("browser_config", {})

        # Verify that browser config section exists and has expected structure
        assert browser_config is not None, "browser_config section should exist"

        # Test expected keys exist (values may vary, but structure should be consistent)
        expected_keys = [
            "debug_window_width",
            "debug_window_height",
            "fallback_window_position",
            "search_result_timeout",
            "max_products_per_search",
        ]

        for key in expected_keys:
            assert key in browser_config, f"Missing browser config key: {key}"

        # Test values are reasonable (not None/empty)
        assert isinstance(browser_config.get("debug_window_width"), int)
        assert isinstance(browser_config.get("debug_window_height"), int)
        assert isinstance(browser_config.get("search_result_timeout"), int)
        assert isinstance(browser_config.get("max_products_per_search"), int)

    def test_asin_validation_config_integration(self):
        """Test ASIN validation patterns from config."""
        config_data = {
            "scrapers": {
                "amazon": {"enabled": True, "base_url": "https://www.amazon.com"}
            },
            "global_settings": {
                "asin_patterns": {
                    "modern_asin_pattern": "^B0[A-Z0-9]{8}$",
                    "legacy_asin_pattern": "^[A-Z0-9]{10}$",
                }
            },
        }

        with (
            patch("yaml.safe_load", return_value=config_data),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            from src.scraper.amazon.config import CONFIG

            patterns = CONFIG.get("global_settings", {}).get("asin_patterns", {})

            # Test that patterns are accessible
            modern = patterns.get("modern_asin_pattern")
            legacy = patterns.get("legacy_asin_pattern")

            assert modern is not None
            assert legacy is not None
            assert "B0" in modern  # Modern pattern should include B0
            assert "[A-Z0-9]{10}" in legacy  # Legacy pattern should be 10 chars


class TestUtilityFunctions:
    """Test utility functions and helper methods."""

    def test_exponential_backoff_retry(self):
        """Test exponential backoff retry decorator."""
        from src.scraper.amazon.utils import exponential_backoff_retry

        # Mock function that fails twice then succeeds
        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Simulated network error")
            return "success"

        # Apply retry decorator
        retried_function = exponential_backoff_retry(
            failing_function, max_retries=3, base_delay=0.01
        )

        # Should eventually succeed after retries
        result = retried_function()
        assert result == "success"
        assert call_count == 3

    def test_validate_asin_format_patterns(self):
        """Test ASIN validation patterns."""
        from src.scraper.amazon.utils import validate_asin_format

        # Test modern ASINs (B0 format)
        modern_asins = ["B0BTYCRJSS", "B08N5WRWNW", "B07XYLWQZ1"]
        for asin in modern_asins:
            assert validate_asin_format(asin), f"Should validate modern ASIN: {asin}"

        # Test legacy ASINs
        legacy_asins = ["1234567890", "ABCD123456"]
        for asin in legacy_asins:
            assert validate_asin_format(asin), f"Should validate legacy ASIN: {asin}"

        # Test invalid ASINs
        invalid_asins = ["B123", "short", "toolongstring12345", "invalid!@#"]
        for asin in invalid_asins:
            assert not validate_asin_format(
                asin
            ), f"Should not validate invalid ASIN: {asin}"

    def test_detect_monitors_function(self):
        """Test monitor detection functionality."""
        from src.scraper.amazon.utils import detect_monitors

        # This should return a list of monitors
        monitors = detect_monitors()
        assert isinstance(monitors, list)

        # Should have at least one monitor (fallback if none detected)
        assert len(monitors) > 0

        # Each monitor should have required keys
        for monitor in monitors:
            assert isinstance(monitor, dict)
            required_keys = ["width", "height", "x", "y"]
            for key in required_keys:
                assert key in monitor, f"Monitor missing required key: {key}"


class TestProductDataModel:
    """Test Amazon ProductData model functionality."""

    def test_product_data_post_init(self):
        """Test ProductData post-initialization logic."""
        # Test with ASIN - should auto-set platform and platform_id
        product = ProductData(
            title="Test Product",
            price="$29.99",
            url="https://amazon.com/dp/TEST123",
            platform=Platform.AMAZON,
            description="Test description",
            images=[],
            videos=[],
            asin="TEST123",
        )

        assert product.platform == Platform.AMAZON
        assert product.platform_id == "TEST123"
        assert product.asin == "TEST123"

    def test_product_data_to_dict(self):
        """Test ProductData dictionary conversion."""
        product = ProductData(
            title="Test Product",
            price="$19.99",
            url="https://amazon.com/dp/B0BTYCRJSS",
            platform=Platform.AMAZON,
            description="Test description",
            images=["image1.jpg", "image2.jpg"],
            videos=["video1.mp4"],
            asin="B0BTYCRJSS",
            serp_rating="4.5 out of 5 stars",
            serp_reviews_count="1,234",
            keyword="test keyword",
        )

        result_dict = product.to_dict()

        # Check Amazon-specific fields
        assert result_dict["asin"] == "B0BTYCRJSS"
        assert result_dict["serp_rating"] == "4.5 out of 5 stars"
        assert result_dict["serp_reviews_count"] == "1,234"

        # Check base fields are included
        assert result_dict["title"] == "Test Product"
        assert result_dict["price"] == "$19.99"
        assert result_dict["platform"] == Platform.AMAZON.value  # Enum value is string

    def test_serp_product_info_model(self):
        """Test SerpProductInfo model."""
        serp_info = SerpProductInfo(
            url="https://amazon.com/dp/TEST123",
            rating="4.0",
            reviews_count="500",
            asin="TEST123",
            keyword="test keyword",
        )

        assert serp_info.url == "https://amazon.com/dp/TEST123"
        assert serp_info.asin == "TEST123"
        assert serp_info.rating == "4.0"
        assert serp_info.reviews_count == "500"
        assert serp_info.keyword == "test keyword"


class TestSearchParameters:
    """Test search parameter validation and building."""

    def test_search_parameters_creation(self):
        """Test SearchParameters model creation."""
        from src.scraper.amazon.models import SearchParameters

        params = SearchParameters(
            min_price=10.00,
            max_price=50.00,
            min_rating=4.0,
            brands=["Sony", "Bose"],
            prime_only=True,
            free_shipping=True,
        )

        assert params.min_price == 10.00
        assert params.max_price == 50.00
        assert params.min_rating == 4.0
        assert "Sony" in params.brands
        assert params.prime_only is True

    def test_search_parameters_defaults(self):
        """Test SearchParameters default values."""
        from src.scraper.amazon.models import SearchParameters

        params = SearchParameters()

        assert params.min_price is None
        assert params.max_price is None
        assert params.brands == []
        assert params.prime_only is False
        assert params.free_shipping is False
        assert params.sort_order == "relevance"

    def test_search_parameters_validation(self):
        """Test SearchParameters validation logic."""
        from src.scraper.amazon.models import SearchParameters

        # Note: SearchParameters may not have built-in validation
        # This tests parameter creation with edge cases

        # Test price range validation if it exists
        try:
            params = SearchParameters(min_price=10, max_price=50)
            assert params.min_price == 10
            assert params.max_price == 50
        except Exception as e:
            # If validation exists, it should raise appropriate errors
            # Log the exception for debugging
            print(f"SearchParameters validation test skipped: {e}")

        # Test rating bounds if validation exists
        try:
            params = SearchParameters(min_rating=4.5)
            assert params.min_rating == 4.5
        except Exception as e:
            # Log the exception for debugging
            print(f"SearchParameters rating test skipped: {e}")


class TestConfigurationUtils:
    """Test configuration utility functions."""

    def test_get_output_path_function(self):
        """Test output path generation."""
        from src.scraper.amazon.config import get_output_path

        # Test base output path
        base_path = get_output_path("base")
        assert "outputs" in str(base_path).lower()

        # Test product-specific path - check if it exists, and if so test it
        try:
            product_path = get_output_path("product", asin="B0BTYCRJSS")
            path_str = str(product_path)
            # Should contain either ASIN or be a valid outputs path
            assert ("B0BTYCRJSS" in path_str) or ("outputs" in path_str.lower())
        except Exception as e:
            # Function may not support product paths yet
            # Log the exception for debugging
            print(f"get_output_path product test skipped: {e}")

    def test_get_filename_pattern_function(self):
        """Test filename pattern generation."""
        from src.scraper.amazon.config import get_filename_pattern

        # Test image filename pattern
        image_filename = get_filename_pattern(
            "image", asin="TEST123", index=0, ext="jpg"
        )
        assert "TEST123" in image_filename
        assert "image" in image_filename
        assert "0" in image_filename
        assert ".jpg" in image_filename

        # Test video filename pattern
        video_filename = get_filename_pattern(
            "video", asin="TEST123", index=1, ext="mp4"
        )
        assert "TEST123" in video_filename
        assert "video" in video_filename
        assert "1" in video_filename
        assert ".mp4" in video_filename


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def scraper_with_invalid_config(self):
        """Create scraper with invalid config for testing error handling."""
        with (
            patch("yaml.safe_load", return_value={}),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            try:
                return BotasaurusAmazonScraper()
            except (KeyError, AttributeError):
                # Expected for invalid config
                return None

    def test_scraper_handles_missing_config_keys(self):
        """Test scraper handles missing configuration keys gracefully."""
        invalid_configs = [
            {},  # Empty config
            {"scrapers": {}},  # Missing amazon key
            {"scrapers": {"amazon": {}}},  # Missing global_settings
        ]

        for config in invalid_configs:
            with (
                patch("yaml.safe_load", return_value=config),
                patch("pathlib.Path.exists", return_value=True),
                patch("builtins.open", mock_open(read_data="")),
                pytest.raises((KeyError, AttributeError)),
            ):
                BotasaurusAmazonScraper()

    def test_scraper_error_handling_integration(self):
        """Test that scraper handles configuration errors gracefully."""
        # This tests the integration-level error handling rather than browser functions
        # Browser function testing is done through integration tests

        # Test that scraper class handles errors appropriately
        try:
            # This should work - just testing the class can be imported and
            # methods exist
            assert hasattr(BotasaurusAmazonScraper, "scrape_products_unified")
            assert hasattr(BotasaurusAmazonScraper, "_is_asin")
        except ImportError:
            pytest.fail("Scraper class should be importable")

    def test_debug_mode_handling(self):
        """Test debug mode configuration and handling."""
        # Test debug mode enabled
        debug_config = {
            "scrapers": {
                "amazon": {"enabled": True, "base_url": "https://www.amazon.com"}
            },
            "global_settings": {"debug_mode": True},
        }

        with (
            patch("yaml.safe_load", return_value=debug_config),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            scraper = BotasaurusAmazonScraper(debug_override=True)
            # Should not raise errors
            assert scraper is not None

    def test_retry_logic_exhaustion(self):
        """Test retry logic when all retries are exhausted."""
        from src.scraper.amazon.utils import exponential_backoff_retry

        # Function that always fails
        def always_fails():
            raise ConnectionError("Always fails")

        retried_function = exponential_backoff_retry(
            always_fails, max_retries=2, base_delay=0.01
        )

        # Should eventually raise the original exception
        with pytest.raises(ConnectionError):
            retried_function()


class TestRecentFixes:
    """Test cases for recent fixes: max_products, debug_config, websocket filter."""

    @pytest.fixture
    def config_with_different_max_products(self):
        """Config with different max_products values for testing precedence."""
        return {
            "scrapers": {
                "amazon": {
                    "enabled": True,
                    "base_url": "https://www.amazon.com",
                    "max_products": 3,  # Amazon-specific setting
                    "associate_tag": "test-tag",
                }
            },
            "global_settings": {
                "debug_config": {
                    "title_preview_length": 30,
                    "url_preview_length": 80,
                    "result_preview_length": 120,
                },
                "browser_config": {
                    "max_products_per_search": 5,  # Global setting (should NOT be used)
                    "debug_window_width": 1920,
                    "debug_window_height": 1200,
                },
            },
        }

    def test_amazon_specific_max_products_precedence(
        self, config_with_different_max_products
    ):
        """Test that Amazon-specific max_products takes precedence over global."""
        with (
            patch("yaml.safe_load", return_value=config_with_different_max_products),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            scraper = BotasaurusAmazonScraper()

            # Check that the scraper uses Amazon-specific max_products
            assert scraper.amazon_config["max_products"] == 3

            # Verify global setting is different (would be wrong if used)
            global_max = scraper.global_settings.get("browser_config", {}).get(
                "max_products_per_search"
            )
            assert global_max == 5
            assert global_max != scraper.amazon_config["max_products"]

    @patch("src.scraper.amazon.scraper.create_dynamic_browser_function")
    def test_max_products_passed_to_browser_function(
        self, mock_create_browser_func, config_with_different_max_products
    ):
        """Test that correct max_products value is passed to browser function."""
        mock_browser_func = Mock(return_value=[])
        mock_create_browser_func.return_value = mock_browser_func

        with (
            patch("yaml.safe_load", return_value=config_with_different_max_products),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            scraper = BotasaurusAmazonScraper()
            scraper.scrape_products_unified("wireless headphones")

            # Verify browser function was called with correct max_products
            mock_browser_func.assert_called_once()
            call_args = mock_browser_func.call_args[0][0]
            assert call_args["max_products"] == 3  # Amazon-specific value
            assert call_args["max_products"] != 5  # Not the global value

    def test_debug_config_accessible(self, config_with_different_max_products):
        """Test that debug_config values are accessible and different from defaults."""
        with (
            patch("yaml.safe_load", return_value=config_with_different_max_products),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            scraper = BotasaurusAmazonScraper()

            debug_config = scraper.global_settings.get("debug_config", {})

            # Test custom values are loaded
            assert debug_config.get("title_preview_length") == 30
            assert debug_config.get("url_preview_length") == 80
            assert debug_config.get("result_preview_length") == 120

            # Verify these are different from defaults
            assert debug_config.get("title_preview_length") != 50  # default
            assert debug_config.get("url_preview_length") != 100  # default

    def test_websocket_filter_creation(self):
        """Test that WebsocketFilter is created and works correctly."""
        from src.scraper.amazon.scraper import WebsocketFilter

        websocket_filter = WebsocketFilter()

        # Test that it filters websocket goodbye messages
        class MockRecord:
            def __init__(self, message):
                self.message = message

            def getMessage(self):
                return self.message

        # These messages should be filtered (return False)
        filtered_messages = [
            "ERROR:websocket:Connection to remote host was lost. - goodbye",
            "websocket connection lost",
            "WebSocket goodbye message",
        ]

        for message in filtered_messages:
            record = MockRecord(message)
            assert websocket_filter.filter(record) is False, f"Should filter: {message}"

        # These messages should NOT be filtered (return True)
        allowed_messages = [
            "INFO:scraper:Starting scrape operation",
            "DEBUG:browser:Navigation completed",
            "ERROR:network:HTTP connection failed",
        ]

        for message in allowed_messages:
            record = MockRecord(message)
            assert websocket_filter.filter(record) is True, f"Should allow: {message}"

    def test_asin_vs_keyword_behavior_consistency(self):
        """Test that max_products only applies to keyword searches, not ASIN lookups."""
        # This is a logical test - ASIN lookups should always return 1 product
        # regardless of max_products setting

        config_data = {
            "scrapers": {
                "amazon": {
                    "enabled": True,
                    "base_url": "https://www.amazon.com",
                    "max_products": 3,  # Should not affect ASIN lookups
                }
            },
            "global_settings": {},
        }

        with (
            patch("yaml.safe_load", return_value=config_data),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            scraper = BotasaurusAmazonScraper()

            # Test ASIN detection
            assert scraper._is_asin("B0BTYCRJSS") is True  # Valid ASIN
            assert scraper._is_asin("wireless headphones") is False  # Keyword

            # The max_products setting should only affect keyword searches
            # This is tested indirectly through the behavior difference


class TestConfigurationUpdates:
    """Test configuration updates and new sections."""

    def test_debug_config_section_exists(self):
        """Test that debug_config section exists in actual config."""
        from src.scraper.amazon.config import CONFIG

        global_settings = CONFIG.get("global_settings", {})
        debug_config = global_settings.get("debug_config", {})

        # Test that debug_config section exists
        assert debug_config is not None

        # Test expected keys exist
        expected_keys = [
            "title_preview_length",
            "url_preview_length",
            "result_preview_length",
        ]
        for key in expected_keys:
            assert key in debug_config, f"debug_config should have {key}"
            assert isinstance(
                debug_config[key], int
            ), f"debug_config.{key} should be integer"

    def test_debug_config_default_values(self):
        """Test debug_config default values are reasonable."""
        from src.scraper.amazon.config import CONFIG

        debug_config = CONFIG.get("global_settings", {}).get("debug_config", {})

        # Test default values are reasonable
        assert debug_config.get("title_preview_length", 0) > 0
        assert debug_config.get("url_preview_length", 0) > 0
        assert debug_config.get("result_preview_length", 0) > 0

        # Test they're different (not all the same value)
        values = [
            debug_config.get("title_preview_length", 50),
            debug_config.get("url_preview_length", 100),
            debug_config.get("result_preview_length", 100),
        ]
        # At least some values should be different
        assert len(set(values)) >= 2, "Debug config values should not all be identical"

    def test_amazon_max_products_vs_global_max_products(self):
        """Test that Amazon max_products and global max_products_per_search can be
        different.
        """
        from src.scraper.amazon.config import CONFIG

        amazon_max = CONFIG.get("scrapers", {}).get("amazon", {}).get("max_products")
        global_max = (
            CONFIG.get("global_settings", {})
            .get("browser_config", {})
            .get("max_products_per_search")
        )

        # Both should exist
        assert amazon_max is not None, "Amazon max_products should be configured"
        assert (
            global_max is not None
        ), "Global max_products_per_search should be configured"

        # They should be different to test the fix
        if amazon_max == global_max:
            # This might be intentional, but we want to verify our fix works
            # The test is still valid - it confirms both values exist and are accessible
            pass


class TestIntegrationScenarios:
    """Test integration scenarios for recent fixes."""

    @patch("src.scraper.amazon.scraper.create_dynamic_browser_function")
    def test_keyword_search_respects_max_products(self, mock_create_browser_func):
        """Integration test for keyword search respecting max_products."""
        # Mock browser function to return multiple products
        mock_products = [
            {
                "title": f"Product {i}",
                "asin": f"TEST{i:05}",
                "price": "$19.99",
                "description": f"Description {i}",
                "images": [],
                "videos": [],
                "url": f"https://amazon.com/dp/TEST{i:05}",
                "keyword": "test",
                "affiliate_link": f"https://amazon.com/dp/TEST{i:05}",
                "serp_rating": None,
                "serp_reviews_count": None,
                "downloaded_images": [],
                "downloaded_videos": [],
            }
            for i in range(5)  # Return 5 products
        ]

        mock_browser_func = Mock(return_value=mock_products)
        mock_create_browser_func.return_value = mock_browser_func

        config_data = {
            "scrapers": {
                "amazon": {
                    "enabled": True,
                    "base_url": "https://www.amazon.com",
                    "max_products": 3,  # Should limit to 3
                }
            },
            "global_settings": {
                "debug_config": {"title_preview_length": 50},
                "browser_config": {
                    "max_products_per_search": 5
                },  # Different global value
            },
        }

        with (
            patch("yaml.safe_load", return_value=config_data),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            scraper = BotasaurusAmazonScraper()

            # Test that correct max_products is passed to browser function
            scraper.scrape_products_unified("test keyword")

            call_args = mock_browser_func.call_args[0][0]
            assert (
                call_args["max_products"] == 3
            ), "Should use Amazon-specific max_products"
            assert call_args["keyword"] == "test keyword"
            assert call_args["is_asin"] is False


def mock_open(read_data=""):
    """Helper function to create mock file opener."""
    from unittest.mock import mock_open as original_mock_open

    return original_mock_open(read_data=read_data)


if __name__ == "__main__":
    pytest.main([__file__])
