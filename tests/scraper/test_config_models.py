"""Tests for Pydantic scraper configuration models.

Tests validate type-safe configuration models, field constraints, and defaults.
"""

import pytest
from pydantic import ValidationError

from src.scraper.config_models import (
    AmazonScraperConfig,
    ASINPatterns,
    BrowserConfig,
    CSSSelectors,
    DebugConfig,
    DebugSettings,
    DownloadConfig,
    FilterParameters,
    HTTPHeaders,
    ImageConfig,
    MediaConfig,
    OutputConfig,
    RateLimitingConfig,
    RetryConfig,
    ScraperConfig,
    SearchParameters,
    SystemTimeouts,
    ValidationConfig,
    VideoConfig,
)

pytestmark = pytest.mark.unit


class TestRetryConfig:
    """Test retry configuration model."""

    def test_retry_config_defaults(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.default_max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
        assert config.use_jitter is True
        assert config.jitter_factor == 0.5

    def test_retry_config_custom_values(self):
        """Test custom retry configuration values."""
        config = RetryConfig(
            default_max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            backoff_factor=3.0,
        )
        assert config.default_max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_factor == 3.0

    def test_retry_config_validation(self):
        """Test retry configuration field validation."""
        # Negative max_retries should fail
        with pytest.raises(ValidationError):
            RetryConfig(default_max_retries=-1)

        # Zero base_delay should fail (gt=0)
        with pytest.raises(ValidationError):
            RetryConfig(base_delay=0)

        # Invalid jitter_factor (must be 0-1)
        with pytest.raises(ValidationError):
            RetryConfig(jitter_factor=1.5)


class TestDownloadConfig:
    """Test download configuration model."""

    def test_download_config_defaults(self):
        """Test default download configuration values."""
        config = DownloadConfig()
        assert config.download_timeout == 30
        assert config.video_download_timeout == 300
        assert config.retry_video_downloads == 2
        assert config.download_chunk_size == 8192
        assert config.validation_range_bytes == "0-1023"
        assert config.concurrent_image_downloads == 5
        assert config.concurrent_video_downloads == 3

    def test_download_config_custom_values(self):
        """Test custom download configuration values."""
        config = DownloadConfig(
            download_timeout=60,
            video_download_timeout=600,
            concurrent_image_downloads=10,
            concurrent_video_downloads=5,
        )
        assert config.download_timeout == 60
        assert config.video_download_timeout == 600
        assert config.concurrent_image_downloads == 10
        assert config.concurrent_video_downloads == 5

    def test_download_config_validation(self):
        """Test download configuration field validation."""
        # Zero timeout should fail (gt=0)
        with pytest.raises(ValidationError):
            DownloadConfig(download_timeout=0)

        # Negative concurrent downloads should fail
        with pytest.raises(ValidationError):
            DownloadConfig(concurrent_image_downloads=0)


class TestVideoConfig:
    """Test video configuration model."""

    def test_video_config_defaults(self):
        """Test default video configuration values."""
        config = VideoConfig()
        assert config.min_dimension == 640
        assert config.min_duration == 1.0
        assert config.max_videos_per_product == 10
        assert config.mute_video_tabs is True
        assert config.enable_metadata_extraction is True
        assert config.enable_m3u8_monitoring is False
        assert config.m3u8_download_timeout == 120
        assert config.network_capture_timeout == 20

    def test_video_config_custom_values(self):
        """Test custom video configuration values."""
        config = VideoConfig(
            min_dimension=1080,
            min_duration=2.0,
            enable_m3u8_monitoring=True,
        )
        assert config.min_dimension == 1080
        assert config.min_duration == 2.0
        assert config.enable_m3u8_monitoring is True

    def test_video_config_validation(self):
        """Test video configuration field validation."""
        # Zero min_dimension should fail
        with pytest.raises(ValidationError):
            VideoConfig(min_dimension=0)

        # Zero min_duration should fail
        with pytest.raises(ValidationError):
            VideoConfig(min_duration=0)


class TestSearchParameters:
    """Test search parameters model."""

    def test_search_parameters_defaults(self):
        """Test default search parameter values."""
        params = SearchParameters()
        assert params.min_price is None
        assert params.max_price is None
        assert params.min_rating is None
        assert params.prime_only is False
        assert params.free_shipping is False
        assert params.brands == []
        assert params.sort_order == "relevanceblender"
        assert params.category is None
        assert params.include_sponsored is False
        assert params.skip_unavailable is True

    def test_search_parameters_custom_values(self):
        """Test custom search parameter values."""
        params = SearchParameters(
            min_price=10.0,
            max_price=100.0,
            min_rating=4.0,
            prime_only=True,
            brands=["Apple", "Samsung"],
        )
        assert params.min_price == 10.0
        assert params.max_price == 100.0
        assert params.min_rating == 4.0
        assert params.prime_only is True
        assert params.brands == ["Apple", "Samsung"]

    def test_search_parameters_validation(self):
        """Test search parameter field validation."""
        # Negative price should fail
        with pytest.raises(ValidationError):
            SearchParameters(min_price=-10.0)

        # Rating out of range should fail
        with pytest.raises(ValidationError):
            SearchParameters(min_rating=6.0)


class TestAmazonScraperConfig:
    """Test Amazon scraper configuration model."""

    def test_amazon_config_defaults(self):
        """Test default Amazon configuration values."""
        config = AmazonScraperConfig()
        assert config.enabled is True
        assert config.base_url == "https://www.amazon.com"
        assert config.keywords == ["keyboard"]
        assert config.max_products == 2
        assert config.associate_tag == ""
        assert isinstance(config.default_search_parameters, SearchParameters)
        assert isinstance(config.filter_parameters, FilterParameters)
        assert isinstance(config.http_headers, HTTPHeaders)

    def test_amazon_config_custom_values(self):
        """Test custom Amazon configuration values."""
        config = AmazonScraperConfig(
            keywords=["mouse", "headphones"],
            max_products=5,
        )
        assert config.keywords == ["mouse", "headphones"]
        assert config.max_products == 5

    def test_amazon_config_validation(self):
        """Test Amazon configuration field validation."""
        # Zero max_products should fail
        with pytest.raises(ValidationError):
            AmazonScraperConfig(max_products=0)

    def test_affiliate_links_defaults_to_enabled(self):
        """A config that says nothing about affiliate links still warns.

        Defaulting to disabled would turn a forgotten tag into silent revenue
        loss instead of the WARN that exists to catch it.
        """
        assert AmazonScraperConfig().affiliate_links.enabled is True

    def test_affiliate_links_parses_from_config(self):
        """The typed path must carry the flag, not drop it.

        `extra="ignore"` is Pydantic's default, so before this field existed
        the block was silently discarded here while the runtime read it from
        the raw dict.
        """
        config = AmazonScraperConfig(affiliate_links={"enabled": False})
        assert config.affiliate_links.enabled is False

    def test_affiliate_links_rejects_unknown_key(self):
        """A typo inside the block must fail loudly, not do nothing."""
        with pytest.raises(ValidationError):
            AmazonScraperConfig(affiliate_links={"enabld": False})


class TestScraperConfig:
    """Test top-level scraper configuration model."""

    def test_scraper_config_structure(self):
        """Test scraper configuration structure."""
        config = ScraperConfig()
        assert hasattr(config, "global_settings")
        assert hasattr(config, "amazon")
        assert hasattr(config.global_settings, "debug_mode")
        assert hasattr(config.global_settings, "download_config")
        assert hasattr(config.global_settings, "video_config")
        assert config.amazon.enabled is True

    def test_scraper_config_nested_access(self):
        """Test nested configuration access."""
        config = ScraperConfig()
        # Access nested download config
        assert config.global_settings.download_config.concurrent_image_downloads == 5
        assert config.global_settings.download_config.concurrent_video_downloads == 3
        # Access nested video config
        assert config.global_settings.video_config.min_dimension == 640
        assert config.global_settings.video_config.enable_m3u8_monitoring is False

    def test_scraper_config_custom_values(self):
        """Test scraper configuration with custom values."""
        from src.scraper.config_models import GlobalScraperSettings

        global_settings = GlobalScraperSettings(debug_mode=True)
        amazon_config = AmazonScraperConfig(max_products=10)

        config = ScraperConfig(
            global_settings=global_settings,
            amazon=amazon_config,
        )

        assert config.global_settings.debug_mode is True
        assert config.amazon.max_products == 10


class TestBrowserConfig:
    """Test browser configuration model."""

    def test_browser_config_defaults(self):
        """Test default browser configuration values."""
        config = BrowserConfig()
        assert config.debug_window_width == 1920
        assert config.debug_window_height == 1200
        assert config.fallback_window_position == [0, 0, 1920, 1080]
        assert config.search_result_timeout == 10
        assert config.max_products_per_search == 5

    def test_browser_config_validation(self):
        """Test browser configuration field validation."""
        # Zero width should fail
        with pytest.raises(ValidationError):
            BrowserConfig(debug_window_width=0)


class TestValidationConfig:
    """Test validation configuration model."""

    def test_validation_config_defaults(self):
        """Test default validation configuration values."""
        config = ValidationConfig()
        assert config.essential_fields == []
        assert config.min_total_media == 3
        assert config.min_images_if_no_video == 5
        assert config.min_images_with_video == 2
        assert config.media_validation_timeout == 30
        assert config.validation_report_top_issues == 10

    def test_validation_config_custom_values(self):
        """Test custom validation configuration values."""
        config = ValidationConfig(
            essential_fields=["title", "price"],
            min_total_media=5,
        )
        assert config.essential_fields == ["title", "price"]
        assert config.min_total_media == 5


class TestCSSSelectors:
    """Test CSS selectors model."""

    def test_css_selectors_defaults(self):
        """Test default CSS selector values."""
        selectors = CSSSelectors()
        assert "#productTitle" in selectors.product_title_selectors
        assert "h1.a-size-large" in selectors.product_title_selectors
        assert (
            selectors.search_result_card == "div[data-component-type='s-search-result']"
        )

    def test_css_selectors_custom_values(self):
        """Test custom CSS selector values."""
        custom_selectors = ["#custom-title", ".custom-class"]
        selectors = CSSSelectors(
            product_title_selectors=custom_selectors,
        )
        assert selectors.product_title_selectors == custom_selectors


class TestASINPatterns:
    """Test ASIN patterns model."""

    def test_asin_patterns_defaults(self):
        """Test default ASIN pattern values."""
        patterns = ASINPatterns()
        assert patterns.modern_asin_pattern == "^B0[A-Z0-9]{8}$"
        assert patterns.legacy_asin_pattern == "^[A-Z0-9]{10}$"
        assert patterns.url_asin_pattern == "/dp/([A-Z0-9]{10})"


class TestHTTPHeaders:
    """Test HTTP headers model."""

    def test_http_headers_defaults(self):
        """Test default HTTP header values."""
        headers = HTTPHeaders()
        assert "User-Agent" in headers.video_validation
        assert "User-Agent" in headers.media_download
        assert "User-Agent" in headers.standard
        assert "Mozilla" in headers.standard["User-Agent"]


class TestOutputConfig:
    """Test output configuration model."""

    def test_output_config_defaults(self):
        """Test default output configuration values."""
        config = OutputConfig()
        assert config.base_directory == "outputs"
        assert "product_file" in config.file_patterns
        assert "image_file" in config.file_patterns
        assert "video_file" in config.file_patterns

    def test_output_config_custom_values(self):
        """Test custom output configuration values."""
        config = OutputConfig(
            base_directory="custom_outputs",
        )
        assert config.base_directory == "custom_outputs"


class TestImageConfig:
    """Test image configuration model."""

    def test_image_config_defaults(self):
        """Test default image configuration values."""
        config = ImageConfig()
        assert config.min_high_res_dimension == 1500
        assert config.min_high_res_file_size == 10000
        assert config.very_high_res_dimension == 2000
        assert config.max_images_per_product == 10

    def test_image_config_validation(self):
        """Test image configuration field validation."""
        # Zero dimension should fail
        with pytest.raises(ValidationError):
            ImageConfig(min_high_res_dimension=0)


class TestMediaConfig:
    """Test media configuration model."""

    def test_media_config_defaults(self):
        """Test default media configuration values."""
        config = MediaConfig()
        assert config.default_image_extension == ".jpg"
        assert "images-amazon.com" in config.amazon_media_domains
        assert config.amazon_high_res_suffix == "._AC_SL2000_.jpg"
        assert config.high_res_upgrade_dimension == 2000

    def test_media_config_custom_values(self):
        """Test custom media configuration values."""
        config = MediaConfig(
            default_image_extension=".png",
            amazon_media_domains=["custom.com"],
        )
        assert config.default_image_extension == ".png"
        assert config.amazon_media_domains == ["custom.com"]


class TestRateLimitingConfig:
    """Test rate limiting configuration model."""

    def test_rate_limiting_defaults(self):
        """Test default rate limiting configuration values."""
        config = RateLimitingConfig()
        assert config.video_validation_delay == [0.5, 1.5]
        assert config.debug_pause_duration == 5

    def test_rate_limiting_validation(self):
        """Test rate limiting configuration field validation."""
        # Negative pause duration should fail
        with pytest.raises(ValidationError):
            RateLimitingConfig(debug_pause_duration=-1)


class TestDebugSettings:
    """Test debug settings model."""

    def test_debug_settings_defaults(self):
        """Test default debug settings values."""
        settings = DebugSettings()
        assert settings.create_media_validation_reports is True
        assert settings.save_screenshots is False
        assert settings.save_error_screenshots is True


class TestDebugConfig:
    """Test debug configuration model."""

    def test_debug_config_defaults(self):
        """Test default debug configuration values."""
        config = DebugConfig()
        assert config.title_preview_length == 50
        assert config.url_preview_length == 100
        assert config.result_preview_length == 100

    def test_debug_config_validation(self):
        """Test debug configuration field validation."""
        # Zero length should fail
        with pytest.raises(ValidationError):
            DebugConfig(title_preview_length=0)


class TestSystemTimeouts:
    """Test system timeouts model."""

    def test_system_timeouts_defaults(self):
        """Test default system timeout values."""
        timeouts = SystemTimeouts()
        assert timeouts.system_command_timeout == 5
        assert timeouts.head_request_timeout == 10

    def test_system_timeouts_validation(self):
        """Test system timeouts field validation."""
        # Zero timeout should fail
        with pytest.raises(ValidationError):
            SystemTimeouts(system_command_timeout=0)


class TestFilterParameters:
    """Test filter parameters model."""

    def test_filter_parameters_defaults(self):
        """Test default filter parameter values."""
        params = FilterParameters()
        assert params.price_to_cents_multiplier == 100
        assert 4.0 in params.rating_codes
        assert params.prime_filter_code == "p_85:2470955011"
        assert params.free_shipping_filter_code == "p_76:419122011"

    def test_filter_parameters_custom_values(self):
        """Test custom filter parameter values."""
        custom_codes = {4.5: "custom_code"}
        params = FilterParameters(rating_codes=custom_codes)
        assert params.rating_codes == custom_codes
