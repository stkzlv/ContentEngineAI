"""Pydantic models for scraper configuration.

Modern, typed configuration models for the scraper system following the same
pattern as the video pipeline configuration.
"""

from pydantic import BaseModel, ConfigDict, Field


class RetryConfig(BaseModel):
    """Exponential backoff retry configuration."""

    default_max_retries: int = Field(default=3, ge=0)
    base_delay: float = Field(default=1.0, gt=0)
    max_delay: float = Field(default=60.0, gt=0)
    backoff_factor: float = Field(default=2.0, gt=0)
    use_jitter: bool = Field(default=True)
    jitter_factor: float = Field(default=0.5, ge=0, le=1)


class RateLimitingConfig(BaseModel):
    """Rate limiting and delay configuration."""

    video_validation_delay: list[float] = Field(default=[0.5, 1.5])
    debug_pause_duration: int = Field(default=5, ge=0)


class ImageConfig(BaseModel):
    """Image processing and validation configuration."""

    min_high_res_dimension: int = Field(default=1500, gt=0)
    min_high_res_file_size: int = Field(default=10000, gt=0)
    very_high_res_dimension: int = Field(default=2000, gt=0)
    max_images_per_product: int = Field(default=10, gt=0)


class DebugSettings(BaseModel):
    """Debug file generation controls."""

    create_media_validation_reports: bool = Field(default=True)
    save_screenshots: bool = Field(default=False)
    save_error_screenshots: bool = Field(default=True)


class VideoConfig(BaseModel):
    """Video processing configuration."""

    min_dimension: int = Field(default=640, gt=0)
    min_duration: float = Field(default=1.0, gt=0)
    max_videos_per_product: int = Field(default=10, gt=0)
    mute_video_tabs: bool = Field(default=True)
    enable_metadata_extraction: bool = Field(default=True)
    enable_m3u8_monitoring: bool = Field(default=False)
    m3u8_download_timeout: int = Field(default=120, gt=0)
    network_capture_timeout: int = Field(default=20, gt=0)


class DownloadConfig(BaseModel):
    """HTTP download configuration."""

    download_timeout: int = Field(default=30, gt=0)
    video_download_timeout: int = Field(default=300, gt=0)
    retry_video_downloads: int = Field(default=2, ge=0)
    download_chunk_size: int = Field(default=8192, gt=0)
    validation_range_bytes: str = Field(default="0-1023")
    concurrent_image_downloads: int = Field(default=5, gt=0)
    concurrent_video_downloads: int = Field(default=3, gt=0)


class SystemTimeouts(BaseModel):
    """System command and network operation timeouts."""

    system_command_timeout: int = Field(default=5, gt=0)
    head_request_timeout: int = Field(default=10, gt=0)


class MediaConfig(BaseModel):
    """Media file handling configuration."""

    default_image_extension: str = Field(default=".jpg")
    amazon_media_domains: list[str] = Field(
        default=["images-amazon.com", "m.media-amazon.com", "media-amazon.com"]
    )
    amazon_high_res_suffix: str = Field(default="._AC_SL2000_.jpg")
    high_res_upgrade_dimension: int = Field(default=2000, gt=0)
    js_context_chars: int = Field(default=500, gt=0)
    valid_http_status_codes: list[int] = Field(default=[200, 206])
    min_file_size_absolute: int = Field(default=1000, gt=0)


class DebugConfig(BaseModel):
    """Debug output formatting configuration."""

    title_preview_length: int = Field(default=50, gt=0)
    url_preview_length: int = Field(default=100, gt=0)
    result_preview_length: int = Field(default=100, gt=0)


class ValidationConfig(BaseModel):
    """Product validation rules."""

    essential_fields: list[str] = Field(default_factory=list)
    min_total_media: int = Field(default=3, ge=0)
    min_images_if_no_video: int = Field(default=5, ge=0)
    min_images_with_video: int = Field(default=2, ge=0)
    media_validation_timeout: int = Field(default=30, gt=0)
    validation_report_top_issues: int = Field(default=10, gt=0)


class BrowserConfig(BaseModel):
    """Browser window behavior and timeouts."""

    debug_window_width: int = Field(default=1920, gt=0)
    debug_window_height: int = Field(default=1200, gt=0)
    fallback_window_position: list[int] = Field(default=[0, 0, 1920, 1080])
    search_result_timeout: int = Field(default=10, gt=0)
    max_products_per_search: int = Field(default=5, gt=0)
    page_load_timeout_ms: int = Field(default=60000, gt=0)
    script_execution_timeout_ms: int = Field(default=30000, gt=0)
    element_selection_timeout: int = Field(default=10, gt=0)
    max_title_selector_attempts: int = Field(default=10, gt=0)
    search_result_wait_reduced: int = Field(default=5, gt=0)


class BatchProcessingConfig(BaseModel):
    """Batch processing loop configuration."""

    max_scrape_attempts: int = Field(
        default=50, gt=0, description="Safety limit to prevent infinite scraping loops"
    )
    prefetch_multiplier: int = Field(
        default=3,
        gt=0,
        description="Multiplier for prefetching to handle validation failures",
    )
    max_batch_size: int = Field(
        default=15, gt=0, description="Maximum products to fetch in a single batch"
    )


class CSSSelectors(BaseModel):
    """CSS selector configuration."""

    product_title_selectors: list[str] = Field(
        default=[
            "#productTitle",
            "h1.a-size-large",
            ".product-title",
            "h1[data-automation-id='product-title']",
        ]
    )
    search_result_card: str = Field(
        default="div[data-component-type='s-search-result']"
    )


class ASINPatterns(BaseModel):
    """ASIN validation patterns."""

    modern_asin_pattern: str = Field(default="^B0[A-Z0-9]{8}$")
    legacy_asin_pattern: str = Field(default="^[A-Z0-9]{10}$")
    url_asin_pattern: str = Field(default="/dp/([A-Z0-9]{10})")


class OutputConfig(BaseModel):
    """Output directory and file pattern configuration."""

    base_directory: str = Field(default="outputs")
    file_patterns: dict[str, str] = Field(
        default={
            "product_file": "{keyword}_products.json",
            "image_file": "{asin}_image_{index}.{ext}",
            "video_file": "{asin}_video_{index}.{ext}",
        }
    )


class GlobalScraperSettings(BaseModel):
    """Global scraper settings."""

    debug_mode: bool = Field(default=False)
    output_config: OutputConfig = Field(default_factory=lambda: OutputConfig())
    retry_config: RetryConfig = Field(default_factory=lambda: RetryConfig())
    rate_limiting: RateLimitingConfig = Field(
        default_factory=lambda: RateLimitingConfig()
    )
    image_config: ImageConfig = Field(default_factory=lambda: ImageConfig())
    debug_settings: DebugSettings = Field(default_factory=lambda: DebugSettings())
    video_config: VideoConfig = Field(default_factory=lambda: VideoConfig())
    download_config: DownloadConfig = Field(default_factory=lambda: DownloadConfig())
    system_timeouts: SystemTimeouts = Field(default_factory=lambda: SystemTimeouts())
    media_config: MediaConfig = Field(default_factory=lambda: MediaConfig())
    debug_config: DebugConfig = Field(default_factory=lambda: DebugConfig())
    validation_config: ValidationConfig = Field(
        default_factory=lambda: ValidationConfig()
    )
    count_products_with_media: bool = Field(default=True)
    browser_config: BrowserConfig = Field(default_factory=lambda: BrowserConfig())
    batch_processing: BatchProcessingConfig = Field(
        default_factory=lambda: BatchProcessingConfig()
    )
    css_selectors: CSSSelectors = Field(default_factory=lambda: CSSSelectors())
    asin_patterns: ASINPatterns = Field(default_factory=lambda: ASINPatterns())


class SearchParameters(BaseModel):
    """Amazon search parameters."""

    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    min_rating: float | None = Field(default=None, ge=1, le=5)
    prime_only: bool = Field(default=False)
    free_shipping: bool = Field(default=False)
    brands: list[str] = Field(default=[])
    sort_order: str = Field(default="relevanceblender")
    category: str | None = Field(default=None)
    include_sponsored: bool = Field(default=False)
    skip_unavailable: bool = Field(default=True)


class FilterParameters(BaseModel):
    """Amazon filter parameters and codes."""

    price_to_cents_multiplier: int = Field(default=100)
    rating_codes: dict[float, str] = Field(
        default={
            4.0: "2661618011",
            3.0: "2661617011",
            2.0: "2661616011",
            1.0: "2661615011",
        }
    )
    prime_filter_code: str = Field(default="p_85:2470955011")
    free_shipping_filter_code: str = Field(default="p_76:419122011")


class HTTPHeaders(BaseModel):
    """HTTP headers for different request types."""

    video_validation: dict[str, str] = Field(
        default={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "video/webm,video/ogg,video/*;q=0.9,"
                "application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "identity",
            "Referer": "https://www.amazon.com/",
        }
    )
    media_download: dict[str, str] = Field(
        default={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.amazon.com/",
        }
    )
    standard: dict[str, str] = Field(
        default={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        }
    )


class AffiliateLinksConfig(BaseModel):
    """Whether this install participates in an affiliate program.

    Declared here so the typed config path carries the flag instead of
    dropping it: the model's default is ``extra="ignore"``, so an undeclared
    key vanishes without an error. ``extra="forbid"`` on this block turns a
    typo *inside* it (``enabld: false``) into a startup failure rather than a
    setting that silently never applies.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True)


class AmazonScraperConfig(BaseModel):
    """Amazon-specific scraper configuration."""

    enabled: bool = Field(default=True)
    base_url: str = Field(default="https://www.amazon.com")
    keywords: list[str] = Field(default=["keyboard"])
    max_products: int = Field(default=2, gt=0)
    associate_tag: str = Field(default="")
    affiliate_links: AffiliateLinksConfig = Field(
        default_factory=lambda: AffiliateLinksConfig()
    )
    default_search_parameters: SearchParameters = Field(
        default_factory=lambda: SearchParameters()
    )
    filter_parameters: FilterParameters = Field(
        default_factory=lambda: FilterParameters()
    )
    http_headers: HTTPHeaders = Field(default_factory=lambda: HTTPHeaders())


class ScraperConfig(BaseModel):
    """Top-level scraper configuration combining global and platform settings."""

    global_settings: GlobalScraperSettings = Field(
        default_factory=lambda: GlobalScraperSettings()
    )
    amazon: AmazonScraperConfig = Field(default_factory=lambda: AmazonScraperConfig())
