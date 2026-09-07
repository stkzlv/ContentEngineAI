"""Pydantic models for scraper configuration.

Modern, typed configuration models for the scraper system following the same
pattern as the video pipeline configuration.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetryConfig(BaseModel):
    """Exponential backoff retry configuration."""

    model_config = ConfigDict(extra="forbid")

    default_max_retries: int = Field(default=3, ge=0)
    base_delay: float = Field(default=1.0, gt=0)
    max_delay: float = Field(default=60.0, gt=0)
    backoff_factor: float = Field(default=2.0, gt=0)
    use_jitter: bool = Field(default=True)
    jitter_factor: float = Field(default=0.5, ge=0, le=1)


class RateLimitingConfig(BaseModel):
    """Rate limiting and delay configuration."""

    model_config = ConfigDict(extra="forbid")

    video_validation_delay: list[float] = Field(default=[0.5, 1.5])
    debug_pause_duration: int = Field(default=5, ge=0)
    # Validated here on every load and read through
    # `ThrottleSettings.from_config`, which does its own validation of the
    # same values; keep these defaults equal to `ThrottleSettings`'.
    inter_input_delay_sec: list[float] = Field(default=[2.0, 5.0])
    throttle_backoff_base_sec: float = Field(default=60.0, gt=0)
    throttle_backoff_max_sec: float = Field(default=600.0, gt=0)
    throttle_max_attempts: int = Field(default=6, ge=1)
    throttle_max_total_wait_sec: float = Field(default=3600.0, ge=0)
    dead_query_after: int = Field(default=3, ge=1)


class ImageConfig(BaseModel):
    """Image processing and validation configuration."""

    model_config = ConfigDict(extra="forbid")

    min_high_res_dimension: int = Field(default=1500, gt=0)
    min_high_res_file_size: int = Field(default=10000, gt=0)
    very_high_res_dimension: int = Field(default=2000, gt=0)
    max_images_per_product: int = Field(default=10, gt=0)


class DebugSettings(BaseModel):
    """Debug file generation controls."""

    model_config = ConfigDict(extra="forbid")

    create_media_validation_reports: bool = Field(default=True)
    save_screenshots: bool = Field(default=False)
    save_error_screenshots: bool = Field(default=True)


class VideoConfig(BaseModel):
    """Video processing configuration."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    download_timeout: int = Field(default=30, gt=0)
    video_download_timeout: int = Field(default=300, gt=0)
    retry_video_downloads: int = Field(default=2, ge=0)
    download_chunk_size: int = Field(default=8192, gt=0)
    # None resolves to `system_timeouts.head_request_timeout` at the site,
    # which is what the dict-walk did when the key was absent.
    validation_timeout: int | None = Field(default=None, gt=0)
    min_image_file_size: int = Field(default=10000, gt=0)
    validation_range_bytes: str = Field(default="0-1023")
    concurrent_image_downloads: int = Field(default=5, gt=0)
    concurrent_video_downloads: int = Field(default=3, gt=0)


class SystemTimeouts(BaseModel):
    """System command and network operation timeouts."""

    model_config = ConfigDict(extra="forbid")

    system_command_timeout: int = Field(default=5, gt=0)
    head_request_timeout: int = Field(default=10, gt=0)
    system_profiler_timeout: int = Field(default=10, gt=0)


class MediaConfig(BaseModel):
    """Media file handling configuration."""

    model_config = ConfigDict(extra="forbid")

    default_image_extension: str = Field(default=".jpg")
    amazon_media_domains: list[str] = Field(
        default=["images-amazon.com", "m.media-amazon.com", "media-amazon.com"]
    )
    amazon_high_res_suffix: str = Field(default="._AC_SL2000_.jpg")
    high_res_upgrade_dimension: int = Field(default=2000, gt=0)
    js_context_chars: int = Field(default=500, gt=0)
    valid_http_status_codes: list[int] = Field(default=[200, 206])
    min_file_size_absolute: int = Field(default=1000, gt=0)
    ffprobe_timeout_sec: int = Field(default=30, gt=0)


class DebugConfig(BaseModel):
    """Debug output formatting configuration."""

    model_config = ConfigDict(extra="forbid")

    title_preview_length: int = Field(default=50, gt=0)
    url_preview_length: int = Field(default=100, gt=0)
    result_preview_length: int = Field(default=100, gt=0)


class ValidationConfig(BaseModel):
    """Product validation rules."""

    model_config = ConfigDict(extra="forbid")

    essential_fields: list[str] = Field(default_factory=list)
    min_total_media: int = Field(default=3, ge=0)
    min_images_if_no_video: int = Field(default=5, ge=0)
    min_images_with_video: int = Field(default=2, ge=0)
    media_validation_timeout: int = Field(default=30, gt=0)
    validation_report_top_issues: int = Field(default=10, gt=0)


class BrowserConfig(BaseModel):
    """Browser window behavior and timeouts."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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
    # The standalone loop's page ceiling; the batch reads max_retry_pages.
    max_pages: int = Field(default=7, gt=0)
    max_retry_pages: int = Field(default=5, ge=0)


class CSSSelectors(BaseModel):
    """CSS selector configuration."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    modern_asin_pattern: str = Field(default="^B0[A-Z0-9]{8}$")
    legacy_asin_pattern: str = Field(default="^[A-Z0-9]{10}$")
    url_asin_pattern: str = Field(default="/dp/([A-Z0-9]{10})")


class OutputConfig(BaseModel):
    """Output directory and file pattern configuration."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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
    # Botasaurus task retries; only ever read by the browser config builder.
    retries: int = Field(default=3, ge=0)
    # Forwarded to the browser as its proxy setting; unset means none.
    proxy: str | None = Field(default=None)


class BatchLoggingConfig(BaseModel):
    """Formatting of the batch summary lines."""

    model_config = ConfigDict(extra="forbid")

    separator_char: str = Field(default="=")
    separator_width: int = Field(default=60, gt=0)
    duration_decimal_places: int = Field(default=2, ge=0)
    media_stats_decimal_places: int = Field(default=2, ge=0)


class BatchSection(BaseModel):
    """The top-level `batch:` block of `config/scraper.yaml`.

    `keywords` keeps the raw shape -- a flat list, or a dict keyed by pillar
    -- because `read_keyword_pillars` is the one reader that folds it.
    """

    model_config = ConfigDict(extra="forbid")

    product_ids: list[str] = Field(default_factory=list)
    keywords: list[str] | dict[str, list[str]] = Field(default_factory=list)
    fail_fast: bool = Field(default=False)
    products_per_keyword: int = Field(default=2, gt=0)
    logging: BatchLoggingConfig = Field(default_factory=lambda: BatchLoggingConfig())


class SearchParameters(BaseModel):
    """Amazon search parameters."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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


class ScrapersSection(BaseModel):
    """The `scrapers:` block; one platform today, refusing any other key."""

    model_config = ConfigDict(extra="forbid")

    amazon: AmazonScraperConfig = Field(default_factory=lambda: AmazonScraperConfig())


class ScraperConfig(BaseModel):
    """Top-level scraper configuration combining global and platform settings."""

    model_config = ConfigDict(extra="forbid")

    global_settings: GlobalScraperSettings = Field(
        default_factory=lambda: GlobalScraperSettings()
    )
    batch: BatchSection = Field(default_factory=lambda: BatchSection())
    amazon: AmazonScraperConfig = Field(default_factory=lambda: AmazonScraperConfig())

    @model_validator(mode="before")
    @classmethod
    def _fold_the_files_shape(cls, data: Any) -> Any:
        """Accept the YAML's own shape (`scrapers.amazon`) as well as the model's.

        Raised inside validation so every refusal surfaces as a
        `ValidationError`: the loaders re-raise that and fall back on
        anything else, and a plain `ValueError` from outside validation was
        swallowed into the defaults (review finding).
        """
        if not isinstance(data, dict):
            raise ValueError("scraper config must be a mapping")
        if "scrapers" not in data:
            return data
        if "amazon" in data:
            raise ValueError(
                "scraper config carries both a top-level `amazon` block and "
                "`scrapers.amazon`; keep one"
            )
        scrapers = ScrapersSection.model_validate(data["scrapers"] or {})
        # Every other top-level key is passed through so this model's own
        # `extra="forbid"` refuses it; picking the sections by name would let
        # a misspelled one load as its defaults.
        return {
            **{k: v for k, v in data.items() if k != "scrapers"},
            "amazon": scrapers.amazon,
        }

    @classmethod
    def from_legacy_dict(cls, config: Any) -> "ScraperConfig":
        """Validate a loaded file in either shape.

        Every submodel refuses unknown keys, so a misspelled key in the file
        or a section the models do not describe fails here, at load, instead
        of being read back as a default somewhere downstream (#125). An empty
        or non-mapping file is refused too: an empty file is a truncated
        write, not a request for the defaults.
        """
        return cls.model_validate(config)

    def to_runtime_dict(self) -> dict[str, Any]:
        """The dict consumers walk, in the YAML's shape, every key present.

        Built from the validated model, so a reader that indexes it can
        never fall through to a default of its own.
        """
        data = self.model_dump()
        return {
            "global_settings": data["global_settings"],
            "batch": data["batch"],
            "scrapers": {"amazon": data["amazon"]},
        }
