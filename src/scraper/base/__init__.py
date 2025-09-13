"""Base infrastructure for multi-platform scraper architecture.

This package provides the common foundation that all platform-specific scrapers
build upon, including models, configuration, utilities, and interfaces.

Public API:
    - Platform: Enum of supported platforms
    - BaseProductData: Base product data model
    - BaseSearchParameters: Base search parameters model
    - BaseScraper: Abstract base class for scrapers
    - ScraperRegistry: Registry for platform scrapers
    - PlatformConfigManager: Multi-platform configuration manager
    - BaseDownloader: Base downloader functionality
    - BaseBrowserConfig: Base browser configuration
"""

# Core models and interfaces
# Browser utilities
from .browser_utils import (
    BaseBrowserConfig,
    BrowserDetection,
    create_browser_config,
    get_default_chrome_options,
)

# Configuration management
from .config import (
    PlatformConfigManager,
    get_config_manager,
    get_filename_pattern,
    get_output_path,
)

# Download functionality
from .downloader import BaseDownloader, create_downloader
from .models import (
    BaseProductData,
    BaseScraper,
    BaseSearchParameters,
    Platform,
    ProductStatus,
    ScrapeResult,
    ScraperRegistry,
    register_scraper,
)

# Utilities
from .utils import (
    detect_monitors,
    exponential_backoff_retry,
    format_duration,
    get_optimal_browser_position,
    human_delay,
    is_valid_product_data,
    normalize_price,
    sanitize_filename,
)

__all__ = [
    # Core models and interfaces
    "BaseProductData",
    "BaseSearchParameters",
    "BaseScraper",
    "Platform",
    "ProductStatus",
    "ScrapeResult",
    "ScraperRegistry",
    "register_scraper",
    # Configuration management
    "PlatformConfigManager",
    "get_config_manager",
    "get_filename_pattern",
    "get_output_path",
    # Utilities
    "detect_monitors",
    "exponential_backoff_retry",
    "format_duration",
    "get_optimal_browser_position",
    "human_delay",
    "is_valid_product_data",
    "normalize_price",
    "sanitize_filename",
    # Download functionality
    "BaseDownloader",
    "create_downloader",
    # Browser utilities
    "BaseBrowserConfig",
    "BrowserDetection",
    "create_browser_config",
    "get_default_chrome_options",
]
