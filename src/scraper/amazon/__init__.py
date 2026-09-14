"""Amazon scraper module for ContentEngineAI.

This module provides comprehensive Amazon product scraping capabilities using
the Botasaurus framework with built-in anti-detection and performance optimization.

Main Components:
- BotasaurusAmazonScraper: Main scraper class with high-level interface
- ProductData, SearchParameters: Data models for products and search
- Media extraction: High-resolution images and video extraction
- Download functionality: Automated media file downloads
- Browser automation: Advanced browser automation with multi-monitor support

Usage:
    from src.scraper.amazon import BotasaurusAmazonScraper

    scraper = BotasaurusAmazonScraper()
    products = scraper.scrape_products(["B0BTYCRJSS"])  # ASIN
    products = scraper.scrape_products(["wireless headphones"])  # Search

**The re-exports resolve on first access, not at import.** A package's
`__init__` runs whenever any submodule is imported, so eagerly re-exporting
`browser_functions` here loaded Botasaurus and its Chromium driver into every
process that merely wanted `models` -- which `src/pipeline/config.py` does.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type checkers only; a module `__getattr__` returns `Any`, which would
    # silence mypy at every call site that uses a re-exported name.
    from .browser_functions import (
        create_dynamic_browser_function,
        scrape_amazon_products_browser_impl,
        scrape_single_product,
    )
    from .config import (
        CONFIG,
        get_default_search_parameters,
        get_filename_pattern,
        get_output_path,
        load_browser_config_from_yaml,
    )
    from .downloader import download_file_sync, download_media_files
    from .media_extractor import (
        extract_functional_videos_with_validation,
        extract_high_res_images_botasaurus,
        is_valid_high_res_image,
        is_valid_video_url,
        modify_amazon_image_for_high_res,
    )
    from .models import ProductData, SearchParameters, SerpProductInfo
    from .scraper import BotasaurusAmazonScraper, main
    from .search_builder import SearchParameterBuilder
    from .utils import (
        detect_monitors,
        exponential_backoff_retry,
        get_optimal_browser_position,
        is_valid_product_data,
        validate_asin_format,
    )

_EXPORTS: dict[str, str] = {
    "create_dynamic_browser_function": "browser_functions",
    "scrape_amazon_products_browser_impl": "browser_functions",
    "scrape_single_product": "browser_functions",
    "CONFIG": "config",
    "get_default_search_parameters": "config",
    "get_filename_pattern": "config",
    "get_output_path": "config",
    "load_browser_config_from_yaml": "config",
    "download_file_sync": "downloader",
    "download_media_files": "downloader",
    "extract_functional_videos_with_validation": "media_extractor",
    "extract_high_res_images_botasaurus": "media_extractor",
    "is_valid_high_res_image": "media_extractor",
    "is_valid_video_url": "media_extractor",
    "modify_amazon_image_for_high_res": "media_extractor",
    "ProductData": "models",
    "SearchParameters": "models",
    "SerpProductInfo": "models",
    "BotasaurusAmazonScraper": "scraper",
    "main": "scraper",
    "SearchParameterBuilder": "search_builder",
    "detect_monitors": "utils",
    "exponential_backoff_retry": "utils",
    "get_optimal_browser_position": "utils",
    "is_valid_product_data": "utils",
    "validate_asin_format": "utils",
}


def __getattr__(name: str) -> Any:
    """Resolve a re-export on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value  # cached, so the lookup happens once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


# Expose main public interface
__all__ = [
    # Main scraper
    "BotasaurusAmazonScraper",
    "main",
    # Data models
    "ProductData",
    "SearchParameters",
    "SerpProductInfo",
    # Configuration
    "CONFIG",
    "get_output_path",
    "get_filename_pattern",
    "get_default_search_parameters",
    # Search
    "SearchParameterBuilder",
    # Validation
    "validate_asin_format",
    "is_valid_product_data",
    # Media functionality
    "extract_high_res_images_botasaurus",
    "extract_functional_videos_with_validation",
    "download_media_files",
    # Browser automation (for advanced usage)
    "scrape_amazon_products_browser_impl",
    "scrape_single_product",
]

# Module metadata
__version__ = "2.2.0"
__author__ = "ContentEngineAI"
__description__ = "Advanced Amazon scraper with anti-detection and media extraction"
