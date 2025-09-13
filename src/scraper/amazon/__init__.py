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
"""

# Main scraper class - primary interface
# Browser automation
from .browser_functions import (
    create_dynamic_browser_function,
    scrape_amazon_products_browser_impl,
    scrape_single_product,
)

# Configuration and utilities
from .config import (
    CONFIG,
    get_default_search_parameters,
    get_filename_pattern,
    get_output_path,
    load_browser_config_from_yaml,
)
from .downloader import download_file_sync, download_media_files

# Media extraction and download
from .media_extractor import (
    extract_functional_videos_with_validation,
    extract_high_res_images_botasaurus,
    is_valid_high_res_image,
    is_valid_video_url,
    modify_amazon_image_for_high_res,
)

# Data models
from .models import ProductData, SearchParameters, SerpProductInfo
from .scraper import BotasaurusAmazonScraper, main

# Search functionality
from .search_builder import SearchParameterBuilder
from .utils import (
    detect_monitors,
    exponential_backoff_retry,
    get_optimal_browser_position,
    is_valid_product_data,
    validate_asin_format,
)

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
__version__ = "2.0.0"
__author__ = "ContentEngineAI"
__description__ = "Advanced Amazon scraper with anti-detection and media extraction"
