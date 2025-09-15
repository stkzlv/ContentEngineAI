#!/usr/bin/env python3
"""Botasaurus-powered Amazon scraper for ContentEngineAI

This module provides advanced web scraping capabilities for Amazon products using
the Botasaurus framework with built-in anti-detection and performance optimization.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

# Import base classes for multi-platform support
from ..base import BaseScraper, Platform, register_scraper

# Import browser automation functions
from .browser_functions import create_dynamic_browser_function
from .config import (
    CONFIG,
    get_default_search_parameters,
    get_output_path,
)

# Import download function (media extraction handled by browser functions)
from .downloader import download_media_files

# Import data models and config from separate modules
from .models import ProductData, SearchParameters
from .utils import validate_asin_format


# Custom logging filter to suppress websocket cleanup messages
class WebsocketFilter(logging.Filter):
    """Filter out harmless websocket disconnection messages during cleanup"""

    def filter(self, record):
        message = record.message if hasattr(record, "message") else record.getMessage()

        # Filter out websocket goodbye messages that appear during Botasaurus cleanup
        # Also filter general websocket connection messages that are not critical
        return not (
            "websocket" in message.lower()
            and (
                "goodbye" in message.lower()
                or "connection" in message.lower()
                and "lost" in message.lower()
                or "connection to remote host was lost" in message.lower()
            )
        )


# Global debug mode - will be set from YAML config
DEBUG_MODE = False


# Media extraction and download functions are now imported from separate modules


# Global variables for YAML-driven configuration
_BROWSER_CONFIG = {}


# Browser function will be created dynamically with runtime configuration


@register_scraper(Platform.AMAZON)
class BotasaurusAmazonScraper(BaseScraper):
    """Amazon scraper using Botasaurus framework

    Features:
    - Built-in anti-detection
    - Automatic caching
    - Robust error handling
    - High-resolution image extraction
    - Video extraction
    - Quality control
    """

    @property
    def platform(self) -> Platform:
        """Return the platform this scraper handles."""
        return Platform.AMAZON

    def validate_product_id(self, product_id: str) -> bool:
        """Validate Amazon ASIN format."""
        return self._validate_asin_format(product_id)

    def scrape_single_product(self, product_id: str) -> ProductData | None:
        """Scrape a single product by ASIN."""
        products = self.scrape_products_unified(product_id)
        return products[0] if products else None

    def __init__(
        self,
        config_path: str = "config/scrapers.yaml",
        debug_override: bool = None,
        debug_options: dict = None,
    ):
        """Initialize scraper with configuration

        Args:
        ----
            config_path: Path to YAML configuration file
            debug_override: Override debug mode setting from CLI
            debug_options: Dictionary of debug options for detailed analysis

        """
        global DEBUG_MODE

        self.config = self._load_config(config_path)
        self.amazon_config = self.config["scrapers"]["amazon"]
        self.global_settings = self.config["global_settings"]
        self.debug_options = debug_options or {}

        # Override debug mode if specified (CLI takes precedence over config)
        if debug_override is not None:
            global DEBUG_MODE
            original_debug_mode = DEBUG_MODE
            DEBUG_MODE = debug_override

            # Update browser configuration with new DEBUG_MODE without reloading YAML
            # (This prevents YAML from overriding CLI arguments)
            global _BROWSER_CONFIG
            _BROWSER_CONFIG.update(
                {
                    "headless": not DEBUG_MODE,
                    "close_on_crash": not DEBUG_MODE,
                }
            )

            if original_debug_mode != DEBUG_MODE:
                print(
                    f"🔧 [CLI OVERRIDE] Debug mode set to {DEBUG_MODE} "
                    f"(overriding config value: {original_debug_mode})"
                )

        self.logger = logging.getLogger(__name__)

        # Enhanced debug setup
        if DEBUG_MODE:
            self.logger.setLevel(logging.DEBUG)
            self.logger.info(
                "🐛 DEBUG MODE ENABLED - Enhanced logging and diagnostics active"
            )
            self.logger.info(
                f"📊 Config loaded: {len(self.amazon_config)} Amazon settings"
            )
            min_high_res = self.global_settings.get("image_config", {}).get(
                "min_high_res_dimension", 1500
            )
            self.logger.info(f"🎯 Min High-Res Dimension: {min_high_res}")
            self.logger.info(f"⚙️ Browser config: {_BROWSER_CONFIG}")
        else:
            self.logger.setLevel(logging.INFO)

    def _load_config(self, path: str) -> dict[str, Any]:
        """Load YAML configuration file"""
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def scrape_products_unified(
        self, keyword: str, search_params: SearchParameters | None = None
    ) -> list[ProductData]:
        """Unified method to scrape products in a single browser session"""
        try:
            self.logger.info(f"Starting unified scrape for keyword: {keyword}")

            # Prepare data for the unified browser function
            data = {
                "keyword": keyword,
                "is_asin": self._is_asin(keyword),
                "search_params": search_params,
                "debug_mode": DEBUG_MODE,
                "debug_options": self.debug_options,
                "max_products": self.amazon_config.get("max_products", 5),
            }

            # Use the dynamic Botasaurus browser function with current debug settings
            if DEBUG_MODE:
                print(
                    f"🔧 [DEBUG] Creating dynamic browser function with "
                    f"DEBUG_MODE={DEBUG_MODE}"
                )

            try:
                browser_func = create_dynamic_browser_function(DEBUG_MODE)
                if DEBUG_MODE:
                    print(f"🔧 [DEBUG] browser_func type: {type(browser_func)}")
                    print(f"🔧 [DEBUG] browser_func: {browser_func}")
                    print(f"🔧 [DEBUG] Calling browser_func with data: {data}")
                results = browser_func(data)
                print(
                    f"🔧 [DEBUG] browser_func returned {len(results) if results else 0} products"
                )
            except Exception as e:
                if DEBUG_MODE:
                    print(f"❌ [DEBUG] Error in browser function: {e}")
                    import traceback

                    print(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")
                raise

            # Start media downloads for scraped products
            if results:
                if DEBUG_MODE:
                    self.logger.info(
                        f"🚀 Starting media downloads for {len(results)} products"
                    )

                # Prepare media download data for all products
                media_download_tasks = []
                for result in results:
                    # Ensure result is a dictionary (graceful error handling)
                    if not isinstance(result, dict):
                        if DEBUG_MODE:
                            self.logger.warning(
                                f"⚠️ Skipping non-dict result in media orchestration: "
                                f"{type(result)}"
                            )
                        continue

                    if DEBUG_MODE:
                        self.logger.debug(
                            f"📋 Checking product: ASIN={result.get('asin')}, "
                            f"images={len(result.get('images', []))}, "
                            f"videos={len(result.get('videos', []))}"
                        )

                    if result.get("asin") and (
                        result.get("images") or result.get("videos")
                    ):
                        media_download_tasks.append(
                            {
                                "asin": result["asin"],
                                "images": result.get("images", []),
                                "videos": result.get("videos", []),
                                "platform": "amazon",
                                "debug_mode": DEBUG_MODE,
                            }
                        )
                        if DEBUG_MODE:
                            self.logger.info(
                                f"✅ Added {result['asin']} to media download queue"
                            )

                if DEBUG_MODE:
                    self.logger.info(
                        f"📦 Total media download tasks prepared: "
                        f"{len(media_download_tasks)}"
                    )

                # Download media for all products if any have media
                # (with graceful degradation)
                if media_download_tasks:
                    media_download_success = 0
                    media_download_partial = 0

                    if DEBUG_MODE:
                        self.logger.info(
                            f"🚀 [MEDIA ORCHESTRATION] Starting media downloads for "
                            f"{len(media_download_tasks)} products"
                        )
                        for i, task in enumerate(media_download_tasks):
                            self.logger.info(
                                f"   • Task {i+1}: ASIN={task['asin']}, "
                                f"Images={len(task['images'])}, "
                                f"Videos={len(task['videos'])}"
                            )

                    try:
                        # Execute media downloads and get results
                        if DEBUG_MODE:
                            self.logger.info(
                                f"🔄 [MEDIA ORCHESTRATION] Calling "
                                f"download_media_files with "
                                f"{len(media_download_tasks)} tasks"
                            )

                        # Call download_media_files for each product individually to avoid batching issues
                        download_results = []
                        for task in media_download_tasks:
                            if DEBUG_MODE:
                                self.logger.info(
                                    f"🔽 [INDIVIDUAL DOWNLOAD] Processing ASIN: {task['asin']}"
                                )
                            result = download_media_files(
                                [task]
                            )  # Pass single item in list
                            if isinstance(result, list):
                                download_results.extend(result)
                            else:
                                download_results.append(result)

                        # Debug: show raw results structure with verbose logging
                        if DEBUG_MODE:
                            print("\n=== BOTASAURUS DOWNLOAD RESULTS DEBUG ===")
                            print(f"Type: {type(download_results)}")
                            length_str = (
                                len(download_results)
                                if isinstance(download_results, list)
                                else "N/A"
                            )
                            print(f"Length: {length_str}")
                            print(f"Content: {download_results}")
                            print("=" * 50)

                        # Handle Botasaurus task results gracefully
                        if not download_results:
                            self.logger.warning(
                                "⚠️ No media download results returned, "
                                "continuing without media"
                            )
                            download_results = []

                        # Botasaurus @task decorator always returns a list of results
                        # Each task in media_download_tasks gets one result
                        if not isinstance(download_results, list):
                            download_results = (
                                [download_results] if download_results else []
                            )

                        # Create mapping for easy lookup with error handling
                        download_map = {}
                        if DEBUG_MODE:
                            length_str = (
                                len(download_results)
                                if isinstance(download_results, list)
                                else "N/A"
                            )
                            self.logger.debug(
                                f"🐛 [DEBUG] Processing download_results: "
                                f"type={type(download_results)}, length={length_str}"
                            )

                        # Process Botasaurus @task results - when input is a list,
                        # output is a list of results (one per input item)
                        for i, result in enumerate(download_results):
                            if DEBUG_MODE:
                                self.logger.debug(
                                    f"🐛 [DEBUG] Processing result {i}: "
                                    f"type={type(result)}"
                                )

                            if isinstance(result, dict) and result.get("asin"):
                                asin = result.get("asin")
                                download_map[asin] = result
                                if DEBUG_MODE:
                                    img_count = len(result.get("downloaded_images", []))
                                    vid_count = len(result.get("downloaded_videos", []))
                                    self.logger.debug(
                                        f"✅ [DEBUG] Mapped download result for ASIN: "
                                        f"{asin} "
                                        f"(images: {img_count}, videos: {vid_count})"
                                    )
                            elif DEBUG_MODE:
                                # Get result preview length from config
                                debug_config = CONFIG.get("global_settings", {}).get(
                                    "debug_config", {}
                                )
                                result_preview_length = debug_config.get(
                                    "result_preview_length", 100
                                )
                                result_preview = str(result)[:result_preview_length]
                                self.logger.debug(
                                    f"⚠️ [DEBUG] Skipping invalid result {i}: "
                                    f"{type(result)}, preview: {result_preview}..."
                                )

                        # Update results with download information
                        # (graceful degradation)
                        for result in results:
                            asin = result.get("asin")
                            if asin in download_map:
                                download_info = download_map[asin]
                                # Safely extract download info with defaults
                                result["downloaded_images"] = download_info.get(
                                    "downloaded_images", []
                                )
                                result["downloaded_videos"] = download_info.get(
                                    "downloaded_videos", []
                                )

                                total_images = download_info.get("total_images", 0)
                                total_videos = download_info.get("total_videos", 0)

                                if total_images > 0 or total_videos > 0:
                                    media_download_success += 1
                                    if DEBUG_MODE:
                                        self.logger.info(
                                            f"📁 ASIN {asin}: {total_images} images, "
                                            f"{total_videos} videos downloaded"
                                        )
                                else:
                                    media_download_partial += 1
                                    if DEBUG_MODE:
                                        self.logger.debug(
                                            f"📁 ASIN {asin}: Media extraction "
                                            f"attempted but no files downloaded"
                                        )
                            else:
                                # Initialize empty media lists for products
                                # without downloads
                                result["downloaded_images"] = []
                                result["downloaded_videos"] = []
                                media_download_partial += 1

                        if DEBUG_MODE:
                            self.logger.info(
                                f"📊 Media download summary: {media_download_success} "
                                f"successful, {media_download_partial} partial/failed"
                            )

                    except Exception as e:
                        self.logger.warning(
                            f"⚠️ Media download failed ({e}), continuing with "
                            f"product data only"
                        )
                        # Graceful degradation: ensure all products have empty
                        # media lists
                        for result in results:
                            result.setdefault("downloaded_images", [])
                            result.setdefault("downloaded_videos", [])

            # Convert to ProductData objects
            products = []
            for result in results:
                product = ProductData(
                    title=result["title"],
                    price=result["price"],
                    description=result["description"],
                    images=result["images"],
                    videos=result["videos"],
                    affiliate_link=result["affiliate_link"],
                    url=result["url"],
                    platform=Platform.AMAZON,  # Required by BaseProductData
                    asin=result["asin"],
                    keyword=result["keyword"],
                    serp_rating=result["serp_rating"],
                    serp_reviews_count=result["serp_reviews_count"],
                    downloaded_images=result["downloaded_images"],
                    downloaded_videos=result["downloaded_videos"],
                )
                products.append(product)
                # Get title preview length from config
                debug_config = self.global_settings.get("debug_config", {})
                title_preview_length = debug_config.get("title_preview_length", 50)
                self.logger.info(
                    f"Successfully scraped: {product.title[:title_preview_length]}..."
                )

            # Final verification for media files
            if DEBUG_MODE:
                global_settings = CONFIG.get("global_settings", {})
                count_products_with_media = global_settings.get(
                    "count_products_with_media", False
                )
                max_products = (
                    CONFIG.get("scrapers", {}).get("amazon", {}).get("max_products", 5)
                )

                self.logger.info(
                    "🔍 [FINAL VERIFICATION] Checking scraped products and media files..."
                )

                # Filter products to only include those with actual downloaded media files
                products_with_media = []
                products_without_media = []

                for i, product in enumerate(products):
                    # Check for actual file existence on disk instead of trusting download results
                    from pathlib import Path

                    from ...utils.outputs_paths import (
                        get_product_directory,
                        get_product_images_directory,
                        get_product_videos_directory,
                    )

                    product_dir = get_product_directory(product.asin)
                    images_dir = get_product_images_directory(product.asin)
                    videos_dir = get_product_videos_directory(product.asin)

                    # Count actual files that exist on disk
                    actual_images = []
                    actual_videos = []

                    if images_dir.exists():
                        actual_images = list(images_dir.glob("*.jpg")) + list(
                            images_dir.glob("*.png")
                        )

                    if videos_dir.exists():
                        actual_videos = list(videos_dir.glob("*.mp4")) + list(
                            videos_dir.glob("*.mov")
                        )

                    img_count = len(actual_images)
                    vid_count = len(actual_videos)

                    self.logger.info(
                        f"🔍 [FINAL VERIFICATION] Product {i+1}: ASIN={product.asin}, "
                        f"Actual files on disk: {img_count} images, {vid_count} videos"
                    )

                    if img_count > 0 or vid_count > 0:
                        products_with_media.append(product)
                        self.logger.info(
                            f"✅ [FINAL VERIFICATION] Product {product.asin} has downloaded media files"
                        )
                    else:
                        products_without_media.append(product)
                        self.logger.error(
                            f"❌ [FINAL VERIFICATION] Product {product.asin} has NO downloaded media files - FILTERING OUT"
                        )
                        # Clean up data.json for products without media files
                        try:
                            product_dir = get_product_directory(product.asin)
                            data_file = product_dir / "data.json"
                            if data_file.exists():
                                data_file.unlink()
                                self.logger.info(
                                    f"🧹 Cleaned up data.json for filtered product: {product.asin}"
                                )
                        except Exception as cleanup_error:
                            self.logger.warning(
                                f"Could not clean up data.json for {product.asin}: {cleanup_error}"
                            )

                if count_products_with_media:
                    if len(products_with_media) == max_products:
                        self.logger.info(
                            f"✅ [FINAL VERIFICATION] SUCCESS: Got exactly {max_products} products "
                            f"with downloaded media files!"
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ [FINAL VERIFICATION] WARNING: Expected {max_products} products with media, "
                            f"but only {len(products_with_media)} have downloaded media files. "
                            f"Filtered out {len(products_without_media)} products without media."
                        )
                else:
                    self.logger.info(
                        f"🔍 [FINAL VERIFICATION] Traditional mode: {len(products)} products scraped, "
                        f"{len(products_with_media)} with media files"
                    )

            # Return only products with actual media files when count_products_with_media is enabled
            final_products = (
                products_with_media if count_products_with_media else products
            )
            self.logger.info(
                f"Completed unified scrape: {len(final_products)} products for {keyword} "
                f"({len(products_without_media)} filtered out for no media)"
            )
            return final_products

        except Exception as e:
            self.logger.error(f"Error in unified scrape for {keyword}: {e}")
            return []

    def scrape_products(
        self, keywords: list[str], search_params: SearchParameters | None = None
    ) -> list[ProductData]:
        """Main method to scrape products for given keywords

        Args:
        ----
            keywords: List of keywords or ASINs to scrape
            search_params: Optional search parameters for filtering

        Returns:
        -------
            List of ProductData objects

        """
        all_products = []

        for keyword in keywords:
            self.logger.info(f"Starting scrape for keyword: {keyword}")

            # Use the unified scraping method
            products = self.scrape_products_unified(keyword, search_params)
            all_products.extend(products)

        # Save results
        if all_products:
            self._save_products(all_products)

        return all_products

    def _is_asin(self, keyword: str) -> bool:
        """Check if a keyword looks like an Amazon ASIN"""
        return self._validate_asin_format(keyword.strip())

    def _validate_asin_format(self, asin: str) -> bool:
        """Validate proper ASIN format: B0[A-Z0-9]{8} (requirement #10)"""
        return validate_asin_format(asin)

    def _save_products(self, products: list[ProductData]) -> None:
        """Save scraped products to product-centric JSON structure"""
        if not products:
            if DEBUG_MODE:
                self.logger.info("⚠️ No products to save")
            return

        # Convert ProductData objects to dictionaries and save manually
        # since Botasaurus output function isn't being called properly
        from .botasaurus_output import write_scraped_data_output

        # Convert ProductData objects to dictionaries
        product_dicts = []
        for product in products:
            product_dict = self._product_to_dict(product)
            product_dicts.append(product_dict)

        if DEBUG_MODE:
            self.logger.info(f"📄 Saving {len(product_dicts)} products manually")

        # Call the output function directly
        write_scraped_data_output({"manual_save": True}, product_dicts)

    def _product_to_dict(self, product: ProductData) -> dict[str, Any]:
        """Convert ProductData to dictionary for JSON serialization"""
        return {
            "title": product.title,
            "price": product.price,
            "description": product.description,
            "images": product.images,
            "videos": product.videos,
            "affiliate_link": product.affiliate_link,
            "url": product.url,
            "asin": product.asin,
            "keyword": product.keyword,
            "platform": (
                product.platform.value
                if hasattr(product.platform, "value")
                else product.platform
            ),
            "serp_rating": product.serp_rating,
            "serp_reviews_count": product.serp_reviews_count,
            "downloaded_images": product.downloaded_images,
            "downloaded_videos": product.downloaded_videos,
        }

    def cleanup(self) -> None:
        """Cleanup resources after scraping to prevent memory leaks"""
        try:
            # Clean up browser instances if they exist
            if hasattr(self, "_browser_func"):
                try:
                    self._browser_func.close()
                    if DEBUG_MODE:
                        self.logger.info("🧹 Browser instances cleaned up")
                except Exception as e:
                    if DEBUG_MODE:
                        self.logger.debug(f"Browser cleanup warning: {e}")

            # Clean up media download tasks
            try:
                download_media_files.close()
                if DEBUG_MODE:
                    self.logger.info("🧹 Media download tasks cleaned up")
            except Exception as e:
                if DEBUG_MODE:
                    self.logger.debug(f"Media download cleanup warning: {e}")

        except Exception as e:
            self.logger.debug(f"General cleanup warning: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit with automatic cleanup"""
        self.cleanup()


def main():
    """Command-line interface for the Botasaurus Amazon scraper"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Botasaurus Amazon Scraper for ContentEngineAI"
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        required=False,
        help="Keywords or ASINs to scrape (overrides config file)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging and browser visibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (more detailed than debug)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean output directory before scraping"
    )
    parser.add_argument(
        "--pause-on-error",
        action="store_true",
        help="Pause execution when errors occur (debug mode only)",
    )
    parser.add_argument(
        "--save-screenshots",
        action="store_true",
        help="Save screenshots at key steps (debug mode only)",
    )
    parser.add_argument(
        "--save-page-source",
        action="store_true",
        help="Save HTML page source for analysis (debug mode only)",
    )
    parser.add_argument(
        "--analyze-images",
        action="store_true",
        help="Deep analysis of all images found on page (debug mode only)",
    )
    parser.add_argument(
        "--dump-image-urls",
        action="store_true",
        help="Save all discovered image URLs to file (debug mode only)",
    )

    # Search parameter arguments
    parser.add_argument(
        "--min-price",
        type=float,
        metavar="PRICE",
        help="Minimum price filter (e.g., 10.99)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        metavar="PRICE",
        help="Maximum price filter (e.g., 99.99)",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        metavar="RATING",
        help="Minimum rating filter (1-5 stars, e.g., 4.0)",
    )
    parser.add_argument(
        "--prime-only", action="store_true", help="Filter for Prime eligible items only"
    )
    parser.add_argument(
        "--free-shipping",
        action="store_true",
        help="Filter for items with free shipping",
    )
    parser.add_argument(
        "--brands",
        nargs="+",
        metavar="BRAND",
        help="Filter by brand names (e.g., --brands Apple Samsung)",
    )
    parser.add_argument(
        "--sort",
        choices=[
            "relevance",
            "price-low",
            "price-high",
            "rating",
            "newest",
            "featured",
        ],
        default="relevance",
        help="Sort order for search results",
    )
    parser.add_argument(
        "--category", metavar="ID", help="Category ID for filtering (advanced usage)"
    )

    args = parser.parse_args()

    # Load keywords from config if not provided via CLI
    if not args.keywords:
        try:
            # Handle working directory changes from Botasaurus
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config/scrapers.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                amazon_config = config.get("scrapers", {}).get("amazon", {})
                config_keywords = amazon_config.get("keywords", [])

                if config_keywords:
                    args.keywords = config_keywords
                    print(
                        f"📝 Using keywords from config file: "
                        f"{', '.join(config_keywords)}"
                    )
                else:
                    print(
                        "❌ No keywords provided via CLI and no keywords found in "
                        "config file"
                    )
                    print(
                        "💡 Either use --keywords 'your keyword' or add keywords "
                        "to config/scrapers.yaml"
                    )
                    return
            else:
                print("❌ No keywords provided via CLI and config file not found")
                print("💡 Use --keywords 'your keyword' to specify what to scrape")
                return
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            print("💡 Use --keywords 'your keyword' to specify what to scrape")
            return

    # Setup debug mode early - before scraper instantiation
    # Check config file for debug mode if no CLI flag provided
    config_debug_mode = False
    if not args.debug and not args.verbose:
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config/scrapers.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                config_debug_mode = config.get("global_settings", {}).get(
                    "debug_mode", False
                )
        except Exception:
            config_debug_mode = False

    if args.debug or args.verbose or config_debug_mode:
        global DEBUG_MODE
        DEBUG_MODE = True

        # Setup file logging for scraper
        from ...utils.outputs_paths import get_logs_directory

        log_dir = get_logs_directory()
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "scraper.log"

        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        log_level = logging.DEBUG

        # Set up console handler
        console_handler = logging.StreamHandler()
        if args.verbose:
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            print("🔍 Verbose mode enabled - detailed logging active")
        else:
            console_formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
            if config_debug_mode and not args.debug:
                print(
                    "🔧 Debug mode enabled from config - browser visibility and detailed logging active"
                )
            else:
                print(
                    "🔧 Debug mode enabled - browser visibility and detailed logging active"
                )

        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)

        # Set up file handler (overwrite mode)
        file_handler = logging.FileHandler(
            log_file,
            mode="w",  # Overwrite file on each run
            encoding="utf-8",
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)

        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(log_level)

        # Apply websocket filter to suppress cleanup messages
        websocket_filter = WebsocketFilter()
        logging.getLogger().addFilter(websocket_filter)
        logging.getLogger("websocket").addFilter(websocket_filter)

        print("🔧 Debug mode set globally for browser visibility")

        if args.pause_on_error:
            print("⏸️ Pause-on-error enabled - execution will pause when errors occur")
        if args.save_screenshots:
            print("📸 Screenshot saving enabled - key steps will be captured")
        if args.save_page_source:
            print("📄 Page source saving enabled - HTML will be saved for analysis")
        if args.analyze_images:
            print("🔍 Deep image analysis enabled - all images will be analyzed")
        if args.dump_image_urls:
            print("📝 Image URL dumping enabled - all URLs will be saved to file")
    else:
        logging.basicConfig(level=logging.INFO)
        # Apply websocket filter to suppress cleanup messages
        websocket_filter = WebsocketFilter()
        logging.getLogger().addFilter(websocket_filter)
        logging.getLogger("websocket").addFilter(websocket_filter)

    if args.clean:
        import re
        import shutil

        # Clean all scraper outputs - comprehensive cleanup
        # Use absolute path to handle Botasaurus working directory changes
        project_root = Path(__file__).parent.parent.parent.parent
        base_output_path = project_root / get_output_path("base")
        if base_output_path.exists():
            print(f"🧹 Cleaning all scraper outputs in: {base_output_path}")

            # Remove all product directories (ASIN patterns and test IDs)
            # Amazon ASINs: typically 10 chars like B0XXXXXXXX, but also catch test IDs
            # Pattern matches: 10-char ASINs, or any alphanumeric that looks like
            # product
            asin_pattern = re.compile(r"^([A-Z0-9]{10}|TEST[A-Z0-9]+)$")

            for item in base_output_path.iterdir():
                if item.is_dir():
                    # Remove ASIN directories
                    if asin_pattern.match(item.name):
                        shutil.rmtree(item)
                        print(f"🧹 Cleaned product directory: {item}")
                    # Remove other scraper directories (but preserve logs, reports)
                    elif item.name in ["cache", "temp", "screenshots"]:
                        shutil.rmtree(item)
                        print(f"🧹 Cleaned scraper directory: {item}")
                elif (
                    item.is_file()
                    and item.suffix
                    in [
                        ".json",
                        ".csv",
                        ".xlsx",
                        ".html",
                    ]
                    and not item.name.startswith("report")
                ):
                    item.unlink()
                    print(f"🧹 Cleaned scraper file: {item}")

            print("✅ Cleanup completed - all scraper outputs removed")

    if args.debug:
        print("🐛 Debug mode enabled")
        from ...utils.outputs_paths import get_temp_directory

        temp_dir = get_temp_directory()
        print(f"📂 Debug files will be saved to: {temp_dir}")

    # Create SearchParameters from CLI arguments with config defaults
    # Start with default parameters from config
    search_params = get_default_search_parameters()

    # Override with CLI arguments if provided
    cli_overrides = {}
    if args.min_price is not None:
        cli_overrides["min_price"] = args.min_price
    if args.max_price is not None:
        cli_overrides["max_price"] = args.max_price
    if args.min_rating is not None:
        cli_overrides["min_rating"] = args.min_rating
    if args.prime_only:
        cli_overrides["prime_only"] = args.prime_only
    if args.free_shipping:
        cli_overrides["free_shipping"] = args.free_shipping
    if args.brands:
        cli_overrides["brands"] = args.brands
    if args.category:
        cli_overrides["category"] = args.category

    # Handle sort order mapping
    if args.sort != "relevance":
        sort_mapping = {
            "relevance": "relevanceblender",
            "price-low": "price-asc-rank",
            "price-high": "price-desc-rank",
            "rating": "review-rank",
            "newest": "date-desc-rank",
            "featured": "featured-rank",
        }
        cli_overrides["sort_order"] = sort_mapping[args.sort]

    # Apply CLI overrides to search parameters
    if cli_overrides:
        search_params = SearchParameters(
            min_price=cli_overrides.get("min_price", search_params.min_price),
            max_price=cli_overrides.get("max_price", search_params.max_price),
            min_rating=cli_overrides.get("min_rating", search_params.min_rating),
            prime_only=cli_overrides.get("prime_only", search_params.prime_only),
            free_shipping=cli_overrides.get(
                "free_shipping", search_params.free_shipping
            ),
            brands=cli_overrides.get("brands", search_params.brands),
            sort_order=cli_overrides.get("sort_order", search_params.sort_order),
            category=cli_overrides.get("category", search_params.category),
        )

    # Validate search parameters
    validation_errors = search_params.validate()
    if validation_errors:
        print("❌ Invalid search parameters:")
        for error in validation_errors:
            print(f"   • {error}")
        return

    # Show search parameters in debug mode
    if args.debug:
        print("🔍 Search parameters configured:")
        if search_params.min_price or search_params.max_price:
            price_range = (
                f"${search_params.min_price or 0:.2f}-${search_params.max_price or '∞'}"
            )
            print(f"   • Price range: {price_range}")
        if search_params.min_rating:
            print(f"   • Minimum rating: {search_params.min_rating}+ stars")
        if search_params.prime_only:
            print("   • Prime only: Yes")
        if search_params.free_shipping:
            print("   • Free shipping: Yes")
        if search_params.brands:
            print(f"   • Brands: {', '.join(search_params.brands)}")
        if search_params.sort_order != "relevanceblender":
            print(f"   • Sort: {search_params.sort_order}")

        # Show config vs CLI override status
        config_defaults = get_default_search_parameters()
        if cli_overrides:
            print(f"   • CLI overrides applied: {list(cli_overrides.keys())}")
        if (
            search_params.min_price != config_defaults.min_price
            or search_params.max_price != config_defaults.max_price
        ):
            print(
                f"   • Config defaults: ${config_defaults.min_price or 0:.2f}-"
                f"${config_defaults.max_price or '∞'}"
            )

    # Initialize and run scraper with debug override and debug options
    try:
        # Collect debug options
        debug_options = {
            "save_screenshots": args.save_screenshots if args.debug else False,
            "save_page_source": args.save_page_source if args.debug else False,
            "analyze_images": args.analyze_images if args.debug else False,
            "dump_image_urls": args.dump_image_urls if args.debug else False,
            "pause_on_error": args.pause_on_error if args.debug else False,
        }

        # Only pass debug_override if explicitly set via CLI (not default False)
        debug_override = args.debug if args.debug else None
        scraper = BotasaurusAmazonScraper(
            debug_override=debug_override, debug_options=debug_options
        )
        products = scraper.scrape_products(args.keywords, search_params)

        if products:
            print("\n✅ Scraping successful!")
            print(f"📊 Products scraped: {len(products)}")
            print(f"🏷️  Keywords: {', '.join(args.keywords)}")
        else:
            print("\n❌ No products scraped")

    except Exception as e:
        print(f"\n💥 Scraper failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        raise


if __name__ == "__main__":
    # Suppress module import warnings when running with -m
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
    main()
