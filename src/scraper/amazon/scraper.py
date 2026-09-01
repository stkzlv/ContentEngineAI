#!/usr/bin/env python3
"""Botasaurus-powered Amazon scraper for ContentEngineAI

This module provides advanced web scraping capabilities for Amazon products using
the Botasaurus framework with built-in anti-detection and performance optimization.
"""

import argparse
import logging
import shutil
import warnings
from pathlib import Path
from typing import Any

import yaml
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.scraper.base.keyword_pillars import pillar_for, read_keyword_pillars

from ...utils.logging_setup import setup_debug_logging
from ...utils.outputs_paths import get_logs_directory
from ...utils.url_shortener import load_url_shortener_settings
from ..base import BaseScraper, Platform, register_scraper
from ..base.models import BaseProductData, BaseSearchParameters
from .browser_functions import (
    create_batch_browser_function,
    create_dynamic_browser_function,
)
from .config import CONFIG, get_default_search_parameters, get_output_path
from .constants import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_SCRAPE_ATTEMPTS,
    DEFAULT_MIN_IMAGES_IF_NO_VIDEO,
    DEFAULT_MIN_IMAGES_WITH_VIDEO,
    DEFAULT_MIN_TOTAL_MEDIA,
    DEFAULT_PREFETCH_MULTIPLIER,
    HIGH_RES_DIMENSION,
)
from .downloader import download_media_files
from .models import ProductData, SearchParameters
from .utils import validate_asin_format

# Initialize logging BEFORE Botasaurus imports to capture early errors
log_dir = get_logs_directory()
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "scraper.log"

# Setup minimal logging (will be reconfigured in main with debug settings)
setup_debug_logging(
    log_file=log_file,
    debug_mode=False,
    verbose=False,
    component_name="AmazonScraper",
    # This runs at import, not at the start of a scrape -- every producer,
    # publisher and batch invocation reaches it, and so does `--help`.
    # `main()` writes the marker once a scrape is actually starting.
    mark_run=False,
)

# Suppress websocket errors (before any browser imports)
ws_logger = logging.getLogger("websocket")
ws_logger.setLevel(logging.CRITICAL)
ws_logger.propagate = False


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


# Suppress frozen runpy warning at module level
warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
warnings.filterwarnings(
    "ignore", message=".*found in sys.modules.*", category=RuntimeWarning
)

# Global debug mode - will be set from YAML config
DEBUG_MODE = False


# Media extraction and download functions are now imported from separate modules


# Module-level logger
logger = logging.getLogger(__name__)

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
        config_path: str = "config/scraper.yaml",
        debug_override: bool = None,
        debug_options: dict = None,
        output_dir: str | None = None,
        profile_uses_videos: bool | None = None,
    ):
        """Initialize scraper with configuration

        Args:
        ----
            config_path: Path to YAML configuration file
            debug_override: Override debug mode setting from CLI
            debug_options: Dictionary of debug options for detailed analysis
            output_dir: Custom output directory (overrides config base_directory)
            profile_uses_videos: Whether the target video profile uses scraped
                videos. When False, validation ignores videos and requires
                enough images for image-only processing. None keeps default.

        """
        self.output_dir = output_dir
        self.profile_uses_videos = profile_uses_videos

        # Set module-level output dir override so Botasaurus callbacks use it
        if output_dir:
            from .botasaurus_output import set_output_dir

            set_output_dir(output_dir)
        global DEBUG_MODE

        self.config = self._load_config(config_path)
        self.amazon_config = self.config["scrapers"]["amazon"]
        self.global_settings = self.config["global_settings"]
        self.debug_options = debug_options or {}
        # Built on first use by pillar_for_keyword.
        self._keyword_pillars: dict[str, str] | None = None
        # Loaded once, and at construction rather than at first use, so a
        # malformed `config/url_shortener.yaml` is reported before a scrape
        # starts instead of after the browser work is paid for.
        self.url_shortener_settings = load_url_shortener_settings()

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
                    "headless": False,  # Headed under Xvfb; Botasaurus headless bug
                    "close_on_crash": not DEBUG_MODE,
                }
            )

            if original_debug_mode != DEBUG_MODE:
                logger.debug(
                    "[CLI OVERRIDE] Debug mode set to %s "
                    "(overriding config value: %s)",
                    DEBUG_MODE,
                    original_debug_mode,
                )

        # Store as instance variables to avoid global reads throughout methods
        self.debug_mode = DEBUG_MODE
        self.browser_config = _BROWSER_CONFIG

        self.logger = logging.getLogger(__name__)

        # Apply WebSocket filter to suppress harmless connection messages
        websocket_filter = WebsocketFilter()
        self.logger.addFilter(websocket_filter)

        # Also apply to root websocket logger
        websocket_logger = logging.getLogger("websocket")
        websocket_logger.addFilter(websocket_filter)
        websocket_logger.setLevel(logging.WARNING)

        # Enhanced debug setup
        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
            self.logger.info(
                "DEBUG MODE ENABLED - Enhanced logging and diagnostics active"
            )
            self.logger.info(
                "Config loaded: %d Amazon settings",
                len(self.amazon_config),
            )
            min_high_res = self.global_settings.get("image_config", {}).get(
                "min_high_res_dimension", HIGH_RES_DIMENSION
            )
            self.logger.info("Min High-Res Dimension: %s", min_high_res)
            self.logger.info("Browser config: %s", self.browser_config)
        else:
            self.logger.setLevel(logging.INFO)

    def _load_config(self, path: str) -> dict[str, Any]:
        """Load YAML configuration file"""
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            return dict(yaml.safe_load(f) or {})

    def scrape_products_unified(
        self,
        keyword: str,
        search_params: SearchParameters | None = None,
        max_products: int | None = None,
    ) -> list[ProductData]:
        """Unified method to scrape products in a single browser session

        Continues scraping until max_products that pass validation are collected.
        """
        try:
            self.logger.info("Starting unified scrape for keyword: %s", keyword)

            # Use provided max_products or fall back to config
            products_limit = (
                max_products
                if max_products is not None
                else self.amazon_config.get("max_products", 5)
            )

            # Check if count_products_with_media is enabled
            global_settings = CONFIG.get("global_settings", {})
            count_products_with_media = global_settings.get(
                "count_products_with_media", False
            )

            # If count_products_with_media is enabled, loop until target is reached
            if count_products_with_media:
                # Pagination is for keyword searches only. A URL or an ASIN
                # names one product, so every later page re-resolves the same
                # one: a listing that fails media validation spent all of
                # `max_pages` browser sessions re-fetching it before returning
                # empty. The run continued to the next input, so nothing was
                # lost; what it cost was seven sessions per bad entry and the
                # delay that put on the rest of an --input-file batch. The
                # global batch gates this the same way (`is_keyword` in
                # _execute_scraping_phase).
                if self._is_asin(keyword) or self._is_url(keyword):
                    return self._scrape_single_pass(
                        keyword,
                        search_params,
                        products_limit,
                        filter_validated=True,
                    )
                return self._scrape_until_validated_count_reached(
                    keyword, search_params, products_limit
                )

            # Otherwise use traditional single-pass scraping without filtering
            return self._scrape_single_pass(
                keyword, search_params, products_limit, filter_validated=False
            )

        except Exception as e:
            self.logger.error("Error in unified scrape for %s: %s", keyword, e)
            return []

    def _scrape_until_validated_count_reached(
        self,
        keyword: str,
        search_params: SearchParameters | None,
        target_count: int,
    ) -> list[ProductData]:
        """Loop scraping until target_count validated products are collected.

        Paginates through search result pages when products on the current
        page fail validation. Stops when the target is reached or `max_pages`
        is passed. `max_scrape_attempts` also stops it, but `total_raw_scraped`
        counts validated products rather than raw ones (it and
        `validated_products` grow by the same `batch`), so that guard fires
        only when the limit is below `target_count` and never bounds a
        listing that fails validation.
        """
        validated_products: list[ProductData] = []
        total_raw_scraped = 0

        # Get batch processing config values
        batch_cfg = CONFIG.get("global_settings", {}).get("batch_processing", {})
        max_attempts = batch_cfg.get("max_scrape_attempts", DEFAULT_MAX_SCRAPE_ATTEMPTS)
        prefetch_multiplier = batch_cfg.get(
            "prefetch_multiplier", DEFAULT_PREFETCH_MULTIPLIER
        )
        max_batch_size = batch_cfg.get("max_batch_size", DEFAULT_MAX_BATCH_SIZE)
        max_pages = batch_cfg.get("max_pages", 7)

        current_page = 1

        self.logger.info(
            "Target: %d products that pass validation requirements",
            target_count,
        )

        while len(validated_products) < target_count:
            if total_raw_scraped >= max_attempts:
                self.logger.warning(
                    "Reached max scrape attempts (%d raw products). "
                    "Stopping with %d/%d validated.",
                    max_attempts,
                    len(validated_products),
                    target_count,
                )
                break

            if current_page > max_pages:
                self.logger.warning(
                    "Reached max pages (%d). Stopping with %d/%d validated.",
                    max_pages,
                    len(validated_products),
                    target_count,
                )
                break

            remaining = target_count - len(validated_products)
            batch_size = min(remaining * prefetch_multiplier, max_batch_size)

            if self.debug_mode:
                self.logger.info(
                    "Progress: %d/%d validated | Page %d | "
                    "Requesting %d more products...",
                    len(validated_products),
                    target_count,
                    current_page,
                    batch_size,
                )

            # Scrape a batch from the current page
            batch = self._scrape_single_pass(
                keyword,
                search_params,
                batch_size,
                target_download_count=remaining,
                page=current_page,
            )

            if not batch:
                # No validated products from this page. Try the next page
                # unless the browser returned nothing at all (exhausted results).
                # We detect exhausted results by checking if raw products were
                # found: _scrape_single_pass calls _validate_and_convert_products
                # which logs rejections. If we're on page 1 and got 0, the search
                # itself may have returned products that all failed validation.
                # Move to next page to find better candidates.
                self.logger.info(
                    "No validated products on page %d, trying next page...",
                    current_page,
                )
                current_page += 1
                continue

            total_raw_scraped += len(batch)
            validated_products.extend(batch)

            if self.debug_mode:
                self.logger.info(
                    "Batch complete: +%d validated products (total: %d/%d)",
                    len(batch),
                    len(validated_products),
                    target_count,
                )

            # Move to next page for the next iteration if still needed
            current_page += 1

        # Trim to exact count if we over-collected
        if len(validated_products) > target_count:
            validated_products = validated_products[:target_count]

        self.logger.info(
            "Scraping complete: %d validated products collected",
            len(validated_products),
        )

        return validated_products

    def _scrape_single_pass(
        self,
        keyword: str,
        search_params: SearchParameters | None,
        products_limit: int,
        filter_validated: bool = True,
        target_download_count: int | None = None,
        page: int = 1,
    ) -> list[ProductData]:
        """Single-pass scraping with download and validation

        Args:
        ----
            keyword: Search keyword or ASIN
            search_params: Search parameters for filtering
            products_limit: Number of products to scrape
            filter_validated: If True, return only products that pass validation
            target_download_count: Max products to download media for (None = all)
            page: Search results page number (1-based)

        Returns:
        -------
            List of ProductData objects (filtered if filter_validated=True)

        """
        try:
            # Prepare data for the unified browser function
            data = {
                "keyword": keyword,
                "is_asin": self._is_asin(keyword),
                "is_url": self._is_url(keyword),
                "search_params": search_params,
                "debug_mode": self.debug_mode,
                "debug_options": self.debug_options,
                "max_products": products_limit,
                "page": page,
            }

            # Use the dynamic Botasaurus browser function with current debug settings
            if self.debug_mode:
                self.logger.debug(
                    "[DEBUG] Creating dynamic browser function with " "DEBUG_MODE=%s",
                    self.debug_mode,
                )

            try:
                browser_func = create_dynamic_browser_function(self.debug_mode)
                if self.debug_mode:
                    self.logger.debug(
                        "[DEBUG] browser_func type: %s",
                        type(browser_func),
                    )
                    self.logger.debug("[DEBUG] browser_func: %s", browser_func)
                    self.logger.debug(
                        "[DEBUG] Calling browser_func with data: %s",
                        data,
                    )
                results = self._scrape_with_retry(browser_func, data)
                self.logger.debug(
                    "[DEBUG] browser_func returned %d products",
                    len(results) if results else 0,
                )
            except Exception as e:
                if self.debug_mode:
                    self.logger.error("[DEBUG] Error in browser function: %s", e)
                    import traceback

                    self.logger.error("[DEBUG] Traceback: %s", traceback.format_exc())
                raise

            # Download media for scraped products
            if results:
                self._orchestrate_media_downloads(results, target_download_count)

            # Convert to ProductData and validate media requirements
            return self._validate_and_convert_products(results, filter_validated)

        except Exception as e:
            self.logger.error("Error in single pass scrape for %s: %s", keyword, e)
            return []

    def _orchestrate_media_downloads(
        self, results: list[dict], target_download_count: int | None
    ) -> None:
        """Download media files for scraped products.

        Mutates ``results`` in place, adding ``downloaded_images`` and
        ``downloaded_videos`` keys to each result dict.

        Args:
        ----
            results: List of raw product dicts from browser scraping
            target_download_count: Max products to download media for (None = all)

        """
        if self.debug_mode:
            self.logger.info(
                "Starting media downloads for %d products",
                len(results),
            )

        # Prepare media download data for all products
        media_download_tasks = []
        for result in results:
            if self.debug_mode:
                self.logger.debug(
                    "Checking product: ASIN=%s, images=%d, videos=%d",
                    result.get("asin"),
                    len(result.get("images", [])),
                    len(result.get("videos", [])),
                )

            if result.get("asin") and (result.get("images") or result.get("videos")):
                media_download_tasks.append(
                    {
                        "asin": result["asin"],
                        "images": result.get("images", []),
                        "videos": result.get("videos", []),
                        "platform": "amazon",
                        "debug_mode": self.debug_mode,
                        "output_dir": self.output_dir,
                    }
                )
                if self.debug_mode:
                    self.logger.info("Added %s to media download queue", result["asin"])

        # Limit downloads to target_download_count if specified
        if (
            target_download_count is not None
            and len(media_download_tasks) > target_download_count
        ):
            if self.debug_mode:
                task_count = len(media_download_tasks)
                self.logger.info(
                    "Limiting downloads: %d -> %d products",
                    task_count,
                    target_download_count,
                )
            media_download_tasks = media_download_tasks[:target_download_count]

        if self.debug_mode:
            self.logger.info(
                "Total media download tasks prepared: %d",
                len(media_download_tasks),
            )

        if not media_download_tasks:
            # Ensure all products have empty media lists
            for result in results:
                result.setdefault("downloaded_images", [])
                result.setdefault("downloaded_videos", [])
            return

        media_download_success = 0
        media_download_partial = 0

        if self.debug_mode:
            self.logger.info(
                "[MEDIA ORCHESTRATION] Starting media downloads for " "%d products",
                len(media_download_tasks),
            )
            for i, task in enumerate(media_download_tasks):
                self.logger.info(
                    "   • Task %d: ASIN=%s, Images=%d, Videos=%d",
                    i + 1,
                    task["asin"],
                    len(task["images"]),
                    len(task["videos"]),
                )

        try:
            if self.debug_mode:
                self.logger.info(
                    "[MEDIA ORCHESTRATION] Calling "
                    "download_media_files with %d tasks",
                    len(media_download_tasks),
                )

            # Download per product individually to avoid batching issues
            download_results = []
            for task in media_download_tasks:
                if self.debug_mode:
                    self.logger.info(
                        "[INDIVIDUAL DOWNLOAD] Processing ASIN: %s",
                        task["asin"],
                    )
                dl_result = download_media_files([task])
                if isinstance(dl_result, list):
                    download_results.extend(dl_result)
                else:
                    download_results.append(dl_result)

            if self.debug_mode:
                self.logger.debug("=== BOTASAURUS DOWNLOAD RESULTS DEBUG ===")
                self.logger.debug("Type: %s", type(download_results))
                length_str = (
                    len(download_results)
                    if isinstance(download_results, list)
                    else "N/A"
                )
                self.logger.debug("Length: %s", length_str)
                self.logger.debug("Content: %s", download_results)
                self.logger.debug("=" * 50)

            if not download_results:
                self.logger.warning(
                    "No media download results returned, " "continuing without media"
                )

            # Create mapping for easy lookup
            download_map = {}
            if self.debug_mode:
                length_str = (
                    len(download_results)
                    if isinstance(download_results, list)
                    else "N/A"
                )
                self.logger.debug(
                    "[DEBUG] Processing download_results: " "type=%s, length=%s",
                    type(download_results),
                    length_str,
                )

            for i, dl_result in enumerate(download_results):
                if self.debug_mode:
                    self.logger.debug(
                        "[DEBUG] Processing result %d: type=%s",
                        i,
                        type(dl_result),
                    )

                if isinstance(dl_result, dict) and dl_result.get("asin"):
                    asin = dl_result.get("asin")
                    download_map[asin] = dl_result
                    if self.debug_mode:
                        img_count = len(dl_result.get("downloaded_images", []))
                        vid_count = len(dl_result.get("downloaded_videos", []))
                        self.logger.debug(
                            "[DEBUG] Mapped download result for ASIN: "
                            "%s (images: %d, videos: %d)",
                            asin,
                            img_count,
                            vid_count,
                        )
                elif self.debug_mode:
                    debug_config = CONFIG.get("global_settings", {}).get(
                        "debug_config", {}
                    )
                    result_preview_length = debug_config.get(
                        "result_preview_length", 100
                    )
                    result_preview = str(dl_result)[:result_preview_length]
                    self.logger.debug(
                        "[DEBUG] Skipping invalid result %d: " "%s, preview: %s...",
                        i,
                        type(dl_result),
                        result_preview,
                    )

            # Update results with download information
            for result in results:
                asin = result.get("asin")
                if asin in download_map:
                    download_info = download_map[asin]
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
                        if self.debug_mode:
                            self.logger.info(
                                "ASIN %s: %d images, %d videos downloaded",
                                asin,
                                total_images,
                                total_videos,
                            )
                    else:
                        media_download_partial += 1
                        if self.debug_mode:
                            self.logger.debug(
                                "ASIN %s: Media extraction "
                                "attempted but no files downloaded",
                                asin,
                            )
                else:
                    result["downloaded_images"] = []
                    result["downloaded_videos"] = []
                    media_download_partial += 1

            if self.debug_mode:
                self.logger.info(
                    "Media download summary: %d successful, %d partial/failed",
                    media_download_success,
                    media_download_partial,
                )

        except Exception as e:
            self.logger.warning(
                "Media download failed (%s), continuing with " "product data only",
                e,
            )
            for result in results:
                result.setdefault("downloaded_images", [])
                result.setdefault("downloaded_videos", [])

    def _validate_and_convert_products(
        self, results: list[dict], filter_validated: bool
    ) -> list[ProductData]:
        """Convert raw result dicts to ProductData and validate media.

        Args:
        ----
            results: List of product dicts with download info
            filter_validated: If True, return only products meeting media requirements

        Returns:
        -------
            List of ProductData objects (filtered if filter_validated=True)

        """
        from ...utils.outputs_paths import (
            get_product_directory,
            get_product_images_directory,
            get_product_videos_directory,
        )

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
                platform=Platform.AMAZON,
                asin=result["asin"],
                keyword=result["keyword"],
                # The detail page carries these on every arm; the search card
                # only exists on a keyword scrape. Without them a product
                # scraped by ASIN or URL had no rating at all.
                rating=result.get("rating"),
                reviews_count=result.get("reviews_count"),
                serp_rating=result["serp_rating"],
                serp_reviews_count=result["serp_reviews_count"],
                downloaded_images=result["downloaded_images"],
                downloaded_videos=result["downloaded_videos"],
            )
            products.append(product)
            self.logger.info(
                "Successfully scraped: %s - %s", product.asin, product.title
            )

        # Final verification for media files
        global_settings = CONFIG.get("global_settings", {})
        count_products_with_media = global_settings.get(
            "count_products_with_media", False
        )
        max_products = (
            CONFIG.get("scrapers", {}).get("amazon", {}).get("max_products", 5)
        )

        products_with_media = []
        products_without_media = []

        if self.debug_mode:
            self.logger.info(
                "[FINAL VERIFICATION] Checking scraped products and " "media files..."
            )

        for i, product in enumerate(products):
            product_dir = get_product_directory(
                product.asin or "unknown", custom_dir=self.output_dir
            )
            images_dir = get_product_images_directory(
                product.asin or "unknown", custom_outputs_dir=self.output_dir
            )
            videos_dir = get_product_videos_directory(
                product.asin or "unknown", custom_outputs_dir=self.output_dir
            )

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

            if self.debug_mode:
                self.logger.info(
                    "[FINAL VERIFICATION] Product %d: "
                    "ASIN=%s, Actual files: %d images, %d videos",
                    i + 1,
                    product.asin,
                    img_count,
                    vid_count,
                )

            # Get producer-aligned media requirements from config
            validation_config = CONFIG.get("global_settings", {}).get(
                "validation_config", {}
            )
            min_total = validation_config.get(
                "min_total_media", DEFAULT_MIN_TOTAL_MEDIA
            )
            min_imgs_no_vid = validation_config.get(
                "min_images_if_no_video", DEFAULT_MIN_IMAGES_IF_NO_VIDEO
            )
            min_imgs_with_vid = validation_config.get(
                "min_images_with_video", DEFAULT_MIN_IMAGES_WITH_VIDEO
            )

            # When profile doesn't use videos, ignore them for validation
            effective_vid_count = vid_count
            if self.profile_uses_videos is False:
                effective_vid_count = 0

            total_media = img_count + effective_vid_count
            meets_requirements = True
            rejection_reason = ""

            if total_media < min_total:
                meets_requirements = False
                rejection_reason = f"total media {total_media} < {min_total}"
            elif effective_vid_count == 0 and img_count < min_imgs_no_vid:
                meets_requirements = False
                rejection_reason = (
                    f"no usable videos and images {img_count} < {min_imgs_no_vid}"
                )
            elif effective_vid_count > 0 and img_count < min_imgs_with_vid:
                meets_requirements = False
                rejection_reason = (
                    f"has videos but images {img_count} < {min_imgs_with_vid}"
                )

            if meets_requirements:
                products_with_media.append(product)
                if self.debug_mode:
                    self.logger.info(
                        "[FINAL VERIFICATION] Product %s "
                        "meets producer requirements: %d images, "
                        "%d videos, %d total media",
                        product.asin,
                        img_count,
                        vid_count,
                        total_media,
                    )
            else:
                products_without_media.append(product)
                self.logger.warning(
                    "Product %s rejected: %s (%d images, %d videos)",
                    product.asin,
                    rejection_reason,
                    img_count,
                    vid_count,
                )
                try:
                    if product_dir.exists():
                        shutil.rmtree(product_dir)
                        if self.debug_mode:
                            self.logger.info(
                                "Cleaned up product directory for "
                                "filtered product: %s",
                                product.asin,
                            )
                except Exception as cleanup_error:
                    if self.debug_mode:
                        self.logger.warning(
                            "Could not clean up directory for %s: %s",
                            product.asin,
                            cleanup_error,
                        )

        if self.debug_mode:
            if count_products_with_media:
                if len(products_with_media) == max_products:
                    self.logger.info(
                        "[FINAL VERIFICATION] SUCCESS: Got exactly "
                        "%d products with downloaded media!",
                        max_products,
                    )
                else:
                    self.logger.warning(
                        "[FINAL VERIFICATION] Expected "
                        "%d products with media, but only "
                        "%d have media files. Filtered out "
                        "%d without media.",
                        max_products,
                        len(products_with_media),
                        len(products_without_media),
                    )
            else:
                self.logger.info(
                    "[FINAL VERIFICATION] Traditional mode: "
                    "%d scraped, %d with media files",
                    len(products),
                    len(products_with_media),
                )

        final_products = products_with_media if filter_validated else products

        if self.debug_mode:
            self.logger.info(
                "Completed single pass: %d products (%d filtered out)",
                len(final_products),
                len(products_without_media),
            )

        return final_products

    def scrape_batch_browser(
        self,
        inputs: list[str],
        search_params: SearchParameters | None = None,
        start_page: int = 1,
    ) -> list[dict]:
        """Scrape multiple inputs in a single Chrome session (browser phase only).

        Returns raw product dicts per input. No media downloads or validation
        happens here; call process_raw_products() on each result set afterwards.

        Returns list of dicts: [{"input": str, "products": list[dict]}]
        """
        items = []
        for inp in inputs:
            items.append(
                {
                    "keyword": inp,
                    "is_asin": self._is_asin(inp),
                    "is_url": self._is_url(inp),
                    "search_params": search_params,
                    "debug_mode": self.debug_mode,
                    "debug_options": self.debug_options,
                    "max_products": self.amazon_config.get("max_products", 5),
                    "page": start_page,
                }
            )

        batch_func = create_batch_browser_function(self.debug_mode)
        raw_results = batch_func({"items": items})
        return raw_results if raw_results else []

    def pillar_for_keyword(self, keyword: str) -> str | None:
        """Return the pillar the config files this keyword under, if any.

        Read from the scraper's own config so the standalone paths do not
        depend on a caller passing a map they have no reason to hold. The
        batch pipeline builds the same mapping for its own config; both come
        from the same `batch.keywords` block.

        A flat keyword list, the pre-pillar shape, maps nothing.
        """
        if self._keyword_pillars is None:
            _, self._keyword_pillars = read_keyword_pillars(
                (self.config.get("batch") or {}).get("keywords")
            )
        return pillar_for(keyword, self._keyword_pillars)

    def process_raw_products(
        self,
        raw_products: list[dict],
        target_download_count: int | None = None,
        filter_validated: bool = True,
        pillar: str | None = None,
    ) -> list[ProductData]:
        """Download media, validate, and save products from browser scraping.

        `pillar` is applied before the file is written. Assigning it to the
        returned records instead loses it: the caller's objects are discarded
        and the directory is re-read from disk.
        """
        if raw_products:
            self._orchestrate_media_downloads(raw_products, target_download_count)
        products = self._validate_and_convert_products(raw_products, filter_validated)
        if pillar:
            for product in products:
                product.pillar = pillar
        if products:
            self._save_products(products)
        return products

    def scrape_products(
        self, keywords: list[str], search_params: BaseSearchParameters | None = None
    ) -> list[BaseProductData]:
        """Main method to scrape products for given keywords

        Args:
        ----
            keywords: List of keywords or ASINs to scrape
            search_params: Optional search parameters for filtering

        Returns:
        -------
            List of ProductData objects

        """
        all_products: list[BaseProductData] = []

        for keyword in keywords:
            self.logger.info("Starting scrape for keyword: %s", keyword)
            keyword_pillar = self.pillar_for_keyword(keyword)

            # Use the unified scraping method
            # Cast search_params to SearchParameters if it's compatible
            amazon_params = None
            if search_params and hasattr(search_params, "__dict__"):
                # Create SearchParameters from BaseSearchParameters attributes
                from .models import SearchParameters

                try:
                    amazon_params = SearchParameters(**search_params.__dict__)
                except Exception:
                    amazon_params = None

            products = self.scrape_products_unified(keyword, amazon_params)
            if keyword_pillar:
                for product in products:
                    product.pillar = keyword_pillar
            all_products.extend(products)

        # Save results
        if all_products:
            # Cast back to ProductData for _save_products
            product_data_list = [p for p in all_products if isinstance(p, ProductData)]
            self._save_products(product_data_list)

        # Return as list of BaseProductData (ProductData inherits from BaseProductData)
        return all_products

    def _is_asin(self, keyword: str) -> bool:
        """Check if a keyword looks like an Amazon ASIN"""
        return self._validate_asin_format(keyword.strip())

    @staticmethod
    def _is_url(keyword: str) -> bool:
        """Check if a keyword is a URL (shortened or full)."""
        return keyword.strip().startswith(("http://", "https://"))

    def _validate_asin_format(self, asin: str) -> bool:
        """Validate proper ASIN format: B0[A-Z0-9]{8} (requirement #10)"""
        return validate_asin_format(asin)

    @retry(  # type: ignore
        retry=retry_if_exception_type(RuntimeError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _scrape_with_retry(self, browser_func, data):
        """Scrape with retry logic for Amazon error pages"""
        try:
            if self.debug_mode:
                self.logger.debug("[DEBUG] Attempting scrape with retry logic")
            return browser_func(data)
        except RuntimeError as e:
            if "Amazon error page detected" in str(e):
                if self.debug_mode:
                    self.logger.warning(
                        "[DEBUG] Caught Amazon error " "page, will retry: %s",
                        e,
                    )
                raise  # Will trigger retry
            else:
                # Other RuntimeErrors should not retry
                raise

    def _shorten_affiliate_links(self, products: list[ProductData]) -> None:
        """Shorten affiliate links for products if URL shortening is enabled.

        Settings come from the typed model rather than from a second read of
        `config/url_shortener.yaml`. The inline read carried its own defaults
        beside the model's, so the two drifted -- the model still said
        `picsee` after the file had been flipped to `bare` -- and a typo'd key
        fell back to a default rather than being reported.
        """
        try:
            settings = self.url_shortener_settings

            if not settings.enabled or not settings.integration.shorten_on_scrape:
                if self.debug_mode:
                    self.logger.debug("URL shortening disabled, skipping")
                return

            # Get API key from environment (load .env if available)
            import os

            from dotenv import load_dotenv

            load_dotenv()

            provider = settings.provider
            provider_settings = settings.active_provider()

            # Bare provider returns input unchanged; no API key, no network.
            api_key = ""
            if provider != "bare":
                api_key_env_var = provider_settings.api_key_env_var or ""
                api_key = os.getenv(api_key_env_var, "") if api_key_env_var else ""
                if not api_key:
                    if self.debug_mode:
                        self.logger.warning(
                            "%s not found, skipping URL shortening",
                            api_key_env_var or "<no api_key_env_var configured>",
                        )
                    return

            # Import URL shortener utilities
            from ...utils.url_shortener import create_url_shortener

            timeout = settings.api.timeout_sec
            custom_domain = provider_settings.custom_domain
            api_base_url = provider_settings.api_base_url or "https://api.pics.ee"
            max_bulk_size = provider_settings.max_bulk_size
            bulk_timeout_multiplier = provider_settings.bulk_timeout_multiplier

            max_retries = settings.api.max_retries
            retry_delay = settings.api.retry_delay_sec
            retry_backoff = settings.api.retry_backoff_multiplier

            # The bare provider doesn't shorten or retry, so the verbose
            # "Shortening N using ...", custom-domain, and retry-config lines
            # don't apply. Emit one short line instead.
            verbose_log = self.debug_mode and provider != "bare"
            if self.debug_mode and provider == "bare":
                self.logger.info(
                    "URL shortener: bare (no-op, %d affiliate link(s) "
                    "passed through unchanged)",
                    len(products),
                )
            elif verbose_log:
                self.logger.info(
                    "Shortening %d affiliate links using %s",
                    len(products),
                    provider,
                )
                if custom_domain:
                    self.logger.info("   Using custom domain: %s", custom_domain)
                self.logger.info(
                    "   Retry config: %d attempts, %.1fs delay, %.1fx backoff",
                    max_retries,
                    retry_delay,
                    retry_backoff,
                )

            shortener = create_url_shortener(
                provider=provider,
                api_key=api_key,
                timeout=timeout,
                custom_domain=custom_domain,
                api_base_url=api_base_url,
                max_bulk_size=max_bulk_size,
                bulk_timeout_multiplier=bulk_timeout_multiplier,
                max_retries=max_retries,
                retry_delay=retry_delay,
                retry_backoff_multiplier=retry_backoff,
            )

            # Shorten affiliate links
            import asyncio

            async def shorten_all():
                for product in products:
                    if not product.affiliate_link:
                        continue

                    try:
                        result = await shortener.shorten(product.affiliate_link)
                        product.shortened_affiliate_link = result.short_url
                        if verbose_log:
                            self.logger.info(
                                "Shortened: %s -> %s",
                                product.asin,
                                result.short_url,
                            )
                    except Exception as e:
                        self.logger.warning(
                            "Failed to shorten link for %s: %s", product.asin, e
                        )
                        if settings.integration.fallback_to_original:
                            product.shortened_affiliate_link = product.affiliate_link

            # Run async shortening - handle both sync and async contexts
            try:
                # Check if we're already in an event loop
                asyncio.get_running_loop()
                # We're in an async context, create and await task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, shorten_all())
                    future.result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                asyncio.run(shorten_all())

            if verbose_log:
                shortened_count = sum(1 for p in products if p.shortened_affiliate_link)
                self.logger.info(
                    "Shortened %d/%d affiliate links",
                    shortened_count,
                    len(products),
                )

        except Exception as e:
            self.logger.warning("URL shortening failed: %s, using original links", e)
            # Fallback: use original affiliate links
            for product in products:
                if product.affiliate_link and not product.shortened_affiliate_link:
                    product.shortened_affiliate_link = product.affiliate_link

    def _save_products(self, products: list[ProductData]) -> None:
        """Save scraped products to product-centric JSON structure"""
        if not products:
            if self.debug_mode:
                self.logger.info("No products to save")
            return

        # Shorten affiliate links if enabled
        self._shorten_affiliate_links(products)

        # Convert ProductData objects to dictionaries and save manually
        # since Botasaurus output function isn't being called properly
        from .botasaurus_output import write_scraped_data_output

        # Convert ProductData objects to dictionaries
        product_dicts = []
        for product in products:
            product_dict = self._product_to_dict(product)
            product_dicts.append(product_dict)

        if self.debug_mode:
            self.logger.info("Saving %d products manually", len(product_dicts))

        # Call the output function directly
        write_scraped_data_output(
            {"manual_save": True}, product_dicts, output_dir=self.output_dir
        )

    def _product_to_dict(self, product: ProductData) -> dict[str, Any]:
        """Serialise a product for ``data.json``.

        Delegates to the record's own ``to_dict`` rather than restating the
        key set. The two used to be separate hand-written dicts and had
        drifted: ``pillar`` reached the file on the topic path, which writes
        through ``to_dict``, and never on this one, so the field looked wired
        while a resumed run saw no pillar at all.
        """
        return product.to_dict()

    def cleanup(self) -> None:
        """Cleanup resources after scraping to prevent memory leaks"""
        try:
            # Clean up browser instances if they exist
            if hasattr(self, "_browser_func"):
                try:
                    self._browser_func.close()
                    if self.debug_mode:
                        self.logger.info("Browser instances cleaned up")
                except Exception as e:
                    if self.debug_mode:
                        self.logger.debug("Browser cleanup warning: %s", e)

            # Clean up media download tasks
            try:
                download_media_files.close()
                if self.debug_mode:
                    self.logger.info("Media download tasks cleaned up")
            except Exception as e:
                if self.debug_mode:
                    self.logger.debug("Media download cleanup warning: %s", e)

        except Exception as e:
            self.logger.debug("General cleanup warning: %s", e)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit with automatic cleanup"""
        self.cleanup()


def build_argument_parser() -> argparse.ArgumentParser:
    """The scraper CLI parser.

    Extracted from `main` so a test can read what an omitted flag
    resolves to. `load_batch_config` resolves several arguments with an
    `is not None` sentinel, so a flag defaulting to a falsy value rather
    than to `None` makes the configured value unreachable -- and that is
    a property of the parser, not of any one run.
    """
    parser = argparse.ArgumentParser(
        description="Botasaurus Amazon Scraper for ContentEngineAI"
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        required=False,
        help=(
            "Keywords or ASINs to scrape - supports multiple values "
            "for batch mode (overrides config file)"
        ),
    )
    parser.add_argument(
        "--product-ids",
        nargs="+",
        required=False,
        help=(
            "Product IDs (ASINs) for batch scraping - supports multiple "
            "values (e.g., --product-ids B0ABC123 B0DEF456)"
        ),
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        metavar="N",
        help="Global cap on total products to collect across all keywords",
    )
    parser.add_argument(
        "--products-per-keyword",
        type=int,
        default=None,
        metavar="N",
        help="Maximum products to scrape per individual keyword",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero when any product failed, not only when none was "
            "scraped (default: a partial failure exits 0)"
        ),
    )
    parser.add_argument(
        "--fail-fast",
        # `BooleanOptionalAction` with `default=None`, not `store_true`. The
        # loader resolves this with `cli_fail_fast if cli_fail_fast is not
        # None`, so an omitted `store_true` flag arriving as False was
        # indistinguishable from one passed deliberately, and
        # `batch.fail_fast` in the YAML could never win. Same collision as the
        # chunked-keywords defect: a not-supplied sentinel meeting a supplied
        # value.
        #
        # The paired form matters once the default is None: `store_true` can
        # then only produce True or "unset", leaving a user who configured
        # `fail_fast: true` no way to ask for continue-on-error for one run.
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stop batch processing on first failure "
            "(default: batch.fail_fast in config/scraper.yaml, else continue)"
        ),
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

    # Video profile alignment
    parser.add_argument(
        "--profile",
        type=str,
        metavar="NAME",
        help=(
            "Video profile name to align media validation with producer "
            "requirements (e.g., slideshow_images4)"
        ),
    )

    # Batch input/output arguments
    parser.add_argument(
        "--input-file",
        metavar="FILE",
        help=(
            "Read product IDs or URLs from file (one per line), "
            "merged with --product-ids"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        metavar="N",
        help="Process products in batches of N (default: all at once)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help='Override output directory (default: "outputs" from config)',
    )

    return parser


def main():
    """Command-line interface for the Botasaurus Amazon scraper"""
    # Load .env BEFORE anything reads env vars. Without this, AMAZON_ASSOCIATE_TAG
    # (and any other secret in .env) is invisible to build_affiliate_url, which
    # silently falls back to returning the input URL unchanged: untagged
    # affiliate links end up in data.json. The global batch entry point in
    # src/pipeline/global_batch.py loads .env the same way for the same reason.
    from dotenv import load_dotenv

    load_dotenv()

    parser = build_argument_parser()

    args = parser.parse_args()

    # Written here rather than by `setup_debug_logging`, which this module
    # calls at import and so reaches every producer, publisher and batch
    # invocation. This marks an invoked run, not a completed scrape -- a
    # missing `--input-file` or no inputs at all still returns below, after
    # the marker and followed by its own error. `--help` and an argparse
    # error exit inside `parse_args` above and write nothing.
    logging.getLogger("AmazonScraper").info("=== AmazonScraper run starting ===")

    # --input-file: read product IDs/URLs from file and merge with --product-ids
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.is_absolute():
            input_path = Path(__file__).parent.parent.parent.parent / input_path
        if input_path.exists():
            with open(input_path, encoding="utf-8") as f:
                file_ids = [line.strip() for line in f if line.strip()]
            # Deduplicate while preserving order
            existing = list(args.product_ids or [])
            seen = set(existing)
            for fid in file_ids:
                if fid not in seen:
                    existing.append(fid)
                    seen.add(fid)
            args.product_ids = existing
            logger.info(
                "Loaded %d entries from %s (%d unique total)",
                len(file_ids),
                args.input_file,
                len(args.product_ids),
            )
        else:
            logger.error("Input file not found: %s", input_path)
            return

    # Load keywords/product_ids from config if not provided via CLI
    if not args.keywords and not args.product_ids:
        try:
            # Handle working directory changes from Botasaurus
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config/scraper.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                # Check batch configuration first
                batch_config = config.get("batch", {})
                batch_product_ids = batch_config.get("product_ids", [])
                # Flattened: the config groups keywords by pillar, and
                # iterating that dict yields the pillar names. A run with no
                # --keywords searched for "value" and "utility" instead of
                # any configured keyword.
                batch_keywords, _ = read_keyword_pillars(
                    batch_config.get("keywords", [])
                )

                # Use batch config if available
                if batch_product_ids or batch_keywords:
                    # Set to list or None (empty list = None to avoid confusion)
                    args.product_ids = batch_product_ids or None
                    args.keywords = batch_keywords or None

                    if batch_product_ids and batch_keywords:
                        logger.debug(
                            "Using batch mode from config: "
                            "%d product IDs, %d keywords",
                            len(batch_product_ids),
                            len(batch_keywords),
                        )
                    elif batch_product_ids:
                        logger.debug(
                            "Using batch product IDs from config: %s",
                            ", ".join(batch_product_ids),
                        )
                    else:
                        logger.debug(
                            "Using batch keywords from config: %s",
                            ", ".join(batch_keywords),
                        )
                else:
                    # Fallback to single-product mode keywords
                    amazon_config = config.get("scrapers", {}).get("amazon", {})
                    config_keywords = amazon_config.get("keywords", [])

                    if config_keywords:
                        args.keywords = config_keywords
                        logger.debug(
                            "Using keywords from config file: %s",
                            ", ".join(config_keywords),
                        )
                    else:
                        logger.error(
                            "No keywords/product_ids provided via CLI and none "
                            "found in config file"
                        )
                        logger.debug(
                            "Either use --keywords/--product-ids or add to "
                            "batch section in config/scraper.yaml"
                        )
                        return
            else:
                logger.error(
                    "No keywords provided via CLI and " "config file not found"
                )
                logger.debug(
                    "Use --keywords 'your keyword' to " "specify what to scrape"
                )
                return
        except Exception as e:
            logger.error("Error loading config file: %s", e)
            logger.debug("Use --keywords 'your keyword' to " "specify what to scrape")
            return

    # Setup debug mode early - before scraper instantiation
    # Check config file for debug mode if no CLI flag provided
    config_debug_mode = False
    if not args.debug and not args.verbose:
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config/scraper.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                config_debug_mode = config.get("global_settings", {}).get(
                    "debug_mode", False
                )
        except Exception:
            config_debug_mode = False

    # Reconfigure logging with proper debug settings
    # Determine debug mode from CLI or config
    debug_enabled = args.debug or args.verbose or config_debug_mode
    if debug_enabled:
        global DEBUG_MODE
        DEBUG_MODE = True

        # Reconfigure with debug settings
        setup_debug_logging(
            log_file=log_file,
            debug_mode=True,
            verbose=args.verbose,
            component_name="AmazonScraper",
            mark_run=False,  # already marked above; this is the same run
        )

    # Apply websocket filter to suppress cleanup messages
    websocket_filter = WebsocketFilter()
    logging.getLogger().addFilter(websocket_filter)
    logging.getLogger("websocket").addFilter(websocket_filter)

    # Log debug mode status messages only when debug is enabled
    if debug_enabled:
        if args.verbose:
            logger.debug("Verbose mode enabled - detailed logging active")
        elif config_debug_mode and not args.debug:
            logger.debug(
                "Debug mode enabled from config - browser visibility and "
                "detailed logging active"
            )
        else:
            logger.debug(
                "Debug mode enabled - browser visibility and detailed " "logging active"
            )

        logger.debug("Debug mode set globally for browser visibility")

        if args.pause_on_error:
            logger.debug(
                "Pause-on-error enabled - execution " "will pause when errors occur"
            )
        if args.save_screenshots:
            logger.debug("Screenshot saving enabled - " "key steps will be captured")
        if args.save_page_source:
            logger.debug(
                "Page source saving enabled - " "HTML will be saved for analysis"
            )
        if args.analyze_images:
            logger.debug("Deep image analysis enabled - " "all images will be analyzed")
        if args.dump_image_urls:
            logger.debug(
                "Image URL dumping enabled - " "all URLs will be saved to file"
            )

    if args.clean:
        import re
        import shutil

        # Clean all scraper outputs - comprehensive cleanup
        # Use absolute path to handle Botasaurus working directory changes
        project_root = Path(__file__).parent.parent.parent.parent
        base_output_path = project_root / get_output_path("base")
        if base_output_path.exists():
            logger.info("Cleaning all scraper outputs in: %s", base_output_path)

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
                        logger.debug("Cleaned product directory: %s", item)
                    # Remove other scraper directories (but preserve logs, reports)
                    elif item.name in ["cache", "temp", "screenshots"]:
                        shutil.rmtree(item)
                        logger.debug("Cleaned scraper directory: %s", item)
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
                    logger.debug("Cleaned scraper file: %s", item)

            logger.info("Cleanup completed - all scraper outputs removed")

    if args.debug:
        logger.debug("Debug mode enabled")
        from ...utils.outputs_paths import get_temp_directory

        temp_dir = get_temp_directory()
        logger.debug("Debug files will be saved to: %s", temp_dir)

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
        logger.error("Invalid search parameters:")
        for error in validation_errors:
            logger.error("   %s", error)
        return

    # Show search parameters in debug mode
    if args.debug:
        logger.debug("Search parameters configured:")
        if search_params.min_price or search_params.max_price:
            price_range = (
                f"${search_params.min_price or 0:.2f}-${search_params.max_price or '∞'}"
            )
            logger.debug("   Price range: %s", price_range)
        if search_params.min_rating:
            logger.debug("   Minimum rating: %s+ stars", search_params.min_rating)
        if search_params.prime_only:
            logger.debug("   Prime only: Yes")
        if search_params.free_shipping:
            logger.debug("   Free shipping: Yes")
        if search_params.brands:
            logger.debug("   Brands: %s", ", ".join(search_params.brands))
        if search_params.sort_order != "relevanceblender":
            logger.debug("   Sort: %s", search_params.sort_order)

        # Show config vs CLI override status
        config_defaults = get_default_search_parameters()
        if cli_overrides:
            logger.debug("   CLI overrides applied: %s", list(cli_overrides.keys()))
        if (
            search_params.min_price != config_defaults.min_price
            or search_params.max_price != config_defaults.max_price
        ):
            config_price_range = (
                f"${config_defaults.min_price or 0:.2f}-"
                f"${config_defaults.max_price or '∞'}"
            )
            logger.debug("   Config defaults: %s", config_price_range)

    # Detect batch mode: --product-ids provided OR multiple keywords
    is_batch_mode = bool(args.product_ids) or (args.keywords and len(args.keywords) > 1)

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

        # Resolve profile-aware media validation
        profile_uses_videos = None
        if getattr(args, "profile", None):
            try:
                from src.video.config_adapter import load_video_config_modular

                video_config = load_video_config_modular()
                profile_obj = video_config.video_profiles.get(args.profile)
                if profile_obj:
                    profile_uses_videos = profile_obj.use_scraped_videos
                    logger.info(
                        "Media validation aligned with profile '%s': " "videos %s",
                        args.profile,
                        "enabled" if profile_uses_videos else "disabled",
                    )
                else:
                    logger.warning(
                        "Profile '%s' not found, using default validation",
                        args.profile,
                    )
            except Exception as e:
                logger.warning(
                    "Could not load video config for profile alignment: %s", e
                )

        scraper = BotasaurusAmazonScraper(
            debug_override=debug_override,
            debug_options=debug_options,
            output_dir=getattr(args, "output_dir", None),
            profile_uses_videos=profile_uses_videos,
        )

        # Products successfully scraped, across whichever mode ran. Used to set
        # a non-zero exit code when nothing was scraped, so CI/cron see it.
        products_scraped = 0
        # Bound only by the batch arm; the single-keyword arm counts what it
        # returned and has no per-product failures to report.
        summary = None

        # Batch mode: use BatchController for multiple products
        if is_batch_mode:
            from .batch_controller import BatchController
            from .config import load_batch_config
            from .models import BatchSummary

            # --batch-size: split product_ids into chunks and run sequentially
            batch_size = getattr(args, "batch_size", None)
            all_product_ids = list(args.product_ids or [])
            all_keywords = list(args.keywords or [])

            if batch_size and all_product_ids:
                chunks = [
                    all_product_ids[i : i + batch_size]
                    for i in range(0, len(all_product_ids), batch_size)
                ]
                logger.info(
                    "Splitting %d products into %d batches of %d",
                    len(all_product_ids),
                    len(chunks),
                    batch_size,
                )
            else:
                chunks = [all_product_ids] if all_product_ids else [[]]

            total_summary: BatchSummary | None = None
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    logger.info(
                        "Batch %d/%d (%d products)",
                        chunk_idx + 1,
                        len(chunks),
                        len(chunk) if chunk else 0,
                    )

                # Load batch configuration with CLI precedence
                batch_config = load_batch_config(
                    cli_product_ids=chunk,
                    # `[]`, not `None`, for the later chunks. The loader reads
                    # `None` as "the CLI named no keywords" and falls back to
                    # the configured list, so a chunked `--product-ids` run
                    # searched every keyword in `scraper.yaml` from the second
                    # chunk on -- silently, since the log reads like a normal
                    # keyword run. The keywords belong to the first chunk
                    # because they are searched once for the whole run, not
                    # once per chunk.
                    cli_keywords=all_keywords if chunk_idx == 0 else [],
                    cli_fail_fast=args.fail_fast,
                    cli_max_products=args.max_products,
                    cli_products_per_keyword=args.products_per_keyword,
                )

                # Override search parameters in batch config with CLI parameters
                batch_config.search_params = search_params

                # Instantiate and run BatchController
                controller = BatchController(scraper, batch_config)
                summary = controller.run_batch()

                if total_summary is None:
                    total_summary = summary
                else:
                    total_summary.total_attempted += summary.total_attempted
                    total_summary.product_ids_attempted += summary.product_ids_attempted
                    total_summary.successful += summary.successful
                    total_summary.failed += summary.failed
                    total_summary.failed_products.extend(summary.failed_products)
                    # Merged like the product-level failures: without this a
                    # keyword lost in any chunk after the first is invisible
                    # to --strict, which is the loss the field exists for.
                    total_summary.failed_keywords.extend(summary.failed_keywords)
                    total_summary.duration_sec += summary.duration_sec

            if total_summary is None:
                logger.warning("No batches were processed")
                raise SystemExit(1)
            summary = total_summary
            products_scraped = summary.successful

            # Display final summary
            logger.info("--- SCRAPER SUMMARY ---")
            logger.info(
                "Products: %d attempted, %d successful, %d failed",
                summary.total_attempted,
                summary.successful,
                summary.failed,
            )
            if summary.successful_products:
                logger.info("Successful: %s", ", ".join(summary.successful_products))
            if summary.failed_products:
                logger.info("Failed: %s", ", ".join(summary.failed_products))
            images = summary.media_stats.get("total_images", 0)
            videos = summary.media_stats.get("total_videos", 0)
            logger.info("Images: %d, Videos: %d", images, videos)
            logger.info("Duration: %.1fs", summary.duration_sec)
            logger.info("---")

        # Single-product mode: use existing scraper.scrape_products()
        else:
            products = scraper.scrape_products(args.keywords, search_params)
            products_scraped = len(products)

            logger.info("--- SCRAPER SUMMARY ---")
            if products:
                logger.info(
                    "Products: %d scraped for keywords: %s",
                    len(products),
                    ", ".join(args.keywords),
                )
            else:
                logger.info("Products: 0 scraped")
            logger.info("---")

        if products_scraped == 0:
            logger.error("Scraper failed: 0 products scraped")
            raise SystemExit(1)

        # A partial failure exits 0 by default, matching the global batch:
        # a run that lost one product of twenty has done most of what was
        # asked. `--strict` is for a caller that would rather investigate
        # than lose a product silently.
        # Both kinds of loss: a product id that yielded nothing, and a
        # keyword whose search returned nothing or raised. The keyword arm
        # records no per-product result, so counting only `failed` would
        # make --strict a no-op on exactly the runs the docs use as
        # examples.
        failed = getattr(summary, "failed", 0)
        lost_keywords = list(getattr(summary, "failed_keywords", []) or [])
        if args.strict and (failed or lost_keywords):
            logger.error(
                "Scraper failed under --strict: %d scraped, %d products "
                "failed, %d keywords produced nothing%s",
                products_scraped,
                failed,
                len(lost_keywords),
                f" ({', '.join(lost_keywords)})" if lost_keywords else "",
            )
            raise SystemExit(1)

    except Exception as e:
        logger.error("Scraper failed: %s", e)
        if args.debug:
            import traceback

            logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Suppress module import warnings when running with -m
    import sys
    import warnings

    # Suppress frozen runpy warning that occurs when module is in sys.modules
    # before execution (common when using python -m package.module)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
    warnings.filterwarnings(
        "ignore", message=".*found in sys.modules.*", category=RuntimeWarning
    )

    main()
