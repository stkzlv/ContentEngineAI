#!/usr/bin/env python3
"""Botasaurus-powered Amazon scraper for ContentEngineAI

This module provides advanced web scraping capabilities for Amazon products using
the Botasaurus framework with built-in anti-detection and performance optimization.
"""

import asyncio
import concurrent.futures
import logging
import os
import shutil
import time
import traceback
import warnings
from typing import Any

import yaml

from src.scraper.base.keyword_pillars import (
    pillar_for,
    read_keyword_pillars,
)
from src.scraper.config_models import ScraperConfig
from src.utils.logging_setup import log_context
from src.utils.outputs_paths import get_project_root

from ...utils.url_shortener import load_url_shortener_settings
from ..base import BaseScraper, Platform, register_scraper
from ..base.models import BaseProductData, BaseSearchParameters
from ..base.throttle import (
    ThrottleSettings,
    ThrottleTracker,
    Verdict,
    is_error_page_failure,
)
from .browser_functions import (
    create_batch_browser_function,
    create_dynamic_browser_function,
)
from .config import CONFIG
from .constants import (
    DEFAULT_MIN_IMAGES_IF_NO_VIDEO,
    DEFAULT_MIN_IMAGES_WITH_VIDEO,
    DEFAULT_MIN_TOTAL_MEDIA,
    HIGH_RES_DIMENSION,
)
from .downloader import download_media_files
from .models import ProductData, SearchParameters

# The botasaurus_output/config/batch_controller relatives below are imported
# function-locally for readability only: this module already imports
# `browser_functions` above, so Botasaurus is resident by the time they run.
# Nothing in this package reaches into the video package any more -- the one
# call that did resolved a whole video config for a single boolean, which the
# CLI now takes as a flag and the batch computes for itself.
from .utils import validate_asin_format

# Logging is configured by `main()`, not here: an entry point owns its
# logging, an imported module does not. Configuring it at import to catch
# Botasaurus load errors pointed the ROOT logger at the production
# scraper.log for everything that imports `ProductData` -- the producer, the
# publisher, the batch, the test suite -- so one pytest session appended
# 430 KB to a real scrape history and rotated the oldest copy away. The
# window before `main()` runs is given up: a record there reaches logging's
# last-resort handler on stderr rather than the file.

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

    # Whether the target video profile uses scraped videos; None when unknown.
    # A class default so an instance built without __init__ (tests) has it.
    profile_uses_videos: bool | None = None

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
        # A per-run limit the batch sets (its per-keyword share); None means
        # the configured `scrapers.amazon.max_products`.
        self.run_max_products: int | None = None
        self.debug_options = debug_options or {}
        # Built on first use by pillar_for_keyword.
        self._keyword_pillars: dict[str, str] | None = None
        # Loaded once, and at construction rather than at first use, so a
        # malformed `config/url_shortener.yaml` is reported before a scrape
        # starts instead of after the browser work is paid for.
        self.url_shortener_settings = load_url_shortener_settings()
        # One tracker per scraper instance, which is one per run. Sharing it
        # across runs would let an earlier run's successes rule a later run's
        # first failure a dead query.
        self.throttle = ThrottleTracker(
            settings=ThrottleSettings.from_config(
                self.global_settings.get("rate_limiting")
            )
        )

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
        project_root = get_project_root()
        config_path = project_root / path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # Validated: a misspelled key, or an empty file, fails here, and
        # every key a reader indexes is present with the model's default.
        self.settings = ScraperConfig.from_legacy_dict(raw)
        return self.settings.to_runtime_dict()

    @property
    def effective_max_products(self) -> int:
        """The run's limit when the batch set one, else the configured one."""
        # getattr: tests build the scraper with `__new__` and set only what
        # they drive, and this property is reached from several of them.
        run_limit = getattr(self, "run_max_products", None)
        if run_limit is not None:
            return int(run_limit)
        return int(self.settings.amazon.max_products)

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
                else self.effective_max_products
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
        `validated_products` grow by the same `batch`), so that guard binds
        only when the limit is below `target_count` and never bounds a
        listing that fails validation. It is checked before each page, so a
        page that yields more than the remaining limit overshoots it.
        """
        validated_products: list[ProductData] = []
        total_raw_scraped = 0

        # Get batch processing config values
        batch_cfg = self.settings.global_settings.batch_processing
        max_attempts = batch_cfg.max_scrape_attempts
        prefetch_multiplier = batch_cfg.prefetch_multiplier
        max_batch_size = batch_cfg.max_batch_size
        max_pages = batch_cfg.max_pages

        current_page = 1

        self.logger.info(
            "Target: %d products that pass validation requirements",
            target_count,
        )

        while len(validated_products) < target_count:
            if total_raw_scraped >= max_attempts:
                self.logger.warning(
                    "Reached max scrape attempts (limit: %d validated "
                    "products). Stopping with %d/%d validated.",
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
                if (
                    keyword in self.throttle.dead_queries
                    or keyword in self.throttle.exhausted_inputs
                ):
                    # Amazon is answering this query with its error page in a
                    # run where other inputs got through. Every later page
                    # re-resolves the same query, costs a fresh Chrome
                    # session, and gets the same verdict, so paginating buys
                    # nothing. Same reasoning as the URL gate above: an input
                    # that cannot work does not get seven attempts at it.
                    self.logger.warning(
                        "Stopping pagination for %r: the throttle has given "
                        "up on it, so each later page would open a browser "
                        "session straight into the same failure.",
                        keyword,
                    )
                    break
                # No validated products from this page, so try the next one.
                # Exhaustion is not detected: `_scrape_single_pass` returns an
                # empty list both for a page of products that all failed
                # validation and for a page that held none, and the two are
                # indistinguishable here. Either way the loop advances until
                # `max_pages`, which is what makes the gate above worth having
                # for an input that can only ever resolve to one product.
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
                "extract_videos": self.profile_uses_videos is not False,
            }

            # Use the dynamic Botasaurus browser function with current debug settings
            if self.debug_mode:
                self.logger.debug(
                    "Creating dynamic browser function with " "DEBUG_MODE=%s",
                    self.debug_mode,
                )

            try:
                browser_func = create_dynamic_browser_function(self.debug_mode)
                if self.debug_mode:
                    self.logger.debug(
                        "browser_func type: %s",
                        type(browser_func),
                    )
                    self.logger.debug("browser_func: %s", browser_func)
                    self.logger.debug(
                        "Calling browser_func with data: %s",
                        data,
                    )
                results = self._scrape_with_retry(browser_func, data)
                self.logger.debug(
                    "browser_func returned %d products",
                    len(results) if results else 0,
                )
            except Exception as e:
                if self.debug_mode:
                    self.logger.debug("Error in browser function: %s", e)

                    self.logger.debug("Traceback: %s", traceback.format_exc())
                raise

            # Download media for scraped products
            if results:
                self._orchestrate_media_downloads(results, target_download_count)

            # Convert to ProductData and validate media requirements
            return self._validate_and_convert_products(
                results, filter_validated, products_limit
            )

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
        skip_videos = self.profile_uses_videos is False
        if skip_videos and any(result.get("videos") for result in results):
            self.logger.info(
                "Video downloads skipped: the target profile is image-only"
            )
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
                        "videos": [] if skip_videos else result.get("videos", []),
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
                # Both entry points reach the downloader through here, so
                # this is where its records get their product id.
                with log_context(product_id=task["asin"]):
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
                    "Processing download_results: " "type=%s, length=%s",
                    type(download_results),
                    length_str,
                )

            for i, dl_result in enumerate(download_results):
                if self.debug_mode:
                    self.logger.debug(
                        "Processing result %d: type=%s",
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
                            "Mapped download result for ASIN: "
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
                        "Skipping invalid result %d: " "%s, preview: %s...",
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
        self,
        results: list[dict],
        filter_validated: bool,
        products_limit: int | None = None,
    ) -> list[ProductData]:
        """Convert raw result dicts to ProductData and validate media.

        Args:
        ----
            results: List of product dicts with download info
            filter_validated: If True, return only products meeting media requirements
            products_limit: The caller's own limit for the final-verification
                summary; the run's limit when None

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
        # The caller's explicit limit when it has one; the run's otherwise.
        max_products = (
            products_limit
            if products_limit is not None
            else self.effective_max_products
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
                    "max_products": self.effective_max_products,
                    "page": start_page,
                    "extract_videos": self.profile_uses_videos is not False,
                }
            )

        batch_func = create_batch_browser_function(self.debug_mode, self.throttle)
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

    def _scrape_with_retry(self, browser_func, data, sleep=time.sleep):
        """Scrape one input, waiting out a rate limit and skipping a dead query.

        This used to be a fixed tenacity policy: three attempts, at most ten
        seconds apart. The block it exists for was measured clearing after
        several minutes, so every attempt landed inside it and the input was
        lost. Worse, the same policy was spent on a query that returns the
        error page permanently, which no amount of waiting fixes.

        The tracker decides which of the two it is from what the rest of the
        run did, so the schedule reaches minutes and a dead query stops
        consuming it. `sleep` is injectable so a test does not wait.
        """
        input_label = str(data.get("keyword") or data.get("input") or "unknown")

        while True:
            try:
                if self.debug_mode:
                    self.logger.debug("Attempting scrape with retry logic")
                result = browser_func(data)
            except RuntimeError as e:
                if not is_error_page_failure(e):
                    raise

                verdict = self.throttle.record_error_page(input_label)
                if verdict is Verdict.RETRY:
                    wait = self.throttle.backoff_sec(input_label)
                    self.logger.warning(
                        "Amazon returned its error page for %r. Nothing else "
                        "has succeeded since, so treating it as a rate limit "
                        "and waiting %.0fs before retrying.",
                        input_label,
                        wait,
                    )
                    sleep(wait)
                    continue
                raise

            # Same rule as the batch loop: an empty result is not evidence.
            # The wrapper returns one for every non-error-page exception, so
            # a keyword whose page died on a CDP timeout used to become the
            # run's proof that the connection works.
            if result:
                self.throttle.record_success(input_label)
            return result

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


def main() -> None:
    """Run the scraper's command line, which lives in `cli.py`.

    Kept here because `python -m src.scraper.amazon.scraper` is the documented
    invocation and the package re-exports this name.
    """
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    # Suppress module import warnings when running with -m
    import warnings

    # Suppress frozen runpy warning that occurs when module is in sys.modules
    # before execution (common when using python -m package.module)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
    warnings.filterwarnings(
        "ignore", message=".*found in sys.modules.*", category=RuntimeWarning
    )

    main()
