"""Browser automation functions for Amazon scraper.

This module contains browser automation functions that handle the actual
web scraping logic using Botasaurus framework.
"""

import contextlib
import logging
import os
import platform
import re
import shutil
from typing import Any

from botasaurus.browser import Driver, browser

from ..base.display import resolve_debug_display
from .botasaurus_output import get_browser_config_for_outputs
from .config import _BROWSER_CONFIG, CONFIG
from .media_extractor import (
    extract_functional_videos_with_validation,
    extract_high_res_images_botasaurus,
)
from .product_extractor import (  # noqa: F401
    extract_product_data_from_page,
    extract_serp_product_info,
)
from .search_builder import SearchParameterBuilder
from .utils import build_affiliate_url, detect_monitors, get_optimal_browser_position

# Module-level logger for browser functions
logger = logging.getLogger(__name__)


def scrape_amazon_products_browser_impl(
    driver: Driver, data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Unified function to scrape Amazon products in a single browser session"""
    DEBUG_MODE = data.get("debug_mode", False)

    logger.info(
        "[DEBUG] scrape_amazon_products_browser called with keyword: %s",
        data.get("keyword"),
    )
    keyword = data["keyword"]
    is_asin = data.get("is_asin", False)
    is_url = data.get("is_url", False)
    logger.info("[DEBUG] is_asin: %s, is_url: %s", is_asin, is_url)
    products: list[dict[str, Any]] = []

    # Get Amazon base URL from config (used by both ASIN and search paths)

    base_url = (
        CONFIG.get("scrapers", {})
        .get("amazon", {})
        .get("base_url", "https://www.amazon.com")
    )

    # Initialize variables used by final debug verification (set per-branch below)
    global_settings = CONFIG.get("global_settings", {})
    count_products_with_media = global_settings.get("count_products_with_media", False)
    products_with_media_count = 0
    max_products = data.get("max_products", 1)

    if is_url:
        # Direct URL navigation — follow redirects (e.g. shortened URLs → Amazon)
        logger.info("Navigating to URL: %s", keyword)
        driver.google_get(keyword, bypass_cloudflare=True)
        driver.short_random_sleep()

        # Extract ASIN from the final (redirected) URL
        final_url = driver.current_url
        asin_match = re.search(r"/dp/([A-Z0-9]{10})", final_url)
        if not asin_match:
            asin_match = re.search(r"/gp/product/([A-Z0-9]{10})", final_url)
        if not asin_match:
            logger.warning("Could not extract ASIN from redirected URL: %s", final_url)
            return products

        asin = asin_match.group(1)
        logger.info("Resolved URL → ASIN: %s (from %s)", asin, final_url)

        # Extract product data using the resolved ASIN
        product_data = extract_product_data_from_page(
            driver,
            asin,
            asin,
            debug_mode=DEBUG_MODE,
            debug_options=data.get("debug_options"),
        )
        if product_data:
            products.append(product_data)

    elif is_asin:
        # Direct product page scraping for ASIN
        product_url = f"{base_url}/dp/{keyword}"
        logger.info("[DEBUG] Calling scrape_single_product for ASIN: %s", keyword)

        # Initialize variables needed by debug verification
        global_settings = CONFIG.get("global_settings", {})
        count_products_with_media = global_settings.get(
            "count_products_with_media", False
        )
        products_with_media_count = 0

        # For single ASIN, max_products is always 1
        max_products = 1

        # Note: scrape_single_product is now a method in BotasaurusAmazonScraper class
        # This function now focuses on the browser-level logic
        # The actual product scraping will be delegated to the class method

        if DEBUG_MODE:
            logger.info("Scraping product: %s", keyword)

        # Use google_get for organic navigation
        driver.google_get(product_url, bypass_cloudflare=True)

        # Force browser maximization programmatically for debug mode
        if DEBUG_MODE:
            try:
                # Use Botasaurus-compatible window sizing method
                # Get window dimensions from config
                global_settings = CONFIG.get("global_settings", {})
                browser_config = global_settings.get("browser_config", {})
                width = browser_config.get("debug_window_width", 1920)
                height = browser_config.get("debug_window_height", 1200)

                driver.run_js(f"window.resizeTo({width}, {height});")
                logger.debug(
                    "[DEBUG] Set browser window size to %sx%s via JavaScript",
                    width,
                    height,
                )
            except Exception as e:
                logger.warning("[DEBUG] Could not set window size: %s", e)

        driver.short_random_sleep()

        # Check for regional redirect
        from .utils import detect_regional_redirect

        redirected, redirect_info = detect_regional_redirect(driver, product_url)
        if redirected and DEBUG_MODE:
            logger.warning("Regional redirect detected: %s", redirect_info)

        # Extract product data directly here for browser function
        product_data = extract_product_data_from_page(
            driver,
            keyword,
            keyword,
            debug_mode=DEBUG_MODE,
            debug_options=data.get("debug_options"),
        )
        if product_data:
            products.append(product_data)
            # Count product with media for verification
            if count_products_with_media:
                has_images = product_data.get("images", [])
                has_videos = product_data.get("videos", [])
                if has_images or has_videos:
                    products_with_media_count += 1
    else:
        # Search-based scraping with anti-detection and advanced parameters
        search_params = data.get("search_params")  # SearchParameters object

        # Build search URL with parameters
        url_builder = SearchParameterBuilder(base_url)
        page = data.get("page", 1)
        search_url = url_builder.build_search_url(keyword, search_params, page=page)

        if DEBUG_MODE:
            url_builder.log_search_parameters(keyword, search_params)
            logger.info("Searching: %s", search_url)
            # Take debug screenshot if enabled in config
            try:
                save_screenshots = (
                    CONFIG.get("global_settings", {})
                    .get("debug_settings", {})
                    .get("save_screenshots", False)
                )
                if save_screenshots:
                    driver.save_screenshot()  # Debug screenshot
            except Exception as e:
                if DEBUG_MODE:
                    logger.warning("[DEBUG] Screenshot failed: %s", e)

        # Use google_get for organic navigation pattern
        if DEBUG_MODE:
            import time

            nav_start = time.time()
            logger.debug("[DEBUG] Navigating to search URL...")

        try:
            if DEBUG_MODE:
                logger.debug("[DEBUG] Starting navigation to search page...")

            logger.info("Navigating to search URL: %s", search_url)

            # CRITICAL WORKAROUND: Botasaurus headless mode bug
            # Browser starts with no tabs - must create initial tab manually
            try:
                # Attempt to access browser to ensure it's initialized
                _ = driver._browser
                # Check if we have any tabs
                if not driver._browser.targets or not any(
                    t._target.type_ == "page" for t in driver._browser.targets
                ):
                    logger.debug("Creating initial tab for browser")
                    # Create new tab using Chrome DevTools Protocol
                    driver._browser.connection.send(
                        "Target.createTarget", {"url": "about:blank"}
                    )
            except Exception as e:
                logger.debug("Tab creation attempt: %s", e)

            # Use google_get for organic navigation pattern (same as working version)
            driver.google_get(search_url, bypass_cloudflare=True)

            # Get page info
            current_url = driver.current_url
            page_title = driver.title[:80] if driver.title else "No title"
            logger.info("Loaded page: %s", page_title)
            if DEBUG_MODE:
                logger.debug("Current URL: %s", current_url)

            if DEBUG_MODE:
                logger.info("[DEBUG] Navigation completed successfully")

        except Exception as e:
            import traceback

            logger.error("[DEBUG] Navigation failed: %s: %s", type(e).__name__, e)
            logger.debug("[DEBUG] Traceback: %s", traceback.format_exc())
            return []

        # Force browser maximization programmatically for debug mode
        if DEBUG_MODE:
            try:
                # Use Botasaurus-compatible window sizing method
                # Get window dimensions from config
                global_settings = CONFIG.get("global_settings", {})
                browser_config = global_settings.get("browser_config", {})
                width = browser_config.get("debug_window_width", 1920)
                height = browser_config.get("debug_window_height", 1200)

                driver.run_js(f"window.resizeTo({width}, {height});")
                logger.debug(
                    "[DEBUG] Set browser window size to %sx%s via JavaScript",
                    width,
                    height,
                )
            except Exception as e:
                logger.warning("[DEBUG] Could not set window size: %s", e)

        if DEBUG_MODE:
            logger.debug("[DEBUG] Short sleep before continuing...")
        driver.short_random_sleep()

        if DEBUG_MODE:
            nav_time = time.time() - nav_start
            logger.debug("[DEBUG] Navigation completed in %.2f seconds", nav_time)

            # Add page info for debugging and error detection
            try:
                current_url = driver.current_url
                page_title = driver.title[:50] if driver.title else "No title"
                logger.debug("[DEBUG] Current URL: %s", current_url)
                logger.debug("[DEBUG] Page title: %s", page_title)

                # Check for Amazon error pages
                if page_title and "Sorry! Something went wrong!" in driver.title:
                    logger.warning(
                        "[DEBUG] Detected Amazon error page - triggering retry"
                    )
                    raise RuntimeError(
                        "Amazon error page detected: Sorry! Something went wrong!"
                    )

            except RuntimeError:
                # Re-raise the error page detection
                raise
            except Exception as e:
                logger.warning("[DEBUG] Could not get page info: %s", e)

        # Check for CAPTCHA or bot detection
        if driver.is_bot_detected():
            logger.error("Bot detection triggered!")
            logger.error("[DEBUG] Bot detection triggered - returning 0 products")
            if DEBUG_MODE:
                # Take error screenshot if enabled in config (default: enabled)
                try:
                    save_error_screenshots = (
                        CONFIG.get("global_settings", {})
                        .get("debug_settings", {})
                        .get("save_error_screenshots", True)
                    )
                    if save_error_screenshots:
                        driver.save_screenshot()
                        logger.warning(
                            "[DEBUG] Bot detection triggered" " - screenshot saved"
                        )
                    else:
                        logger.warning(
                            "[DEBUG] Bot detection triggered" " - screenshot disabled"
                        )
                except Exception:
                    driver.save_screenshot()  # Fallback to always save
                    logger.warning(
                        "[DEBUG] Bot detection triggered" " - screenshot saved"
                    )
                logger.debug(
                    "[DEBUG] Continuing without manual intervention for browser "
                    "visibility testing"
                )
            return []

        # Get product cards from search results (with timeout to prevent long waits)
        if DEBUG_MODE:
            logger.info("[DEBUG] Searching for product cards...")

        try:
            # Get search result selector from config - try more specific selectors first

            # More comprehensive list of selectors for Amazon search results
            product_selectors = [
                # Products with ASIN
                "div[data-component-type='s-search-result'][data-asin]",
                "div[data-component-type='s-search-result']",  # All search results
                "[data-component-type='s-search-result']",  # Any element type
                "div[data-asin]:not([data-asin=''])",  # Any div with non-empty ASIN
                "[data-asin]:not([data-asin=''])",  # Any element with non-empty ASIN
                "div.s-result-item[data-asin]",  # Classic result item
                ".s-result-item[data-asin]",  # Result item class
                "div.s-result-item",  # Any result item
                ".s-result-item",  # Result item class fallback
                "div[class*='result']",  # Any div with 'result' in class
                "h2 a[href*='/dp/']",  # Product title links
                "a[href*='/dp/']",  # Any product links
            ]

            product_cards = []
            search_selector = None  # Track which selector worked

            # First, wait for any product content to load
            # (critical for headless mode)
            try:
                driver.wait_for_element(product_selectors[0])
            except Exception as e:
                # Continue even if first selector fails
                logger.debug("Initial selector wait failed: %s", e)

            for selector in product_selectors:
                if DEBUG_MODE:
                    logger.info("[DEBUG] Trying selector: %s", selector)
                try:
                    # Now try to select without wait since we already waited
                    cards = driver.select_all(selector)
                    if cards and len(cards) > 0:
                        if "h2 a" in selector or "a[href*='/dp/']" in selector:
                            # If we selected links, get their parent containers
                            parent_cards = []
                            for card in cards[
                                :10
                            ]:  # Limit to first 10 to avoid too many
                                # Try to find a reasonable parent container
                                parent = card
                                # Walk up to find a container with meaningful content
                                for _ in range(5):  # Max 5 levels up
                                    parent = (
                                        parent.parent
                                        if hasattr(parent, "parent") and parent.parent
                                        else parent
                                    )
                                    if parent and hasattr(parent, "get_attribute"):
                                        data_asin = parent.get_attribute("data-asin")
                                        if data_asin and data_asin.strip():
                                            break
                                if parent:
                                    parent_cards.append(parent)
                            product_cards = parent_cards
                        else:
                            product_cards = cards

                        search_selector = selector
                        if DEBUG_MODE:
                            logger.info("[DEBUG] Using selector: %s", selector)
                            logger.info(
                                "[DEBUG] Found %d product cards",
                                len(product_cards),
                            )
                        break
                except Exception as e:
                    if DEBUG_MODE:
                        logger.error(
                            "[DEBUG] Selector '%s' failed: %s",
                            selector,
                            str(e)[:100],
                        )
                    continue

            # If no cards found, try one final wait for basic content
            if not product_cards:
                if DEBUG_MODE:
                    logger.warning(
                        "[DEBUG] No product cards found with immediate selectors, "
                        "trying with wait..."
                    )
                try:
                    # Wait for page to load basic content
                    driver.wait_for_element("body", 3)
                    # Try simple selectors with wait
                    for selector in [
                        "div[data-asin]",
                        "[data-asin]",
                        "a[href*='/dp/']",
                    ]:
                        try:
                            driver.wait_for_element(selector, 2)
                            cards = driver.select_all(selector)
                            if cards:
                                product_cards = cards[:5]  # Limit to first 5
                                search_selector = selector
                                if DEBUG_MODE:
                                    logger.debug(
                                        "[DEBUG] Found %d cards with wait: %s",
                                        len(product_cards),
                                        selector,
                                    )
                                break
                        except Exception as e:
                            if DEBUG_MODE:
                                logger.warning(
                                    "[DEBUG] Wait failed for selector %s: %s",
                                    selector,
                                    e,
                                )
                            continue
                except Exception:
                    if DEBUG_MODE:
                        logger.error("[DEBUG] Even basic wait failed")

            if DEBUG_MODE:
                logger.info(
                    "[DEBUG] Final result: %d product cards",
                    len(product_cards),
                )
        except Exception as e:
            if DEBUG_MODE:
                logger.warning("[DEBUG] Exception in product card search: %s", e)
            product_cards = []

        if not product_cards:
            if DEBUG_MODE:
                logger.error("[DEBUG] No product cards found")
                # Take error screenshot if enabled in config (default: enabled)
                try:
                    save_error_screenshots = (
                        CONFIG.get("global_settings", {})
                        .get("debug_settings", {})
                        .get("save_error_screenshots", True)
                    )
                    if save_error_screenshots:
                        driver.save_screenshot()
                except Exception:
                    driver.save_screenshot()  # Fallback to always save on errors
            return []

        # Extract products from search results
        global_settings = CONFIG.get("global_settings", {})
        browser_config = global_settings.get("browser_config", {})
        default_max = browser_config.get("max_products_per_search", 5)
        max_products = data.get("max_products", default_max)

        # Enhanced collection logic: count products with media files (requirement #20)
        count_products_with_media = global_settings.get(
            "count_products_with_media", False
        )
        products_with_media_count = 0

        # Track processed ASINs to avoid duplicates
        processed_asins = set()

        i = 0
        while i < len(product_cards):
            try:
                # Get current card by index (avoids stale DOM reference)
                card = product_cards[i]

                # Extract product info from card
                serp_info = extract_serp_product_info(card, keyword)
                if DEBUG_MODE:
                    logger.info(
                        "[DEBUG] Processing card %d/%d: serp_info=%s, url=%s",
                        i + 1,
                        len(product_cards),
                        "" if serp_info else "",
                        "" if serp_info and serp_info.url else "",
                    )

                if serp_info and serp_info.url:
                    # Check for duplicate ASIN before processing
                    if serp_info.asin in processed_asins:
                        if DEBUG_MODE:
                            logger.warning(
                                "[DEBUG] Skipping duplicate ASIN: %s",
                                serp_info.asin,
                            )
                        i += 1
                        continue

                    # Mark ASIN as processed
                    processed_asins.add(serp_info.asin)

                    logger.info(
                        "Processing product %d/%d: %s",
                        i + 1,
                        max_products,
                        serp_info.asin,
                    )

                    # Navigate to product page
                    driver.google_get(serp_info.url, bypass_cloudflare=True)
                    driver.short_random_sleep()
                    # Extract full product data
                    product_data = extract_product_data_from_page(
                        driver,
                        serp_info.asin,
                        keyword,
                        serp_info,
                        debug_mode=DEBUG_MODE,
                        debug_options=data.get("debug_options"),
                    )
                    if DEBUG_MODE:
                        result_status = "" if product_data else ""
                        logger.info(
                            "[DEBUG] Product extraction result: %s",
                            result_status,
                        )

                    # Count and validate products with media files
                    if product_data:
                        if count_products_with_media:
                            # Check if product has media URLs
                            has_media_urls = (
                                product_data.get("images")
                                and len(product_data.get("images", [])) > 0
                            ) or (
                                product_data.get("videos")
                                and len(product_data.get("videos", [])) > 0
                            )

                            if has_media_urls:
                                products.append(product_data)
                                products_with_media_count += 1
                                if DEBUG_MODE:
                                    img_count = len(product_data.get("images", []))
                                    vid_count = len(product_data.get("videos", []))
                                    logger.info(
                                        "[DEBUG] Product %d/%d with media "
                                        "(ASIN: %s, %d images, %d videos)",
                                        products_with_media_count,
                                        max_products,
                                        serp_info.asin,
                                        img_count,
                                        vid_count,
                                    )

                                # Stop when we have enough products with media
                                if products_with_media_count >= max_products:
                                    if DEBUG_MODE:
                                        logger.info(
                                            "[DEBUG] Reached target: "
                                            "%d products with media files!",
                                            max_products,
                                        )
                                    break
                            else:
                                if DEBUG_MODE:
                                    logger.warning(
                                        "[DEBUG] Product %d "
                                        "(ASIN: %s) has no media URLs",
                                        i + 1,
                                        serp_info.asin,
                                    )
                        else:
                            # Traditional count: stop when we have max_products total
                            products.append(product_data)
                            if len(products) >= max_products:
                                if DEBUG_MODE:
                                    logger.info(
                                        "[DEBUG] Reached target: " "%d products total",
                                        max_products,
                                    )
                                break

                    # Navigate back to search results page for next product
                    current_count = (
                        products_with_media_count
                        if count_products_with_media
                        else len(products)
                    )
                    if current_count < max_products and i < len(product_cards) - 1:
                        if DEBUG_MODE:
                            logger.debug("[DEBUG] Navigating back to search results...")
                        try:
                            driver.google_get(search_url, bypass_cloudflare=True)
                            driver.short_random_sleep()
                            # Re-find product cards since DOM is fresh
                            driver.wait_for_element(search_selector)
                            new_product_cards = driver.select_all(search_selector)
                            if new_product_cards and len(new_product_cards) > i:
                                product_cards = new_product_cards
                                if DEBUG_MODE:
                                    logger.info(
                                        "[DEBUG] Found %d cards",
                                        len(product_cards),
                                    )
                            else:
                                if DEBUG_MODE:
                                    logger.warning(
                                        "[DEBUG] No product cards" " after navigation"
                                    )
                                break
                        except Exception as e:
                            if DEBUG_MODE:
                                logger.warning("[DEBUG] Navigation failed: %s", e)
                            break  # Break if navigation fails
                else:
                    if DEBUG_MODE:
                        logger.warning(
                            "[DEBUG] Skipping card %d - no valid product info",
                            i + 1,
                        )

                # Move to next card
                i += 1

            except Exception as e:
                if DEBUG_MODE:
                    logger.error("[DEBUG] Error processing card %d: %s", i + 1, e)
                i += 1  # Continue to next card instead of breaking
                continue

    # Final verification
    if DEBUG_MODE:
        logger.info("[DEBUG] Extracted %d products total", len(products))

        # Verify we have the expected number of products
        if count_products_with_media:
            expected_count = max_products
            actual_count = products_with_media_count
            logger.info(
                "[VERIFICATION] Expected %d, got %d",
                expected_count,
                actual_count,
            )

            # Verify each product has media URLs
            for idx, product in enumerate(products):
                asin = product.get("asin", "Unknown")
                img_urls = product.get("images", [])
                vid_urls = product.get("videos", [])
                img_count = len(img_urls)
                vid_count = len(vid_urls)

                logger.info(
                    "Product %d (%s): %dimg, %dvid",
                    idx + 1,
                    asin,
                    img_count,
                    vid_count,
                )

                if img_count == 0 and vid_count == 0:
                    logger.error(
                        "[VERIFICATION] ERROR: Product %s has no media URLs!",
                        asin,
                    )
                else:
                    logger.info("[VERIFICATION] Product %s has media URLs", asin)

            if actual_count == expected_count:
                logger.info(
                    "SUCCESS: Got exactly %d products with media!",
                    expected_count,
                )
            else:
                logger.warning(
                    "WARNING: Expected %d but got %d",
                    expected_count,
                    actual_count,
                )
        else:
            logger.info("Traditional mode: %d products extracted", len(products))

    return products


def _build_browser_config(debug_mode=False):
    """Build Botasaurus browser config dict.

    Shared by both single-keyword and batch browser functions so Chrome
    options stay consistent.
    """
    try:
        current_config = _BROWSER_CONFIG.copy() if _BROWSER_CONFIG else {}
    except Exception:
        current_config = {}

    # Keep Chrome on X11 (Xvfb when normal, Xwayland when debug), never Wayland. If
    # WAYLAND_DISPLAY is set, Chromium can pick the Wayland backend and draw a real
    # window even in normal/invisible mode, defeating the Xvfb virtual display.
    os.environ.pop("WAYLAND_DISPLAY", None)

    force_real_browser = debug_mode

    if force_real_browser:
        display_info = resolve_debug_display()
        if display_info.source == "none":
            logger.warning(
                "No real display found for debug run; falling back to a virtual "
                "display, so the browser window will not be visible. Run from a "
                "desktop session to watch the browser."
            )
            current_config["enable_xvfb_virtual_display"] = True
        else:
            current_config["enable_xvfb_virtual_display"] = False
            os.environ["DISPLAY"] = display_info.display or ":0"
            if display_info.xauthority:
                os.environ["XAUTHORITY"] = display_info.xauthority
            logger.info(
                "Using display %s (source: %s, xauthority: %s)",
                display_info.display,
                display_info.source,
                display_info.xauthority or "none",
            )

        logger.info("Auto-discovering multi-monitor setup...")
        monitors = detect_monitors()
        browser_x, browser_y, browser_width, browser_height = (
            get_optimal_browser_position(monitors)
        )

        logger.info("Detected %d monitor(s)", len(monitors))
        for i, monitor in enumerate(monitors):
            primary_str = " (PRIMARY)" if monitor.get("primary") else ""
            logger.info(
                "   Monitor %d: %dx%d at +%d+%d%s",
                i + 1,
                monitor["width"],
                monitor["height"],
                monitor["x"],
                monitor["y"],
                primary_str,
            )

        logger.info(
            "Browser maximized on primary monitor: %d,%d size: %dx%d",
            browser_x,
            browser_y,
            browser_width,
            browser_height,
        )

        chrome_args = current_config.get("add_arguments", [])
        chrome_args.extend(
            [
                "--start-maximized",
                "--disable-extensions",
                "--disable-plugins",
                f"--window-size={browser_width},{browser_height}",
                f"--window-position={browser_x},{browser_y}",
                "--new-window",
                "--force-device-scale-factor=1",
                "--disable-dev-shm-usage",
                "--disable-features=VizDisplayCompositor",
                "--no-default-browser-check",
                "--no-first-run",
                "--disable-infobars",
                "--remote-debugging-port=0",
                "--enable-logging",
                "--v=1",
            ]
        )
        current_config["add_arguments"] = chrome_args

        current_config.update(
            {
                "block_images": False,
                "cache": False,
                "max_retry": 1,
                "window_size": (browser_width, browser_height),
            }
        )

        logger.info(
            "Debug mode enabled - browser window will be visible on your screen"
        )
        logger.info(
            "Maximized positioning: --window-position=%d,%d --window-size=%d,%d",
            browser_x,
            browser_y,
            browser_width,
            browser_height,
        )
    else:
        # Normal runs: no real display on Wayland, so let Botasaurus start its own Xvfb
        # virtual display (headful but invisible). is_vmish is false on a desktop, so
        # the flag must be set explicitly or no display starts and google_get hangs 60s.
        current_config["enable_xvfb_virtual_display"] = True
        if shutil.which("Xvfb") is None:
            logger.warning(
                "Xvfb binary not found; Botasaurus will fall back to detectable "
                "headless mode and likely crash. Install it: "
                "sudo apt-get install -y xvfb"
            )

    is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER", False)
    is_ci = os.environ.get("CI", False)
    has_display = os.environ.get("DISPLAY") is not None

    if debug_mode:
        logger.info("Environment detection:")
        logger.info("   • Platform: %s", platform.system())
        logger.info("   • Is Docker: %s", is_docker)
        logger.info("   • Is CI: %s", is_ci)
        logger.info("   • Has DISPLAY: %s", has_display)
        logger.info("   • DISPLAY value: %s", os.environ.get("DISPLAY", "Not set"))

    current_config.update(
        {
            "headless": False,
            "close_on_crash": not debug_mode,
            "reuse_driver": False,
            "create_driver": True,
        }
    )

    output_config = get_browser_config_for_outputs()
    current_config.update(output_config)

    chrome_args = current_config.get("add_arguments", [])
    if not any("--timeout" in arg for arg in chrome_args):
        global_settings = CONFIG.get("global_settings", {})
        browser_config = global_settings.get("browser_config", {})
        page_timeout = browser_config.get("page_load_timeout_ms", 60000)
        script_timeout = browser_config.get("script_execution_timeout_ms", 30000)

        chrome_args.extend(
            [
                f"--timeout={page_timeout}",
                f"--script-timeout={script_timeout}",
            ]
        )
    current_config["add_arguments"] = chrome_args

    return current_config


def create_dynamic_browser_function(debug_mode=False):
    """Create browser function with current DEBUG_MODE settings"""
    DEBUG_MODE = debug_mode
    current_config = _build_browser_config(debug_mode)

    @browser(**current_config)
    def scrape_amazon_products_browser(
        driver: Driver, data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        try:
            if DEBUG_MODE:
                logger.debug(
                    "Using Botasaurus explicit wait pattern "
                    "(no implicit wait needed)"
                )

            return scrape_amazon_products_browser_impl(driver, data)
        except Exception as e:
            if DEBUG_MODE:
                logger.error("[DEBUG] Browser function error: %s", e)
                import traceback

                logger.debug(traceback.format_exc())
            return []

    return scrape_amazon_products_browser


def create_batch_browser_function(debug_mode=False):
    """Create a browser function that scrapes multiple inputs in one Chrome session.

    Each input is scraped sequentially within the same browser, avoiding the
    ~15s Chrome startup overhead per keyword that the single-keyword function has.
    """
    current_config = _build_browser_config(debug_mode)

    @browser(**current_config)
    def scrape_batch(driver: Driver, data: dict[str, Any]) -> list[dict[str, Any]]:
        items = data.get("items", [])
        results = []
        for i, item in enumerate(items):
            input_label = item.get("keyword", f"item-{i}")
            try:
                logger.info(
                    "[batch %d/%d] Scraping: %s", i + 1, len(items), input_label
                )
                products = scrape_amazon_products_browser_impl(driver, item)
                results.append({"input": input_label, "products": products or []})
            except Exception as e:
                logger.error(
                    "[batch %d/%d] Failed for %s: %s",
                    i + 1,
                    len(items),
                    input_label,
                    e,
                )
                results.append({"input": input_label, "products": [], "error": str(e)})
        return results

    return scrape_batch


def scrape_single_product(
    driver: Driver,
    product_info: dict[str, Any],
    debug_mode=False,
    debug_options: dict = None,
) -> dict[str, Any]:
    """Helper function to scrape a single product page"""
    DEBUG_MODE = debug_mode

    if DEBUG_MODE:
        logger.info("Scraping product: %s", product_info.get("asin", "Unknown"))

    # Use google_get for organic navigation
    original_url = product_info["url"]
    driver.google_get(original_url, bypass_cloudflare=True)

    # Mute all videos globally to prevent audio during scraping
    with contextlib.suppress(Exception):
        driver.run_js("""
            // Mute all existing videos
            document.querySelectorAll('video').forEach(video => {
                video.muted = true;
                video.volume = 0;
            });

            // Mute all future videos using MutationObserver
            const observer = new MutationObserver((mutations) => {
                document.querySelectorAll('video').forEach(video => {
                    if (!video.muted) {
                        video.muted = true;
                        video.volume = 0;
                    }
                });
            });

            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        """)

    if DEBUG_MODE:
        logger.info("[DEBUG] Browser should be visible now - Amazon page loaded!")
        logger.info("[DEBUG] Current URL: %s", driver.current_url)
        try:
            page_title = driver.title
            logger.info("[DEBUG] Page title: %s", page_title)

            # Check for Amazon error pages
            if page_title and "Sorry! Something went wrong!" in page_title:
                logger.warning(
                    "[DEBUG] Detected Amazon error page in product page - "
                    "triggering retry"
                )
                raise RuntimeError(
                    "Amazon error page detected: Sorry! Something went wrong!"
                )

        except RuntimeError:
            # Re-raise the error page detection
            raise
        except Exception as e:
            logger.info("[DEBUG] Page title: Unable to get (%s)", e)

        # Additional browser visibility information
        logger.info("[DEBUG] Browser window info:")
        try:
            current_url = driver.current_url
            logger.info("   • Current URL: %s", current_url)
        except Exception as e:
            logger.info("   • Current URL: Unable to get (%s)", e)

        try:
            # Check window size using JavaScript (Botasaurus-compatible)
            window_size = driver.run_js(
                "return {width: window.outerWidth, height: window.outerHeight};"
            )
            logger.info(
                "   • Window size: %sx%s",
                window_size.get("width", "Unknown"),
                window_size.get("height", "Unknown"),
            )

            # Check window position using JavaScript
            window_position = driver.run_js(
                "return {x: window.screenX, y: window.screenY};"
            )
            logger.info(
                "   • Window position: %s,%s",
                window_position.get("x", "Unknown"),
                window_position.get("y", "Unknown"),
            )
        except Exception as e:
            logger.info("   • Window info: Unable to get (%s)", e)

        try:
            # Use driver properties instead of methods
            logger.info("   • Browser session active: Yes")
        except Exception as e:
            logger.info("   • Browser session: Error (%s)", e)

        logger.info("   • Driver type: %s", type(driver).__name__)
        logger.info("   • Browser name: %s", getattr(driver, "name", "Unknown"))

        try:
            from .config import CONFIG

            debug_pause = (
                CONFIG.get("global_settings", {})
                .get("rate_limiting", {})
                .get("debug_pause_duration", 5)
            )
        except Exception:
            debug_pause = 5
        logger.debug(
            "[DEBUG] Pausing for %s seconds so you can see the browser...",
            debug_pause,
        )
        import time

        time.sleep(debug_pause)

    # Check for regional redirect first
    from .utils import detect_regional_redirect

    redirected, redirect_info = detect_regional_redirect(driver, original_url)
    if redirected and DEBUG_MODE:
        logger.warning("Regional redirect detected: %s", redirect_info)

    # Extract title
    title = ""
    # Get title selectors from config
    css_selectors = CONFIG.get("global_settings", {}).get("css_selectors", {})
    title_selectors = css_selectors.get(
        "product_title_selectors",
        [
            "#productTitle",
            "h1.a-size-large",
            ".product-title",
            "h1[data-automation-id='product-title']",
        ],
    )

    for selector in title_selectors:
        title_element = driver.select(selector)
        if title_element:
            title = title_element.text.strip()
            if DEBUG_MODE:
                logger.info("Found title: %s...", title[:50])
            break

    # Extract price
    price = ""
    price_selectors = [
        ".a-price-whole",
        ".a-price .a-offscreen",
        "#priceblock_dealprice",
        "#priceblock_ourprice",
        ".a-price-symbol + .a-price-whole",
        ".a-price",
    ]

    for selector in price_selectors:
        price_element = driver.select(selector)
        if price_element:
            price = price_element.text.strip()
            if DEBUG_MODE:
                logger.info("Found price: %s", price)
            break

    # Extract description
    description = ""
    desc_selectors = [
        "#feature-bullets ul",
        "#productDescription",
        ".a-unordered-list.a-vertical",
        "#featurebullets_feature_div",
    ]

    for selector in desc_selectors:
        desc_element = driver.select(selector)
        if desc_element:
            description = desc_element.text.strip()
            if DEBUG_MODE:
                logger.info("Found description: %d chars", len(description))
            break

    # Validate critical product data
    from .utils import is_valid_product_data

    if not is_valid_product_data(title, price):
        if DEBUG_MODE:
            logger.warning(
                "Invalid product data - title: '%s...', price: '%s'",
                title[:50],
                price,
            )
        return {}

    # Extract media
    if DEBUG_MODE:
        logger.info("Starting image extraction...")

    images = extract_high_res_images_botasaurus(driver, debug_options=debug_options)

    if DEBUG_MODE:
        logger.info("Extracted %d images", len(images))
        logger.info("Starting video extraction...")

    videos = extract_functional_videos_with_validation(driver, DEBUG_MODE)

    if DEBUG_MODE:
        logger.info("Extracted %d videos", len(videos))

    # Build result
    return {
        "title": title,
        "price": price,
        "description": description,
        "images": images,
        "videos": videos,
        "url": product_info["url"],
        "asin": product_info.get("asin"),
        "keyword": product_info.get("keyword", ""),
        "serp_rating": product_info.get("rating"),
        "serp_reviews_count": product_info.get("reviews_count"),
        "affiliate_link": build_affiliate_url(product_info["url"]),
        "downloaded_images": [],
        "downloaded_videos": [],
    }
