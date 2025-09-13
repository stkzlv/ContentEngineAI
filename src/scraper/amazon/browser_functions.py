"""Browser automation functions for Amazon scraper.

This module contains browser automation functions that handle the actual
web scraping logic using Botasaurus framework.
"""

import logging
import os
import platform
from typing import Any

from botasaurus.browser import Driver, browser

from .botasaurus_output import get_browser_config_for_outputs
from .config import _BROWSER_CONFIG, CONFIG
from .media_extractor import (
    extract_functional_videos_with_validation,
    extract_high_res_images_botasaurus,
)
from .search_builder import SearchParameterBuilder
from .utils import detect_monitors, get_optimal_browser_position


def scrape_amazon_products_browser_impl(
    driver: Driver, data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Unified function to scrape Amazon products in a single browser session"""
    DEBUG_MODE = data.get("debug_mode", False)

    print(
        f"🔍 [DEBUG] scrape_amazon_products_browser called with keyword: "
        f"{data.get('keyword')}"
    )
    keyword = data["keyword"]
    is_asin = data.get("is_asin", False)
    print(f"🔍 [DEBUG] is_asin: {is_asin}")
    products = []

    # Get Amazon base URL from config (used by both ASIN and search paths)

    base_url = (
        CONFIG.get("scrapers", {})
        .get("amazon", {})
        .get("base_url", "https://www.amazon.com")
    )

    if is_asin:
        # Direct product page scraping for ASIN
        product_url = f"{base_url}/dp/{keyword}"
        print(f"🔍 [DEBUG] Calling scrape_single_product for ASIN: {keyword}")

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
            logging.getLogger(__name__).info(f"📦 Scraping product: {keyword}")

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
                print(
                    f"🖥️ [DEBUG] Set browser window size to "
                    f"{width}x{height} via JavaScript"
                )
            except Exception as e:
                print(f"⚠️ [DEBUG] Could not set window size: {e}")

        driver.short_random_sleep()

        # Check for regional redirect
        from .utils import detect_regional_redirect

        redirected, redirect_info = detect_regional_redirect(driver, product_url)
        if redirected and DEBUG_MODE:
            logging.getLogger(__name__).warning(
                f"🌍 Regional redirect detected: {redirect_info}"
            )

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
        search_url = url_builder.build_search_url(keyword, search_params)

        if DEBUG_MODE:
            url_builder.log_search_parameters(keyword, search_params)
            logging.getLogger(__name__).info(f"🔍 Searching: {search_url}")
            driver.save_screenshot()  # Debug screenshot

        # Use google_get for organic navigation pattern
        if DEBUG_MODE:
            import time

            nav_start = time.time()
            print("🌐 [DEBUG] Navigating to search URL...")

        try:
            if DEBUG_MODE:
                print("🚀 [DEBUG] Starting navigation to search page...")

            # Use google_get for organic navigation pattern (same as working version)
            driver.google_get(search_url, bypass_cloudflare=True)

            if DEBUG_MODE:
                print("✅ [DEBUG] Navigation completed successfully")

        except Exception as e:
            if DEBUG_MODE:
                print(f"❌ [DEBUG] Navigation failed: {e}")
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
                print(
                    f"🖥️ [DEBUG] Set browser window size to "
                    f"{width}x{height} via JavaScript"
                )
            except Exception as e:
                print(f"⚠️ [DEBUG] Could not set window size: {e}")

        if DEBUG_MODE:
            print("💤 [DEBUG] Short sleep before continuing...")
        driver.short_random_sleep()

        if DEBUG_MODE:
            nav_time = time.time() - nav_start
            print(f"⏱️ [DEBUG] Navigation completed in {nav_time:.2f} seconds")

            # Add page info for debugging
            try:
                current_url = driver.current_url
                page_title = driver.title[:50] if driver.title else "No title"
                print(f"🌐 [DEBUG] Current URL: {current_url}")
                print(f"📄 [DEBUG] Page title: {page_title}")
            except Exception as e:
                print(f"⚠️ [DEBUG] Could not get page info: {e}")

        # Check for CAPTCHA or bot detection
        if driver.is_bot_detected():
            logging.getLogger(__name__).error("🚫 Bot detection triggered!")
            if DEBUG_MODE:
                driver.save_screenshot()
                print("⚠️ [DEBUG] Bot detection triggered - screenshot saved")
                print(
                    "💡 [DEBUG] Continuing without manual intervention for browser "
                    "visibility testing"
                )
            return []

        # Get product cards from search results (with timeout to prevent long waits)
        if DEBUG_MODE:
            print("🔍 [DEBUG] Searching for product cards...")

        try:
            # Get timeout from config - use shorter timeout to prevent hanging
            global_settings = CONFIG.get("global_settings", {})
            browser_config = global_settings.get("browser_config", {})
            timeout = browser_config.get(
                "search_result_timeout", 5
            )  # Reduced from 10 to 5

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

            for selector in product_selectors:
                if DEBUG_MODE:
                    print(f"🔍 [DEBUG] Trying selector: {selector}")
                try:
                    # Don't wait for element - just try to find it immediately
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
                            print(f"✅ [DEBUG] Using selector: {selector}")
                            print(
                                f"🔍 [DEBUG] Found {len(product_cards)} product cards"
                            )
                        break
                except Exception as e:
                    if DEBUG_MODE:
                        print(
                            f"❌ [DEBUG] Selector '{selector}' failed: {str(e)[:100]}"
                        )
                    continue

            # If no cards found, try one final wait for basic content
            if not product_cards:
                if DEBUG_MODE:
                    print(
                        "⚠️ [DEBUG] No product cards found with immediate selectors, "
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
                                    print(
                                        f"⏰ [DEBUG] Found {len(product_cards)} cards "
                                        f"with wait: {selector}"
                                    )
                                break
                        except Exception as e:
                            if DEBUG_MODE:
                                print(
                                    f"⚠️ [DEBUG] Wait failed for selector {selector}: "
                                    f"{e}"
                                )
                            continue
                except Exception:
                    if DEBUG_MODE:
                        print("❌ [DEBUG] Even basic wait failed")

            if DEBUG_MODE:
                print(f"🔍 [DEBUG] Final result: {len(product_cards)} product cards")
        except Exception as e:
            if DEBUG_MODE:
                print(f"⚠️ [DEBUG] Exception in product card search: {e}")
            product_cards = []

        if not product_cards:
            if DEBUG_MODE:
                print("❌ [DEBUG] No product cards found")
                driver.save_screenshot()
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
                    print(
                        f"🔍 [DEBUG] Processing card {i+1}/{len(product_cards)}: "
                        f"serp_info={'✓' if serp_info else '✗'}, "
                        f"url={'✓' if serp_info and serp_info.url else '✗'}"
                    )

                if serp_info and serp_info.url:
                    # Check for duplicate ASIN before processing
                    if serp_info.asin in processed_asins:
                        if DEBUG_MODE:
                            print(
                                f"⚠️ [DEBUG] Skipping duplicate ASIN: {serp_info.asin}"
                            )
                        i += 1
                        continue

                    # Mark ASIN as processed
                    processed_asins.add(serp_info.asin)

                    if DEBUG_MODE:
                        print(f"🔍 [DEBUG] Extracting product {i+1}: {serp_info.asin}")

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
                        result_status = "✓" if product_data else "✗"
                        print(f"🔍 [DEBUG] Product extraction result: {result_status}")

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
                                    print(
                                        f"✅ [DEBUG] Product "
                                        f"{products_with_media_count}/{max_products} "
                                        f"with media (ASIN: {serp_info.asin}, "
                                        f"{img_count} images, {vid_count} videos)"
                                    )

                                # Stop when we have enough products with media
                                if products_with_media_count >= max_products:
                                    if DEBUG_MODE:
                                        print(
                                            f"🎯 [DEBUG] Reached target: "
                                            f"{max_products} products with media files!"
                                        )
                                    break
                            else:
                                if DEBUG_MODE:
                                    print(
                                        f"⚠️ [DEBUG] Product {i+1} "
                                        f"(ASIN: {serp_info.asin}) has no media URLs"
                                    )
                        else:
                            # Traditional count: stop when we have max_products total
                            products.append(product_data)
                            if len(products) >= max_products:
                                if DEBUG_MODE:
                                    print(
                                        f"🎯 [DEBUG] Reached target: "
                                        f"{max_products} products total"
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
                            print("🔙 [DEBUG] Navigating back to search results...")
                        try:
                            driver.google_get(search_url, bypass_cloudflare=True)
                            driver.short_random_sleep()
                            # Re-find product cards since DOM is fresh
                            driver.wait_for_element(search_selector, timeout)
                            new_product_cards = driver.select_all(search_selector)
                            if new_product_cards and len(new_product_cards) > i:
                                product_cards = new_product_cards
                                if DEBUG_MODE:
                                    print(
                                        f"🔍 [DEBUG] Found {len(product_cards)} cards"
                                    )
                            else:
                                if DEBUG_MODE:
                                    print(
                                        "⚠️ [DEBUG] No product cards after navigation"
                                    )
                                break
                        except Exception as e:
                            if DEBUG_MODE:
                                print(
                                    f"⚠️ [DEBUG] Navigation failed: {e}"
                                )
                            break  # Break if navigation fails
                else:
                    if DEBUG_MODE:
                        print(f"⚠️ [DEBUG] Skipping card {i+1} - no valid product info")

                # Move to next card
                i += 1

            except Exception as e:
                if DEBUG_MODE:
                    print(f"❌ [DEBUG] Error processing card {i+1}: {e}")
                i += 1  # Continue to next card instead of breaking
                continue

    # Final verification
    if DEBUG_MODE:
        print(f"✅ [DEBUG] Extracted {len(products)} products total")

        # Verify we have the expected number of products
        if count_products_with_media:
            expected_count = max_products
            actual_count = products_with_media_count
            print(
                f"🔍 [VERIFICATION] Expected {expected_count}, got {actual_count}"
            )

            # Verify each product has media URLs
            for idx, product in enumerate(products):
                asin = product.get("asin", "Unknown")
                img_urls = product.get("images", [])
                vid_urls = product.get("videos", [])
                img_count = len(img_urls)
                vid_count = len(vid_urls)

                print(
                    f"🔍 Product {idx+1} ({asin}): {img_count}img, {vid_count}vid"
                )

                if img_count == 0 and vid_count == 0:
                    print(f"❌ [VERIFICATION] ERROR: Product {asin} has no media URLs!")
                else:
                    print(f"✅ [VERIFICATION] Product {asin} has media URLs")

            if actual_count == expected_count:
                print(
                    f"✅ SUCCESS: Got exactly {expected_count} products with media!"
                )
            else:
                print(
                    f"⚠️ WARNING: Expected {expected_count} but got {actual_count}"
                )
        else:
            print(
                f"🔍 Traditional mode: {len(products)} products extracted"
            )

    return products


def extract_product_data_from_page(
    driver: Driver,
    asin: str,
    keyword: str,
    serp_info=None,
    debug_mode=False,
    debug_options=None,
) -> dict[str, Any] | None:
    """Extract product data from a single Amazon product page"""
    DEBUG_MODE = debug_mode

    try:
        # Check for shipping/availability issues first
        shipping_unavailable = False
        unavailable_indicators = [
            "This item cannot be shipped to your selected delivery location",
            "Currently unavailable",
            "We don't know when or if this item will be back in stock",
            "Sorry, this item is not available",
            "not available in your location",
        ]

        for indicator in unavailable_indicators:
            if indicator.lower() in driver.get_text("body").lower():
                shipping_unavailable = True
                if DEBUG_MODE:
                    print(f"⚠️ [DEBUG] Product unavailable: {indicator}")
                break

        if shipping_unavailable:
            return None

        # Extract basic product information
        title = ""
        price = ""
        description = ""

        # Get title selectors from config
        from .config import CONFIG

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
                break

        # Try multiple selectors for price
        price_selectors = [
            ".a-price-whole",
            ".a-price .a-offscreen",
            "#priceblock_dealprice",
            "#priceblock_ourprice",
            ".a-price-symbol + .a-price-whole",
        ]

        for selector in price_selectors:
            price_element = driver.select(selector)
            if price_element:
                price = price_element.text.strip()
                break

        # Extract description
        desc_selectors = [
            "#feature-bullets ul",
            "#productDescription",
            ".a-unordered-list.a-vertical",
        ]

        for selector in desc_selectors:
            desc_element = driver.select(selector)
            if desc_element:
                description = desc_element.text.strip()
                break

        # Validate required fields with enhanced validation BEFORE media extraction
        from .config import CONFIG
        from .utils import is_valid_product_data

        # Get essential fields configuration
        essential_fields = (
            CONFIG.get("global_settings", {})
            .get("validation_config", {})
            .get("essential_fields", [])
        )

        # Extract rating for validation if required
        rating = None
        if "rating" in essential_fields:
            rating_selectors = [
                ".a-icon-alt",
                "[data-hook='average-star-rating'] .a-icon-alt",
                ".reviewCountTextLinkedHistogram .a-icon-alt",
            ]
            for selector in rating_selectors:
                rating_element = driver.select(selector)
                if rating_element:
                    rating_text = rating_element.text
                    if "out of" in rating_text:
                        rating = rating_text.split(" out of")[0]
                    break

        # CRITICAL: Validate product data BEFORE extracting media to avoid waste
        if not is_valid_product_data(
            title, price, description, asin, rating, essential_fields
        ):
            if DEBUG_MODE:
                # Get title preview length from config
                debug_config = CONFIG.get("global_settings", {}).get("debug_config", {})
                title_preview_length = debug_config.get("title_preview_length", 50)
                logging.getLogger(__name__).warning(
                    f"❌ Invalid product data for {asin}: "
                    f"title='{title[:title_preview_length]}...', price='{price}', "
                    f"description={'✓' if description else '✗'}, "
                    f"rating={'✓' if rating else '✗'} - SKIPPING MEDIA EXTRACTION"
                )
            return None

        # ONLY extract media for valid products
        if DEBUG_MODE:
            logging.getLogger(__name__).info(
                f"🖼️ Extracting images for validated product {asin}"
            )

        images = extract_high_res_images_botasaurus(driver, debug_options=debug_options)

        if DEBUG_MODE:
            logging.getLogger(__name__).info(
                f"🎥 Extracting videos for validated product {asin}"
            )

        videos = extract_functional_videos_with_validation(driver)

        # Build product data
        product_data = {
            "title": title,
            "price": price,
            "description": description,
            "images": images,
            "videos": videos,
            "affiliate_link": driver.current_url,  # Current Amazon URL
            "url": driver.current_url,
            "asin": asin,
            "keyword": keyword,
            "serp_rating": serp_info.rating if serp_info else None,
            "serp_reviews_count": serp_info.reviews_count if serp_info else None,
            "downloaded_images": [],  # Will be populated by download task
            "downloaded_videos": [],  # Will be populated by download task
        }

        if DEBUG_MODE:
            logging.getLogger(__name__).info(
                f"✅ Extracted product data for {asin}: {len(images)} images, "
                f"{len(videos)} videos"
            )

        return product_data

    except Exception as e:
        if DEBUG_MODE:
            logging.getLogger(__name__).error(
                f"❌ Error extracting product data for {asin}: {e}"
            )
        return None


def extract_serp_product_info(card_element, keyword: str):
    """Extract product info from search result card"""
    DEBUG_MODE = globals().get("DEBUG_MODE", False)
    CONFIG = globals().get("CONFIG", {})

    try:
        from .models import SerpProductInfo

        # Quick check: skip if this doesn't look like a product card
        # Look for common non-product card indicators
        card_text = card_element.text.lower() if hasattr(card_element, "text") else ""
        skip_indicators = [
            "people also search for",
            "related searches",
            "sponsored brands",
            "advertisement",
            "top brands",
            "frequently bought together",
        ]

        for indicator in skip_indicators:
            if indicator in card_text:
                if DEBUG_MODE:
                    print(f"🚫 [DEBUG] Skipping card - found indicator: {indicator}")
                return None

        # Extract URL with comprehensive selector attempts
        link_element = None
        link_selectors = [
            # Primary title link selectors (most common)
            "h2 a[href*='/dp/']",
            "h3 a[href*='/dp/']",
            "h1 a[href*='/dp/']",
            # Try all links in the card and filter by href
            "a[href*='/dp/']",
            "a[href*='/gp/product/']",
            # Secondary title link patterns
            "[data-cy='title-recipe-title'] a",
            ".s-link-style a[href*='/dp/']",
            ".a-link-normal[href*='/dp/']",
            # Fallback - get any link and validate
            "a",
        ]

        for selector in link_selectors:
            try:
                if selector == "a":
                    # For fallback, get all links and find product links
                    all_links = card_element.select_all(selector)
                    for link in all_links:
                        href = link.get_attribute("href")
                        if href and ("/dp/" in href or "/gp/product/" in href):
                            link_element = link
                            if DEBUG_MODE:
                                # Get URL preview length from config
                                debug_config = CONFIG.get("global_settings", {}).get(
                                    "debug_config", {}
                                )
                                url_preview_length = debug_config.get(
                                    "url_preview_length", 100
                                )
                                print(
                                    f"✅ [DEBUG] Found: {href[:url_preview_length]}..."
                                )
                            break
                    if link_element:
                        break
                else:
                    link_element = card_element.select(selector)
                    if link_element:
                        # Verify it's actually a product link
                        href = link_element.get_attribute("href")
                        if href and ("/dp/" in href or "/gp/product/" in href):
                            if DEBUG_MODE:
                                print(
                                    f"✅ [DEBUG] Found link with selector: {selector}"
                                )
                            break
                        else:
                            if DEBUG_MODE and href:
                                # Get URL preview length from config
                                debug_config = CONFIG.get("global_settings", {}).get(
                                    "debug_config", {}
                                )
                                url_preview_length = debug_config.get(
                                    "url_preview_length", 100
                                )
                                print(
                                    f"⚠️ [DEBUG] Skip: {href[:url_preview_length]}"
                                )
                            link_element = None
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ [DEBUG] Error with selector {selector}: {e}")
                continue

        if not link_element:
            if DEBUG_MODE:
                print("❌ [DEBUG] No valid product link found in card")
            return None

        url = link_element.get_attribute("href")
        if url and not url.startswith("http"):
            from .config import CONFIG

            base_url = (
                CONFIG.get("scrapers", {})
                .get("amazon", {})
                .get("base_url", "https://www.amazon.com")
            )
            url = f"{base_url}{url}"

        # Extract ASIN from URL with multiple patterns
        asin = None
        if "/dp/" in url:
            asin = url.split("/dp/")[1].split("/")[0].split("?")[0]
        elif "/gp/product/" in url:
            asin = url.split("/gp/product/")[1].split("/")[0].split("?")[0]
        else:
            # Try to extract ASIN from any part of URL using regex
            import re

            asin_match = re.search(r"/([A-Z0-9]{10})(?:/|$|\?)", url)
            if asin_match:
                asin = asin_match.group(1)

        if DEBUG_MODE:
            # Get URL preview length from config
            debug_config = CONFIG.get("global_settings", {}).get("debug_config", {})
            url_preview_length = debug_config.get("url_preview_length", 100)
            print(f"🔍 [DEBUG] URL: {url[:url_preview_length]}...")
            print(f"🔍 [DEBUG] Extracted ASIN: {asin or 'None'}")

        if not asin:
            if DEBUG_MODE:
                print(f"❌ [DEBUG] No ASIN found in URL: {url}")
            return None

        # Extract rating with multiple selector attempts
        rating = None
        rating_selectors = [
            ".a-icon-alt",
            "[aria-label*='stars']",
            ".a-star-mini .a-icon-alt",
            ".a-icon-row .a-icon-alt",
        ]

        for selector in rating_selectors:
            rating_element = card_element.select(selector)
            if rating_element:
                rating_text = (
                    rating_element.get_attribute("aria-label")
                    or rating_element.text
                    or ""
                )
                # Extract rating from various formats
                if "out of" in rating_text:
                    rating = rating_text.split(" out of")[0].strip()
                elif "stars" in rating_text.lower():
                    # Format like "4.5 stars"
                    import re

                    match = re.search(r"([\d.]+)\s*stars?", rating_text.lower())
                    if match:
                        rating = match.group(1)
                if rating:
                    break

        # Extract reviews count with multiple selector attempts
        reviews_count = None
        reviews_selectors = [
            ".a-size-base",
            ".a-link-normal .a-size-base",
            "[aria-label*='ratings']",
            ".a-row .a-size-small",
        ]

        for selector in reviews_selectors:
            reviews_element = card_element.select(selector)
            if reviews_element:
                reviews_text = reviews_element.text or ""
                # Clean and validate reviews count (numbers with commas)
                clean_text = (
                    reviews_text.replace(",", "").replace("(", "").replace(")", "")
                )
                if clean_text.isdigit():
                    reviews_count = reviews_text.strip()
                    break

        return SerpProductInfo(
            url=url,
            rating=rating,
            reviews_count=reviews_count,
            asin=asin,
            keyword=keyword,
        )

    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ [DEBUG] Exception in extract_serp_product_info: {e}")
        return None


def create_dynamic_browser_function(debug_mode=False):
    """Create browser function with current DEBUG_MODE settings"""
    DEBUG_MODE = debug_mode

    # Get browser config, with fallback if not initialized
    try:
        current_config = _BROWSER_CONFIG.copy() if _BROWSER_CONFIG else {}
    except Exception:
        current_config = {}

    # Debug mode automatically enables browser visibility
    force_real_browser = DEBUG_MODE

    if force_real_browser:
        # User explicitly wants real browser window - force it immediately
        current_config["enable_xvfb_virtual_display"] = False
        os.environ["DISPLAY"] = ":0"  # Force main display

        # Auto-detect multi-monitor setup and calculate optimal position
        print("🔍 [DEBUG] Auto-discovering multi-monitor setup...")
        monitors = detect_monitors()
        browser_x, browser_y, browser_width, browser_height = (
            get_optimal_browser_position(monitors)
        )

        # Log monitor discovery results
        print(f"🖥️ [DEBUG] Detected {len(monitors)} monitor(s):")
        for i, monitor in enumerate(monitors):
            primary_str = " (PRIMARY)" if monitor.get("primary") else ""
            print(
                f"   Monitor {i+1}: {monitor['width']}x{monitor['height']} at "
                f"+{monitor['x']}+{monitor['y']}{primary_str}"
            )

        print(
            f"🎯 [DEBUG] Browser maximized on primary monitor: {browser_x},"
            f"{browser_y} size: {browser_width}x{browser_height}"
        )

        # Configure Chrome arguments with optimal positioning and stable flags
        chrome_args = current_config.get("add_arguments", [])
        chrome_args.extend(
            [
                # Window positioning and display
                "--start-maximized",  # Primary maximization flag
                "--disable-extensions",  # Prevent extensions from interfering
                # with window size
                "--disable-plugins",  # Prevent plugins from affecting window behavior
                f"--window-size={browser_width},{browser_height}",  # Explicit size
                f"--window-position={browser_x},{browser_y}",  # Position on primary
                "--new-window",  # Force new window instead of tab
                "--force-device-scale-factor=1",  # Prevent scaling issues
                # Stability and performance (official flags)
                "--disable-dev-shm-usage",  # Prevent /dev/shm issues
                "--disable-features=VizDisplayCompositor",  # Improve window
                # positioning reliability
                # User experience
                "--no-default-browser-check",  # Prevent default browser popup
                "--no-first-run",  # Skip first-run setup
                "--disable-infobars",  # Disable notification bars
                # Development and debugging
                "--remote-debugging-port=0",  # Enable debugging
                "--enable-logging",  # Enable Chrome logging
                "--v=1",  # Verbose logging
            ]
        )
        current_config["add_arguments"] = chrome_args

        # Override config settings for browser visibility mode
        current_config.update(
            {
                "block_images": False,  # Don't block images in debug/visible
                # browser mode
                "cache": False,  # Disable cache for fresh data in debug mode
                "max_retry": 1,  # Fewer retries for faster feedback in debug mode
                "window_size": (browser_width, browser_height),  # Set explicit
                # window size for Botasaurus
            }
        )

        # Force environment to use main display
        os.environ["DISPLAY"] = ":0.0"  # Use the full display specification

        print(
            "👁️ [DEBUG] Debug mode enabled - browser window will be visible "
            "on your screen"
        )
        print(f"🖥️ [DEBUG] Using display: {os.environ.get('DISPLAY')}")
        print("🔧 [DEBUG] Virtual display disabled: enable_xvfb_virtual_display=False")
        print(
            f"🖼️ [DEBUG] Maximized positioning: --window-position={browser_x},"
            f"{browser_y} --window-size={browser_width},{browser_height}"
        )

    # Detect environment for better debugging
    is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER", False)
    is_ci = os.environ.get("CI", False)
    has_display = os.environ.get("DISPLAY") is not None

    if DEBUG_MODE:
        print("🔍 [ENV DEBUG] Environment detection:")
        print(f"   • Platform: {platform.system()}")
        print(f"   • Is Docker: {is_docker}")
        print(f"   • Is CI: {is_ci}")
        print(f"   • Has DISPLAY: {has_display}")
        print(f"   • DISPLAY value: {os.environ.get('DISPLAY', 'Not set')}")

    # Force update debug-related settings with current DEBUG_MODE
    current_config.update(
        {
            "headless": not DEBUG_MODE,  # Show browser if debug mode
            "close_on_crash": not DEBUG_MODE,  # Keep browser open on crash
            # if debug mode
        }
    )

    # Add custom output configuration to direct outputs to our outputs/ directory
    output_config = get_browser_config_for_outputs()
    current_config.update(output_config)

    # NOTE: Botasaurus manages its own output directory structure by design.
    # We use custom output functions instead of trying to override the output directory.

    # Add timeout configuration to prevent hanging (via Chrome args)
    chrome_args = current_config.get("add_arguments", [])
    if not any("--timeout" in arg for arg in chrome_args):
        chrome_args.extend(
            [
                "--timeout=60000",  # 60 second timeout
                "--script-timeout=30000",  # 30 second script timeout
            ]
        )
    current_config["add_arguments"] = chrome_args

    # Create the browser function with current configuration
    @browser(**current_config)
    def scrape_amazon_products_browser(
        driver: Driver, data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        try:
            # Set driver timeouts to prevent hanging
            try:
                driver.set_page_load_timeout(60)  # 60 seconds for page load
                driver.implicitly_wait(10)  # 10 seconds implicit wait
            except Exception as timeout_err:
                if DEBUG_MODE:
                    print(f"⚠️ [DEBUG] Could not set driver timeouts: {timeout_err}")

            return scrape_amazon_products_browser_impl(driver, data)
        except Exception as e:
            if DEBUG_MODE:
                print(f"❌ [DEBUG] Browser function error: {e}")
                import traceback

                traceback.print_exc()
            # Return empty list on timeout/error to prevent hanging
            return []

    return scrape_amazon_products_browser


def scrape_single_product(
    driver: Driver,
    product_info: dict[str, Any],
    debug_mode=False,
    debug_options: dict = None,
) -> dict[str, Any]:
    """Helper function to scrape a single product page"""
    DEBUG_MODE = debug_mode

    if DEBUG_MODE:
        logging.getLogger(__name__).info(
            f"📦 Scraping product: {product_info.get('asin', 'Unknown')}"
        )

    # Use google_get for organic navigation
    original_url = product_info["url"]
    driver.google_get(original_url, bypass_cloudflare=True)

    if DEBUG_MODE:
        print("🔍 [DEBUG] Browser should be visible now - Amazon page loaded!")
        print(f"🔍 [DEBUG] Current URL: {driver.current_url}")
        try:
            page_title = driver.title
            print(f"🔍 [DEBUG] Page title: {page_title}")
        except Exception as e:
            print(f"🔍 [DEBUG] Page title: Unable to get ({e})")

        # Additional browser visibility information
        print("🔍 [DEBUG] Browser window info:")
        try:
            current_url = driver.current_url
            print(f"   • Current URL: {current_url}")
        except Exception as e:
            print(f"   • Current URL: Unable to get ({e})")

        try:
            # Check window size using JavaScript (Botasaurus-compatible)
            window_size = driver.run_js(
                "return {width: window.outerWidth, height: window.outerHeight};"
            )
            print(
                f"   • Window size: {window_size.get('width', 'Unknown')}x"
                f"{window_size.get('height', 'Unknown')}"
            )

            # Check window position using JavaScript
            window_position = driver.run_js(
                "return {x: window.screenX, y: window.screenY};"
            )
            print(
                f"   • Window position: {window_position.get('x', 'Unknown')},"
                f"{window_position.get('y', 'Unknown')}"
            )
        except Exception as e:
            print(f"   • Window info: Unable to get ({e})")

        try:
            # Use driver properties instead of methods
            print("   • Browser session active: Yes")
        except Exception as e:
            print(f"   • Browser session: Error ({e})")

        print(f"   • Driver type: {type(driver).__name__}")
        print(f"   • Browser name: {getattr(driver, 'name', 'Unknown')}")

        try:
            from .config import CONFIG

            debug_pause = (
                CONFIG.get("global_settings", {})
                .get("rate_limiting", {})
                .get("debug_pause_duration", 5)
            )
        except Exception:
            debug_pause = 5
        print(
            f"⏰ [DEBUG] Pausing for {debug_pause} seconds so you can see "
            f"the browser..."
        )
        import time

        time.sleep(debug_pause)

    # Check for regional redirect first
    from .utils import detect_regional_redirect

    redirected, redirect_info = detect_regional_redirect(driver, original_url)
    if redirected and DEBUG_MODE:
        logging.getLogger(__name__).warning(
            f"🌍 Regional redirect detected: {redirect_info}"
        )

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
                logging.getLogger(__name__).info(f"📝 Found title: {title[:50]}...")
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
                logging.getLogger(__name__).info(f"💲 Found price: {price}")
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
                logging.getLogger(__name__).info(
                    f"📄 Found description: {len(description)} chars"
                )
            break

    # Validate critical product data
    from .utils import is_valid_product_data

    if not is_valid_product_data(title, price):
        if DEBUG_MODE:
            logging.getLogger(__name__).warning(
                f"❌ Invalid product data - title: '{title[:50]}...', price: '{price}'"
            )
        return {}

    # Extract media
    if DEBUG_MODE:
        logging.getLogger(__name__).info("🖼️ Starting image extraction...")

    images = extract_high_res_images_botasaurus(driver, debug_options=debug_options)

    if DEBUG_MODE:
        logging.getLogger(__name__).info(f"🖼️ Extracted {len(images)} images")
        logging.getLogger(__name__).info("🎥 Starting video extraction...")

    videos = extract_functional_videos_with_validation(driver)

    if DEBUG_MODE:
        logging.getLogger(__name__).info(f"🎥 Extracted {len(videos)} videos")

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
        "affiliate_link": product_info["url"],
        "downloaded_images": [],
        "downloaded_videos": [],
    }
