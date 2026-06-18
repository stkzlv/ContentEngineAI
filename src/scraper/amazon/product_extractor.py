"""Product data extraction from Amazon pages.

This module handles extraction of product information (title, price,
description, media) from individual product pages and SERP cards.
"""

import logging
import re
from typing import Any

from botasaurus.browser import Driver

from .config import CONFIG
from .media_extractor import (
    extract_functional_videos_with_validation,
    extract_high_res_images_botasaurus,
)
from .utils import build_affiliate_url, is_valid_product_data

logger = logging.getLogger(__name__)


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
        # Check for shipping/availability issues
        unavailable_indicators = [
            "This item cannot be shipped to your selected delivery location",
            "Currently unavailable",
            "We don't know when or if this item will be back in stock",
            "Sorry, this item is not available",
            "not available in your location",
        ]

        for indicator in unavailable_indicators:
            if indicator.lower() in driver.get_text("body").lower():
                if DEBUG_MODE:
                    logger.info(f"Shipping restriction detected: {indicator}")
                    logger.info(
                        "Continuing to extract media " "despite shipping restriction"
                    )
                break

        # Extract basic product information
        title = ""
        price = ""
        description = ""

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

        # Validate required fields BEFORE media extraction
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

        # Validate product data BEFORE extracting media
        if not is_valid_product_data(
            title, price, description, asin, rating, essential_fields
        ):
            if DEBUG_MODE:
                debug_config = CONFIG.get("global_settings", {}).get("debug_config", {})
                title_preview_length = debug_config.get("title_preview_length", 50)
                logger.warning(
                    f"Invalid product data for {asin}: "
                    f"title='{title[:title_preview_length]}...', "
                    f"price='{price}', "
                    f"description={'' if description else ''}, "
                    f"rating={'' if rating else ''} "
                    f"- SKIPPING MEDIA EXTRACTION"
                )
            return None

        # ONLY extract media for valid products
        logger.info("Extracting images for %s", asin)
        images = extract_high_res_images_botasaurus(driver, debug_options=debug_options)

        logger.info("Extracting videos for %s", asin)
        videos = extract_functional_videos_with_validation(driver, DEBUG_MODE)

        # Build product data
        product_data = {
            "title": title,
            "price": price,
            "description": description,
            "images": images,
            "videos": videos,
            "affiliate_link": build_affiliate_url(driver.current_url),
            "url": driver.current_url,
            "asin": asin,
            "keyword": keyword,
            "serp_rating": serp_info.rating if serp_info else None,
            "serp_reviews_count": (serp_info.reviews_count if serp_info else None),
            "downloaded_images": [],
            "downloaded_videos": [],
        }

        if DEBUG_MODE:
            logger.info(
                f"Extracted product data for {asin}: "
                f"{len(images)} images, {len(videos)} videos"
            )

        return product_data

    except Exception as e:
        if DEBUG_MODE:
            logger.error(f"Error extracting product data for {asin}: {e}")
        return None


def extract_serp_product_info(card_element, keyword: str):
    """Extract product info from search result card"""
    from .models import SerpProductInfo

    try:
        # Quick check: skip non-product cards
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
                return None

        # Extract URL with comprehensive selector attempts
        link_element = None
        link_selectors = [
            "h2 a[href*='/dp/']",
            "h3 a[href*='/dp/']",
            "h1 a[href*='/dp/']",
            "a[href*='/dp/']",
            "a[href*='/gp/product/']",
            "[data-cy='title-recipe-title'] a",
            ".s-link-style a[href*='/dp/']",
            ".a-link-normal[href*='/dp/']",
            "a",
        ]

        for selector in link_selectors:
            try:
                if selector == "a":
                    all_links = card_element.select_all(selector)
                    for link in all_links:
                        href = link.get_attribute("href")
                        if href and ("/dp/" in href or "/gp/product/" in href):
                            link_element = link
                            break
                    if link_element:
                        break
                else:
                    link_element = card_element.select(selector)
                    if link_element:
                        href = link_element.get_attribute("href")
                        if href and ("/dp/" in href or "/gp/product/" in href):
                            break
                        else:
                            link_element = None
            except Exception:  # noqa: S112
                continue

        if not link_element:
            return None

        url = link_element.get_attribute("href")
        if url and not url.startswith("http"):
            base_url = (
                CONFIG.get("scrapers", {})
                .get("amazon", {})
                .get("base_url", "https://www.amazon.com")
            )
            url = f"{base_url}{url}"

        # Extract ASIN from URL
        asin = None
        if "/dp/" in url:
            asin = url.split("/dp/")[1].split("/")[0].split("?")[0]
        elif "/gp/product/" in url:
            asin = url.split("/gp/product/")[1].split("/")[0].split("?")[0]
        else:
            asin_match = re.search(r"/([A-Z0-9]{10})(?:/|$|\?)", url)
            if asin_match:
                asin = asin_match.group(1)

        if not asin:
            return None

        # Extract rating
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
                if "out of" in rating_text:
                    rating = rating_text.split(" out of")[0].strip()
                elif "stars" in rating_text.lower():
                    match = re.search(r"([\d.]+)\s*stars?", rating_text.lower())
                    if match:
                        rating = match.group(1)
                if rating:
                    break

        # Extract reviews count
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

    except Exception:
        return None
