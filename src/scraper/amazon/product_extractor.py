"""Product data extraction from Amazon pages.

This module handles extraction of product information (title, price,
description, media) from individual product pages and SERP cards.
"""

import logging
import re
from typing import Any

from botasaurus.browser import Driver

from .config import CONFIG, get_settings
from .media_extractor import (
    extract_functional_videos_with_validation,
    extract_high_res_images_botasaurus,
)
from .utils import build_affiliate_url, is_valid_product_data

logger = logging.getLogger(__name__)


def _normalize_price(raw: str) -> str:
    """Extract a clean dot-decimal price string from messy element text.

    Handles both US ("$1,234.56") and European ("1.234,56 EUR", "19,95")
    grouping/decimal conventions, since Amazon's locale domains differ. Drops
    currency symbols and stray text, then infers which separator is the decimal:

    - Both `.` and `,` present: the last one is the decimal, the other grouping.
    - One separator, appearing more than once: all grouping ("1.234.567").
    - One separator followed by exactly 3 digits: grouping ("1.234" -> "1234"),
      since prices use 2 decimal places, not 3.
    - One separator otherwise: decimal ("44,99" -> "44.99", "0.50" -> "0.50").

    Returns a dot-decimal string ("1234.56", "44") or "" when no number is
    present. `.a-price-whole` (a fallback selector) carries only the integer
    part, so it normalizes to whole dollars with no cents.
    """
    # Keep digits and separators only; drop currency symbols, text, whitespace.
    s = re.sub(r"[^\d.,]", "", raw).strip(".,")
    if not s:
        return ""

    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        decimal = "." if s.rfind(".") > s.rfind(",") else ","
        s = s.replace("," if decimal == "." else ".", "").replace(decimal, ".")
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        parts = s.split(sep)
        if len(parts) > 2 or len(parts[1]) == 3:
            s = "".join(parts)  # grouping separator
        else:
            s = parts[0] + "." + parts[1]  # decimal separator

    match = re.search(r"\d+(?:\.\d+)?", s)
    return match.group(0) if match else ""


def _price_from_parts(whole_raw: str, fraction_raw: str | None) -> str:
    """Combine `.a-price-whole` and `.a-price-fraction` text into a price.

    `.a-price-whole` holds only the integer part (its text includes the nested
    decimal span, so a trailing newline and dot), and `.a-price-fraction` holds
    the cents. Used as a fallback when `.a-offscreen` is missing, so the price
    keeps its cents instead of truncating to whole dollars. Returns a
    dot-decimal string, or "" when there's no whole number.
    """
    whole = _normalize_price(whole_raw)
    if not whole:
        return ""
    fraction = re.sub(r"\D", "", fraction_raw) if fraction_raw else ""
    return f"{whole}.{fraction}" if fraction else whole


_RATING_SELECTORS = (
    "[data-hook='average-star-rating'] .a-icon-alt",
    ".reviewCountTextLinkedHistogram .a-icon-alt",
    ".a-icon-alt",
)

_REVIEWS_COUNT_SELECTORS = (
    "#acrCustomerReviewText",
    "[data-hook='total-review-count']",
)


# A rating is a number, possibly with a comma as the decimal separator on a
# localised page. Anything else is a different element's text.
_RATING_VALUE = re.compile(r"\d+(?:[.,]\d+)?$")


def _extract_detail_rating(driver) -> str | None:
    """Read the star rating from a product detail page.

    The specific hooks come first: `.a-icon-alt` matches every star widget on
    the page, including a review's own rating, so leading with it can read a
    single review instead of the product average.

    The candidate must look like a number. `.a-icon-alt` is unscoped, and the
    separator is a bare substring, so "Producto de Amazon Renewed" would
    otherwise be read as a rating of `Producto` -- and a wrong truthy rating is
    worse than none, because it also suppresses the fallback to the card's.

    `wait=None` because these are page furniture, not something to wait for:
    the driver's default polls for four seconds per miss, and a listing with no
    reviews misses every selector here.
    """
    for selector in _RATING_SELECTORS:
        element = driver.select(selector, wait=None)
        if not element:
            continue
        text = (element.text or "").strip()
        # "4.5 out of 5 stars", localised as "4,5 de 5 estrellas".
        for separator in (" out of", " de "):
            if separator not in text:
                continue
            candidate = text.split(separator)[0].strip()
            if _RATING_VALUE.match(candidate):
                return candidate
    return None


def _extract_detail_reviews_count(driver) -> str | None:
    """Read the review count from a product detail page.

    Returned as the page writes it ("1,234 ratings"). This does not match
    `serp_reviews_count`, which the card path stores digits-only, so a consumer
    parsing either has to handle both shapes. Nothing reads it numerically yet.
    """
    for selector in _REVIEWS_COUNT_SELECTORS:
        element = driver.select(selector, wait=None)
        if element:
            text = (element.text or "").strip()
            if text:
                return text
    return None


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
                    logger.info("Shipping restriction detected: %s", indicator)
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

        # Break on text, not on the element: the page carries `productTitle`
        # twice, once as the span that holds the title and once as a hidden
        # input whose text is empty (see docs/notes/scraper.md).
        for selector in title_selectors:
            title_element = driver.select(selector)
            if title_element and title_element.text.strip():
                title = title_element.text.strip()
                break

        # Try multiple selectors for price. `.a-offscreen` carries the full
        # clean price ("$44.99"). Scope to the core price block and skip
        # `.a-text-price` (the struck-through list/was price) so we don't read
        # the wrong number; `driver.select` returns the first match, and an
        # unscoped `.a-price .a-offscreen` can land on a list or per-unit price.
        price_selectors = [
            "#corePrice_feature_div .a-price:not(.a-text-price) .a-offscreen",
            "#corePriceDisplay_desktop_feature_div"
            " .a-price:not(.a-text-price) .a-offscreen",
            "#priceblock_dealprice",
            "#priceblock_ourprice",
            ".a-price:not(.a-text-price) .a-offscreen",
            ".a-price .a-offscreen",
        ]

        for selector in price_selectors:
            price_element = driver.select(selector)
            if price_element:
                normalized = _normalize_price(price_element.text)
                if normalized:
                    price = normalized
                    break

        # Fallback when no `.a-offscreen` price is present: reconstruct from the
        # split whole/fraction spans so the price keeps its cents.
        if not price:
            whole_element = driver.select(".a-price-whole")
            if whole_element:
                fraction_element = driver.select(".a-price-fraction")
                price = _price_from_parts(
                    whole_element.text,
                    fraction_element.text if fraction_element else None,
                )

        # Extract description
        # The last one is scoped: unscoped it matches any vertical list on the
        # page, which falling through an empty first match now makes reachable,
        # and this text becomes narration.
        desc_selectors = [
            "#feature-bullets ul",
            "#productDescription",
            "#feature-bullets .a-unordered-list.a-vertical",
        ]

        for selector in desc_selectors:
            desc_element = driver.select(selector)
            if desc_element and desc_element.text.strip():
                description = desc_element.text.strip()
                break

        # Validate required fields BEFORE media extraction
        essential_fields = (
            CONFIG.get("global_settings", {})
            .get("validation_config", {})
            .get("essential_fields", [])
        )

        # Read the rating from the detail page. This used to run only when
        # `rating` was an essential field, and the value was thrown away after
        # validation -- so a scrape that never sees a search card (an ASIN or a
        # URL) produced a record with no rating at all, while the same product
        # scraped by keyword carried one from the card.
        rating = _extract_detail_rating(driver)
        reviews_count = _extract_detail_reviews_count(driver)

        # Validate product data BEFORE extracting media
        if not is_valid_product_data(
            title, price, description, asin, rating, essential_fields
        ):
            if DEBUG_MODE:
                title_preview_length = (
                    get_settings().global_settings.debug_config.title_preview_length
                )
                # Both of these used to be `"" if x else ""`, so the one line
                # that explains a rejection printed nothing either way.
                logger.warning(
                    "Invalid product data for %s: title='%s...', price='%s', "
                    "description=%s, rating=%s - SKIPPING MEDIA EXTRACTION",
                    asin,
                    title[:title_preview_length],
                    price,
                    "present" if description else "MISSING",
                    "present" if rating else "MISSING",
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
            "rating": rating,
            "reviews_count": reviews_count,
            "serp_rating": serp_info.rating if serp_info else None,
            "serp_reviews_count": (serp_info.reviews_count if serp_info else None),
            "downloaded_images": [],
            "downloaded_videos": [],
        }

        if DEBUG_MODE:
            logger.info(
                "Extracted product data for %s: %s images, %s videos",
                asin,
                len(images),
                len(videos),
            )

        return product_data

    except Exception as e:
        # WARNING, ungated: the product page was reached and its product is
        # lost to the run here; under the debug gate a normal run recorded
        # nothing at all for the loss (#466's class, one call down).
        logger.warning("Error extracting product data for %s: %s", asin, e)
        return None


# A search-result card is on screen when it is inspected, so a selector
# that matches nothing now will not match later. The driver's element
# lookups default to a four-second wait, and the link chain below tries nine
# selectors, so a sponsored or placeholder card cost about forty seconds
# to skip. None means one lookup, no waiting.
CARD_LOOKUP_WAIT = None


# One script call returns everything the classification needs. Each element
# lookup is a DOM round trip (about half a second here), and the selector
# chains below make seventeen of them, which is why a skipped card still cost
# eight seconds with the waits gone. The chains stay as the fallback for an
# element that cannot run a script.
CARD_FACTS_JS = """(el) => {
  const hrefs = Array.from(el.querySelectorAll("a[href]")).map(
    (a) => a.getAttribute("href") || ""
  );
  const isProduct = (h) => h.includes("/dp/") || h.includes("/gp/product/");
  const link = hrefs.find(isProduct) || null;
  const ratings = Array.from(
    el.querySelectorAll(
      ".a-icon-alt, [aria-label*='stars'], .a-star-mini .a-icon-alt, " +
        ".a-icon-row .a-icon-alt"
    )
  ).map((n) => n.getAttribute("aria-label") || n.textContent || "");
  const reviews = Array.from(
    el.querySelectorAll(
      ".a-size-base, .a-link-normal .a-size-base, " +
        "[aria-label*='ratings'], .a-row .a-size-small"
    )
  ).map((n) => n.textContent || "");
  return { text: el.innerText || el.textContent || "", link, ratings, reviews };
}"""

_SKIP_INDICATORS = (
    "people also search for",
    "related searches",
    "sponsored brands",
    "advertisement",
    "top brands",
    "frequently bought together",
)

_CARD_LINK_SELECTORS = (
    "h2 a[href*='/dp/']",
    "h3 a[href*='/dp/']",
    "h1 a[href*='/dp/']",
    "a[href*='/dp/']",
    "a[href*='/gp/product/']",
    "[data-cy='title-recipe-title'] a",
    ".s-link-style a[href*='/dp/']",
    ".a-link-normal[href*='/dp/']",
)
_CARD_RATING_SELECTORS = (
    ".a-icon-alt",
    "[aria-label*='stars']",
    ".a-star-mini .a-icon-alt",
    ".a-icon-row .a-icon-alt",
)
_CARD_REVIEWS_SELECTORS = (
    ".a-size-base",
    ".a-link-normal .a-size-base",
    "[aria-label*='ratings']",
    ".a-row .a-size-small",
)


def _is_product_href(href: str | None) -> bool:
    if not href:
        return False
    return "/dp/" in href or "/gp/product/" in href


def _card_facts_by_script(card_element) -> dict[str, Any] | None:
    run_js = getattr(card_element, "run_js", None)
    if run_js is None:
        return None
    try:
        facts = run_js(CARD_FACTS_JS)
    except Exception:
        return None
    return facts if isinstance(facts, dict) else None


def _card_facts_by_selectors(card_element) -> dict[str, Any]:
    """The same facts through element lookups, none of them waiting."""
    text = card_element.text if hasattr(card_element, "text") else ""
    link = None
    for selector in _CARD_LINK_SELECTORS:
        try:
            element = card_element.select(selector, wait=CARD_LOOKUP_WAIT)
        except Exception:  # noqa: S112
            continue
        href = element.get_attribute("href") if element else None
        if _is_product_href(href):
            link = href
            break
    if link is None:
        try:
            for anchor in card_element.select_all("a", wait=CARD_LOOKUP_WAIT):
                href = anchor.get_attribute("href")
                if _is_product_href(href):
                    link = href
                    break
        except Exception:  # noqa: S110
            pass
    ratings = []
    for selector in _CARD_RATING_SELECTORS:
        element = card_element.select(selector, wait=CARD_LOOKUP_WAIT)
        if element:
            ratings.append(element.get_attribute("aria-label") or element.text or "")
    reviews = []
    for selector in _CARD_REVIEWS_SELECTORS:
        element = card_element.select(selector, wait=CARD_LOOKUP_WAIT)
        if element:
            reviews.append(element.text or "")
    return {"text": text or "", "link": link, "ratings": ratings, "reviews": reviews}


def _parse_rating(candidates: list[str]) -> str | None:
    """The first candidate that reads as a rating; a badge's text before the
    star row (an `.a-icon-alt` that says "Amazon Prime") is passed over.
    """
    for rating_text in candidates:
        if "out of" in rating_text:
            found = rating_text.split(" out of")[0].strip()
            if found:
                return found
        elif "stars" in rating_text.lower():
            match = re.search(r"([\d.]+)\s*stars?", rating_text.lower())
            if match:
                return match.group(1)
    return None


def _parse_reviews(candidates: list[str]) -> str | None:
    for text in candidates:
        clean = text.replace(",", "").replace("(", "").replace(")", "").strip()
        if clean.isdigit():
            return text.strip()
    return None


def _asin_from(url: str) -> str | None:
    if "/dp/" in url:
        return url.split("/dp/")[1].split("/")[0].split("?")[0] or None
    if "/gp/product/" in url:
        return url.split("/gp/product/")[1].split("/")[0].split("?")[0] or None
    match = re.search(r"/([A-Z0-9]{10})(?:/|$|\?)", url)
    return match.group(1) if match else None


def extract_serp_product_info(card_element, keyword: str):
    """Extract product info from a search result card, in one round trip."""
    from .models import SerpProductInfo

    try:
        facts = _card_facts_by_script(card_element)
        if facts is None:
            facts = _card_facts_by_selectors(card_element)
        text = str(facts.get("text") or "").lower()
        if any(indicator in text for indicator in _SKIP_INDICATORS):
            return None
        url = facts.get("link")
        if not url:
            return None
        if not url.startswith("http"):
            base_url = (
                CONFIG.get("scrapers", {})
                .get("amazon", {})
                .get("base_url", "https://www.amazon.com")
            )
            url = f"{base_url}{url}"
        asin = _asin_from(url)
        if not asin:
            return None
        return SerpProductInfo(
            url=url,
            rating=_parse_rating([str(r) for r in facts.get("ratings") or []]),
            reviews_count=_parse_reviews(list(facts.get("reviews") or [])),
            asin=asin,
            keyword=keyword,
        )
    except Exception:
        return None
