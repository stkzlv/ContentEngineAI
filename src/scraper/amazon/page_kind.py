"""Name the page a navigation landed on, and raise on a block (REQ-SCR-082).

The block the scraper met most was not the error page the throttle watched
for but a page titled just `Amazon.com` with no results, and the scraper read
it as a keyword with no products after waiting out every card selector. One
classifier now runs after every navigation, from the URL, the title and a
bounded read of the body, so a block is named within seconds.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.scraper.base.throttle import (
    AMAZON_ERROR_PAGE_TITLE,
    BLOCK_KINDS,
    AmazonBlockError,
    AmazonErrorPageError,
    PageKind,
)

logger = logging.getLogger(__name__)

BARE_TITLE = "Amazon.com"
CAPTCHA_URL_MARKER = "validatecaptcha"
CAPTCHA_TEXT = (
    "enter the characters you see below",
    "type the characters you see in this image",
)
INTERSTITIAL_MARKERS = ("click the button below to continue shopping", "bm-verify")
NOT_FOUND_TITLE = "page not found"
NOT_FOUND_TEXT = "we couldn't find that page"

RESULTS_SELECTOR = '[data-component-type="s-search-result"], .s-result-item[data-asin]'
"""What marks a results page, for the classifier and the bounded wait."""

BODY_CHARS = 5000
"""How much body text one read takes; every marker sits near the top."""

BARE_TITLE_RECHECK_SEC = 2.0
"""One short wait before a bare title is called a block, for a slow render."""

# One script, so a classification costs one round trip to the browser.
_PAGE_FACTS_JS = f"""
const body = document.body ? (document.body.innerText || '') : '';
return {{
    url: location.href,
    title: document.title || '',
    body: body.slice(0, {BODY_CHARS}),
    has_results: !!document.querySelector({RESULTS_SELECTOR!r}),
    has_product: !!document.getElementById('productTitle'),
    captcha_form: !!document.querySelector('form[action*="validateCaptcha"]'),
}};
"""


@dataclass(frozen=True)
class PageFacts:
    """What the classifier reads from a page."""

    url: str = ""
    title: str = ""
    body: str = ""
    has_results: bool = False
    has_product: bool = False
    captcha_form: bool = False


def classify(facts: PageFacts) -> PageKind:
    """The page kind, block markers first so a block never reads as a page."""
    title = facts.title.strip()
    url = facts.url.lower()
    body = facts.body.lower()
    if AMAZON_ERROR_PAGE_TITLE in title:
        return PageKind.ERROR_PAGE
    if (
        CAPTCHA_URL_MARKER in url
        or facts.captcha_form
        or any(m in body for m in CAPTCHA_TEXT)
    ):
        return PageKind.CAPTCHA
    if any(m in body or m in url for m in INTERSTITIAL_MARKERS):
        return PageKind.INTERSTITIAL
    if facts.has_results:
        return PageKind.RESULTS
    if facts.has_product:
        return PageKind.PRODUCT
    if NOT_FOUND_TITLE in title.lower() or NOT_FOUND_TEXT in body:
        return PageKind.NOT_FOUND
    if title == BARE_TITLE:
        return PageKind.BARE_TITLE
    return PageKind.UNKNOWN


def read_page_facts(driver: Any) -> PageFacts:
    """Read the facts in one script; an unreadable page reads as empty."""
    try:
        raw = driver.run_js(_PAGE_FACTS_JS)
    except Exception as exc:  # noqa: BLE001 - a driver that cannot be read
        logger.debug("Could not read the page: %s", exc)
        raw = None
    if not isinstance(raw, dict):
        # A driver whose script returned nothing still has a title, which
        # is enough to see the error page.
        return PageFacts(title=_read(driver, "title"), url=_read(driver, "current_url"))
    return PageFacts(
        url=str(raw.get("url") or ""),
        title=str(raw.get("title") or ""),
        body=str(raw.get("body") or ""),
        has_results=bool(raw.get("has_results")),
        has_product=bool(raw.get("has_product")),
        captcha_form=bool(raw.get("captcha_form")),
    )


def _read(driver: Any, attribute: str) -> str:
    try:
        return str(getattr(driver, attribute) or "")
    except Exception as exc:  # noqa: BLE001 - a driver that cannot be read
        logger.debug("Could not read the page %s: %s", attribute, exc)
        return ""


def raise_if_blocked(
    driver: Any, sleep: Callable[[float], None] = time.sleep
) -> PageKind:
    """Classify the page the driver is on; raise if it is a block.

    Returns the kind otherwise, so a caller can log what it saw. A bare title
    is read a second time after a short wait before it counts, since a page
    still rendering has the same title and no results yet.
    """
    kind = classify(read_page_facts(driver))
    if kind is PageKind.BARE_TITLE:
        sleep(BARE_TITLE_RECHECK_SEC)
        kind = classify(read_page_facts(driver))
    logger.debug("Page kind: %s", kind.value)
    if kind in BLOCK_KINDS:
        logger.warning("Block page: %s", kind.value)
        if kind is PageKind.ERROR_PAGE:
            raise AmazonErrorPageError()
        raise AmazonBlockError(kind)
    return kind
