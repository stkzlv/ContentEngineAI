"""Regression tests for scrape_amazon_products_browser_impl execution paths.

The function runs inside a Botasaurus @browser wrapper with a live Driver, so every
other test mocks it out and its body never executes under pytest. A redundant local
`import time` inside an `if DEBUG_MODE:` block once shadowed the module-level import,
making `time` function-local for the whole function. On the non-debug path the local
import never ran, so `time.monotonic()` raised UnboundLocalError and every keyword
scrape crashed at runtime, while debug runs (which take the if-branch first) worked.

These tests drive the impl directly with a mock Driver so the body actually runs. They
cover all three input branches (search keyword, URL, ASIN) on the non-debug path, which
is where the time bug fired and where the shared `count_products_with_media` /
`products_with_media_count` / `max_products` variables are read after the branch (the
"initialize before the branch" footgun documented in CLAUDE.md).
"""

from unittest.mock import MagicMock

import pytest

from src.scraper.amazon import browser_functions
from src.scraper.amazon.browser_functions import scrape_amazon_products_browser_impl


@pytest.mark.parametrize("debug_mode", [False, True])
def test_keyword_search_reaches_navigation_without_unbound_local(debug_mode):
    """Keyword scrape must reach navigation without UnboundLocalError on time.

    Pre-fix this raised UnboundLocalError("cannot access local variable 'time'") on
    the non-debug path (debug_mode=False). google_get is made to raise so execution is
    bounded: the function computes nav_start (time.monotonic) before the call and uses
    time again in the failure handler, then returns []. That exercises every `time.*`
    use on the path without needing a real page or extraction pipeline.
    """
    driver = MagicMock()
    driver.google_get.side_effect = RuntimeError("nav blocked")
    data = {"keyword": "bluetooth speaker", "debug_mode": debug_mode}

    result = scrape_amazon_products_browser_impl(driver, data)

    assert result == []
    driver.google_get.assert_called_once()


def test_url_input_non_debug_path_runs(monkeypatch):
    """URL input on the non-debug path runs without raising.

    Bounds execution by stubbing product extraction to return nothing, so the function
    resolves the ASIN from current_url and returns [] without a real page.
    """
    monkeypatch.setattr(
        browser_functions, "extract_product_data_from_page", lambda *a, **k: None
    )
    driver = MagicMock()
    driver.current_url = "https://www.amazon.com/dp/B0TEST1234/"
    data = {"keyword": "https://tr.ee/short", "is_url": True, "debug_mode": False}

    result = scrape_amazon_products_browser_impl(driver, data)

    assert result == []


def test_asin_input_non_debug_path_runs(monkeypatch):
    """ASIN input on the non-debug path runs without raising."""
    monkeypatch.setattr(
        browser_functions, "extract_product_data_from_page", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "src.scraper.amazon.utils.detect_regional_redirect",
        lambda *a, **k: (False, None),
    )
    driver = MagicMock()
    data = {"keyword": "B0TEST1234", "is_asin": True, "debug_mode": False}

    result = scrape_amazon_products_browser_impl(driver, data)

    assert result == []
