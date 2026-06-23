"""Regression tests for scrape_amazon_products_browser_impl execution paths.

The function runs inside a Botasaurus @browser wrapper with a live Driver, so every
other test mocks it out and its body never executes under pytest. A redundant local
`import time` inside an `if DEBUG_MODE:` block once shadowed the module-level import,
making `time` function-local for the whole function. On the non-debug path the local
import never ran, so `time.monotonic()` raised UnboundLocalError and every keyword
scrape crashed at runtime, while debug runs (which take the if-branch first) worked.

These tests drive the impl directly with a mock Driver so the body actually runs,
guarding the search path in both debug modes.
"""

from unittest.mock import MagicMock

import pytest

from src.scraper.amazon.browser_functions import scrape_amazon_products_browser_impl


def _make_driver() -> MagicMock:
    """Mock Driver whose google_get raises, bounding execution.

    The function computes nav_start (time.monotonic) before google_get and uses time
    again in the navigation-failure handler, then returns []. Raising here exercises
    every `time.*` use on the path without needing a real page or extraction pipeline.
    """
    driver = MagicMock()
    driver.google_get.side_effect = RuntimeError("nav blocked")
    return driver


@pytest.mark.parametrize("debug_mode", [False, True])
def test_keyword_search_reaches_navigation_without_unbound_local(debug_mode):
    """Keyword scrape must reach navigation without UnboundLocalError on time.

    Pre-fix this raised UnboundLocalError("cannot access local variable 'time'") on
    the non-debug path (debug_mode=False). The function returns [] when navigation
    fails; the assertion is that it returns cleanly rather than crashing.
    """
    driver = _make_driver()
    data = {"keyword": "bluetooth speaker", "debug_mode": debug_mode}

    result = scrape_amazon_products_browser_impl(driver, data)

    assert result == []
    driver.google_get.assert_called_once()
