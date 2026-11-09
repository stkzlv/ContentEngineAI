"""A page that is not a results page is given up in seconds (REQ-SCR-085).

Each of twelve result-card selectors waited the driver's default four
seconds, so a blocked or empty search page cost about a minute before it
counted as a keyword with no products. A page the classifier did not see as
results now gets one bounded wait for the results to render, and on a page
known to be loaded each selector waits the short bound.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.scraper.amazon import browser_functions, product_extractor
from src.scraper.amazon.browser_functions import scrape_amazon_products_browser_impl
from src.scraper.amazon.config import get_settings
from src.scraper.amazon.page_kind import RESULTS_SELECTOR
from src.scraper.base.throttle import PageKind

pytestmark = [pytest.mark.unit, pytest.mark.req("REQ-SCR-085")]

BROWSER = get_settings().global_settings.browser_config


def _search(monkeypatch, kind: PageKind, driver: MagicMock) -> list:
    monkeypatch.setattr(browser_functions, "raise_if_blocked", lambda _d: kind)
    driver.is_bot_detected.return_value = False
    return scrape_amazon_products_browser_impl(
        driver, {"keyword": "phone stand", "debug_mode": False}
    )


def test_a_page_without_results_is_given_up_after_one_bounded_wait(monkeypatch):
    driver = MagicMock()
    driver.wait_for_element.side_effect = RuntimeError("not found")

    assert _search(monkeypatch, PageKind.UNKNOWN, driver) == []

    driver.wait_for_element.assert_called_once_with(
        RESULTS_SELECTOR, BROWSER.search_result_wait_reduced
    )
    driver.select_all.assert_not_called()


def test_results_that_render_late_are_still_read(monkeypatch):
    driver = MagicMock()
    driver.select_all.return_value = []

    _search(monkeypatch, PageKind.UNKNOWN, driver)

    driver.wait_for_element.assert_called_once()
    assert driver.select_all.called


def test_a_results_page_waits_only_the_short_bound_per_selector(monkeypatch):
    driver = MagicMock()
    driver.select_all.return_value = []

    _search(monkeypatch, PageKind.RESULTS, driver)

    driver.wait_for_element.assert_not_called()
    waits = {call.args[1] for call in driver.select_all.call_args_list}
    assert waits == {BROWSER.element_selection_timeout}


def test_the_bound_is_a_few_seconds():
    assert BROWSER.element_selection_timeout <= 2
    assert BROWSER.search_result_wait_reduced <= 5


def test_a_product_page_reads_its_body_once_and_bounds_each_selector():
    driver = MagicMock()
    driver.get_text.return_value = "In stock"
    driver.select.return_value = None
    driver.run_js.return_value = None

    product_extractor.extract_product_data_from_page(
        driver, "B0TEST0001", "phone stand", extract_videos=False
    )

    assert driver.get_text.call_count == 1
    field_waits = {
        call.args[1] for call in driver.select.call_args_list if len(call.args) > 1
    }
    assert field_waits == {BROWSER.element_selection_timeout}
    unbounded = [call for call in driver.select.call_args_list if len(call.args) < 2]
    assert all("wait" in call.kwargs for call in unbounded)
