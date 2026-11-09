"""One page limit for both entry points, paced page retries, a plain search.

The standalone loop stopped at `max_pages` (code default 7) and the batch at
`max_retry_pages` (5); one key now bounds both. A later result page is
another search, so it waits the same pause as a new input. The site runs no
Cloudflare, so the search navigation no longer asks the driver to check for it.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.pipeline.phases import scraping as scraping_phase
from src.scraper.amazon import browser_functions
from src.scraper.amazon.config import get_settings
from src.scraper.base.throttle import PageKind, ThrottleSettings, ThrottleTracker
from tests.pipeline.test_cap_skips_published import _config, _FakeScraper, _scrape

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_scraper(monkeypatch):
    _FakeScraper.pages = {
        ("phone stand", n): [{"asin": f"B0TEST{n}"}] for n in range(1, 9)
    }
    _FakeScraper.invalid = {f"B0TEST{n}" for n in range(1, 9)}
    monkeypatch.setattr(
        "src.scraper.amazon.scraper.BotasaurusAmazonScraper", _FakeScraper
    )
    monkeypatch.setattr(scraping_phase, "media_on_disk", lambda *_a: (0, 0))
    monkeypatch.setattr(
        "src.scraper.amazon.config.CONFIG",
        {"global_settings": {"batch_processing": {"max_pages": 3}}},
    )
    return _FakeScraper


@pytest.mark.req("REQ-SCR-033")
def test_the_bundled_limit_is_one_key():
    batch = get_settings().global_settings.batch_processing

    assert batch.max_pages == 5
    assert not hasattr(batch, "max_retry_pages")


@pytest.mark.req("REQ-SCR-033")
@pytest.mark.asyncio
async def test_the_batch_stops_at_max_pages(fake_scraper):
    await _scrape(_config())

    assert fake_scraper.calls == [
        ("phone stand", 1),
        ("phone stand", 2),
        ("phone stand", 3),
    ]


@pytest.mark.req("REQ-SCR-008")
@pytest.mark.asyncio
async def test_the_batch_paces_each_page_retry(fake_scraper, monkeypatch):
    pauses: list[float] = []
    monkeypatch.setattr(scraping_phase, "sleep", pauses.append)

    await _scrape(_config())

    assert len(pauses) == 2


@pytest.mark.req("REQ-SCR-008")
def test_the_standalone_loop_paces_each_later_page(monkeypatch):
    from src.scraper.amazon.scraper import BotasaurusAmazonScraper
    from src.scraper.config_models import ScraperConfig

    scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
    scraper.settings = ScraperConfig()
    scraper.debug_mode = False
    scraper.logger = logging.getLogger("test")
    scraper.throttle = ThrottleTracker(
        settings=ThrottleSettings(inter_input_delay_sec=(3.0, 3.0))
    )
    monkeypatch.setattr(scraper, "_scrape_single_pass", lambda *_a, **_k: [])
    pauses: list[float] = []
    monkeypatch.setattr("src.scraper.amazon.scraper.time.sleep", pauses.append)

    scraper._scrape_until_validated_count_reached("phone stand", None, target_count=1)

    assert pauses == [3.0] * (
        scraper.settings.global_settings.batch_processing.max_pages - 1
    )


def test_the_search_navigation_skips_the_cloudflare_check(monkeypatch):
    monkeypatch.setattr(
        browser_functions, "raise_if_blocked", lambda _d: PageKind.UNKNOWN
    )
    driver = MagicMock()
    driver.is_bot_detected.return_value = False
    driver.wait_for_element.side_effect = RuntimeError("no results")

    browser_functions.scrape_amazon_products_browser_impl(
        driver, {"keyword": "phone stand", "debug_mode": False}
    )

    search_calls = [
        c for c in driver.google_get.call_args_list if "/s?" in str(c.args[0])
    ]
    assert search_calls
    assert all("bypass_cloudflare" not in c.kwargs for c in search_calls)


def test_no_search_navigation_passes_the_cloudflare_flag():
    """Both search navigations: the first one and the return to the results."""
    import inspect

    source = inspect.getsource(browser_functions)

    assert source.count("google_get(search_url)") == 2
    assert "google_get(search_url, bypass_cloudflare" not in source


def test_the_dead_single_product_helper_is_gone():
    assert not hasattr(browser_functions, "scrape_single_product")
