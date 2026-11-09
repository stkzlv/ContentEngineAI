"""A captcha or bare-title page is retried in a fresh session, then stops the run.

Run logs showed every later search in a blocked browser session meeting the
same page, while a fresh session started a minute later got through. Both
entry points retry such an input once in a new session and stop browsing
when the block comes back with no success between (REQ-SCR-083).
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from src.pipeline.phases import scraping as scraping_phase
from src.scraper.amazon.batch_controller import BatchController
from src.scraper.amazon.models import BatchConfig, SearchParameters
from src.scraper.base.throttle import (
    AmazonBlockError,
    PageKind,
    ThrottleSettings,
    ThrottleTracker,
)
from tests.pipeline.test_cap_skips_published import _config, _FakeScraper, _scrape

KEYWORDS = ["phone stand", "desk lamp", "cable organizer"]
_NO_PACING = ThrottleSettings(
    inter_input_delay_sec=(0.0, 0.0), session_block_retry_wait_sec=0.0
)


class _BlockingScraper(_FakeScraper):
    """The cap test's fake, whose browser calls can meet a bare-title page.

    `blocks` holds one flag per browser call, in call order. A blocked call
    records the block with the tracker, as the real session loop does.
    """

    blocks: list[bool] = []
    retry_wait_sec: float = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.throttle = ThrottleTracker(  # type: ignore[assignment]
            settings=ThrottleSettings(
                inter_input_delay_sec=(0.0, 0.0),
                session_block_retry_wait_sec=self.retry_wait_sec,
            )
        )

    def scrape_batch_browser(
        self, inputs, search_params=None, start_page=1, skip_asins=()
    ):
        if self.blocks and self.blocks.pop(0):
            self.calls.append((inputs[0], start_page))
            self.throttle.record_session_block(inputs[0], PageKind.BARE_TITLE)
            return [{"input": inputs[0], "products": [], "failure_kind": "bare_title"}]
        results = super().scrape_batch_browser(
            inputs, search_params, start_page, skip_asins
        )
        for entry in results:
            if entry["products"]:
                self.throttle.record_success(entry["input"])
        return results


@pytest.fixture
def blocking_scraper(monkeypatch):
    _BlockingScraper.pages = {
        (k, 1): [{"asin": f"B0TEST{n}"}] for n, k in enumerate(KEYWORDS)
    }
    _BlockingScraper.invalid = set()
    _BlockingScraper.blocks = []
    _BlockingScraper.retry_wait_sec = 0.0
    monkeypatch.setattr(
        "src.scraper.amazon.scraper.BotasaurusAmazonScraper", _BlockingScraper
    )
    monkeypatch.setattr(scraping_phase, "media_on_disk", lambda *_a: (0, 0))
    monkeypatch.setattr(
        "src.scraper.amazon.config.CONFIG",
        {"global_settings": {"batch_processing": {"max_retry_pages": 1}}},
    )
    return _BlockingScraper


@pytest.mark.unit
@pytest.mark.req("REQ-SCR-083")
class TestTheBatch:
    @pytest.mark.asyncio
    async def test_a_blocked_input_is_retried_in_a_fresh_session(
        self, blocking_scraper
    ):
        blocking_scraper.blocks = [True, False]

        summary = await _scrape(_config(keywords=KEYWORDS))

        assert summary.successful_products == ["B0TEST0"]
        assert blocking_scraper.calls == [("phone stand", 1), ("phone stand", 1)]

    @pytest.mark.asyncio
    async def test_the_fresh_session_waits_first(self, blocking_scraper, monkeypatch):
        pauses: list[float] = []
        monkeypatch.setattr(scraping_phase, "sleep", pauses.append)
        blocking_scraper.retry_wait_sec = 60.0
        blocking_scraper.blocks = [True, False]

        await _scrape(_config(keywords=KEYWORDS))

        assert pauses == [60.0]

    @pytest.mark.asyncio
    async def test_a_second_block_in_a_row_stops_browsing(self, blocking_scraper):
        blocking_scraper.blocks = [True, True]

        summary = await _scrape(_config(keywords=KEYWORDS))

        assert summary.successful_products == []
        assert summary.failed_products == ["phone stand"]
        assert blocking_scraper.calls == [("phone stand", 1), ("phone stand", 1)]


def _standalone_scraper():
    from src.scraper.amazon.scraper import BotasaurusAmazonScraper
    from src.scraper.config_models import ScraperConfig

    scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
    scraper.settings = ScraperConfig()
    scraper.debug_mode = False
    scraper.logger = logging.getLogger("test")
    scraper.throttle = ThrottleTracker(settings=_NO_PACING)
    return scraper


@pytest.mark.unit
@pytest.mark.req("REQ-SCR-083")
class TestTheStandaloneScraper:
    def test_a_blocked_call_is_made_again(self):
        """Each call starts its own browser, so the retry is a fresh session."""
        scraper = _standalone_scraper()
        outcomes = [AmazonBlockError(PageKind.CAPTCHA), [{"asin": "B0TEST0001"}]]

        def browser_func(_data):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        result = scraper._scrape_with_retry(browser_func, {"keyword": "phone stand"})

        assert result == [{"asin": "B0TEST0001"}]
        assert outcomes == []

    def test_a_second_block_in_a_row_is_raised(self):
        scraper = _standalone_scraper()

        def browser_func(_data):
            raise AmazonBlockError(PageKind.BARE_TITLE)

        with pytest.raises(AmazonBlockError):
            scraper._scrape_with_retry(browser_func, {"keyword": "phone stand"})
        assert scraper.throttle.stopped

    def test_the_keyword_loop_stops_once_browsing_has_stopped(self):
        scraper = Mock()
        scraper.logger = logging.getLogger("test")
        scraper.throttle = ThrottleTracker(settings=_NO_PACING)

        def unified(keyword, **_kwargs):
            scraper.throttle.record_session_block(keyword, PageKind.BARE_TITLE)
            scraper.throttle.record_session_block(keyword, PageKind.BARE_TITLE)
            return []

        scraper.scrape_products_unified.side_effect = unified
        config = BatchConfig(
            product_ids=[],
            keywords=KEYWORDS,
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=3,
            products_per_keyword=1,
        )

        BatchController(scraper, config).run_batch()

        assert scraper.scrape_products_unified.call_count == 1

    def test_the_product_id_loop_stops_once_browsing_has_stopped(self):
        scraper = Mock()
        scraper.logger = logging.getLogger("test")
        scraper.throttle = ThrottleTracker(settings=_NO_PACING)

        def unified(keyword, **_kwargs):
            scraper.throttle.record_session_block(keyword, PageKind.CAPTCHA)
            scraper.throttle.record_session_block(keyword, PageKind.CAPTCHA)
            return []

        scraper.scrape_products_unified.side_effect = unified
        config = BatchConfig(
            product_ids=["B0TEST0001", "B0TEST0002", "B0TEST0003"],
            keywords=[],
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=3,
            products_per_keyword=1,
        )

        BatchController(scraper, config).run_batch()

        assert scraper.scrape_products_unified.call_count == 1

    def test_the_summary_names_the_stop(self, caplog):
        scraper = Mock()
        scraper.logger = logging.getLogger("test.summary")
        scraper.throttle = ThrottleTracker(settings=_NO_PACING)

        def unified(keyword, **_kwargs):
            scraper.throttle.record_session_block(keyword, PageKind.BARE_TITLE)
            scraper.throttle.record_session_block(keyword, PageKind.BARE_TITLE)
            return []

        scraper.scrape_products_unified.side_effect = unified
        config = BatchConfig(
            product_ids=[],
            keywords=KEYWORDS,
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=1,
            products_per_keyword=1,
        )

        with caplog.at_level(logging.WARNING, logger="test.summary"):
            BatchController(scraper, config).run_batch()

        assert "Stopped browsing at phone stand" in caplog.text

    def test_pagination_stops_once_browsing_has_stopped(self, monkeypatch):
        """Each later page would open a fresh browser into the same block."""
        scraper = _standalone_scraper()
        pages: list[int] = []

        def single_pass(keyword, _params, _batch_size, **kwargs):
            pages.append(kwargs.get("page", 1))
            scraper.throttle.record_session_block(keyword, PageKind.BARE_TITLE)
            return []

        monkeypatch.setattr(scraper, "_scrape_single_pass", single_pass)

        result = scraper._scrape_until_validated_count_reached(
            "phone stand", None, target_count=1
        )

        assert result == []
        assert pages == [1, 2]
