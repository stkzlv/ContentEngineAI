"""The page classifier and the handling of each block kind (#812).

The page facts below are made up to carry each kind's markers, the way the
site's pages carry them: the bare-title block is a page titled just
`Amazon.com` with no results, the captcha posts to `validateCaptcha`, and the
interstitial asks the visitor to click through to continue shopping.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.scraper.amazon import browser_functions
from src.scraper.amazon.browser_functions import (
    scrape_amazon_products_browser_impl,
    scrape_batch_items,
)
from src.scraper.amazon.page_kind import PageFacts, classify, raise_if_blocked
from src.scraper.base.throttle import (
    AmazonBlockError,
    AmazonErrorPageError,
    PageKind,
    ThrottleSettings,
    ThrottleTracker,
    Verdict,
    is_error_page_failure,
    session_block_kind,
)

SEARCH_URL = "https://www.amazon.com/s?k=phone+stand"

PAGES = {
    PageKind.RESULTS: PageFacts(
        url=SEARCH_URL, title="Amazon.com : phone stand", has_results=True
    ),
    PageKind.PRODUCT: PageFacts(
        url="https://www.amazon.com/dp/B0TEST0001",
        title="Amazon.com: A Phone Stand",
        has_product=True,
    ),
    PageKind.ERROR_PAGE: PageFacts(
        url=SEARCH_URL, title="Sorry! Something went wrong!"
    ),
    PageKind.CAPTCHA: PageFacts(
        url="https://www.amazon.com/errors/validateCaptcha",
        title="Amazon.com",
        body="Enter the characters you see below",
        captcha_form=True,
    ),
    PageKind.INTERSTITIAL: PageFacts(
        url=SEARCH_URL,
        title="Amazon.com",
        body="Click the button below to continue shopping",
    ),
    PageKind.BARE_TITLE: PageFacts(url=SEARCH_URL, title="Amazon.com"),
    PageKind.NOT_FOUND: PageFacts(
        url="https://www.amazon.com/dp/B0TEST0002",
        title="Page Not Found",
        body="Sorry! We couldn't find that page.",
    ),
    PageKind.UNKNOWN: PageFacts(url="https://example.test/", title="Something else"),
}


class _Driver:
    """Serves one page-facts dict per read, the last one repeating."""

    def __init__(self, *pages: PageFacts) -> None:
        self.pages = list(pages)
        self.reads = 0

    def run_js(self, _script: str) -> dict:
        facts = self.pages[min(self.reads, len(self.pages) - 1)]
        self.reads += 1
        return {
            "url": facts.url,
            "title": facts.title,
            "body": facts.body,
            "has_results": facts.has_results,
            "has_product": facts.has_product,
            "captcha_form": facts.captcha_form,
        }


@pytest.mark.unit
@pytest.mark.req("REQ-SCR-082")
class TestTheClassifier:
    @pytest.mark.parametrize("kind", list(PageKind))
    def test_each_kind_is_named(self, kind):
        assert classify(PAGES[kind]) is kind

    def test_a_captcha_titled_like_the_bare_page_is_a_captcha(self):
        """Both are titled `Amazon.com`; the captcha's own markers decide."""
        facts = PageFacts(url=SEARCH_URL, title="Amazon.com", captcha_form=True)

        assert classify(facts) is PageKind.CAPTCHA

    @pytest.mark.parametrize(
        "kind",
        [
            PageKind.ERROR_PAGE,
            PageKind.CAPTCHA,
            PageKind.INTERSTITIAL,
            PageKind.BARE_TITLE,
        ],
    )
    def test_a_block_raises_with_its_kind(self, kind):
        with pytest.raises(AmazonBlockError) as raised:
            raise_if_blocked(_Driver(PAGES[kind]), sleep=lambda _s: None)

        assert raised.value.kind is kind

    @pytest.mark.parametrize(
        "kind", [PageKind.RESULTS, PageKind.PRODUCT, PageKind.NOT_FOUND]
    )
    def test_a_page_that_is_not_a_block_returns_its_kind(self, kind):
        assert raise_if_blocked(_Driver(PAGES[kind])) is kind

    def test_a_bare_title_is_read_again_before_it_counts(self):
        pauses: list[float] = []
        driver = _Driver(PAGES[PageKind.BARE_TITLE], PAGES[PageKind.RESULTS])

        kind = raise_if_blocked(driver, sleep=pauses.append)

        assert kind is PageKind.RESULTS
        assert len(pauses) == 1
        assert driver.reads == 2

    def test_the_error_page_keeps_its_own_type(self):
        """The throttle tracker's backoff matches it, as before."""
        with pytest.raises(AmazonErrorPageError) as raised:
            raise_if_blocked(_Driver(PAGES[PageKind.ERROR_PAGE]))

        assert is_error_page_failure(raised.value)

    def test_only_the_session_blocks_skip_the_backoff(self):
        interstitial = AmazonBlockError(PageKind.INTERSTITIAL)
        bare = AmazonBlockError(PageKind.BARE_TITLE)

        assert is_error_page_failure(interstitial)
        assert session_block_kind(interstitial) is None
        assert not is_error_page_failure(bare)
        assert session_block_kind(bare) is PageKind.BARE_TITLE


@pytest.mark.unit
@pytest.mark.req("REQ-SCR-083")
class TestSessionBlocks:
    def test_the_first_is_retried_in_a_fresh_session(self):
        tracker = ThrottleTracker()

        assert tracker.record_session_block("a", PageKind.BARE_TITLE) is Verdict.RETRY
        assert not tracker.stopped

    def test_a_second_in_a_row_stops_the_run(self):
        tracker = ThrottleTracker()
        tracker.record_session_block("a", PageKind.BARE_TITLE)

        assert tracker.record_session_block("a", PageKind.CAPTCHA) is Verdict.STOP
        assert tracker.stopped
        assert any("Stopped browsing at a" in line for line in tracker.summary_lines())

    def test_a_success_between_resets_the_count(self):
        tracker = ThrottleTracker()
        tracker.record_session_block("a", PageKind.BARE_TITLE)
        tracker.record_success("a")

        assert tracker.record_session_block("b", PageKind.BARE_TITLE) is Verdict.RETRY

    def test_the_retry_count_is_configurable(self):
        tracker = ThrottleTracker(
            settings=ThrottleSettings.from_config({"session_block_retries": 0})
        )

        assert tracker.record_session_block("a", PageKind.CAPTCHA) is Verdict.STOP

    def test_the_session_ends_at_the_blocked_input(self, monkeypatch):
        """Later inputs in the same session would meet the same page."""
        seen: list[str] = []

        def impl(_driver, item):
            seen.append(item["keyword"])
            raise AmazonBlockError(PageKind.BARE_TITLE)

        monkeypatch.setattr(
            browser_functions, "scrape_amazon_products_browser_impl", impl
        )
        tracker = ThrottleTracker()

        results = scrape_batch_items(
            MagicMock(),
            [{"keyword": "a"}, {"keyword": "b"}],
            tracker,
            sleep=lambda _s: None,
        )

        assert seen == ["a"]
        assert results == [
            {
                "input": "a",
                "products": [],
                "error": "Amazon block page detected: bare_title",
                "failure_kind": "bare_title",
            }
        ]
        assert not tracker.stopped


@pytest.mark.unit
@pytest.mark.req("REQ-SCR-087")
class TestAnEmptyPageSaysWhatItWas:
    def test_a_search_page_with_no_cards_logs_its_kind(self, monkeypatch, caplog):
        monkeypatch.setattr(
            browser_functions, "raise_if_blocked", lambda _d: PageKind.UNKNOWN
        )
        driver = MagicMock()
        driver.is_bot_detected.return_value = False
        driver.select_all.return_value = []

        with caplog.at_level(logging.INFO, logger=browser_functions.__name__):
            result = scrape_amazon_products_browser_impl(
                driver, {"keyword": "phone stand", "debug_mode": False}
            )

        assert result == []
        assert "Search page yielded nothing (page kind: unknown)" in caplog.text

    def test_a_results_page_that_gave_nothing_counts_its_cards(
        self, monkeypatch, caplog
    ):
        from types import SimpleNamespace

        cards = [MagicMock(name="B0TESTPUB")]
        monkeypatch.setattr(
            browser_functions, "raise_if_blocked", lambda _d: PageKind.RESULTS
        )
        monkeypatch.setattr(
            browser_functions,
            "extract_serp_product_info",
            lambda card, _kw: SimpleNamespace(
                asin=card._mock_name,
                url=f"https://www.amazon.com/dp/{card._mock_name}",
            ),
        )
        driver = MagicMock()
        driver.is_bot_detected.return_value = False
        driver.select_all.return_value = cards
        data = {
            "keyword": "phone stand",
            "debug_mode": False,
            "skip_asins": ["B0TESTPUB"],
        }

        with caplog.at_level(logging.INFO, logger=browser_functions.__name__):
            result = scrape_amazon_products_browser_impl(driver, data)

        assert result == []
        assert (
            "Search page yielded nothing (page kind: results): 1 card(s), "
            "1 skipped as published"
        ) in caplog.text
