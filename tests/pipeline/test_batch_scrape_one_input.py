"""The batch scrapes one input at a time and stops at `max_products`.

The scraping phase used to browse every input before validating any of them,
so a run capped at one product still made a search per input, and those extra
visits drew the site's block. Each input is now searched, validated and
counted before the next one is searched, the way the standalone scraper's
keyword loop stops (REQ-SCR-038).
"""

from __future__ import annotations

import pytest

from src.pipeline.phases import scraping as scraping_phase
from tests.pipeline.test_cap_skips_published import _config, _FakeScraper, _scrape

KEYWORDS = ["phone stand", "desk lamp", "cable organizer"]


@pytest.fixture
def fake_scraper(monkeypatch):
    """The cap test's fake: pages keyed by (input, page), as the browser serves."""
    _FakeScraper.pages = {}
    _FakeScraper.invalid = set()
    monkeypatch.setattr(
        "src.scraper.amazon.scraper.BotasaurusAmazonScraper", _FakeScraper
    )
    monkeypatch.setattr(scraping_phase, "media_on_disk", lambda *_a: (0, 0))
    monkeypatch.setattr(
        "src.scraper.amazon.config.CONFIG",
        {"global_settings": {"batch_processing": {"max_retry_pages": 2}}},
    )
    return _FakeScraper


def _one_valid_product_per_keyword() -> dict[tuple[str, int], list[dict]]:
    return {(k, 1): [{"asin": f"B0TEST{n}"}] for n, k in enumerate(KEYWORDS)}


@pytest.mark.unit
@pytest.mark.req("REQ-SCR-084")
class TestTheBatchScrapesOneInputAtATime:
    @pytest.mark.asyncio
    async def test_a_validated_first_input_makes_one_search(self, fake_scraper):
        fake_scraper.pages = _one_valid_product_per_keyword()

        summary = await _scrape(_config(keywords=KEYWORDS))

        assert summary.successful_products == ["B0TEST0"]
        assert fake_scraper.calls == [("phone stand", 1)]

    @pytest.mark.asyncio
    async def test_an_input_that_fails_validation_moves_to_the_next(self, fake_scraper):
        """Its retry page comes first, then the next input, then nothing more."""
        fake_scraper.pages = {
            ("phone stand", 1): [{"asin": "B0TESTBAD"}],
            ("desk lamp", 1): [{"asin": "B0TESTGOOD"}],
            ("cable organizer", 1): [{"asin": "B0TESTLAST"}],
        }
        fake_scraper.invalid = {"B0TESTBAD"}

        summary = await _scrape(_config(keywords=KEYWORDS))

        assert summary.successful_products == ["B0TESTGOOD"]
        assert fake_scraper.calls == [
            ("phone stand", 1),
            ("phone stand", 2),
            ("desk lamp", 1),
        ]

    @pytest.mark.asyncio
    async def test_each_browser_call_names_one_input(self, fake_scraper):
        fake_scraper.pages = _one_valid_product_per_keyword()

        summary = await _scrape(_config(keywords=KEYWORDS, max_products=3))

        assert summary.successful_products == ["B0TEST0", "B0TEST1", "B0TEST2"]
        assert fake_scraper.batch_sizes == [1, 1, 1]

    @pytest.mark.asyncio
    async def test_inputs_after_the_first_are_paced(self, fake_scraper, monkeypatch):
        pauses: list[float] = []
        monkeypatch.setattr(scraping_phase, "sleep", pauses.append)
        fake_scraper.pages = _one_valid_product_per_keyword()

        await _scrape(_config(keywords=KEYWORDS, max_products=3))

        assert len(pauses) == 2

    @pytest.mark.asyncio
    async def test_no_pause_once_the_cap_is_met(self, fake_scraper, monkeypatch):
        pauses: list[float] = []
        monkeypatch.setattr(scraping_phase, "sleep", pauses.append)
        fake_scraper.pages = _one_valid_product_per_keyword()

        await _scrape(_config(keywords=KEYWORDS))

        assert pauses == []

    @pytest.mark.asyncio
    async def test_a_validated_product_id_fills_the_cap_before_any_search(
        self, fake_scraper
    ):
        """The same mixed inputs stop the standalone run before its keywords."""
        fake_scraper.pages = {
            ("B0TESTID01", 1): [{"asin": "B0TESTID01"}],
            **_one_valid_product_per_keyword(),
        }

        summary = await _scrape(_config(product_ids=["B0TESTID01"], keywords=KEYWORDS))

        assert summary.successful_products == ["B0TESTID01"]
        assert fake_scraper.calls == [("B0TESTID01", 1)]
