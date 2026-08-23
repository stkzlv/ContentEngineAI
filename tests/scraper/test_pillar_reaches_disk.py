"""Tests that a keyword's pillar reaches `data.json`.

The pillar was assigned to the in-memory record *after* the file had been
written, on all three scraper paths, so every scrape wrote `pillar: null`.
Nothing failed: the record the caller held was correct, and the caller is what
every existing test looked at. Only the file was wrong, and the producer reads
the file.

So these assert on the written bytes. A test that checked the returned records
would have passed against the broken version on all three paths.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scraper.amazon.models import ProductData


def _product(asin: str = "B0TEST0001") -> ProductData:
    return ProductData(
        title="A product",
        price="$10",
        url=f"https://www.amazon.com/dp/{asin}",
        platform=None,
        asin=asin,
    )


def _scraper(tmp_path: Path, keywords_block):
    """A real scraper with its config stubbed and its output redirected."""
    from src.scraper.amazon.scraper import BotasaurusAmazonScraper

    scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
    scraper.config = {"batch": {"keywords": keywords_block}}
    scraper._keyword_pillars = None
    scraper.output_dir = tmp_path
    scraper.debug_mode = False
    scraper.logger = MagicMock()
    return scraper


def _written(tmp_path: Path, asin: str) -> dict:
    path = tmp_path / asin / "data.json"
    assert path.exists(), f"no data.json written at {path}"
    data = json.loads(path.read_text())
    if isinstance(data, list):
        data = data[0]
    return data


@pytest.mark.unit
class TestPillarResolution:
    def test_a_configured_keyword_resolves_to_its_pillar(self, tmp_path):
        scraper = _scraper(tmp_path, {"value": ["smart plug"], "utility": ["ssd"]})
        assert scraper.pillar_for_keyword("smart plug") == "value"
        assert scraper.pillar_for_keyword("ssd") == "utility"

    def test_an_unconfigured_keyword_resolves_to_nothing(self, tmp_path):
        scraper = _scraper(tmp_path, {"value": ["smart plug"]})
        assert scraper.pillar_for_keyword("something else") is None

    def test_a_flat_keyword_list_maps_nothing(self, tmp_path):
        """The pre-pillar config shape. It must load, not raise."""
        scraper = _scraper(tmp_path, ["smart plug"])
        assert scraper.pillar_for_keyword("smart plug") is None

    def test_a_missing_batch_block_maps_nothing(self, tmp_path):
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        scraper.config = {}
        scraper._keyword_pillars = None
        assert scraper.pillar_for_keyword("smart plug") is None


@pytest.mark.unit
class TestPillarOnDisk:
    """The three paths that write `data.json`, each asserted from the file."""

    def test_process_raw_products_writes_the_pillar(self, tmp_path):
        """The global batch's path."""
        scraper = _scraper(tmp_path, {"value": ["smart plug"]})
        product = _product()

        with (
            patch.object(scraper, "_orchestrate_media_downloads"),
            patch.object(
                scraper, "_validate_and_convert_products", return_value=[product]
            ),
            patch.object(scraper, "_shorten_affiliate_links"),
        ):
            scraper.process_raw_products([{"asin": "B0TEST0001"}], pillar="value")

        assert _written(tmp_path, "B0TEST0001")["pillar"] == "value"

    def test_process_raw_products_without_a_pillar_writes_null(self, tmp_path):
        """An unconfigured keyword must not invent one."""
        scraper = _scraper(tmp_path, {"value": ["smart plug"]})
        product = _product()

        with (
            patch.object(scraper, "_orchestrate_media_downloads"),
            patch.object(
                scraper, "_validate_and_convert_products", return_value=[product]
            ),
            patch.object(scraper, "_shorten_affiliate_links"),
        ):
            scraper.process_raw_products([{"asin": "B0TEST0001"}])

        assert _written(tmp_path, "B0TEST0001")["pillar"] is None

    def test_scrape_products_writes_the_pillar(self, tmp_path):
        """The standalone single-keyword path, which is the issue's repro."""
        scraper = _scraper(tmp_path, {"value": ["smart plug"]})
        product = _product()

        with (
            patch.object(scraper, "scrape_products_unified", return_value=[product]),
            patch.object(scraper, "_shorten_affiliate_links"),
        ):
            scraper.scrape_products(["smart plug"], None)

        assert _written(tmp_path, "B0TEST0001")["pillar"] == "value"

    def test_the_batch_controller_writes_the_pillar(self, tmp_path):
        """The standalone multi-keyword path.

        It never called `_save_products` at all, so the file was whatever the
        browser callback wrote mid-scrape -- before the pillar existed and
        before the media downloads finished.
        """
        from src.scraper.amazon.batch_controller import BatchController
        from src.scraper.amazon.models import BatchConfig

        scraper = _scraper(tmp_path, {"value": ["smart plug"]})
        product = _product()

        from src.scraper.amazon.models import SearchParameters

        config = BatchConfig(
            product_ids=[],
            keywords=["smart plug"],
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=1,
            products_per_keyword=1,
            keyword_pillar_map={"smart plug": "value"},
        )
        controller = BatchController(scraper, config)

        with (
            patch.object(scraper, "scrape_products_unified", return_value=[product]),
            patch.object(scraper, "_shorten_affiliate_links"),
        ):
            controller._process_keywords()

        assert _written(tmp_path, "B0TEST0001")["pillar"] == "value"
