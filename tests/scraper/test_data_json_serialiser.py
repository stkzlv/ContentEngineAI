"""Tests that one serialiser writes `data.json` for every path.

Two hand-written dicts used to produce this file, and they had drifted: a field
added to the dataclass reached the file on the topic path and vanished on the
scraper path, with nothing to say which. A field that persists on one path and
not the other looks wired from every angle except the file on disk.
"""

import dataclasses
import json

import pytest

from src.scraper.amazon.models import ProductData
from src.scraper.base.models import BaseProductData

# The keys a scraped `data.json` carried before the serialisers were merged.
# Pinned rather than derived: anything dropped from this set stops a consumer
# that reads the file, and the merge must only ever add.
LEGACY_SCRAPED_KEYS = {
    "affiliate_link",
    "asin",
    "description",
    "downloaded_images",
    "downloaded_videos",
    "images",
    "keyword",
    "platform",
    "price",
    "rating",
    "serp_rating",
    "serp_reviews_count",
    "shortened_affiliate_link",
    "title",
    "url",
    "videos",
}


def _product(**extra) -> ProductData:
    payload = {
        "title": "A product",
        "price": "$10",
        "url": "https://www.amazon.com/dp/B0TEST0001",
        "platform": None,
        "asin": "B0TEST0001",
    }
    payload.update(extra)
    return ProductData(**payload)


@pytest.mark.unit
class TestOneSerialiser:
    def test_the_scraper_writes_through_the_records_own_serialiser(self):
        """Not a second dict that happens to agree today."""
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        product = _product()
        written = BotasaurusAmazonScraper._product_to_dict(None, product)
        assert written == product.to_dict()

    def test_every_dataclass_field_reaches_the_file(self):
        """The drift was a field on the model that no serialiser carried.

        Comparing the two serialisers to each other cannot catch that, because
        they can agree and both be incomplete.
        """
        serialised = set(_product().to_dict())
        declared = {f.name for f in dataclasses.fields(ProductData)}
        assert declared - serialised == set()

    def test_no_key_a_consumer_reads_was_dropped(self):
        assert LEGACY_SCRAPED_KEYS - set(_product().to_dict()) == set()

    def test_every_serialised_key_can_be_read_back(self):
        """Batch discovery does `ProductData(**loaded)` with no filter.

        A key in the file that is not a dataclass field raises
        `unexpected keyword argument` there, and the caller logs it and skips
        the product. The other direction is asserted above; this is the one the
        round trip actually depends on, and the tests' own `_reload` filters
        the payload so it would not notice.
        """
        declared = {f.name for f in dataclasses.fields(ProductData)}
        assert set(_product().to_dict()) - declared == set()


@pytest.mark.unit
class TestFieldsThatUsedToVanish:
    def test_pillar_reaches_a_scraped_record(self):
        """It was set in memory during a run and never written.

        A resume or a batch discovery pass re-reads the file, so the pillar the
        video was rendered under was lost every time.
        """
        assert _product(pillar="utility").to_dict()["pillar"] == "utility"

    def test_topic_reaches_a_record(self):
        assert _product(topic="Why wifi drops").to_dict()["topic"] == "Why wifi drops"

    @pytest.mark.parametrize(
        "field", ["brand", "category", "platform_id", "search_position"]
    )
    def test_the_other_absent_fields_reach_the_file(self, field):
        assert field in _product().to_dict()


@pytest.mark.unit
class TestRoundTrip:
    """A record read back from disk carries plain strings, not enums.

    `ProductData(**loaded)` does no coercion, so re-serialising one used to
    raise `AttributeError` on `.value`. The scraper's own dict guarded against
    it; the shared one has to as well or the merge trades one silent failure
    for a loud one.
    """

    def _reload(self, product: ProductData) -> ProductData:
        raw = json.loads(json.dumps(product.to_dict()))
        fields = {f.name for f in dataclasses.fields(ProductData)}
        return ProductData(**{k: v for k, v in raw.items() if k in fields})

    def test_a_loaded_record_serialises_again(self):
        assert self._reload(_product()).to_dict()["platform"] == "amazon"

    def test_the_enum_value_survives_the_trip(self):
        product = _product(pillar="value")
        assert self._reload(product).to_dict()["pillar"] == "value"

    def test_status_survives_too(self):
        """`status` is the second enum, and it fails the same way."""
        assert self._reload(_product()).to_dict()["status"] == "unknown"


@pytest.mark.unit
class TestBaseRecord:
    def test_a_non_amazon_record_serialises_without_the_amazon_fields(self):
        """The base class has no ASIN, and must not grow one."""
        base = BaseProductData(
            title="t", price="1", url="u", platform=None, description="d"
        )
        assert "asin" not in base.to_dict()
        assert base.to_dict()["pillar"] is None


@pytest.mark.unit
class TestPillarValueReachesTheFile:
    """The key existing is not the same as the value arriving.

    The batch path sets `pillar` on the record; this asserts the serialiser
    carries whatever was set through to the dict the scraper writes. Which
    paths set it at all is a separate question, tracked separately.
    """

    def test_a_pillar_set_after_construction_is_written(self):
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        product = _product()
        product.pillar = "utility"
        assert BotasaurusAmazonScraper._product_to_dict(None, product)["pillar"] == (
            "utility"
        )

    def test_an_unset_pillar_is_written_as_null_not_omitted(self):
        """A key that is present and null says the pillar was not attached.

        An absent key cannot distinguish that from a serialiser that never
        carried the field, which is how this went unnoticed.
        """
        written = _product().to_dict()
        assert "pillar" in written
        assert written["pillar"] is None


@pytest.mark.unit
class TestBothStandaloneArmsWriteTheSameKeys:
    """A product scraped by ASIN and by keyword must produce one shape.

    The `--product-ids` arm never called `_save_products`, so its file was
    whatever the Botasaurus output callback wrote mid-scrape: the raw
    extractor dict, missing ten of the canonical keys and carrying an empty
    `downloaded_images` because the callback fires before the media
    downloads run. Nothing downstream knows to expect two shapes.
    """

    @staticmethod
    def _run(tmp_path, *, by_keyword: bool):
        from unittest.mock import MagicMock, patch

        from src.scraper.amazon.batch_controller import BatchController
        from src.scraper.amazon.models import (
            BatchConfig,
            ProductData,
            SearchParameters,
        )
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        scraper.config = {"batch": {"keywords": {"value": ["smart plug"]}}}
        scraper._keyword_pillars = None
        scraper.output_dir = str(tmp_path)
        scraper.debug_mode = False
        scraper.logger = MagicMock()

        product = ProductData(
            title="A product",
            price="$10",
            url="https://www.amazon.com/dp/B0TEST0002",
            platform=None,
            asin="B0TEST0002",
        )
        config = BatchConfig(
            product_ids=[] if by_keyword else ["B0TEST0002"],
            keywords=["smart plug"] if by_keyword else [],
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=1,
            products_per_keyword=1,
        )
        controller = BatchController(scraper, config)

        with (
            patch.object(scraper, "scrape_products_unified", return_value=[product]),
            patch.object(scraper, "_shorten_affiliate_links"),
        ):
            if by_keyword:
                controller._process_keywords()
            else:
                controller._process_product_ids()

        written = tmp_path / "B0TEST0002" / "data.json"
        assert written.exists(), "the arm wrote no data.json"
        return set(json.loads(written.read_text())[0])

    def test_the_two_arms_agree(self, tmp_path):
        by_id = self._run(tmp_path / "ids", by_keyword=False)
        by_keyword = self._run(tmp_path / "kw", by_keyword=True)
        assert by_id == by_keyword

    def test_the_product_ids_arm_writes_every_declared_field(self, tmp_path):
        from dataclasses import fields

        from src.scraper.amazon.models import ProductData

        written = self._run(tmp_path, by_keyword=False)
        assert {f.name for f in fields(ProductData)} <= written
