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
