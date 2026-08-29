"""A product scraped by ASIN or URL must carry a rating, like one by keyword.

`ProductData.__post_init__` sources `rating` from `serp_rating`, and a search
card is the only thing that sets `serp_rating`. A scrape that goes straight to
a detail page has no card, so `rating` and `reviews_count` were always `None`
-- confirmed on a real scrape of a product that carried both when reached by
keyword.

The detail page was already being read for a rating, but only when `rating`
was configured as an essential field, and the value was discarded after
validation instead of being put on the record.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.scraper.amazon.product_extractor import (
    _extract_detail_rating,
    _extract_detail_reviews_count,
)


class FakeDriver:
    """Answers `select` from a selector -> text map, like the real driver."""

    def __init__(self, texts: dict[str, str], current_url: str = ""):
        self.texts = texts
        self.current_url = current_url

    def select(self, selector: str):
        text = self.texts.get(selector)
        return SimpleNamespace(text=text) if text is not None else None

    def get_text(self, _selector: str) -> str:
        # The extractor scans the page body for availability warnings before
        # reading anything else.
        return ""


class TestRating:
    def test_the_star_widget_is_read(self):
        driver = FakeDriver({".a-icon-alt": "4.5 out of 5 stars"})

        assert _extract_detail_rating(driver) == "4.5"

    def test_a_localised_page_is_read(self):
        """A regional redirect is a documented outcome of a URL scrape."""
        driver = FakeDriver({".a-icon-alt": "4,5 de 5 estrellas"})

        assert _extract_detail_rating(driver) == "4,5"

    def test_the_product_average_wins_over_a_review_widget(self):
        """`.a-icon-alt` matches every star widget on the page.

        Including the one inside a customer review, so leading with it can put
        one reviewer's score on the record as the product's rating. The
        specific hooks are tried first for that reason.
        """
        driver = FakeDriver(
            {
                "[data-hook='average-star-rating'] .a-icon-alt": "4.5 out of 5 stars",
                ".a-icon-alt": "1.0 out of 5 stars",
            }
        )

        assert _extract_detail_rating(driver) == "4.5"

    def test_a_matched_element_without_a_score_falls_through(self):
        """The old code broke on the first match and returned nothing.

        So a page whose first star widget carried no parseable score yielded no
        rating even when a later selector would have found one.
        """
        driver = FakeDriver(
            {
                "[data-hook='average-star-rating'] .a-icon-alt": "See all reviews",
                ".a-icon-alt": "4.7 out of 5 stars",
            }
        )

        assert _extract_detail_rating(driver) == "4.7"

    def test_no_star_widget_is_not_an_error(self):
        assert _extract_detail_rating(FakeDriver({})) is None


class TestReviewsCount:
    def test_the_count_is_read_as_written(self):
        """The search card supplies this field in the same shape."""
        driver = FakeDriver({"#acrCustomerReviewText": "1,234 ratings"})

        assert _extract_detail_reviews_count(driver) == "1,234 ratings"

    def test_an_empty_element_falls_through(self):
        driver = FakeDriver(
            {
                "#acrCustomerReviewText": "   ",
                "[data-hook='total-review-count']": "12 ratings",
            }
        )

        assert _extract_detail_reviews_count(driver) == "12 ratings"

    def test_no_count_is_not_an_error(self):
        assert _extract_detail_reviews_count(FakeDriver({})) is None


def test_the_extractor_puts_both_on_the_record(monkeypatch):
    """The regression guard: the values were read and then dropped.

    Every arm -- keyword, ASIN and URL -- reaches
    `extract_product_data_from_page`, so this is the one place that decides
    whether the fields exist at all.
    """
    from src.scraper.amazon import product_extractor as extractor

    driver = FakeDriver(
        {
            "#productTitle": "A product",
            ".a-price .a-offscreen": "$19.99",
            "#feature-bullets ul": "A description long enough to be real.",
            "[data-hook='average-star-rating'] .a-icon-alt": "4.6 out of 5 stars",
            "#acrCustomerReviewText": "88 ratings",
        },
        current_url="https://www.amazon.com/dp/B0TEST1234",
    )

    monkeypatch.setattr(
        extractor, "extract_high_res_images_botasaurus", lambda *a, **k: ["img"]
    )
    monkeypatch.setattr(
        extractor, "extract_functional_videos_with_validation", lambda *a, **k: []
    )

    data = extractor.extract_product_data_from_page(
        driver, "B0TEST1234", "B0TEST1234", serp_info=None
    )

    assert data is not None, "the fake page should pass validation"
    assert data["rating"] == "4.6"
    assert data["reviews_count"] == "88 ratings"
    # No search card on this arm, which is the whole point.
    assert data["serp_rating"] is None


def test_the_record_keeps_the_detail_values(monkeypatch, tmp_path):
    """`_validate_and_convert_products` is where the dict becomes a record.

    A key added to the extractor's dict and not read here goes no further than
    the browser callback's raw output, which every arm overwrites.
    """
    from src.scraper.amazon.scraper import BotasaurusAmazonScraper

    result: dict[str, object] = {
        "title": "A product",
        "price": "19.99",
        "description": "A description.",
        "images": [],
        "videos": [],
        "affiliate_link": "https://www.amazon.com/dp/B0TEST1234",
        "url": "https://www.amazon.com/dp/B0TEST1234",
        "asin": "B0TEST1234",
        "keyword": "B0TEST1234",
        "rating": "4.6",
        "reviews_count": "88 ratings",
        "serp_rating": None,
        "serp_reviews_count": None,
        "downloaded_images": [],
        "downloaded_videos": [],
    }

    import logging

    scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
    scraper.logger = logging.getLogger("test")
    scraper.debug_mode = False
    scraper.output_dir = tmp_path
    scraper.profile_uses_videos = False
    products = BotasaurusAmazonScraper._validate_and_convert_products(
        scraper, [result], filter_validated=False
    )

    assert len(products) == 1
    assert products[0].rating == "4.6"
    assert products[0].reviews_count == "88 ratings"


def test_a_search_card_still_supplies_the_rating(monkeypatch):
    """The card remains the fallback for a page whose widget is unreadable."""
    from src.scraper.amazon.models import ProductData
    from src.scraper.base import Platform

    product = ProductData(
        title="A product",
        price="19.99",
        url="https://www.amazon.com/dp/B0TEST1234",
        platform=Platform.AMAZON,
        asin="B0TEST1234",
        rating=None,
        serp_rating="4.9",
    )

    assert product.rating == "4.9"


@pytest.mark.parametrize("field", ["rating", "reviews_count"])
def test_the_serialiser_emits_both(field):
    """`to_dict` is what reaches `data.json`; a dropped key is silent."""
    from src.scraper.amazon.models import ProductData
    from src.scraper.base import Platform

    product = ProductData(
        title="A product",
        price="19.99",
        url="https://www.amazon.com/dp/B0TEST1234",
        platform=Platform.AMAZON,
        asin="B0TEST1234",
        rating="4.6",
        reviews_count="88 ratings",
    )

    assert product.to_dict()[field] == getattr(product, field)
