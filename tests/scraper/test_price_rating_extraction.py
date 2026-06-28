"""Price and rating extraction cleanup.

Price: `.a-price-whole` text includes Amazon's nested `.a-price-decimal` span,
yielding values with a trailing newline and dot. `_normalize_price` pulls a
clean numeric string.

Rating: the detail-page rating selector is unreliable, so `rating` is often
empty while the search-results card captured `serp_rating`. ProductData falls
back to `serp_rating` so consumers see a value.
"""

import pytest

from src.scraper.amazon.models import ProductData
from src.scraper.amazon.product_extractor import _normalize_price, _price_from_parts
from src.scraper.base import Platform


def _product(rating, serp_rating):
    return ProductData(
        title="Test product",
        price="19.99",
        url="https://www.amazon.com/dp/B0TEST1234",
        platform=Platform.AMAZON,
        asin="B0TEST1234",
        rating=rating,
        serp_rating=serp_rating,
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        # US grouping/decimal (comma thousands, dot decimal)
        ("44\n.", "44"),  # .a-price-whole with nested decimal span
        ("20\n.", "20"),
        ("$44.99", "44.99"),  # .a-offscreen clean full price
        ("1,299.99", "1299.99"),  # thousands separator
        ("$1,234.50", "1234.50"),
        ("  $19.95  ", "19.95"),
        ("$0.50/Count", "0.50"),  # per-unit suffix dropped
        # European grouping/decimal (dot thousands, comma decimal)
        ("19,95 EUR", "19.95"),
        ("44,99", "44.99"),
        ("1.234,56", "1234.56"),
        ("2.999.999,99", "2999999.99"),
        ("1.299", "1299"),  # lone 3-digit group is thousands, not .299
        ("1.234.567", "1234567"),  # repeated separator is grouping
        # Degenerate input
        ("", ""),  # no number
        ("Currently unavailable", ""),
    ],
)
def test_normalize_price(raw, expected):
    assert _normalize_price(raw) == expected


@pytest.mark.parametrize(
    "whole,fraction,expected",
    [
        ("44\n.", "99", "44.99"),  # split spans keep cents
        ("44\n.", "99\n", "44.99"),  # fraction span has stray whitespace
        ("1,299", "00", "1299.00"),  # grouping in the whole part
        ("1.299", "50", "1299.50"),  # European grouping in the whole part
        ("44", None, "44"),  # no fraction span -> whole dollars
        ("44", "", "44"),  # empty fraction -> whole dollars
        ("", "99", ""),  # no whole number -> empty
        ("Currently unavailable", "99", ""),
    ],
)
def test_price_from_parts(whole, fraction, expected):
    assert _price_from_parts(whole, fraction) == expected


def test_rating_falls_back_to_serp_rating():
    """Rating is None on the detail page but serp_rating was captured."""
    assert _product(rating=None, serp_rating="4.9").rating == "4.9"


def test_explicit_rating_is_not_overwritten():
    """An explicit rating (if a caller ever sets one) wins over serp_rating."""
    assert _product(rating="4.5", serp_rating="4.9").rating == "4.5"


def test_no_rating_source_stays_none():
    assert _product(rating=None, serp_rating=None).rating is None


def test_data_json_serializer_carries_rating():
    """The data.json serializer must emit the (fallback-populated) rating.

    The model fallback is useless if `_product_to_dict` drops the field, which
    is what shipped before this guard. `_product_to_dict` reads only `product`,
    so it's safe to call with a dummy `self`.
    """
    from src.scraper.amazon.scraper import BotasaurusAmazonScraper

    p = _product(rating=None, serp_rating="4.2")  # rating falls back to 4.2
    d = BotasaurusAmazonScraper._product_to_dict(None, p)
    assert d["rating"] == "4.2"
