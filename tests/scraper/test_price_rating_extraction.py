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
from src.scraper.amazon.product_extractor import _normalize_price
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
        ("44\n.", "44"),  # .a-price-whole with nested decimal span
        ("20\n.", "20"),
        ("$44.99", "44.99"),  # .a-offscreen clean full price
        ("1,299.99", "1299.99"),  # thousands separator
        ("  $19.95  ", "19.95"),
        ("", ""),  # no number
        ("Currently unavailable", ""),
    ],
)
def test_normalize_price(raw, expected):
    assert _normalize_price(raw) == expected


def test_rating_falls_back_to_serp_rating():
    """Rating is None on the detail page but serp_rating was captured."""
    assert _product(rating=None, serp_rating="4.9").rating == "4.9"


def test_explicit_rating_is_not_overwritten():
    """A real detail-page rating wins over serp_rating."""
    assert _product(rating="4.5", serp_rating="4.9").rating == "4.5"


def test_no_rating_source_stays_none():
    assert _product(rating=None, serp_rating=None).rating is None
