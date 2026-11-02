"""Every affiliate link is `<host>/dp/<ASIN>?tag=<tag>` (REQ-SCR-051).

Only a `/dp/<ASIN>` path was canonicalised; a `/gp/product/` link, a mobile
`/gp/aw/d/` link or a URL with no ASIN in the path kept its query string and
only had the tag appended.
"""

from __future__ import annotations

import pytest

from src.scraper.amazon.utils import build_affiliate_url

CANONICAL = "https://www.amazon.com/dp/B0TEST0001?tag=tag-20"


@pytest.mark.req("REQ-SCR-051")
@pytest.mark.parametrize(
    "url",
    [
        "https://www.amazon.com/dp/B0TEST0001",
        "https://www.amazon.com/Lamp-Name/dp/B0TEST0001/ref=sr_1_3?th=1",
        "https://www.amazon.com/gp/product/B0TEST0001?psc=1",
        "https://www.amazon.com/gp/aw/d/B0TEST0001#reviews",
    ],
)
def test_every_product_form_is_canonical(url: str) -> None:
    assert build_affiliate_url(url, "tag-20") == CANONICAL


@pytest.mark.req("REQ-SCR-051")
def test_a_url_without_an_asin_uses_the_product_s() -> None:
    url = "https://www.amazon.com/s?k=desk+lamp&ref=nb"
    assert build_affiliate_url(url, "tag-20", asin="B0TEST0001") == CANONICAL


def test_the_marketplace_host_is_kept() -> None:
    assert (
        build_affiliate_url("https://www.amazon.co.uk/gp/product/B0TEST0001", "t-21")
        == "https://www.amazon.co.uk/dp/B0TEST0001?tag=t-21"
    )


def test_no_asin_anywhere_still_gets_the_tag() -> None:
    assert (
        build_affiliate_url("https://www.amazon.com/s?k=lamp", "tag-20")
        == "https://www.amazon.com/s?k=lamp&tag=tag-20"
    )
