"""A search keyword reaches the site as typed, encoded once (#811)."""

from urllib.parse import parse_qs, urlparse

import pytest

from src.scraper.amazon.search_builder import SearchParameterBuilder


def _keyword(url: str) -> str:
    return parse_qs(urlparse(url).query)["k"][0]


@pytest.mark.req("REQ-SCR-080")
@pytest.mark.parametrize("keyword", ["red running shoes", "usb-c hub 4k", "a+b"])
def test_a_keyword_round_trips_through_the_url(keyword: str) -> None:
    url = SearchParameterBuilder("https://www.amazon.com").build_search_url(keyword)

    assert _keyword(url) == keyword
    assert "%2B" not in url or "+" in keyword


@pytest.mark.req("REQ-SCR-080")
def test_spaces_are_sent_as_plus_signs() -> None:
    url = SearchParameterBuilder("https://www.amazon.com").build_search_url(
        "red running shoes", page=2
    )

    assert "k=red+running+shoes" in url
