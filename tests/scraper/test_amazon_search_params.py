"""Tests for the Amazon SearchParameters dataclass defaults.

This is `src.scraper.amazon.models.SearchParameters` (the dataclass the global
batch uses via scraper_filters), not the Pydantic `config_models` one.
"""

from src.scraper.amazon.models import SearchParameters


def test_default_sort_order_is_valid_amazon_token():
    # The global batch builds scraper_filters from this bare default, so the
    # default must be a valid Amazon token, not the CLI-friendly "relevance".
    params = SearchParameters()
    assert params.sort_order == "relevanceblender"


def test_default_params_validate_cleanly():
    assert SearchParameters().validate() == []
