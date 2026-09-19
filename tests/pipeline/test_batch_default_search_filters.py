"""The global batch seeds its search filters the way the standalone CLI does.

The CLI starts from `scrapers.amazon.default_search_parameters` and layers
its flags on top. The batch started from a bare `SearchParameters`, so a
keyword run with no flags and no pipeline.yaml filters searched with no
price or rating bound at all (`basic search (no filters)` on 2026-09-19).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.config import load_global_batch_config, resolve_scraper_filters
from src.scraper.amazon.cli import _build_search_params
from src.scraper.amazon.config import get_default_search_parameters


def _cli_args(**overrides) -> argparse.Namespace:
    """The scraper CLI's namespace with nothing given."""
    args: dict[str, Any] = {
        "min_price": None,
        "max_price": None,
        "min_rating": None,
        "prime_only": False,
        "free_shipping": False,
        "brands": [],
        "category": None,
        "sort": "relevance",
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def _batch_filters(tmp_path: Path, yaml_filters: dict | None = None, **cli):
    path = tmp_path / "pipeline.yaml"
    section: dict = {}
    if yaml_filters is not None:
        section["scraper_filters"] = yaml_filters
    path.write_text(yaml.safe_dump({"global_batch": section}))
    return load_global_batch_config(
        argparse.Namespace(**cli), config_path=path
    ).scraper_filters


class TestBothEntryPointsResolveTheSameDefaults:
    def test_nothing_given_resolves_to_the_scraper_defaults(self, tmp_path):
        batch = _batch_filters(tmp_path)
        cli = _build_search_params(_cli_args())

        assert cli is not None
        assert batch == cli[0] == get_default_search_parameters()

    def test_the_configured_defaults_are_real_filters(self, tmp_path):
        """The shipped scraper config carries a price and rating bound; a
        batch that resolved to None for both would be the old behaviour.
        """
        batch = _batch_filters(tmp_path)

        assert batch.min_price is not None
        assert batch.min_rating is not None

    def test_a_null_pipeline_field_means_the_scraper_value(self, tmp_path):
        nulls = {
            "min_price": None,
            "max_price": None,
            "min_rating": None,
            "prime_only": None,
        }

        assert _batch_filters(tmp_path, nulls) == get_default_search_parameters()


class TestOverridesLayerInOrder:
    def test_a_pipeline_value_overrides_one_field_only(self, tmp_path):
        defaults = get_default_search_parameters()

        batch = _batch_filters(tmp_path, {"min_price": 5})

        assert batch.min_price == 5
        assert batch.max_price == defaults.max_price
        assert batch.min_rating == defaults.min_rating
        assert batch.sort_order == defaults.sort_order

    def test_a_cli_flag_overrides_the_pipeline_value(self, tmp_path):
        batch = _batch_filters(tmp_path, {"min_price": 5}, min_price=7)

        assert batch.min_price == 7

    def test_a_zero_bound_from_the_cli_is_kept(self):
        """The CLI treats 0 as a real bound; the batch used `or` and lost it."""
        filters = resolve_scraper_filters(
            argparse.Namespace(min_price=0), {"min_price": 5}
        )

        assert filters.min_price == 0

    def test_prime_from_the_pipeline_and_the_cli(self, monkeypatch):
        """With the scraper default at true, so that an explicit false in
        pipeline.yaml has something to override (the shipped default is
        false, and against it this case would pass with no yaml layer).
        """
        from src.scraper.amazon import config as scraper_config
        from src.scraper.amazon.models import SearchParameters

        monkeypatch.setattr(
            scraper_config,
            "get_default_search_parameters",
            lambda: SearchParameters(prime_only=True),
        )

        assert resolve_scraper_filters(argparse.Namespace(), {}).prime_only
        assert not resolve_scraper_filters(
            argparse.Namespace(), {"prime_only": False}
        ).prime_only
        assert resolve_scraper_filters(
            argparse.Namespace(), {"prime_only": None}
        ).prime_only
        assert resolve_scraper_filters(
            argparse.Namespace(prime_only=True), {"prime_only": False}
        ).prime_only
