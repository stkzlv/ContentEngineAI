"""Tests for the one reader of the pillar-keyed keyword config.

Three places folded `batch.keywords` into a keyword list and a pillar map,
each with its own loop, and they disagreed. The CLI never flattened it at all:
iterating the dict yields the pillar names, so a run with no `--keywords`
searched for the literal strings `value`, `novelty` and `utility` while the
fifty-four configured keywords went unsearched.

Lookups were also byte-exact, so a keyword differing only in case or spacing
from its config spelling silently lost its pillar.
"""

import contextlib

import pytest

from src.scraper.base.keyword_pillars import (
    normalize_keyword,
    pillar_for,
    read_keyword_pillars,
)


@pytest.mark.unit
class TestReadingTheConfiguredKeywords:
    def test_a_pillar_dict_flattens_to_its_keywords(self):
        """The defect behind a no-flag run searching for pillar names."""
        keywords, _ = read_keyword_pillars(
            {"value": ["USB C hub", "smart plug"], "utility": ["portable ssd"]}
        )

        assert keywords == ["USB C hub", "smart plug", "portable ssd"]
        assert "value" not in keywords

    def test_the_keyword_list_keeps_its_spelling(self):
        """It is what gets searched; normalizing it would change the query."""
        keywords, _ = read_keyword_pillars({"value": ["USB C Hub"]})

        assert keywords == ["USB C Hub"]

    def test_a_flat_list_is_still_accepted(self):
        """The pre-pillar config shape, which attaches no pillar."""
        keywords, pillars = read_keyword_pillars(["a", "b"])

        assert keywords == ["a", "b"]
        assert pillars == {}

    @pytest.mark.parametrize("raw", [None, "", 42, {"value": "not a list"}])
    def test_an_unusable_shape_yields_nothing_rather_than_raising(self, raw):
        assert read_keyword_pillars(raw) == ([], {})


@pytest.mark.unit
class TestLookupIgnoresPresentation:
    """Case and spacing are how a keyword is written, not which keyword it is.

    A byte-exact lookup dropped the pillar for a variant, and a missing pillar
    is indistinguishable from an unconfigured keyword, so it failed silently.
    """

    @pytest.fixture
    def pillars(self):
        _, pillars = read_keyword_pillars({"value": ["USB C hub"]})
        return pillars

    @pytest.mark.parametrize(
        "written",
        ["USB C hub", "usb c hub", "USB C HUB", "  USB C hub  ", "USB  C   hub"],
    )
    def test_variants_resolve_to_the_same_pillar(self, written, pillars):
        assert pillar_for(written, pillars) == "value"

    def test_a_genuinely_different_keyword_does_not(self, pillars):
        assert pillar_for("usb hub", pillars) is None

    def test_normalize_is_idempotent(self):
        once = normalize_keyword("  USB  C hub ")
        assert normalize_keyword(once) == once


@pytest.mark.unit
class TestEveryLoaderAndLookupAgrees:
    """#208: nothing covered the path from config through to a lookup.

    The two ends were tested and the link between them was not, which is where
    both of the above defects lived.
    """

    def test_every_shipped_keyword_resolves_through_the_real_loader(self):
        """Through `load_batch_config`, not a hand-built config.

        The loader owns the CLI-over-YAML precedence, so building the map by
        hand would skip the code that actually runs.
        """
        import pathlib
        from unittest.mock import patch

        import yaml

        from src.scraper.amazon.config import load_batch_config

        cfg = yaml.safe_load(pathlib.Path("config/scraper.yaml").read_text())
        with patch("src.scraper.amazon.config.CONFIG", cfg):
            config = load_batch_config()

        # Every keyword the config ships must resolve; an unresolved one would
        # render with no pillar and be indistinguishable from an unconfigured
        # keyword.
        unresolved = [k for k in config.keywords if config.pillar_for(k) is None]
        assert not unresolved, f"configured keywords with no pillar: {unresolved}"
        assert len(config.keywords) > 3, "the pillar names leaked in as keywords"

    def test_the_batch_pipeline_loader_resolves_the_same_way(self):
        """The global batch has its own loader and its own lookup site."""
        import argparse

        from src.pipeline.config import load_global_batch_config
        from src.scraper.base.keyword_pillars import pillar_for as lookup

        ns = argparse.Namespace()
        config = load_global_batch_config(ns, "config/pipeline.yaml")

        assert config.keywords, "no keywords loaded"
        for keyword in config.keywords:
            assert (
                lookup(keyword, config.keyword_pillar_map) is not None
            ), f"{keyword!r} lost its pillar on the batch path"
        # The spelling the config ships is not the spelling the map is keyed
        # by, so a raw lookup is the defect this guards.
        mixed = [k for k in config.keywords if k != k.casefold()]
        assert mixed, "expected at least one mixed-case keyword in the config"
        for keyword in mixed:
            assert config.keyword_pillar_map.get(keyword) is None
            assert lookup(keyword, config.keyword_pillar_map) is not None

    def test_the_scrapers_own_lookup_ignores_presentation(self):
        """`pillar_for_keyword` reads the scraper's own config copy."""
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        scraper.config = {"batch": {"keywords": {"value": ["USB C hub"]}}}
        scraper._keyword_pillars = None

        assert scraper.pillar_for_keyword("USB C hub") == "value"
        assert scraper.pillar_for_keyword("usb c hub") == "value"
        assert scraper.pillar_for_keyword("  USB  C hub ") == "value"


@pytest.mark.unit
class TestTheNoFlagRunSearchesKeywords:
    """The defect site: the CLI reads `batch.keywords` when no `--keywords`
    is given, and it assigned the pillar-keyed dict straight through.

    Every later consumer treats it as a sequence, and iterating a dict yields
    its keys, so the run searched for `value`, `novelty` and `utility`. The
    reader is tested above; this pins that the CLI actually calls it.
    """

    def test_config_keywords_reach_args_flattened(self, monkeypatch, tmp_path):
        import sys
        from unittest.mock import MagicMock

        from src.scraper.amazon import scraper as scraper_mod

        seen: dict = {}

        class _Stub:
            def __init__(self, *a, **kw):
                pass

            def scrape_products(self, keywords, search_params=None):
                seen["keywords"] = list(keywords or [])
                return []

            def __getattr__(self, name):
                return MagicMock()

        def _capture_batch_config(*a, **kw):
            # Fifty-four configured keywords means the CLI takes the batch
            # arm, so the resolved list arrives here rather than at
            # scrape_products.
            seen["keywords"] = list(kw.get("cli_keywords") or [])
            raise SystemExit(0)

        monkeypatch.setattr(scraper_mod, "BotasaurusAmazonScraper", _Stub)
        # Imported inside the function, so patch it at its source module.
        from src.scraper.amazon import config as scraper_config

        monkeypatch.setattr(scraper_config, "load_batch_config", _capture_batch_config)
        monkeypatch.setattr(sys, "argv", ["scraper"])
        monkeypatch.setattr(
            scraper_mod, "load_dotenv", lambda *a, **kw: None, raising=False
        )

        with contextlib.suppress(SystemExit):
            scraper_mod.main()

        assert seen.get("keywords"), "the CLI never reached a scrape"
        assert (
            "value" not in seen["keywords"]
        ), "the pillar names were searched as keywords"
        assert any(
            " " in k for k in seen["keywords"]
        ), "expected real multi-word keywords from the config"
