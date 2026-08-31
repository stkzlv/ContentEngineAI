"""Two consecutive batch runs draw different products.

Nothing in the input path varied. `config/pipeline.yaml` carried six keywords,
every one of them also in `config/scraper.yaml`'s fifty-four, so the batch
searched six however many were configured next door. At
`products_per_keyword: 1` a six-product run exhausted that pool exactly, and
two runs an hour apart returned the same three already-published products.

Two things had to change together. Reading one pool is what makes a rotation
worth anything -- rotating a list the cap consumes whole changes nothing --
and rotating is what makes the larger pool reachable, since the run stops at
`max_products` and would otherwise always take the head.

The third part is separate: an already-published product was scraped,
downloaded and rendered before the publish phase dropped it, so the render
cost was paid for output nobody sees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.pipeline.config import (
    _scraper_keyword_pool,
    keywords_for_run,
    load_global_batch_config,
)


@pytest.mark.unit
class TestTheRotationIsWorthHaving:
    """Guards the property the issue asks for, not the implementation."""

    def test_consecutive_days_are_disjoint(self):
        pool = [f"kw{i}" for i in range(54)]

        first = keywords_for_run(pool, 10, day_ordinal=739000)
        second = keywords_for_run(pool, 10, day_ordinal=739001)

        assert set(first) & set(second) == set()

    def test_the_pool_is_covered_rather_than_sampled(self):
        """A rotation that never reaches the tail is no better than the head."""
        pool = [f"kw{i}" for i in range(54)]

        seen = set()
        for day in range(739000, 739000 + 6):
            seen |= set(keywords_for_run(pool, 9, day_ordinal=day))

        assert seen == set(pool)

    def test_a_full_width_slice_still_rotates_nothing(self):
        """The trap this nearly shipped with.

        Taking the whole pool makes the start offset a multiple of the length,
        which is zero, so the rotation is the identity. That is why the slice
        width is what the run consumes rather than the pool size.
        """
        pool = [f"kw{i}" for i in range(10)]

        assert keywords_for_run(pool, 10, day_ordinal=739000) == pool
        assert keywords_for_run(pool, 10, day_ordinal=739001) == pool

    def test_asking_for_more_than_the_pool_holds_returns_it_once(self):
        pool = ["a", "b", "c"]

        got = keywords_for_run(pool, 10, day_ordinal=739000)

        assert sorted(got) == ["a", "b", "c"]

    @pytest.mark.parametrize("count", [0, -1])
    def test_a_non_positive_count_yields_nothing(self, count):
        assert keywords_for_run(["a", "b"], count, day_ordinal=739000) == []

    def test_an_empty_pool_yields_nothing(self):
        assert keywords_for_run([], 5, day_ordinal=739000) == []


@pytest.mark.unit
class TestTheBatchDrawsTheScraperPool:
    def test_the_bundled_config_no_longer_shadows_the_scraper(self):
        """`global_batch.keywords` is empty, so the fallback is load-bearing."""
        import yaml

        with open("config/pipeline.yaml", encoding="utf-8") as handle:
            batch = yaml.safe_load(handle)["global_batch"]

        assert not batch.get("keywords"), (
            "pipeline.yaml carries its own keyword list again; if that is "
            "deliberate, it shadows the scraper's pool for every batch run"
        )

    def test_the_scraper_pool_is_the_larger_one(self):
        """States the asymmetry the fallback exists for."""
        pool, pillars = _scraper_keyword_pool()

        assert len(pool) > 20
        assert pillars

    def test_a_no_flag_run_reaches_the_wide_pool(self):
        config = load_global_batch_config(argparse.Namespace())

        assert config.keywords
        assert len(config.keyword_pillar_map) > 20
        # The run takes a slice, not the whole pool, or there is no rotation.
        assert len(config.keywords) < len(_scraper_keyword_pool()[0])

    def test_cli_keywords_are_not_rotated(self):
        """A keyword typed on the command line was asked for by name."""
        args = argparse.Namespace(keywords=["one", "two"])

        config = load_global_batch_config(args)

        assert config.keywords == ["one", "two"]

    def test_the_pool_follows_the_config_directory(self, tmp_path):
        """Pointing the batch elsewhere must not read the repo's scraper file.

        The fallback first defaulted to `config/scraper.yaml` regardless, so
        loading any other pipeline.yaml silently picked up this repo's
        keywords -- which an existing test caught by asserting that a config
        with no keywords produces no pillars.
        """
        import yaml

        (tmp_path / "pipeline.yaml").write_text(
            yaml.safe_dump({"global_batch": {"product_ids": ["B0ABCDEFGH"]}}),
            encoding="utf-8",
        )

        config = load_global_batch_config(
            argparse.Namespace(), str(tmp_path / "pipeline.yaml")
        )

        assert config.keyword_pillar_map == {}
        assert config.keywords == []

    def test_a_sibling_scraper_config_is_read(self, tmp_path):
        import yaml

        (tmp_path / "pipeline.yaml").write_text(
            yaml.safe_dump({"global_batch": {}}), encoding="utf-8"
        )
        (tmp_path / "scraper.yaml").write_text(
            yaml.safe_dump({"batch": {"keywords": {"value": ["sideloaded"]}}}),
            encoding="utf-8",
        )

        config = load_global_batch_config(
            argparse.Namespace(), str(tmp_path / "pipeline.yaml")
        )

        assert config.keywords == ["sideloaded"]
        assert config.keyword_pillar_map == {"sideloaded": "value"}

    def test_a_missing_scraper_config_is_not_fatal(self, tmp_path):
        pool, pillars = _scraper_keyword_pool(str(tmp_path / "absent.yaml"))

        assert pool == []
        assert pillars == {}

    def test_unreadable_scraper_config_is_not_fatal(self, tmp_path):
        bad = tmp_path / "scraper.yaml"
        bad.write_text("batch: [unclosed\n", encoding="utf-8")

        pool, pillars = _scraper_keyword_pool(str(bad))

        assert pool == []
        assert pillars == {}


@pytest.mark.unit
class TestAlreadyPublishedProductsAreNotRendered:
    """The filter sits before the render, which is where the cost is."""

    @staticmethod
    def _pipeline(tmp_path, platforms=None, force=False):
        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        pipeline = GlobalPipelineOrchestrator.__new__(GlobalPipelineOrchestrator)
        pipeline.config = MagicMock()
        pipeline.config.outputs_dir = tmp_path
        pipeline.config.platforms = platforms
        pipeline.config.force = force
        return pipeline

    @staticmethod
    def _history(tmp_path, entries: dict):
        (tmp_path / "publish_history.json").write_text(
            json.dumps({"posts": entries}), encoding="utf-8"
        )

    @staticmethod
    def _product(asin: str):
        data = MagicMock()
        data.asin = asin
        return (Path(f"outputs/{asin}"), data)

    def test_a_fully_published_product_is_dropped(self, tmp_path):
        self._history(
            tmp_path,
            {
                "B0AAA": {"post_id": "1"},
                "B0AAA:youtube": {"post_id": "1"},
                "B0AAA:tiktok": {"post_id": "2"},
                "B0AAA:instagram": {"post_id": "3"},
            },
        )
        pipeline = self._pipeline(tmp_path)

        kept = pipeline._drop_already_published([self._product("B0AAA")])

        assert kept == []

    def test_a_partially_published_product_is_kept(self, tmp_path):
        """It still has somewhere to go, so the render is not wasted."""
        self._history(tmp_path, {"B0BBB:youtube": {"post_id": "1"}})
        pipeline = self._pipeline(tmp_path)

        kept = pipeline._drop_already_published([self._product("B0BBB")])

        assert len(kept) == 1

    def test_an_unpublished_product_is_kept(self, tmp_path):
        self._history(tmp_path, {})
        pipeline = self._pipeline(tmp_path)

        kept = pipeline._drop_already_published([self._product("B0CCC")])

        assert len(kept) == 1

    def test_the_filter_respects_the_run_s_own_platforms(self, tmp_path):
        """Published on youtube only, and youtube is all this run targets."""
        self._history(tmp_path, {"B0DDD:youtube": {"post_id": "1"}})
        pipeline = self._pipeline(tmp_path, platforms=["youtube"])

        kept = pipeline._drop_already_published([self._product("B0DDD")])

        assert kept == []

    def test_force_renders_it_anyway(self, tmp_path):
        self._history(
            tmp_path,
            {
                "B0EEE:youtube": {"post_id": "1"},
                "B0EEE:tiktok": {"post_id": "2"},
                "B0EEE:instagram": {"post_id": "3"},
            },
        )
        pipeline = self._pipeline(tmp_path, force=True)

        kept = pipeline._drop_already_published([self._product("B0EEE")])

        assert len(kept) == 1

    def test_a_record_with_no_asin_is_kept(self, tmp_path):
        """A topic carries no ASIN and cannot be looked up."""
        self._history(tmp_path, {})
        pipeline = self._pipeline(tmp_path)
        data = MagicMock()
        data.asin = None

        kept = pipeline._drop_already_published([(Path("outputs/topic-x"), data)])

        assert len(kept) == 1

    def test_no_history_file_keeps_everything(self, tmp_path):
        pipeline = self._pipeline(tmp_path)

        kept = pipeline._drop_already_published([self._product("B0FFF")])

        assert len(kept) == 1


@pytest.mark.unit
class TestTheForceFlagExists:
    """The filter needs an escape hatch, and the other paths already have one."""

    def test_the_batch_parser_accepts_force(self):
        from src.pipeline.global_batch import create_argument_parser

        args = create_argument_parser().parse_args(["--force"])

        assert args.force is True

    def test_it_reaches_the_config(self):
        config = load_global_batch_config(argparse.Namespace(force=True))

        assert config.force is True

    def test_it_defaults_off(self):
        config = load_global_batch_config(argparse.Namespace())

        assert config.force is False
