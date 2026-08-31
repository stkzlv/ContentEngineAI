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
        # Every flag the filter reads has to be set explicitly: an unset
        # attribute on a MagicMock is truthy, so leaving `skip_publish` alone
        # short-circuits the guard and every assertion below passes vacuously.
        pipeline.config.skip_publish = False
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

    def test_a_published_topic_is_still_rendered(self, tmp_path):
        """The silent one, and the reason it is silent.

        A topic's id is `topic_product_id(title)`, a pure function of the
        title, and the batch records its publishes under exactly that. So a
        guard that treats a topic like a product drops it on every run after
        the first -- permanently, with no failure and no skip. The bundled
        config ships two topics at one per run, so the tutorial arm would
        have stopped producing on day three.

        The test this replaces passed a record with `asin = None`, a shape no
        topic ever has, so it asserted nothing about topics at all.
        """
        from src.video.producer.topic_input import topic_product_id

        topic_id = topic_product_id("Why your wifi keeps dropping")
        self._history(
            tmp_path,
            {
                f"{topic_id}:youtube": {"post_id": "1"},
                f"{topic_id}:tiktok": {"post_id": "2"},
                f"{topic_id}:instagram": {"post_id": "3"},
            },
        )
        pipeline = self._pipeline(tmp_path)
        data = MagicMock()
        data.asin = topic_id

        kept = pipeline._drop_already_published([(Path(f"outputs/{topic_id}"), data)])

        assert len(kept) == 1, "a published topic was dropped and never renders again"

    def test_a_record_with_no_asin_is_kept(self, tmp_path):
        self._history(tmp_path, {})
        pipeline = self._pipeline(tmp_path)
        data = MagicMock()
        data.asin = None

        kept = pipeline._drop_already_published([(Path("outputs/x"), data)])

        assert len(kept) == 1

    def test_a_skip_publish_run_renders_everything(self, tmp_path):
        """Nothing to duplicate, and re-rendering is the point of such a run."""
        self._history(
            tmp_path,
            {
                "B0GGG:youtube": {"post_id": "1"},
                "B0GGG:tiktok": {"post_id": "2"},
                "B0GGG:instagram": {"post_id": "3"},
            },
        )
        pipeline = self._pipeline(tmp_path)
        pipeline.config.skip_publish = True

        kept = pipeline._drop_already_published([self._product("B0GGG")])

        assert len(kept) == 1

    def test_a_narrowed_publisher_config_changes_the_decision(
        self, tmp_path, monkeypatch
    ):
        """The guard must ask what the publish phase will actually target.

        Asserted through the decision, not through the returned list. The
        bundled `default_platforms` *is* the hardcoded triple, so comparing
        against it passes against the implementation this replaced -- the
        same shape of vacuous test as the `asin = None` one above.
        """
        import yaml

        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "publisher.yaml").write_text(
            yaml.safe_dump({"default_platforms": ["youtube", "instagram"]}),
            encoding="utf-8",
        )
        self._history(
            tmp_path,
            {
                "B0HHH:youtube": {"post_id": "1"},
                "B0HHH:instagram": {"post_id": "2"},
            },
        )
        pipeline = self._pipeline(tmp_path)
        monkeypatch.chdir(tmp_path)

        kept = pipeline._drop_already_published([self._product("B0HHH")])

        assert kept == [], (
            "the product is published everywhere this install targets, so it "
            "must not be rendered; a hardcoded triple would demand tiktok"
        )

    def test_platform_names_are_matched_case_insensitively(self, tmp_path):
        """`record_publish` writes lowercase; config validation does not.

        A `platforms: [YouTube]` install would look up a key nothing ever
        writes, keep every product, and publish duplicates.
        """
        self._history(tmp_path, {"B0III:youtube": {"post_id": "1"}})
        pipeline = self._pipeline(tmp_path, platforms=["YouTube"])

        kept = pipeline._drop_already_published([self._product("B0III")])

        assert kept == []

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


@pytest.mark.unit
class TestARunThatFindsNothingNewIsNotAFailure:
    """A dropped product is not a lost one.

    The drop happens after the scraping summary is built, so the products
    never reach `total_attempted` and `end_to_end_success` is zero. Without a
    distinct outcome the run reported `PIPELINE FAILED ... 0 failed, 0
    skipped` and exited 1 -- a contradiction, on a correct result, that would
    page whoever watches the cron. The rotation makes it a normal outcome: it
    walks the whole pool in under a week, so from the second week the same
    keywords return the same already-published top results.
    """

    @staticmethod
    def _summary(production, failures=0):
        from src.pipeline.config import PipelineSummary, ScrapingPhaseSummary

        scraping = ScrapingPhaseSummary(10, 10, 0, ["B0X"], [], {}, 1.0)
        return PipelineSummary(
            scraping=scraping,
            production=production,
            publishing=None,
            end_to_end_success=production.successful,
            partial_success=10,
            total_failures=failures,
            total_duration_sec=1.0,
        )

    @staticmethod
    def _production(**kwargs):
        from src.pipeline.config import ProductionPhaseSummary

        base: dict = {
            "total_attempted": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "failed_products": [],
            "skipped_products": [],
            "profile_distribution": None,
            "duration_sec": 0.0,
        }
        base.update(kwargs)
        return ProductionPhaseSummary(**base)

    def test_everything_already_published_exits_zero(self):
        summary = self._summary(
            self._production(already_published=3, already_published_products=["a"])
        )

        assert summary.outcome() == "nothing new"
        assert summary.exit_code() == 0

    def test_strict_does_not_fail_it_either(self):
        """`strict` catches a product asked for that does not exist.

        Nothing was asked for here, so there is nothing for it to catch.
        """
        summary = self._summary(
            self._production(already_published=3, already_published_products=["a"])
        )

        assert summary.exit_code(strict=True) == 0

    def test_a_genuinely_empty_run_still_fails(self):
        """The verdict this must not weaken."""
        summary = self._summary(self._production())

        assert summary.outcome() == "failed"
        assert summary.exit_code() == 1

    def test_a_real_failure_alongside_drops_still_fails(self):
        summary = self._summary(
            self._production(
                total_attempted=1,
                failed=1,
                failed_products=["X"],
                already_published=2,
                already_published_products=["a", "b"],
            ),
            failures=1,
        )

        assert summary.outcome() == "failed"
        assert summary.exit_code() == 1

    def test_a_partial_run_with_drops_is_still_lost(self):
        summary = self._summary(
            self._production(
                total_attempted=2,
                successful=1,
                skipped=1,
                skipped_products=["Y"],
                already_published=1,
                already_published_products=["a"],
            )
        )

        assert summary.outcome() == "lost"
        assert summary.exit_code() == 0
        assert summary.exit_code(strict=True) == 1

    def test_the_summary_names_them(self):
        """A run that rendered nothing has to say why."""
        summary = self._summary(
            self._production(already_published=1, already_published_products=["B0ZZZ"])
        )

        assert "B0ZZZ" in summary.format()


@pytest.mark.unit
class TestKeywordsPerRunIsValidated:
    """Its sibling `topics_per_run` is; this was not.

    A string raised a bare TypeError naming no key, a negative silently
    searched nothing, and `0` fell through to the default and searched ten --
    the opposite of what `topics_per_run: 0` means in the same file.
    """

    @staticmethod
    def _configs(tmp_path, value):
        import yaml

        (tmp_path / "scraper.yaml").write_text(
            yaml.safe_dump({"batch": {"keywords": {"value": ["a", "b", "c"]}}}),
            encoding="utf-8",
        )
        block: dict = {}
        if value is not None:
            block["keywords_per_run"] = value
        (tmp_path / "pipeline.yaml").write_text(
            yaml.safe_dump({"global_batch": block}), encoding="utf-8"
        )
        return str(tmp_path / "pipeline.yaml")

    @pytest.mark.parametrize("value", ["10", 3.5, [1]])
    def test_a_non_integer_is_refused_by_name(self, tmp_path, value):
        path = self._configs(tmp_path, value)

        with pytest.raises(ValueError, match="keywords_per_run"):
            load_global_batch_config(argparse.Namespace(), path)

    def test_a_negative_is_refused_by_name(self, tmp_path):
        path = self._configs(tmp_path, -1)

        with pytest.raises(ValueError, match="keywords_per_run"):
            load_global_batch_config(argparse.Namespace(), path)

    def test_zero_means_no_keyword_search(self, tmp_path):
        """`topics_per_run: 0` means no topics, so this must match."""
        path = self._configs(tmp_path, 0)

        config = load_global_batch_config(argparse.Namespace(), path)

        assert config.keywords == []

    def test_absent_falls_back_to_what_the_run_consumes(self, tmp_path):
        path = self._configs(tmp_path, None)

        config = load_global_batch_config(argparse.Namespace(), path)

        assert config.keywords


@pytest.mark.unit
class TestTheDropReachesTheVerdict:
    """Driven through `run_pipeline`, because the wiring was the defect.

    The `nothing new` outcome was added, tested, and documented in three
    places while never firing in the case it exists for. When the guard drops
    *every* product there is nothing to render, so `run_pipeline` takes the
    early-return branch, and that summary did not carry the drop count -- so
    the run still reported `PIPELINE FAILED ... 0 failed, 0 skipped` and
    exited 1.

    Every test written for it built `ProductionPhaseSummary` by hand and
    called `outcome()` directly, so all of them passed against the broken
    wiring. This one asserts the verdict the operator actually sees.
    """

    @pytest.mark.asyncio
    async def test_a_run_whose_products_are_all_published_exits_zero(
        self, tmp_path, monkeypatch
    ):
        from unittest.mock import AsyncMock, MagicMock

        from src.pipeline.config import ScrapingPhaseSummary
        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        asin = "B0PUBLISHED"
        product_dir = tmp_path / asin
        product_dir.mkdir()
        (tmp_path / "publish_history.json").write_text(
            json.dumps(
                {
                    "posts": {
                        f"{asin}:youtube": {"post_id": "1"},
                        f"{asin}:tiktok": {"post_id": "2"},
                        f"{asin}:instagram": {"post_id": "3"},
                    }
                }
            ),
            encoding="utf-8",
        )

        pipeline = GlobalPipelineOrchestrator.__new__(GlobalPipelineOrchestrator)
        pipeline.config = MagicMock()
        pipeline.config.outputs_dir = tmp_path
        pipeline.config.platforms = None
        pipeline.config.force = False
        # Not `--skip-publish`: that short-circuits the guard, which would
        # disable the very thing under test.
        pipeline.config.skip_publish = False
        pipeline.config.process_all_products = True
        pipeline.config.topics = []
        pipeline.config.keywords = ["anything"]
        pipeline.config.strict = False
        pipeline.config.resume = False

        from src.pipeline.config import PipelineState

        pipeline.state = PipelineState.create_new(pipeline.config)
        pipeline._save_state = MagicMock()  # type: ignore[method-assign]

        record = MagicMock()
        record.asin = asin

        monkeypatch.setattr(
            "src.video.producer.cli.discover_products_for_batch",
            lambda *a, **k: [(product_dir, record)],
        )
        pipeline._execute_scraping_phase = AsyncMock(  # type: ignore[method-assign]
            return_value=ScrapingPhaseSummary(1, 1, 0, [asin], [], {}, 1.0)
        )
        pipeline._notify_webhook = AsyncMock()  # type: ignore[method-assign]
        pipeline._execute_publishing_phase = AsyncMock(  # type: ignore[method-assign]
            return_value=None
        )

        summary = await pipeline.run_pipeline()

        assert summary.production.already_published == 1
        assert summary.outcome() == "nothing new"
        assert summary.exit_code() == 0
        assert summary.exit_code(strict=True) == 0

    @pytest.mark.asyncio
    async def test_a_resume_does_not_republish_a_dropped_product(
        self, tmp_path, monkeypatch
    ):
        """The one path that reaches publishing without passing production.

        A resume rebuilds the publish list from the saved run, which still
        names the products the guard just dropped, so it logged "Skipping 1
        already-published product" and then handed it to publishing anyway.

        That was not producing a duplicate post, which an earlier version of
        this docstring claimed: the reconstruction composed
        `outputs/<id>/video.mp4`, a name nothing writes, so every product on
        a resumed publish failed at `upload_media` first. The path now
        resolves the render properly, which is what makes the filter matter.
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.pipeline.config import (
            PipelinePhase,
            PipelineState,
            ScrapingPhaseSummary,
        )
        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        published, fresh = "B0PUBLISHED", "B0FRESH0001"
        for asin in (published, fresh):
            (tmp_path / asin).mkdir()
            # A real render name. The resume path resolves the file rather
            # than composing `video.mp4`, which nothing writes.
            (tmp_path / asin / f"video_{asin}_slideshow_images1.mp4").write_bytes(b"")
        (tmp_path / "publish_history.json").write_text(
            json.dumps(
                {
                    "posts": {
                        f"{published}:{p}": {"post_id": "1"}
                        for p in ("youtube", "tiktok", "instagram")
                    }
                }
            ),
            encoding="utf-8",
        )

        pipeline = GlobalPipelineOrchestrator.__new__(GlobalPipelineOrchestrator)
        pipeline.config = MagicMock()
        pipeline.config.outputs_dir = tmp_path
        pipeline.config.platforms = None
        pipeline.config.force = False
        pipeline.config.skip_publish = False
        pipeline.config.process_all_products = True
        pipeline.config.topics = []
        pipeline.config.keywords = ["anything"]
        pipeline.config.strict = False
        pipeline.config.resume = True

        pipeline.state = PipelineState.create_new(pipeline.config)
        pipeline.state.mark_phase_complete(PipelinePhase.PRODUCTION)
        pipeline.state.production_completed_products = [published, fresh]
        pipeline.state.production_summary = {
            "total_attempted": 2,
            "successful": 2,
            "failed": 0,
            "skipped": 0,
            "failed_products": [],
            "skipped_products": [],
            "profile_distribution": None,
            "duration_sec": 1.0,
        }
        pipeline._save_state = MagicMock()  # type: ignore[method-assign]

        records = []
        for asin in (published, fresh):
            record = MagicMock()
            record.asin = asin
            records.append((tmp_path / asin, record))

        monkeypatch.setattr(
            "src.video.producer.cli.discover_products_for_batch",
            lambda *a, **k: records,
        )
        pipeline._execute_scraping_phase = AsyncMock(  # type: ignore[method-assign]
            return_value=ScrapingPhaseSummary(2, 2, 0, [published, fresh], [], {}, 1.0)
        )
        pipeline._notify_webhook = AsyncMock()  # type: ignore[method-assign]
        from src.pipeline.config import PublishingPhaseSummary

        publish = AsyncMock(
            return_value=PublishingPhaseSummary(1, 1, 0, 0, [], [], {}, [], 1.0)
        )
        pipeline._execute_publishing_phase = publish  # type: ignore[method-assign]

        summary = await pipeline.run_pipeline()

        handed_to_publish = publish.call_args.args[0]
        assert [pid for _, pid in handed_to_publish] == [fresh], (
            "a resume handed an already-published product to the publish "
            "phase, which has no guard of its own"
        )
        # The path, not just the id. Asserting ids alone let the resolver be
        # reverted to composing `outputs/<id>/video.mp4` -- a name nothing
        # writes, which fails every product at `upload_media` -- with the
        # whole suite still green.
        assert handed_to_publish == [
            (tmp_path / fresh / f"video_{fresh}_slideshow_images1.mp4", fresh)
        ]
        assert summary.production.already_published == 1

    @pytest.mark.asyncio
    async def test_a_mixed_run_reports_both_the_render_and_the_drop(
        self, tmp_path, monkeypatch
    ):
        """The other carry, which a mutation showed nothing was asserting.

        One product dropped and one rendered takes the *rendering* branch, so
        the count comes from `_execute_production_phase`'s summary rather
        than the early return. Deleting that pair of arguments left the whole
        suite green, which is the same gap the early-return carry had.
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.pipeline.config import ScrapingPhaseSummary
        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        published, fresh = "B0PUBLISHED", "B0FRESH0001"
        for asin in (published, fresh):
            (tmp_path / asin).mkdir()
        (tmp_path / "publish_history.json").write_text(
            json.dumps(
                {
                    "posts": {
                        f"{published}:{p}": {"post_id": "1"}
                        for p in ("youtube", "tiktok", "instagram")
                    }
                }
            ),
            encoding="utf-8",
        )

        pipeline = GlobalPipelineOrchestrator.__new__(GlobalPipelineOrchestrator)
        pipeline.config = MagicMock()
        pipeline.config.outputs_dir = tmp_path
        pipeline.config.platforms = None
        pipeline.config.force = False
        # Not `--skip-publish`: that short-circuits the guard.
        pipeline.config.skip_publish = False
        pipeline.config.process_all_products = True
        pipeline.config.topics = []
        pipeline.config.keywords = ["anything"]
        pipeline.config.strict = False
        pipeline.config.resume = False
        pipeline.config.random_profile = False
        pipeline.config.profile = "slideshow_images1"
        pipeline.config.profile_pool = ["slideshow_images1"]
        pipeline.config.topic_profile_pool = []
        pipeline.config.fail_fast = False
        pipeline.config.pillar = None

        records = []
        for asin in (published, fresh):
            record = MagicMock()
            record.asin = asin
            records.append((tmp_path / asin, record))

        monkeypatch.setattr(
            "src.video.producer.cli.discover_products_for_batch",
            lambda *a, **k: records,
        )
        from src.pipeline.config import PipelineState

        pipeline.state = PipelineState.create_new(pipeline.config)
        pipeline._save_state = MagicMock()  # type: ignore[method-assign]
        pipeline._execute_scraping_phase = AsyncMock(  # type: ignore[method-assign]
            return_value=ScrapingPhaseSummary(2, 2, 0, [published, fresh], [], {}, 1.0)
        )
        pipeline._notify_webhook = AsyncMock()  # type: ignore[method-assign]
        # Only the render is stubbed. `_execute_production_phase` itself runs,
        # so the carry under test is the production code's, not the test's --
        # stubbing the whole phase would assert a copy of the line it is
        # supposed to be proving.
        monkeypatch.setattr(
            "src.video.producer.orchestration.create_video_for_product",
            AsyncMock(return_value=tmp_path / fresh / "video.mp4"),
        )

        from src.pipeline.config import PublishingPhaseSummary

        pipeline._execute_publishing_phase = AsyncMock(  # type: ignore[method-assign]
            return_value=PublishingPhaseSummary(1, 1, 0, 0, [], [], {}, [], 1.0)
        )

        summary = await pipeline.run_pipeline()

        assert summary.production.already_published == 1
        assert summary.production.already_published_products == [published]
        assert summary.production.total_attempted == 1


@pytest.mark.unit
class TestTheDryRunPlanMatchesTheRun:
    """The plan must not promise a filter the run will not apply."""

    @staticmethod
    def _plan(**flags):
        import contextlib
        import io
        from contextlib import redirect_stdout
        from unittest.mock import MagicMock

        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        pipeline = GlobalPipelineOrchestrator.__new__(GlobalPipelineOrchestrator)
        pipeline.config = MagicMock()
        pipeline.config.skip_publish = flags.get("skip_publish", False)
        pipeline.config.force = flags.get("force", False)
        pipeline.config.product_ids = ["B0AAAAAAAA"]
        pipeline.config.keywords = []
        pipeline.config.topics = []
        pipeline.config.platforms = None
        pipeline.config.outputs_dir = Path("outputs")

        buffer = io.StringIO()
        # The plan reads config this fixture does not fill, and the handoff
        # lines print before anything that can raise. Suppressing is what
        # keeps the fixture small -- but the call has to be the real method,
        # or the suppression hides a typo and every assertion here passes on
        # an empty buffer.
        with contextlib.suppress(Exception), redirect_stdout(buffer):
            pipeline.display_execution_plan(MagicMock())
        output = buffer.getvalue()
        assert "PHASE 2: HANDOFF" in output, "the plan never reached the handoff block"
        return output

    def test_a_normal_run_announces_the_filter(self):
        assert "already published" in self._plan()

    def test_a_skip_publish_run_does_not(self):
        """The guard returns early on `--skip-publish`, so promising it lies."""
        assert "already published" not in self._plan(skip_publish=True)

    def test_a_force_run_does_not(self):
        assert "already published" not in self._plan(force=True)


@pytest.mark.unit
class TestTheResumePicksTheSameCutAsThePublisher:
    """A product rendered twice has two files; the resume must not guess.

    `sole_render_for_product` falls back to the alphabetically first render
    when it is given no profiles and no platform, which is what the first
    version of the resume fix did. Every other caller routes it, so a resumed
    publish would have sent a different cut from the one the publish phase
    would have chosen for the same product.
    """

    def test_the_routed_profile_wins_over_the_alphabetical_first(self, tmp_path):
        from src.publisher.video_selector import sole_render_for_product

        asin = "B0TWOCUTS0"
        product_dir = tmp_path / asin
        product_dir.mkdir()
        # "a_profile" sorts first; the publisher routes youtube to "z_profile".
        for profile in ("a_profile", "z_profile"):
            (product_dir / f"video_{asin}_{profile}.mp4").write_bytes(b"")

        unrouted = sole_render_for_product(product_dir)
        routed = sole_render_for_product(
            product_dir, {"youtube": "z_profile"}, "youtube"
        )

        assert unrouted is not None and "a_profile" in unrouted.name
        assert routed is not None and "z_profile" in routed.name
        assert unrouted != routed, (
            "this fixture no longer distinguishes routed from unrouted, so "
            "the assertion below proves nothing"
        )

    def test_the_resume_passes_the_publisher_routing(self, tmp_path, monkeypatch):
        """Reads the call, since the two agree on a single-render directory.

        A directory with one render resolves the same either way, so a test
        driving `run_pipeline` on the usual fixture cannot tell whether the
        routing was passed.
        """
        import inspect

        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        source = inspect.getsource(GlobalPipelineOrchestrator.run_pipeline)
        start = source.index("sole_render_for_product(")
        # Balance the parens rather than cutting at the first `)`, which sits
        # inside `self._publisher_profiles()` and truncated the call.
        depth, end = 0, start
        for offset, char in enumerate(source[start:], start):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = offset + 1
                    break
        call = source[start:end]

        assert "_publisher_profiles()" in call, (
            "the resume resolves a render without the publisher's profile "
            "routing, so it can send a different cut than the publish phase"
        )
        assert "platforms_for_resume" in call
