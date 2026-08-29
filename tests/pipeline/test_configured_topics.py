"""A run with no input flags must be able to produce both content formats.

`config/pipeline.yaml::global_batch.keywords` gives a zero-argument batch its
inputs, and every configured keyword is a physical product. There was no
equivalent for topics, so a topic could only enter a run by being typed on that
day's command line -- which meant the repeatable path, the one a scheduled run
uses, produced product renders and nothing else, and the tutorial arm could not
become part of the cadence.
"""

from __future__ import annotations

import argparse
import contextlib

import pytest
import yaml

from src.pipeline.config import (
    GlobalBatchConfig,
    load_global_batch_config,
    topics_for_run,
    validate_global_batch_config,
)
from src.video.producer.topic_input import TopicInputError, TopicSpec


@pytest.fixture
def real_video_config():
    """The bundled profiles: which ones can render a topic is the thing tested."""
    from src.video.config_adapter import load_video_config_modular

    return load_video_config_modular()


def spec(title: str) -> TopicSpec:
    return TopicSpec(title=title)


class TestTheRotation:
    """Which of the configured topics a given day's run takes."""

    TOPICS = [spec("A"), spec("B"), spec("C")]

    def test_consecutive_days_take_different_topics(self):
        """Taking the top of the list would re-render one topic forever.

        The list exists so a daily cadence works through it. A fixed offset
        would make every other configured topic dead config.
        """
        picked = [
            topics_for_run(self.TOPICS, 1, day_ordinal=d)[0].title
            for d in range(10, 14)
        ]

        assert picked == ["B", "C", "A", "B"]

    def test_a_count_above_one_takes_consecutive_topics(self):
        assert [t.title for t in topics_for_run(self.TOPICS, 2, day_ordinal=10)] == [
            "B",
            "C",
        ]

    def test_a_count_above_the_list_length_is_capped(self):
        """Wrapping returned the same spec twice.

        Two entries with one title render into one directory, so the run wrote
        one video and counted two -- and the batch summary then reported a
        product scraped but never produced.
        """
        picked = topics_for_run(self.TOPICS, 5, day_ordinal=0)

        assert len(picked) == len(self.TOPICS)
        assert len({t.title for t in picked}) == len(picked)

    def test_zero_means_products_only(self):
        assert topics_for_run(self.TOPICS, 0, day_ordinal=10) == []

    def test_a_negative_count_is_not_an_error(self):
        assert topics_for_run(self.TOPICS, -1, day_ordinal=10) == []

    def test_no_configured_topics_is_not_an_error(self):
        assert topics_for_run([], 1, day_ordinal=10) == []

    def test_the_same_day_is_reproducible(self):
        """A resumed or re-run batch must not draw a different topic."""
        first = topics_for_run(self.TOPICS, 1, day_ordinal=42)
        second = topics_for_run(self.TOPICS, 1, day_ordinal=42)

        assert [t.title for t in first] == [t.title for t in second]


def write_config(tmp_path, global_batch: dict):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump({"global_batch": global_batch}), encoding="utf-8")
    return str(path)


class TestTheNoFlagRun:
    """Driven through the loader on a real config file, not a built config.

    A hand-built `GlobalBatchConfig` proves the fields exist. What was missing
    was a path from the config file to them.
    """

    CONFIG = {
        "keywords": ["wireless earbuds"],
        "topics": [
            {
                "title": "Why your wifi keeps dropping",
                "description": "Router placement and channel congestion.",
                "keywords": ["wifi router"],
            }
        ],
        "topics_per_run": 1,
    }

    def test_it_carries_both_arms(self, tmp_path):
        config = load_global_batch_config(
            argparse.Namespace(), write_config(tmp_path, self.CONFIG)
        )

        assert config.keywords == ["wireless earbuds"]
        assert [t.title for t in config.topics] == ["Why your wifi keeps dropping"]

    def test_a_topic_keeps_its_own_search_terms(self, tmp_path):
        """A stock profile inheriting the global product keywords matches
        neither, so the per-topic terms are the whole point of the section.
        """
        config = load_global_batch_config(
            argparse.Namespace(), write_config(tmp_path, self.CONFIG)
        )

        assert config.topics[0].keywords == ["wifi router"]

    def test_zero_per_run_yields_products_only(self, tmp_path):
        config = load_global_batch_config(
            argparse.Namespace(),
            write_config(tmp_path, {**self.CONFIG, "topics_per_run": 0}),
        )

        assert config.topics == []
        assert config.keywords == ["wireless earbuds"]

    def test_no_topics_section_is_not_an_error(self, tmp_path):
        """Every config predating this has none."""
        config = load_global_batch_config(
            argparse.Namespace(), write_config(tmp_path, {"keywords": ["earbuds"]})
        )

        assert config.topics == []

    def test_a_malformed_topic_is_refused_at_load(self, tmp_path):
        """Same rule the topics file follows: a hand-written list that is
        wrong should say so, not quietly render one video fewer.
        """
        with pytest.raises(TopicInputError):
            load_global_batch_config(
                argparse.Namespace(),
                write_config(
                    tmp_path,
                    {"topics": [{"title": "Fine"}, {"titel": "typo"}]},
                ),
            )

    def test_a_non_integer_count_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="topics_per_run"):
            load_global_batch_config(
                argparse.Namespace(),
                write_config(tmp_path, {**self.CONFIG, "topics_per_run": "one"}),
            )


class TestCliInputsStillWin:
    """The configured mix is the default, not an addition to what was asked."""

    def test_named_keywords_do_not_drag_in_the_configured_topics(self, tmp_path):
        """`--keywords` means "this is the input set", as it always has.

        Appending a configured topic to it would make every targeted run
        render something nobody asked for.
        """
        args = argparse.Namespace(keywords=["headphones"])
        config = load_global_batch_config(
            args, write_config(tmp_path, TestTheNoFlagRun.CONFIG)
        )

        assert config.keywords == ["headphones"]
        assert config.topics == []

    def test_a_named_topic_does_not_drag_in_the_configured_keywords(self, tmp_path):
        args = argparse.Namespace(
            topic="Something else", topic_description="", topic_keywords=None
        )
        config = load_global_batch_config(
            args, write_config(tmp_path, TestTheNoFlagRun.CONFIG)
        )

        assert [t.title for t in config.topics] == ["Something else"]
        assert config.keywords == []

    def test_a_named_topic_ignores_the_configured_count(self, tmp_path):
        """`topics_per_run` paces the cadence; it must not silently drop a
        topic the operator named.
        """
        args = argparse.Namespace(
            topic="Named", topic_description="", topic_keywords=None
        )
        config = load_global_batch_config(
            args,
            write_config(tmp_path, {**TestTheNoFlagRun.CONFIG, "topics_per_run": 0}),
        )

        assert [t.title for t in config.topics] == ["Named"]


class TestTheBundledConfigCarriesBoth:
    """The shipped file is what a fresh install's no-flag run reads."""

    def test_a_no_flag_run_produces_both_formats(self):
        config = load_global_batch_config(argparse.Namespace())

        assert config.keywords, "no configured keywords"
        assert config.topics, (
            "a run with no input flags still produces product renders only, "
            "so the tutorial arm cannot be part of the cadence"
        )

    def test_the_configured_topics_are_renderable(self, real_video_config):
        """A topic whose profile pool is empty fails at gather_visuals."""
        config = load_global_batch_config(argparse.Namespace())
        config.skip_publish = True

        validate_global_batch_config(config, real_video_config)

        assert config.topic_profile_pool
        assert config.profile_pool


class TestBothPhasesRun:
    @pytest.mark.asyncio
    async def test_a_mixed_run_scrapes_and_materialises(self, tmp_path, monkeypatch):
        """The branch's central change, driven rather than read.

        An AST check that both method names appear passes on the old `if/else`
        too, since both appear there as well -- so restoring the exclusive form
        left the suite green while a mixed run silently dropped every
        configured keyword.
        """
        from src.pipeline.config import PipelinePhase, ScrapingPhaseSummary
        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        config = GlobalBatchConfig(
            topics=[TopicSpec(title="Why wifi drops")],
            keywords=["wireless earbuds"],
            outputs_dir=tmp_path,
            skip_publish=True,
        )
        orchestrator = GlobalPipelineOrchestrator(config)
        called: list[str] = []

        def summary(pid: str) -> ScrapingPhaseSummary:
            return ScrapingPhaseSummary(
                total_attempted=1,
                successful=1,
                failed=0,
                successful_products=[pid],
                failed_products=[],
                media_stats={},
                duration_sec=0.0,
            )

        def fake_topics():
            called.append("topics")
            return summary("topic-why-wifi-drops-abc")

        async def fake_scrape():
            called.append("scrape")
            return summary("B0ABCDEFGH")

        monkeypatch.setattr(orchestrator, "_materialise_topics_phase", fake_topics)
        monkeypatch.setattr(orchestrator, "_execute_scraping_phase", fake_scrape)
        # Stop after the phase under test; the rest needs a browser and a
        # renderer, and neither says anything about which phases ran.
        monkeypatch.setattr(
            orchestrator,
            "_execute_handoff_phase",
            lambda ids: (_ for _ in ()).throw(RuntimeError("stop here")),
        )

        with contextlib.suppress(RuntimeError):
            await orchestrator.run_pipeline()

        assert called == ["topics", "scrape"], (
            "a mixed run must do both; a choice between them silently drops "
            f"one input kind (ran: {called})"
        )
        assert orchestrator.state.is_phase_completed(PipelinePhase.SCRAPING)
        assert sorted(orchestrator.state.scraping_completed_products) == [
            "B0ABCDEFGH",
            "topic-why-wifi-drops-abc",
        ]

    def test_the_summaries_are_folded_into_one(self):
        from src.pipeline.config import ScrapingPhaseSummary
        from src.pipeline.global_batch import _merge_scraping_summaries

        topics = ScrapingPhaseSummary(
            total_attempted=1,
            successful=1,
            failed=0,
            successful_products=["topic-why-wifi-drops-abc"],
            failed_products=[],
            media_stats={"total_images": 0, "total_videos": 0},
            duration_sec=0.1,
        )
        scraped = ScrapingPhaseSummary(
            total_attempted=2,
            successful=1,
            failed=1,
            successful_products=["B0ABCDEFGH"],
            failed_products=["B0FAILFAIL"],
            media_stats={"total_images": 5, "total_videos": 1},
            duration_sec=12.0,
        )

        merged = _merge_scraping_summaries(topics, scraped)

        assert merged.total_attempted == 3
        assert merged.successful == 2
        assert merged.failed == 1
        assert merged.successful_products == [
            "topic-why-wifi-drops-abc",
            "B0ABCDEFGH",
        ]
        assert merged.failed_products == ["B0FAILFAIL"]
        assert merged.media_stats == {"total_images": 5, "total_videos": 1}
        # Summed, not maxed: the phases run one after the other.
        assert merged.duration_sec == pytest.approx(12.1)

    def test_one_phase_is_returned_unchanged(self):
        """A products-only or topics-only run must be byte-identical to before."""
        from src.pipeline.config import ScrapingPhaseSummary
        from src.pipeline.global_batch import _merge_scraping_summaries

        only = ScrapingPhaseSummary(
            total_attempted=1,
            successful=1,
            failed=0,
            successful_products=["B0ABCDEFGH"],
            failed_products=[],
            media_stats={"total_images": 3, "total_videos": 0},
            duration_sec=4.0,
        )

        assert _merge_scraping_summaries(only, None) is only
        assert _merge_scraping_summaries(None, only) is only


class TestEachRecordDrawsItsOwnProfile:
    def test_a_topic_and_a_product_get_different_pools(self, real_video_config):
        config = GlobalBatchConfig(
            topics=[TopicSpec(title="Why wifi drops")],
            keywords=["wireless earbuds"],
            skip_publish=True,
        )

        validate_global_batch_config(config, real_video_config)

        assert set(config.topic_profile_pool).isdisjoint(
            {"slideshow_images1"}
        ), "a topic drawing a product profile gathers no visuals and the run fails"
        assert "slideshow_images1" in config.profile_pool

    @pytest.mark.asyncio
    async def test_the_production_loop_gives_each_record_its_own_pool(
        self, tmp_path, real_video_config, monkeypatch
    ):
        """Two pools on the config are inert unless the loop picks by record.

        Driven through the real loop with the render stubbed, because the
        shape of the call site is not the behaviour: a wrongly computed `pool`
        variable passes any check that only reads the argument name.
        """
        from src.pipeline.global_batch import GlobalPipelineOrchestrator
        from src.scraper.amazon.models import ProductData
        from src.scraper.base.models import Platform
        from src.video.producer import orchestration
        from src.video.producer.topic_input import build_topic_product

        config = GlobalBatchConfig(
            topics=[TopicSpec(title="Why wifi drops")],
            keywords=["wireless earbuds"],
            outputs_dir=tmp_path,
            skip_publish=True,
        )
        validate_global_batch_config(config, real_video_config)

        used: dict[str, str] = {}

        async def fake_render(*args, **kwargs):
            product = kwargs.get("product") or args[0]
            used[product.asin] = kwargs["profile_name"]
            raise RuntimeError("render stubbed out")

        monkeypatch.setattr(orchestration, "create_video_for_product", fake_render)

        topic = build_topic_product(TopicSpec(title="Why wifi drops"))
        product = ProductData(
            title="Wireless earbuds",
            price="19.99",
            url="https://www.amazon.com/dp/B0ABCDEFGH",
            platform=Platform.AMAZON,
            asin="B0ABCDEFGH",
        )

        await GlobalPipelineOrchestrator(config)._execute_production_phase(
            [(tmp_path / str(topic.asin), topic), (tmp_path / "B0ABCDEFGH", product)]
        )

        assert used[str(topic.asin)] in config.topic_profile_pool, (
            "the topic drew a product profile; it has no product photography, "
            "so gather_visuals finds nothing and the render fails outright"
        )
        assert used["B0ABCDEFGH"] in config.profile_pool
        assert used["B0ABCDEFGH"] not in config.topic_profile_pool


class TestAResumedMixedRunKeepsBothPools:
    """The AST guard checks that `resume_has_products` is assigned, not what.

    Two independent mutations -- `_run_has_product_records` returning False on
    a resume, and `main` assigning False -- each produced the exact defect its
    docstring names, with the whole suite green.
    """

    def _state(self, tmp_path, product_ids):
        from src.pipeline.config import (
            PipelineState,
            save_pipeline_state,
        )

        state = PipelineState.create_new(GlobalBatchConfig(outputs_dir=tmp_path))
        state.scraping_completed_products = product_ids
        save_pipeline_state(state, tmp_path)

    def test_the_product_pool_survives(self, tmp_path, real_video_config):
        from src.video.producer.topic_input import topic_product_id

        self._state(tmp_path, [topic_product_id("Why wifi drops"), "B0ABCDEFGH"])

        config = GlobalBatchConfig(
            outputs_dir=tmp_path,
            resume=True,
            keywords=["wireless earbuds"],
            topics_resume=True,
            resume_has_products=True,
            skip_publish=True,
        )
        validate_global_batch_config(config, real_video_config)

        assert "slideshow_images1" in config.profile_pool, (
            "the resumed scraped products would render from generic stock "
            "footage, ignoring the photography scraped for them"
        )
        assert config.topic_profile_pool == ["slideshow_stock"]

    def test_a_topics_only_resume_still_narrows(self, tmp_path, real_video_config):
        """The opposite direction, so the fix cannot be a blanket widening."""
        from src.video.producer.topic_input import topic_product_id

        self._state(tmp_path, [topic_product_id("Why wifi drops")])

        config = GlobalBatchConfig(
            outputs_dir=tmp_path,
            resume=True,
            keywords=["wireless earbuds"],
            topics_resume=True,
            resume_has_products=False,
            skip_publish=True,
        )
        validate_global_batch_config(config, real_video_config)

        assert config.profile_pool == ["slideshow_stock"]

    def test_both_flags_are_stamped_from_the_state(self, tmp_path):
        """`resume_has_products` has to come from the state, not the config.

        A resume inherits the configured keywords whatever it is resuming, and
        the completed scraping phase already ignored them. Asserted on the
        function `main` calls, because an AST check that an assignment exists
        passes just as well when the assignment is a constant.
        """
        from src.pipeline.global_batch import apply_resume_record_kinds
        from src.video.producer.topic_input import topic_product_id

        self._state(tmp_path, [topic_product_id("Why wifi drops"), "B0ABCDEFGH"])
        config = GlobalBatchConfig(
            outputs_dir=tmp_path, resume=True, keywords=["wireless earbuds"]
        )

        apply_resume_record_kinds(config)

        assert config.topics_resume is True
        assert config.resume_has_products is True

    def test_a_topics_only_state_leaves_the_products_flag_off(self, tmp_path):
        from src.pipeline.global_batch import apply_resume_record_kinds
        from src.video.producer.topic_input import topic_product_id

        self._state(tmp_path, [topic_product_id("Why wifi drops")])
        config = GlobalBatchConfig(
            outputs_dir=tmp_path, resume=True, keywords=["wireless earbuds"]
        )

        apply_resume_record_kinds(config)

        assert config.topics_resume is True
        assert config.resume_has_products is False


class TestCleanOnAMixedRun:
    """`--clean` narrows the sweep to the ids a run names.

    Returning the first non-empty input kind meant a run carrying a topic and
    keywords -- which the bundled config now produces with no flags at all --
    named only the topic, and every product directory the operator asked to
    remove survived.
    """

    def test_keywords_name_nothing_so_the_sweep_is_unnarrowed(self):
        from src.pipeline.global_batch import _named_run_ids

        assert (
            _named_run_ids(
                GlobalBatchConfig(
                    keywords=["wireless earbuds"],
                    topics=[TopicSpec(title="Why wifi drops")],
                )
            )
            == []
        )

    def test_ids_and_topics_are_unioned(self):
        from src.pipeline.global_batch import _named_run_ids
        from src.video.producer.topic_input import topic_product_id

        named = _named_run_ids(
            GlobalBatchConfig(
                product_ids=["B0ABCDEFGH"],
                topics=[TopicSpec(title="Why wifi drops")],
            )
        )

        assert named == ["B0ABCDEFGH", topic_product_id("Why wifi drops")]

    def test_a_topics_only_run_still_names_its_topics(self):
        from src.pipeline.global_batch import _named_run_ids
        from src.video.producer.topic_input import topic_product_id

        assert _named_run_ids(
            GlobalBatchConfig(topics=[TopicSpec(title="Why wifi drops")])
        ) == [topic_product_id("Why wifi drops")]


class TestTheDryRunPlanShowsBothHalves:
    """The plan exists to rule out a run doing something other than it says.

    Suppressing the scraping half whenever a topic is present hid work a mixed
    run will do -- the same defect as printing work it would discard, in the
    other direction.
    """

    def _plan(self, config, real_video_config, capsys):
        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        validate_global_batch_config(config, real_video_config)
        GlobalPipelineOrchestrator(config).display_execution_plan(real_video_config)
        return capsys.readouterr().out

    def test_a_mixed_run_lists_its_keywords(self, tmp_path, real_video_config, capsys):
        out = self._plan(
            GlobalBatchConfig(
                topics=[TopicSpec(title="Why wifi drops")],
                keywords=["wireless earbuds"],
                outputs_dir=tmp_path,
                skip_publish=True,
            ),
            real_video_config,
            capsys,
        )

        assert "wireless earbuds" in out, (
            "the plan promises a run that scrapes nothing while the run "
            "scrapes every configured keyword"
        )
        assert "Why wifi drops" in out

    def test_a_mixed_run_names_the_topic_pool(
        self, tmp_path, real_video_config, capsys
    ):
        """Two pools, so printing one leaves the plan silent about the other."""
        out = self._plan(
            GlobalBatchConfig(
                topics=[TopicSpec(title="Why wifi drops")],
                keywords=["wireless earbuds"],
                outputs_dir=tmp_path,
                skip_publish=True,
            ),
            real_video_config,
            capsys,
        )

        assert "Topic profile pool" in out
        assert "slideshow_stock" in out

    def test_a_topics_only_run_still_hides_the_scraping_half(
        self, tmp_path, real_video_config, capsys
    ):
        """The original defect, in its original direction."""
        out = self._plan(
            GlobalBatchConfig(
                topics=[TopicSpec(title="Why wifi drops")],
                outputs_dir=tmp_path,
                skip_publish=True,
            ),
            real_video_config,
            capsys,
        )

        assert "nothing to scrape" in out
        assert "Keywords to search" not in out
        assert "Filters:" not in out
