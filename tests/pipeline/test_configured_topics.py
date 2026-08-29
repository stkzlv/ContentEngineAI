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

    def test_the_selection_wraps(self):
        """A count larger than the list repeats rather than returning fewer.

        Silently returning three topics for `topics_per_run: 5` would be a run
        quietly doing less than it was configured to do.
        """
        assert len(topics_for_run(self.TOPICS, 5, day_ordinal=0)) == 5

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
    def test_the_orchestrator_does_not_choose_between_them(self):
        """The topic phase used to replace the scraping phase outright.

        Reading it as a choice is what made the two inputs exclusive, and a
        mixed run would silently render the topics and drop every keyword.
        """
        import ast
        from pathlib import Path

        source = Path("src/pipeline/global_batch.py").read_text()
        tree = ast.parse(source)

        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {"_materialise_topics_phase", "_execute_scraping_phase"}
        }

        assert calls == {"_materialise_topics_phase", "_execute_scraping_phase"}
        assert "_merge_scraping_summaries" in source, (
            "the two phases run but their summaries are not combined, so the "
            "saved state records only one of them"
        )

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
