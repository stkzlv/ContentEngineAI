"""Topics are a batch input, not just a producer one (#226).

The Module/Batch Alignment Rule says a flag on one entry point belongs on both.
The producer could render a topic and the batch could not, so the pivot to
topic content had no path to the daily cadence: `make batch-lowpri` could only
produce the product arm.

Each test below covers one of the places the batch assumed an input was a
scraped product. They are separate tests rather than one end-to-end case
because they fail for different reasons and a single assertion would report
whichever broke first.
"""

from __future__ import annotations

import argparse

import pytest

from src.pipeline.config import (
    GlobalBatchConfig,
    load_global_batch_config,
    validate_global_batch_config,
)
from src.pipeline.global_batch import _RUN_DIR_PATTERN, _clean_targets
from src.video.producer.topic_input import TopicSpec, topic_product_id


@pytest.fixture
def real_video_config():
    """The bundled profiles: which ones can render a topic is the thing tested."""
    from src.video.config_adapter import load_video_config_modular

    return load_video_config_modular()


def _args(**overrides) -> argparse.Namespace:
    """The batch CLI namespace, with everything the loader reads present."""
    base = {
        "product_ids": None,
        "keywords": None,
        "topic": None,
        "topic_description": None,
        "topic_keywords": None,
        "topics_file": None,
        "max_products": None,
        "products_per_keyword": None,
        "profile": "slideshow_stock",
        "random_profile": False,
        "profile_pool": None,
        "outputs_dir": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestTopicsAreAnInputSource:
    def test_a_topic_run_does_not_inherit_the_yaml_keywords(self):
        """`cli_has_inputs` counted product ids and keywords only.

        So `--topic` fell through to the YAML branch and the run scraped every
        configured keyword alongside the topic, pairing them for no reason.
        """
        config = load_global_batch_config(_args(topic="Why wifi drops"))

        assert [spec.title for spec in config.topics] == ["Why wifi drops"]
        assert config.keywords == []
        assert config.product_ids == []

    def test_a_run_with_no_flags_still_reads_the_yaml_keywords(self):
        """The guard above must not disable the configured default run."""
        config = load_global_batch_config(_args())

        assert config.topics == []
        assert config.keywords, "the no-flag run lost its YAML keyword list"

    def test_topic_keywords_survive_as_phrases(self):
        """Split on commas, not spaces: 'wifi router' is one search term.

        The stock provider joins its keyword list into a single query, so a
        phrase split into words changes what the render searches for.
        """
        config = load_global_batch_config(
            _args(topic="Why wifi drops", topic_keywords="wifi router, home network")
        )

        assert config.topics[0].keywords == ["wifi router", "home network"]


class TestValidation:
    def test_topics_alone_are_enough_to_run(self, real_video_config):
        validate_global_batch_config(
            GlobalBatchConfig(
                topics=[TopicSpec(title="Why wifi drops")],
                profile="slideshow_stock",
                skip_publish=True,
            ),
            real_video_config,
        )

    def test_a_run_with_no_inputs_at_all_is_still_refused(self, real_video_config):
        with pytest.raises(ValueError, match="No inputs provided") as excinfo:
            validate_global_batch_config(
                GlobalBatchConfig(profile="slideshow_stock", skip_publish=True),
                real_video_config,
            )

        # The message lists what to pass; omitting topics sends the reader to
        # the scraper flags for a run that needs neither.
        assert "--topic" in str(excinfo.value)


class TestCleanCoversTopicDirectories:
    """`--clean` matched an ASIN shape, so topic runs accumulated forever."""

    @pytest.mark.parametrize(
        "name,matches",
        [
            ("B0FC5S16YM", True),
            ("TESTABC123", True),
            ("topic-why-your-wifi-keeps-dropping-1a2b3c4d", True),
            ("logs", False),
            ("coverage", False),
            ("published_products.json", False),
        ],
    )
    def test_which_directories_are_run_directories(self, name, matches):
        assert bool(_RUN_DIR_PATTERN.match(name)) is matches

    def test_a_real_topic_id_matches(self):
        """Pinned to the generator, not to a hand-written example.

        A change to the slug or digest format that stopped matching would
        otherwise leave `--clean` silently walking past topic directories
        again, which is the defect this replaced.
        """
        pid = topic_product_id("Why your wifi keeps dropping")

        assert _RUN_DIR_PATTERN.match(pid), pid

    def test_clean_selects_a_topic_directory(self, tmp_path):
        pid = topic_product_id("Why your wifi keeps dropping")
        (tmp_path / pid).mkdir()
        (tmp_path / "B0FC5S16YM").mkdir()
        (tmp_path / "logs").mkdir()

        targets = {p.name for p in _clean_targets(tmp_path, None)}

        assert targets == {pid, "B0FC5S16YM"}


class TestDiscoveryReturnsTopicsOnlyWhenAsked:
    """The handoff drops what discovery skips.

    `discover_products_for_batch` skips topic directories on purpose: a plain
    `--batch` run means "every product here", and a topic handed to a product
    profile finds no imagery and fails the run rather than skipping it. A run
    whose inputs *are* topics named them, so it opts in.
    """

    @staticmethod
    def _topic_dir(root, title):
        import json

        from src.video.producer.topic_input import (
            TopicSpec,
            build_topic_product,
        )

        product = build_topic_product(TopicSpec(title=title))
        directory = root / str(product.asin)
        directory.mkdir(parents=True)
        (directory / "data.json").write_text(
            json.dumps(product.to_dict()), encoding="utf-8"
        )
        return directory

    def test_a_product_batch_still_skips_topics(self, tmp_path):
        from src.video.producer.cli import discover_products_for_batch

        self._topic_dir(tmp_path, "Why wifi drops")

        assert discover_products_for_batch(tmp_path) == []

    def test_a_topic_run_gets_them_back(self, tmp_path):
        from src.video.producer.cli import discover_products_for_batch

        directory = self._topic_dir(tmp_path, "Why wifi drops")

        found = discover_products_for_batch(tmp_path, include_topics=True)

        assert [path for path, _ in found] == [directory]
        assert [data.topic for _, data in found] == ["Why wifi drops"]


class TestTheTopicPhaseReplacesScraping:
    """A topic has no listing, so the scraping phase must not run at all."""

    def test_it_writes_a_record_and_reports_the_ids_the_handoff_filters_on(
        self, tmp_path
    ):
        import json

        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        config = GlobalBatchConfig(
            topics=[TopicSpec(title="Why your wifi keeps dropping")],
            profile="slideshow_stock",
            outputs_dir=tmp_path,
            skip_publish=True,
        )
        summary = GlobalPipelineOrchestrator(config)._materialise_topics_phase()

        pid = topic_product_id("Why your wifi keeps dropping")
        # The handoff filters discovered directories against these ids, so a
        # summary that omitted them would drop every topic it just wrote.
        assert summary.successful_products == [pid]
        assert summary.successful == 1 and summary.failed == 0

        record = json.loads((tmp_path / pid / "data.json").read_text())
        assert record["topic"] == "Why your wifi keeps dropping"
        assert record["asin"] == pid


class TestAProfileThatCannotRenderATopicIsRefusedUpFront:
    """A product profile does not degrade gracefully on a topic.

    It gathers nothing and `step_gather_visuals` raises, so the run is
    reported FAILED rather than skipped -- and only after the script and the
    voiceover have been paid for. Every case here is refused during
    validation instead.
    """

    def test_the_shortest_invocation_works(self, real_video_config):
        """`--topic X` with no profile at all.

        The default pool is built from the product profiles, every one of
        which fails on a topic, so leaving it to the normal random-profile
        path made the shortest possible command fail deterministically.
        """
        config = GlobalBatchConfig(
            topics=[TopicSpec(title="Why wifi drops")], skip_publish=True
        )
        validate_global_batch_config(config, real_video_config)

        assert config.profile_pool, "a topics run was left with no usable pool"
        assert "slideshow_stock" in config.profile_pool
        assert not any(p.startswith("slideshow_images") for p in config.profile_pool)

    def test_a_named_product_profile_is_refused(self, real_video_config):
        with pytest.raises(ValueError, match="draws no stock media"):
            validate_global_batch_config(
                GlobalBatchConfig(
                    topics=[TopicSpec(title="Why wifi drops")],
                    profile="slideshow_images1",
                    skip_publish=True,
                ),
                real_video_config,
            )

    def test_a_named_stock_profile_is_accepted(self, real_video_config):
        validate_global_batch_config(
            GlobalBatchConfig(
                topics=[TopicSpec(title="Why wifi drops")],
                profile="slideshow_stock",
                skip_publish=True,
            ),
            real_video_config,
        )

    def test_a_pool_carrying_a_product_profile_is_refused(self, real_video_config):
        with pytest.raises(ValueError, match="cannot render a topic"):
            validate_global_batch_config(
                GlobalBatchConfig(
                    topics=[TopicSpec(title="Why wifi drops")],
                    random_profile=True,
                    profile_pool=["slideshow_stock", "slideshow_images1"],
                    profile_pool_from_cli=True,
                    skip_publish=True,
                ),
                real_video_config,
            )


class TestTopicsAreExclusiveWithScraperInputs:
    """A topic run replaces the scraping phase, so anything to scrape is lost.

    Both were accepted and the scraper inputs silently discarded, while the
    dry-run plan printed the keywords as work the run would do.
    """

    @pytest.mark.parametrize(
        "extra,expected",
        [
            ({"keywords": ["wireless earbuds"]}, "--keywords"),
            ({"product_ids": ["B0ABCDEFGH"]}, "--product-ids"),
        ],
    )
    def test_the_combination_is_refused(self, real_video_config, extra, expected):
        with pytest.raises(ValueError, match="Cannot combine topics") as excinfo:
            validate_global_batch_config(
                GlobalBatchConfig(
                    topics=[TopicSpec(title="Why wifi drops")],
                    profile="slideshow_stock",
                    skip_publish=True,
                    **extra,
                ),
                real_video_config,
            )

        assert expected in str(excinfo.value)


class TestTheTwoTopicSourcesAreExclusive:
    def test_naming_both_a_flag_and_a_file_is_refused(self, tmp_path):
        """The producer errors on this pair; the batch used to drop `--topic`.

        Silently taking the file meant a run rendered something other than
        what was asked for.
        """
        from src.video.producer.topic_input import TopicInputError, specs_from_args

        topics_file = tmp_path / "topics.yaml"
        topics_file.write_text("- title: From file\n", encoding="utf-8")

        with pytest.raises(TopicInputError, match="cannot be used together"):
            specs_from_args(
                topic="From flag",
                topic_description=None,
                topic_keywords=None,
                topics_file=topics_file,
            )


class TestCleanRemovesOnlyWhatTheRunNamed:
    """A topics run named its inputs, so it must not wipe the machine.

    `_clean_targets` takes the "remove everything" branch when no product ids
    are given, and topics were not counted as named inputs -- so
    `--topic X --clean` would delete every scraped product directory in
    `outputs/`, along with any rendered-but-unpublished video in them.
    """

    @staticmethod
    def _tree(root):
        for name in ("B0ABCDEFGH", "B0FC5S16YM", "logs"):
            (root / name).mkdir()
        (root / topic_product_id("Why wifi drops")).mkdir()
        (root / topic_product_id("Some other topic")).mkdir()

    def test_a_topics_run_removes_only_its_own_topics(self, tmp_path):
        from src.pipeline.global_batch import _clean_targets, _named_run_ids

        self._tree(tmp_path)
        config = GlobalBatchConfig(
            topics=[TopicSpec(title="Why wifi drops")], outputs_dir=tmp_path
        )

        targets = {p.name for p in _clean_targets(tmp_path, _named_run_ids(config))}

        assert targets == {topic_product_id("Why wifi drops")}

    def test_a_product_run_is_unchanged(self, tmp_path):
        from src.pipeline.global_batch import _clean_targets, _named_run_ids

        self._tree(tmp_path)
        config = GlobalBatchConfig(product_ids=["B0FC5S16YM"], outputs_dir=tmp_path)

        targets = {p.name for p in _clean_targets(tmp_path, _named_run_ids(config))}

        assert targets == {"B0FC5S16YM"}

    def test_a_run_naming_nothing_still_sweeps_every_run_directory(self, tmp_path):
        """The keyword-run behaviour this must not break."""
        from src.pipeline.global_batch import _clean_targets, _named_run_ids

        self._tree(tmp_path)
        config = GlobalBatchConfig(keywords=["wireless earbuds"], outputs_dir=tmp_path)

        targets = {p.name for p in _clean_targets(tmp_path, _named_run_ids(config))}

        assert "logs" not in targets
        assert targets == {
            "B0ABCDEFGH",
            "B0FC5S16YM",
            topic_product_id("Why wifi drops"),
            topic_product_id("Some other topic"),
        }


class TestResumeKeepsTheTopics:
    """`--resume` carries no input flags, so the config has no topics on it.

    Reading `config.topics` to decide whether discovery returns them meant a
    resumed topics run found nothing and reported PIPELINE FAILED.
    """

    def test_discovery_opts_in_from_the_ids_being_handed_off(self, tmp_path):
        import json

        from src.pipeline.global_batch import GlobalPipelineOrchestrator
        from src.video.producer.topic_input import build_topic_product

        pid = topic_product_id("Why wifi drops")
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "data.json").write_text(
            json.dumps(
                build_topic_product(TopicSpec(title="Why wifi drops")).to_dict()
            ),
            encoding="utf-8",
        )

        # A resumed run: no topics on the config, ids from the saved state.
        resumed = GlobalBatchConfig(outputs_dir=tmp_path, skip_publish=True)
        ready = GlobalPipelineOrchestrator(resumed)._execute_handoff_phase([pid])

        assert [p.name for p, _ in ready] == [pid]


class TestTopicsAndProcessAllProducts:
    def test_the_combination_is_refused(self, real_video_config):
        """It would render scraped products from generic stock footage.

        A topics run narrows the pool to stock-sourced profiles, and
        `--process-all-products` sweeps every product directory into the same
        run, so each would be rendered with its own photos ignored -- and then
        published.
        """
        with pytest.raises(ValueError, match="--process-all-products"):
            validate_global_batch_config(
                GlobalBatchConfig(
                    topics=[TopicSpec(title="Why wifi drops")],
                    profile="slideshow_stock",
                    process_all_products=True,
                    skip_publish=True,
                ),
                real_video_config,
            )


class TestAConfiguredPoolIsNotAnInstructionForThisRun:
    """`pipeline.yaml` can set `profile_pool` for the default product run.

    Refusing it made "omit --profile" fail on any install that configured one,
    while the tests passed because they build the config directly.
    """

    def test_a_yaml_pool_is_replaced(self, real_video_config):
        config = GlobalBatchConfig(
            topics=[TopicSpec(title="Why wifi drops")],
            random_profile=True,
            profile_pool=["slideshow_images1", "slideshow_images2"],
            profile_pool_from_cli=False,
            skip_publish=True,
        )
        validate_global_batch_config(config, real_video_config)

        assert config.profile_pool == ["slideshow_stock"]

    def test_a_pool_named_on_the_command_line_is_refused(self, real_video_config):
        with pytest.raises(ValueError, match="cannot render a topic"):
            validate_global_batch_config(
                GlobalBatchConfig(
                    topics=[TopicSpec(title="Why wifi drops")],
                    random_profile=True,
                    profile_pool=["slideshow_images1"],
                    profile_pool_from_cli=True,
                    skip_publish=True,
                ),
                real_video_config,
            )
