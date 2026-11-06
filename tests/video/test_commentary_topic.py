"""A topic marked commentary renders as an opinion take (REQ-CNT-166, REQ-CNT-167).

The record carries `topic_kind: commentary`; the producer swaps the topic
family for the commentary one on a copy of the config, for that record only,
and turns the step list off, so no step visuals are built.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ai.script_generator import select_script_template
from src.scraper.amazon.models import ProductData
from src.video.producer.orchestration import PipelineError, config_for_record
from src.video.producer.topic_input import (
    TopicInputError,
    TopicSpec,
    build_topic_product,
    load_topics_file,
    specs_from_args,
)

pytestmark = pytest.mark.req("REQ-CNT-166", "REQ-CNT-167")


@pytest.fixture
def config():
    from src.video.config_adapter import load_video_config_modular

    return load_video_config_modular()


def _commentary(title: str = "A test take") -> ProductData:
    return build_topic_product(TopicSpec(title=title, kind="commentary"))


class TestTheRecord:
    def test_kind_defaults_to_tutorial(self):
        assert TopicSpec(title="t").kind == "tutorial"

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(TopicInputError, match="kind"):
            TopicSpec(title="t", kind="opinion")

    def test_commentary_reaches_data_json(self):
        data = _commentary().to_dict()
        assert data["topic_kind"] == "commentary"
        assert ProductData(**data).topic_kind == "commentary"

    def test_a_tutorial_topic_has_no_kind(self):
        tutorial = build_topic_product(TopicSpec(title="t")).to_dict()
        assert tutorial["topic_kind"] is None

    def test_topics_file_takes_kind(self, tmp_path: Path):
        path = tmp_path / "topics.yaml"
        path.write_text(
            "- title: A take\n  kind: commentary\n- title: A how-to\n",
            encoding="utf-8",
        )
        assert [s.kind for s in load_topics_file(path)] == ["commentary", "tutorial"]

    def test_topics_file_refuses_a_bad_kind(self, tmp_path: Path):
        path = tmp_path / "topics.yaml"
        path.write_text("- title: A take\n  kind: rant\n", encoding="utf-8")
        with pytest.raises(TopicInputError, match="entry 1"):
            load_topics_file(path)

    def test_the_flag_sets_the_kind(self):
        (spec,) = specs_from_args(
            topic="A take",
            topic_description=None,
            topic_keywords=None,
            topics_file=None,
            topic_kind="commentary",
        )
        assert spec.kind == "commentary"

    @pytest.mark.parametrize(
        ("topic", "topics_file"), [(None, None), (None, Path("topics.yaml"))]
    )
    def test_the_flag_without_a_topic_is_refused(self, topic, topics_file):
        with pytest.raises(TopicInputError, match="--topic-kind"):
            specs_from_args(
                topic=topic,
                topic_description=None,
                topic_keywords=None,
                topics_file=topics_file,
                topic_kind="commentary",
            )


class TestTheConfigSwap:
    def test_a_tutorial_topic_keeps_the_config(self, config):
        record = build_topic_product(TopicSpec(title="t"))
        assert config_for_record(config, record) is config

    def test_commentary_swaps_the_family_on_a_copy(self, config):
        original = config.llm_settings.script_templates.topic_templates[:]
        swapped = config_for_record(config, _commentary())
        t = swapped.llm_settings.script_templates
        assert t.topic_templates == t.commentary_templates == ["commentary_take"]
        assert t.narrator_profile_topic == t.narrator_profile_commentary
        assert t.cta_options_topic == t.cta_options_commentary
        assert t.target_length.topic == t.target_length.commentary
        assert t.topic_routing.enabled is False
        assert swapped.llm_settings.topic_scripts.step_list.enabled is False
        # The shared config is untouched for the next record in a batch.
        assert config.llm_settings.script_templates.topic_templates == original
        assert config.llm_settings.topic_scripts.step_list.enabled is True

    def test_commentary_without_templates_is_refused(self, config):
        config.llm_settings.script_templates.commentary_templates = []
        with pytest.raises(PipelineError, match="commentary_templates"):
            config_for_record(config, _commentary())

    def test_the_take_draws_the_commentary_template(self, config):
        swapped = config_for_record(config, _commentary())
        record = _commentary()
        chosen = select_script_template(
            swapped.llm_settings, record.asin, is_topic=True, title=record.title
        )
        assert chosen.stem == "commentary_take"

    def test_a_product_never_draws_it(self, config):
        for pid in (f"B0TEST{i:04d}" for i in range(40)):
            chosen = select_script_template(config.llm_settings, pid)
            assert chosen.stem != "commentary_take"


def test_both_entry_points_pass_the_flag():
    # Module/batch alignment: the producer and the batch read --topic-kind.
    from src.utils.outputs_paths import get_project_root

    root = get_project_root()
    for path in ("src/video/producer/cli.py", "src/pipeline/config.py"):
        assert "topic_kind" in (root / path).read_text()
    assert '"--topic-kind"' in (root / "src/pipeline/cli.py").read_text()
