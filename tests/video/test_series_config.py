"""Series config and series topics (REQ-CNT-193)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.scraper.amazon.models import ProductData
from src.video.config.series_models import SeriesConfig
from src.video.producer.topic_input import (
    TopicInputError,
    TopicSpec,
    build_topic_product,
    specs_from_mappings,
)

pytestmark = pytest.mark.req("REQ-CNT-193")

SERIES = SeriesConfig(title_prefix="Myth check: ", verdicts=["YES", "NO", "PARTLY"])


def _spec(**kw) -> TopicSpec:
    base = {"title": "Does a magnet wipe a phone", "kind": "commentary"}
    return TopicSpec(**(base | kw))


def test_the_bundled_config_has_no_series():
    from src.video.config import config

    assert config.series == {}


def test_a_series_topic_names_claim_and_verdict():
    with pytest.raises(TopicInputError, match="claim and verdict"):
        _spec(series="myth")
    with pytest.raises(TopicInputError, match="need a series"):
        _spec(claim="A magnet wipes a phone")
    spec = _spec(series="myth", claim="A magnet wipes a phone", verdict="NO")
    assert spec.verdict == "NO"


def test_the_topics_file_accepts_the_series_keys_and_still_refuses_typos():
    entry = {
        "title": "Magnets",
        "kind": "commentary",
        "series": "myth",
        "claim": "A magnet wipes a phone",
        "verdict": "NO",
        "credit": "@viewer",
    }
    (spec,) = specs_from_mappings([entry], "topics.yaml")
    assert (spec.series, spec.credit) == ("myth", "@viewer")
    with pytest.raises(TopicInputError, match="unknown key"):
        specs_from_mappings([entry | {"verdikt": "NO"}], "topics.yaml")


def test_the_record_carries_the_series_fields():
    record = build_topic_product(
        _spec(series="myth", claim="A magnet wipes a phone", verdict="NO")
    )
    data = record.to_dict()
    assert (data["series"], data["claim"], data["verdict"]) == (
        "myth",
        "A magnet wipes a phone",
        "NO",
    )
    assert ProductData(**data).verdict == "NO"


def test_episode_numbering_needs_a_queue_file():
    with pytest.raises(ValidationError, match="queue_file"):
        SeriesConfig(numbering="episode")


def _record(series: str, verdict: str, kind: str = "commentary"):
    return build_topic_product(
        _spec(series=series, claim="A magnet wipes a phone", verdict=verdict, kind=kind)
    )


def test_an_unconfigured_series_or_verdict_is_refused():
    from src.video.config import config
    from src.video.producer.orchestration import PipelineError, config_for_record

    cfg = config.model_copy(deep=True)
    cfg.series = {"myth": SERIES}
    with pytest.raises(PipelineError, match="isn't configured"):
        config_for_record(cfg, _record("other", "NO"))
    with pytest.raises(PipelineError, match="isn't one of"):
        config_for_record(cfg, _record("myth", "MAYBE"))
    assert config_for_record(cfg, _record("myth", "NO")) is not None
    # A tutorial topic with no identity pins takes the early return.
    with pytest.raises(PipelineError, match="isn't one of"):
        config_for_record(cfg, _record("myth", "MAYBE", kind="tutorial"))


def test_an_unquoted_yaml_boolean_label_is_refused_with_the_fix():
    import yaml

    raw = yaml.safe_load("verdicts: [YES, NO]")
    with pytest.raises(ValidationError, match="quote labels"):
        SeriesConfig(**raw)
    assert SeriesConfig(**yaml.safe_load('verdicts: ["YES", "NO"]')).verdicts == [
        "YES",
        "NO",
    ]


def test_an_unquoted_yaml_boolean_in_a_topic_is_refused_with_the_fix():
    import yaml

    entry = yaml.safe_load(
        "title: T\nkind: commentary\nseries: myth\nclaim: c\nverdict: NO"
    )
    with pytest.raises(TopicInputError, match="quote values"):
        specs_from_mappings([entry], "topics.yaml")
