"""One content format per day, decided by date parity (#437).

A no-flag run renders both formats side by side, and with more than one
record per day the publish order decides which slot each post takes -- so the
two formats end up pinned to different times of day, which confounds any
reach comparison between them. `alternate_formats` makes the daily run draw
one side per day, interleaving the formats with posting time held constant.
"""

from __future__ import annotations

import argparse
import datetime as real_datetime
from unittest.mock import patch

import pytest
import yaml

from src.pipeline.config import (
    format_for_run,
    load_global_batch_config,
)

CONFIG = {
    "keywords": ["earbuds", "charger", "tracker", "hub"],
    "topics": [
        {"title": "Topic A"},
        {"title": "Topic B"},
        {"title": "Topic C"},
        {"title": "Topic D"},
    ],
    "topics_per_run": 1,
    "alternate_formats": True,
}


def write_config(tmp_path, global_batch: dict) -> str:
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump({"global_batch": global_batch}), encoding="utf-8")
    return str(path)


def load_on_day(tmp_path, global_batch: dict, ordinal: int):
    """Drive the loader with a pinned calendar day.

    The loader reads `date.today()` itself, so the pin patches the module's
    `date` rather than passing an argument -- there is no CLI surface for
    "pretend it is Tuesday", deliberately.
    """

    class FakeDate(real_datetime.date):
        @classmethod
        def today(cls):
            return real_datetime.date.fromordinal(ordinal)

    with patch("src.pipeline.config.date", FakeDate):
        return load_global_batch_config(
            argparse.Namespace(), write_config(tmp_path, global_batch)
        )


class TestTheParity:
    def test_even_days_draw_topics_odd_days_products(self):
        assert format_for_run(10) == "topic"
        assert format_for_run(11) == "product"

    def test_today_is_the_default(self):
        assert format_for_run() in ("topic", "product")


class TestTheLoader:
    def test_a_topic_day_carries_topics_only(self, tmp_path):
        config = load_on_day(tmp_path, CONFIG, ordinal=100)

        assert config.alternated_format == "topic"
        assert config.topics and not config.keywords and not config.product_ids

    def test_a_product_day_carries_products_only(self, tmp_path):
        config = load_on_day(tmp_path, CONFIG, ordinal=101)

        assert config.alternated_format == "product"
        assert config.keywords and not config.topics

    def test_off_by_default_keeps_the_mixed_run(self, tmp_path):
        plain = {k: v for k, v in CONFIG.items() if k != "alternate_formats"}
        config = load_on_day(tmp_path, plain, ordinal=100)

        assert config.alternated_format is None
        assert config.topics and config.keywords

    def test_a_non_bool_value_is_refused_by_name(self, tmp_path):
        with pytest.raises(ValueError, match="alternate_formats"):
            load_on_day(tmp_path, {**CONFIG, "alternate_formats": "yes"}, ordinal=100)


class TestTheCompressedRotation:
    """The trap the raw ordinal sets: a side only runs every second day, so
    its ordinals share a parity, and `ordinal % len(pool)` over same-parity
    ordinals visits only half the indices of an even-length pool -- topics
    2 and 4 of a four-topic pool would never render.
    """

    def test_consecutive_topic_days_take_consecutive_topics(self, tmp_path):
        titles = []
        for ordinal in (100, 102, 104, 106):
            config = load_on_day(tmp_path, CONFIG, ordinal=ordinal)
            assert config.alternated_format == "topic"
            titles.append(config.topics[0].title)

        # Four topic days over a four-topic pool cover the whole pool.
        assert sorted(titles) == ["Topic A", "Topic B", "Topic C", "Topic D"]

    def test_consecutive_product_days_move_the_keyword_slice(self, tmp_path):
        cfg = {**CONFIG, "keywords_per_run": 1}
        picked = []
        for ordinal in (101, 103, 105, 107):
            config = load_on_day(tmp_path, cfg, ordinal=ordinal)
            assert config.alternated_format == "product"
            picked.extend(config.keywords)

        assert sorted(picked) == ["charger", "earbuds", "hub", "tracker"]


class TestExplicitInputsBypass:
    def test_cli_topics_render_whatever_the_parity_says(self, tmp_path):
        args = argparse.Namespace(topic="Named topic")
        path = write_config(tmp_path, CONFIG)

        class FakeDate(real_datetime.date):
            @classmethod
            def today(cls):
                return real_datetime.date.fromordinal(101)  # a product day

        with patch("src.pipeline.config.date", FakeDate):
            config = load_global_batch_config(args, path)

        assert config.alternated_format is None
        assert [t.title for t in config.topics] == ["Named topic"]

    def test_cli_keywords_search_whatever_the_parity_says(self, tmp_path):
        args = argparse.Namespace(keywords=["usb hub"])
        path = write_config(tmp_path, CONFIG)

        class FakeDate(real_datetime.date):
            @classmethod
            def today(cls):
                return real_datetime.date.fromordinal(100)  # a topic day

        with patch("src.pipeline.config.date", FakeDate):
            config = load_global_batch_config(args, path)

        assert config.alternated_format is None
        assert config.keywords == ["usb hub"]


class TestTheEmptySideFallback:
    """A scheduled run must not be lost to a config gap every second day."""

    def test_a_topic_day_with_no_topics_runs_products(self, tmp_path, caplog):
        cfg = {k: v for k, v in CONFIG.items() if k != "topics"}
        with caplog.at_level("WARNING"):
            config = load_on_day(tmp_path, cfg, ordinal=100)

        assert config.alternated_format == "product"
        assert config.keywords
        assert any("topic side" in r.message for r in caplog.records)

    def test_a_product_day_with_no_keywords_runs_topics(self, tmp_path, caplog):
        cfg = {k: v for k, v in CONFIG.items() if k != "keywords"}
        # The loader falls back to the scraper's keyword pool when the batch
        # has none; point the sibling scraper.yaml at an empty pool so the
        # product side is genuinely empty.
        (tmp_path / "scraper.yaml").write_text(
            yaml.safe_dump({"batch": {"keywords": []}}), encoding="utf-8"
        )
        with caplog.at_level("WARNING"):
            config = load_on_day(tmp_path, cfg, ordinal=101)

        assert config.alternated_format == "topic"
        assert config.topics
        assert any(
            "product side but that side is empty" in r.message for r in caplog.records
        )

    def test_the_fallback_state_rotates_daily_not_half_speed(self, tmp_path):
        """In the fallback state the surviving side runs EVERY day, so the
        compressed ordinal would repeat on consecutive days: the same
        keyword slice searched two days running yields nothing at handoff,
        and the same topic publishes twice. The raw ordinal is the
        consecutive-step one there.
        """
        cfg = {k: v for k, v in CONFIG.items() if k != "topics"}
        cfg["keywords_per_run"] = 1
        picked = [
            load_on_day(tmp_path, cfg, ordinal=o).keywords[0]
            for o in (100, 101, 102, 103)
        ]

        assert len(set(picked)) == 4, f"consecutive days repeated a slice: {picked}"

    def test_the_topic_fallback_state_rotates_daily_too(self, tmp_path):
        cfg = {k: v for k, v in CONFIG.items() if k != "keywords"}
        (tmp_path / "scraper.yaml").write_text(
            yaml.safe_dump({"batch": {"keywords": []}}), encoding="utf-8"
        )
        picked = [
            load_on_day(tmp_path, cfg, ordinal=o).topics[0].title
            for o in (100, 101, 102, 103)
        ]

        assert len(set(picked)) == 4, f"consecutive days repeated a topic: {picked}"

    def test_keywords_per_run_zero_is_an_empty_product_side(self, tmp_path, caplog):
        """`keywords_per_run: 0` slices a non-empty pool to nothing, so a
        product day would build a no-input config and the scheduled run
        would be refused instead of falling back.
        """
        cfg = {**CONFIG, "keywords_per_run": 0}
        with caplog.at_level("WARNING"):
            config = load_on_day(tmp_path, cfg, ordinal=101)

        assert config.alternated_format == "topic"
        assert config.topics


class TestThePlanNamesTheSide:
    def test_dry_run_plan_prints_the_drawn_side(self, tmp_path, capsys):
        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        config = load_on_day(tmp_path, CONFIG, ordinal=100)
        config.dry_run = True
        orchestrator = GlobalPipelineOrchestrator(config)

        from src.video.config_adapter import load_video_config_modular

        orchestrator.display_execution_plan(load_video_config_modular())
        out = capsys.readouterr().out

        assert "today draws the topic side" in out
