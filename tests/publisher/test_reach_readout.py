"""The reach-test readout: day-N views by format arm (REQ-PUB-149 to 152)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.publisher.analytics import (
    PostMetrics,
    _combine,
    load_metrics,
    platform_day_views,
    readout_by_arm,
    save_metrics,
    summarize_post,
)
from src.publisher.late.cli import _iso_date, _load_product_map, cmd_analytics
from src.publisher.models import AnalyticsConfig, PublisherConfig
from src.publisher.product_registry import RegistryEntry, save_registry

REPO = Path(__file__).resolve().parents[2]
PUBLISHED = datetime(2026, 9, 20, 8, 0)


def _row(platform: str, day: int, views: int) -> dict:
    return {"platform": platform, "date": f"2026-09-{20 + day:02d}", "views": views}


def _post(post_id: str, youtube: int | None, tiktok: int | None, **kw) -> PostMetrics:
    by_platform = {}
    if youtube is not None:
        by_platform["youtube"] = {"day_2": youtube, "day_7": youtube}
    if tiktok is not None:
        by_platform["tiktok"] = {"day_2": tiktok, "day_7": tiktok}
    return PostMetrics(
        post_id=post_id,
        published_at=kw.get("published_at", "2026-09-20T08:00:00"),
        views_by_platform=by_platform,
    )


def _readout(metrics, arms, since=None, platforms=("youtube", "tiktok")):
    return readout_by_arm(
        metrics,
        arms,
        since=since,
        platforms=list(platforms),
        breakout_multiple=2.0,
    )


def _line(lines: list[str], prefix: str) -> str:
    return next(line.strip() for line in lines if line.strip().startswith(prefix))


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-149")
class TestPerPlatformDayViews:
    def test_each_platform_is_read_from_its_own_rows(self) -> None:
        rows = [
            _row("youtube", 1, 10),
            _row("tiktok", 1, 100),
            _row("youtube", 3, 20),
            _row("tiktok", 3, 300),
            _row("youtube", 8, 25),
            _row("tiktok", 8, 400),
        ]

        assert platform_day_views(rows, PUBLISHED) == {
            "youtube": {"day_2": 10, "day_7": 20},
            "tiktok": {"day_2": 100, "day_7": 300},
        }

    def test_a_leg_with_no_row_by_the_cutoff_is_unknown(self) -> None:
        rows = [
            _row("youtube", 1, 10),
            _row("youtube", 8, 30),
            _row("tiktok", 5, 90),
            _row("tiktok", 8, 120),
        ]

        views = platform_day_views(rows, PUBLISHED)

        assert views["tiktok"]["day_2"] is None
        assert views["tiktok"]["day_7"] == 90

    def test_the_sweep_stores_them(self) -> None:
        rows = [_row("youtube", 1, 10), _row("youtube", 8, 30)]

        measured = summarize_post("p", PUBLISHED.isoformat(), rows)

        assert measured.views_by_platform == {"youtube": {"day_2": 10, "day_7": 10}}

    def test_a_later_unknown_never_erases_a_stored_figure(self, tmp_path: Path) -> None:
        stored = _post("p", 50, 400)
        fresh = PostMetrics(
            post_id="p",
            views_by_platform={"youtube": {"day_2": None, "day_7": None}},
        )

        merged = _combine(stored, fresh)
        save_metrics([stored], tmp_path)
        save_metrics([fresh], tmp_path)

        assert merged.views_by_platform["youtube"] == {"day_2": 50, "day_7": 50}
        assert load_metrics(tmp_path)[0].views_by_platform["tiktok"]["day_7"] == 400


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-150")
class TestTheReadout:
    def test_medians_ratio_and_verdict_count_only_the_gate_platforms(self) -> None:
        metrics = [
            _post("t1", 100, 300),
            _post("t2", 100, 500),
            _post("t3", 100, 700),
            _post("p1", 100, 900),
            _post("p2", 100, 1100),
        ]
        # Instagram figures that would flip the ratio if they were counted.
        for m in metrics[3:]:
            m.views_by_platform["instagram"] = {"day_2": 0, "day_7": 0}
        for m in metrics[:3]:
            m.views_by_platform["instagram"] = {"day_2": 5000, "day_7": 5000}
        arms = {"t1": "topic", "t2": "topic", "t3": "topic"}
        arms |= {"p1": "product", "p2": "product"}

        lines = _readout(metrics, arms)

        assert "day 7 median 600 (n=3)" in _line(lines, "topic [")
        assert "day 7 median 1100 (n=2)" in _line(lines, "product [")
        assert _line(lines, "ratio day 7") == (
            "ratio day 7: 0.55: proceed, re-forecast revenue down by the ratio"
        )

    @pytest.mark.parametrize(
        ("topic", "verdict"),
        [
            (700, "reach holds: proceed"),
            (699, "proceed, re-forecast revenue down by the ratio"),
            (400, "proceed, re-forecast revenue down by the ratio"),
            (399, "reach premise fails: revisit niche or format"),
        ],
    )
    def test_the_bands(self, topic: int, verdict: str) -> None:
        arms = {"t": "topic", "p": "product"}

        lines = _readout([_post("t", 0, topic), _post("p", 0, 1000)], arms)

        assert _line(lines, "ratio day 7").endswith(verdict)

    def test_a_post_missing_a_gate_platform_is_left_out(self) -> None:
        metrics = [_post("t1", None, 50), _post("t2", 10, 500), _post("p", 10, 500)]
        arms = {"t1": "topic", "t2": "topic", "p": "product"}

        lines = _readout(metrics, arms)

        assert "day 7 median 510 (n=1)" in _line(lines, "topic [2 post(s)]")

    def test_posts_with_no_arm_are_counted_and_named(self) -> None:
        metrics = [_post("t", 1, 1), _post("p", 1, 1), _post("lost", 1, 1)]

        lines = _readout(metrics, {"t": "topic", "p": "product"})

        assert lines[0].endswith("3 post(s), 1 with no arm")
        assert _line(lines, "no arm:") == "no arm: lost"

    def test_since_drops_older_posts(self) -> None:
        metrics = [
            _post("old", 1, 1, published_at="2026-09-13T20:00:00"),
            _post("new", 1, 1, published_at="2026-09-14T08:00:00"),
        ]

        lines = _readout(
            metrics, {"old": "topic", "new": "topic"}, since=datetime(2026, 9, 14)
        )

        assert "since 2026-09-14: 1 post(s)" in lines[0]
        assert "topic [1 post(s)]" in _line(lines, "topic [")

    def test_an_empty_arm_reads_not_measurable(self) -> None:
        lines = _readout([_post("t", 1, 1)], {"t": "topic"})

        assert _line(lines, "ratio day 7") == "ratio day 7: not measurable"


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-152")
class TestSecondaryLines:
    def test_per_platform_medians_cover_every_platform(self) -> None:
        metrics = [_post("t", 10, 300), _post("p", 90, 200)]
        metrics[0].views_by_platform["instagram"] = {"day_2": 7, "day_7": 7}

        lines = _readout(metrics, {"t": "topic", "p": "product"})

        assert _line(lines, "youtube day 7") == (
            "youtube day 7 median: topic 10 (n=1), product 90 (n=1)"
        )
        assert _line(lines, "instagram day 7") == (
            "instagram day 7 median: topic 7 (n=1), product - (n=0)"
        )

    def test_breakouts_are_at_the_multiple_of_the_pooled_median(self) -> None:
        # Pooled day-7 sums: 100, 200, 300, 599, 600 -> median 300, bar 600.
        metrics = [
            _post("t1", 0, 100),
            _post("t2", 0, 600),
            _post("p1", 0, 300),
            _post("p2", 0, 599),
            _post("p3", 0, 200),
        ]
        arms = {"t1": "topic", "t2": "topic", "p1": "product"}
        arms |= {"p2": "product", "p3": "product"}

        line = _line(_readout(metrics, arms), "breakouts")

        assert "pooled median 300 or more" in line
        assert line.endswith("topic 1 of 2, product 0 of 3")


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-151")
class TestPostToProduct:
    def _write(self, tmp_path: Path, history: dict, schedule: list) -> None:
        state = tmp_path / "state"
        state.mkdir()
        (state / "publish_history.json").write_text(json.dumps({"posts": history}))
        (state / "schedule.json").write_text(json.dumps({"entries": schedule}))

    def test_schedule_names_posts_a_republish_dropped_from_history(
        self, tmp_path: Path
    ) -> None:
        self._write(
            tmp_path,
            {"topic-x:youtube": {"product_id": "topic-x", "post_id": "new"}},
            [
                {"product_id": "topic-x", "post_id": "old"},
                {"product_id": "wrong", "post_id": "new"},
            ],
        )

        assert _load_product_map(tmp_path) == {"new": "topic-x", "old": "topic-x"}

    def test_a_missing_or_broken_file_reads_as_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "state").mkdir()
        (tmp_path / "state" / "schedule.json").write_text("{not json")

        assert _load_product_map(tmp_path) == {}


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-150")
class TestTheCommand:
    async def test_rank_only_by_arm_logs_the_readout(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        save_metrics([_post("a", 10, 90), _post("b", 10, 190)], tmp_path)
        state = tmp_path / "state"
        (state / "publish_history.json").write_text(
            json.dumps(
                {
                    "posts": {
                        "topic-a:youtube": {"product_id": "topic-a", "post_id": "a"},
                        "B0B:youtube": {"product_id": "B0B", "post_id": "b"},
                    }
                }
            )
        )
        save_registry(
            [
                RegistryEntry("topic-a", "A", "", "", content_format="topic"),
                RegistryEntry("B0B", "B", "", "", content_format="product"),
            ],
            tmp_path,
        )
        args = argparse.Namespace(
            limit=None,
            rank_only=True,
            by_arm=True,
            since=_iso_date("2026-09-14"),
            outputs_dir=tmp_path,
            debug=False,
        )
        config = PublisherConfig(
            provider="late",
            api_key="sk_live_key_12345",
            analytics_config=AnalyticsConfig(readout_platforms=["tiktok"]),
        )

        with caplog.at_level(logging.INFO):
            await cmd_analytics(args, config, MagicMock())

        assert "Reach readout by arm on tiktok since 2026-09-14" in caplog.text
        assert "ratio day 7: 0.47" in caplog.text

    def test_since_takes_a_date(self) -> None:
        assert _iso_date("2026-09-14") == datetime(2026, 9, 14)
        with pytest.raises(argparse.ArgumentTypeError):
            _iso_date("14/09/2026")


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-150")
class TestConfig:
    def test_the_shipped_gate_is_youtube_and_tiktok(self) -> None:
        raw = yaml.safe_load((REPO / "config" / "publisher.yaml").read_text())
        config = AnalyticsConfig(**raw["analytics"])

        assert config.readout_platforms == ["youtube", "tiktok"]
        assert config.breakout_multiple == 2.0

    @pytest.mark.parametrize(
        "bad",
        [
            {"readout_platforms": []},
            {"readout_platforms": "youtube"},
            {"readout_platforms": ["youtube", ""]},
            {"breakout_multiple": 1},
            {"breakout_multiple": True},
            {"breakout_multiple": "2"},
        ],
    )
    def test_bad_values_are_refused(self, bad: dict) -> None:
        AnalyticsConfig()
        with pytest.raises(ValueError):
            AnalyticsConfig(**bad)
