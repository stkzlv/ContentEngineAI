"""Per-platform quality metrics: stored, kept unknown when unexposed, segmented."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from late import LateError

from src.publisher.analytics import (
    PostMetrics,
    load_metrics,
    metrics_path,
    quality_metrics,
    save_metrics,
    segment_quality,
)
from src.publisher.late.cli import cmd_analytics
from src.publisher.models import AnalyticsConfig, PublisherConfig

# The provider's per-post analytics, shaped as measured on a live post: it
# fills unexposed fields with 0 on every platform.
PLATFORM_ANALYTICS = [
    {
        "platform": "youtube",
        "analytics": {"views": 25, "likes": 0, "impressions": 0, "reach": 0},
    },
    {
        "platform": "tiktok",
        "analytics": {
            "views": 257,
            "likes": 6,
            "shares": 0,
            "impressions": 0,
            "completionRate": 0,
        },
    },
    {
        "platform": "instagram",
        "analytics": {
            "views": 52,
            "reach": 46,
            "igReelsAvgWatchTime": 4330,
            "reelsSkipRate": 70.8,
            "completionRate": 0,
        },
    },
    {"platform": "linkedin", "analytics": {"views": 3}},
]


@pytest.mark.req("REQ-PUB-084")
def test_an_exposed_zero_is_kept_and_an_unexposed_one_is_unknown() -> None:
    q = quality_metrics(PLATFORM_ANALYTICS)

    assert q["youtube"]["views"] == 25
    assert q["youtube"]["likes"] == 0  # exposed: a real zero
    assert "impressions" not in q["youtube"]  # not a quality field at all
    assert q["youtube"]["engagedViews"] is None  # a first-seconds gap, unknown
    assert q["tiktok"]["shares"] == 0
    assert q["tiktok"]["completionRate"] is None  # filled with 0, not exposed
    assert q["instagram"]["reelsSkipRate"] == 70.8
    assert q["instagram"]["saves"] is None  # absent from the response
    assert "linkedin" not in q


def test_odd_input_yields_nothing_or_unknown() -> None:
    assert quality_metrics(None) == {}
    assert quality_metrics(["x", {"platform": None}]) == {}
    odd = quality_metrics([{"platform": "TikTok", "analytics": {"views": True}}])
    assert odd["tiktok"]["views"] is None


@pytest.mark.req("REQ-PUB-084")
@pytest.mark.parametrize(
    "leg",
    [{"syncStatus": "pending"}, {"syncStatus": "unavailable"}, {"status": "failed"}],
)
def test_a_leg_with_no_reading_keeps_the_stored_figures(
    tmp_path: Path, leg: dict
) -> None:
    save_metrics(
        [PostMetrics(post_id="p1", platform_metrics={"tiktok": {"views": 257}})],
        tmp_path,
    )
    fresh = quality_metrics([{"platform": "tiktok", "analytics": {"views": 0}, **leg}])
    save_metrics([PostMetrics(post_id="p1", platform_metrics=fresh)], tmp_path)

    assert fresh == {}
    assert load_metrics(tmp_path)[0].platform_metrics["tiktok"]["views"] == 257


@pytest.mark.req("REQ-PUB-084")
def test_a_failed_reading_keeps_the_stored_figures(tmp_path: Path) -> None:
    first = PostMetrics(
        post_id="p1", platform_metrics=quality_metrics(PLATFORM_ANALYTICS)
    )
    save_metrics([first], tmp_path)
    save_metrics([PostMetrics(post_id="p1")], tmp_path)

    stored = load_metrics(tmp_path)[0].platform_metrics

    assert stored["tiktok"]["views"] == 257


def test_a_fresh_reading_replaces_the_stored_one(tmp_path: Path) -> None:
    save_metrics(
        [PostMetrics(post_id="p1", platform_metrics={"tiktok": {"views": 1}})],
        tmp_path,
    )
    save_metrics(
        [PostMetrics(post_id="p1", platform_metrics={"tiktok": {"views": 9}})],
        tmp_path,
    )

    assert load_metrics(tmp_path)[0].platform_metrics["tiktok"]["views"] == 9


def test_a_file_from_before_the_field_still_loads(tmp_path: Path) -> None:
    path = metrics_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"post_id": "p1", "views_total": 5}]))

    assert load_metrics(tmp_path)[0].platform_metrics == {}


@pytest.mark.req("REQ-PUB-084")
def test_segments_average_known_readings_only() -> None:
    metrics = [
        PostMetrics(post_id="a", platform_metrics={"tiktok": {"views": 100}}),
        PostMetrics(post_id="b", platform_metrics={"tiktok": {"views": 300}}),
        PostMetrics(post_id="c", platform_metrics={"tiktok": {"views": None}}),
    ]
    product_by_post = {"a": "A", "b": "B", "c": "C"}
    labels = {
        "A": {"content_format": "review", "voice_profile": "charon"},
        "B": {"content_format": "review", "voice_profile": "puck"},
        "C": {"content_format": "review"},
    }

    lines = segment_quality(metrics, product_by_post, labels)

    assert "content_format=review [3 post(s)]: tiktok.views=200.0(n=2)" in lines
    assert "voice_profile=charon [1 post(s)]: tiktok.views=100.0(n=1)" in lines


async def _sweep(tmp_path: Path, analytics_side_effect) -> None:
    resource = MagicMock()
    resource.get_post_timeline.return_value = {"timeline": []}
    resource.get_analytics.side_effect = analytics_side_effect
    publisher = MagicMock()
    publisher.authenticate = AsyncMock(return_value=True)
    publisher.list_posts = AsyncMock(return_value=[{"id": "p1", "platforms": []}])
    publisher.client = MagicMock()
    config = PublisherConfig(
        provider="late",
        api_key="sk_live_key_12345",
        analytics_config=AnalyticsConfig(limit=50),
    )
    args = argparse.Namespace(
        limit=None,
        rank_only=False,
        by_arm=False,
        since=None,
        outputs_dir=tmp_path,
        debug=False,
    )
    with (
        patch(
            "src.publisher.late.cli._create_publisher_from_config",
            return_value=publisher,
        ),
        patch("src.publisher.late.cli.timeline_resource", return_value=resource),
        patch("src.publisher.late.cli.ANALYTICS_FAILURES_LOG", tmp_path / "f.log"),
    ):
        await cmd_analytics(args, config, MagicMock())


@pytest.mark.req("REQ-PUB-084")
@pytest.mark.asyncio
async def test_the_sweep_stores_each_posts_platform_metrics(tmp_path: Path) -> None:
    await _sweep(tmp_path, [{"platformAnalytics": PLATFORM_ANALYTICS}])

    stored = load_metrics(tmp_path)[0].platform_metrics
    assert stored["instagram"]["igReelsAvgWatchTime"] == 4330


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        LateError("analytics add-on missing"),
        httpx.ReadError("connection reset"),
        json.JSONDecodeError("Expecting value", "<html>", 0),
    ],
)
async def test_a_failed_analytics_call_still_stores_the_post(
    tmp_path: Path, error: Exception
) -> None:
    await _sweep(tmp_path, error)

    stored = load_metrics(tmp_path)
    assert [m.post_id for m in stored] == ["p1"]
    assert stored[0].platform_metrics == {}


def test_an_unknown_reading_does_not_erase_a_known_one(tmp_path: Path) -> None:
    save_metrics(
        [PostMetrics(post_id="p1", platform_metrics={"tiktok": {"views": 257}})],
        tmp_path,
    )
    save_metrics(
        [PostMetrics(post_id="p1", platform_metrics={"tiktok": {"views": None}})],
        tmp_path,
    )

    assert load_metrics(tmp_path)[0].platform_metrics["tiktok"]["views"] == 257
