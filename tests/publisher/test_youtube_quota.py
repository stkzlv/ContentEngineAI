"""YouTube Analytics refusing a sweep ends its calls for that sweep (REQ-PUB-163).

The provider reads YouTube Analytics through a Google project whose daily
quota it shares between customers. Once the quota is spent every call is
refused until it resets, so a sweep that kept calling made one doomed request
and one warning per post.
"""

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from late import LateRateLimitError

from src.publisher.analytics import PostMetrics
from src.publisher.late.cli import (
    _with_youtube_retention,
    _YouTubeQuota,
    cmd_analytics,
)
from src.publisher.models import AnalyticsConfig, PublisherConfig

LEGS = [{"platform": "youtube", "platformPostId": "vid", "accountId": "acc"}]


class _Refusing:
    def __init__(self) -> None:
        self.calls = 0

    def get_you_tube_daily_views(self, **_: Any) -> Any:
        self.calls += 1
        raise LateRateLimitError("Rate limit exceeded")


class TestRetentionCall:
    def test_rate_limit_keeps_the_post_and_marks_the_quota(self):
        quota = _YouTubeQuota()
        measured = PostMetrics(post_id="p", platform_metrics={"youtube": {"v": 1.0}})

        out = _with_youtube_retention(_Refusing(), measured, LEGS, "", quota)

        assert out is measured
        assert quota.exhausted and quota.skipped == 0

    def test_after_the_refusal_no_further_call_is_made(self):
        resource, quota = _Refusing(), _YouTubeQuota()
        for post_id in ("a", "b", "c"):
            _with_youtube_retention(
                resource, PostMetrics(post_id=post_id), LEGS, "", quota
            )

        assert resource.calls == 1
        assert quota.skipped == 2

    def test_other_failures_do_not_end_the_calls(self):
        class Flaky:
            calls = 0

            def get_you_tube_daily_views(self, **_: Any) -> Any:
                Flaky.calls += 1
                raise ValueError("bad json")

        quota = _YouTubeQuota()
        for post_id in ("a", "b"):
            _with_youtube_retention(
                Flaky(), PostMetrics(post_id=post_id), LEGS, "", quota
            )

        assert Flaky.calls == 2
        assert not quota.exhausted


POSTS = [
    {
        "id": post_id,
        "platforms": [{"platform": "youtube", "publishedAt": "2026-07-01T08:00:00Z"}],
    }
    for post_id in ("a", "b", "c")
]
TIMELINE = {
    "timeline": [
        {"date": "2026-07-02", "views": 100, "platform": "youtube"},
        {"date": "2026-07-08", "views": 400, "platform": "youtube"},
    ]
}


@pytest.mark.asyncio
async def test_the_sweep_stops_calling_after_the_first_refusal(tmp_path: Path):
    resource = MagicMock()
    resource.get_post_timeline.return_value = TIMELINE
    resource.get_analytics.return_value = {"platformAnalytics": LEGS}
    resource.get_you_tube_daily_views.side_effect = LateRateLimitError("quota")
    publisher = MagicMock()
    publisher.authenticate = AsyncMock(return_value=True)
    publisher.list_posts = AsyncMock(return_value=POSTS)
    args = argparse.Namespace(
        limit=None,
        rank_only=False,
        by_arm=False,
        since=None,
        until=None,
        outputs_dir=tmp_path,
        debug=False,
    )
    config = PublisherConfig(
        provider="late",
        api_key="sk_live_key_12345",
        analytics_config=AnalyticsConfig(limit=50),
    )
    with (
        patch(
            "src.publisher.late.cli._create_publisher_from_config",
            return_value=publisher,
        ),
        patch("src.publisher.late.cli.timeline_resource", return_value=resource),
        patch("src.publisher.late.cli._sweep_audience"),
        patch(
            "src.publisher.late.cli.ANALYTICS_FAILURES_LOG",
            tmp_path / "failures.log",
        ),
    ):
        await cmd_analytics(args, config, MagicMock())

    assert resource.get_you_tube_daily_views.call_count == 1
    assert resource.get_analytics.call_count == 3
