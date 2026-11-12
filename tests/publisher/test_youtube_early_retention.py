"""The YouTube early-retention read (REQ-PUB-084).

Once a video is old enough the sweep reads its retention curve once and
stores the share still watching at the configured second, with YouTube's
relative score there; an unusable curve is unknown, never 0.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from late import LateRateLimitError

from src.publisher.analytics import PostMetrics, segment_quality
from src.publisher.audience import youtube_early_retention
from src.publisher.late.cli import _with_youtube_early_retention, _YouTubeQuota
from src.publisher.models import AnalyticsConfig

pytestmark = pytest.mark.req("REQ-PUB-084")

LEGS = [{"platform": "youtube", "platformPostId": "vid", "accountId": "acc"}]
UNKNOWN = {"earlyWatchingPct": None, "earlyRelativeRetention": None}


def _curve(duration: Any = 30, points: Any = None) -> dict[str, Any]:
    if points is None:
        points = [
            {
                "elapsedVideoTimeRatio": r / 100,
                "audienceWatchRatio": 1.2 - r / 100,
                "relativeRetentionPerformance": 0.4,
            }
            for r in range(1, 101)
        ]
    return {"durationSeconds": duration, "retentionCurve": points}


def _days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT10:00:00Z")


class TestReading:
    def test_reads_the_first_point_at_or_after_the_second(self):
        # 3 s of a 30 s video is ratio 0.10: the point closing that interval.
        assert youtube_early_retention(_curve(), 3.0) == {
            "earlyWatchingPct": 110.0,
            "earlyRelativeRetention": 0.4,
        }

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            _curve(duration=None),
            _curve(duration=0),
            _curve(points=[]),
            # The curve skips from 0.02 to 0.50, far past 3 s of 30.
            _curve(
                points=[
                    {"elapsedVideoTimeRatio": 0.02, "audienceWatchRatio": 1.0},
                    {"elapsedVideoTimeRatio": 0.5, "audienceWatchRatio": 0.4},
                ]
            ),
        ],
    )
    def test_an_unusable_curve_is_unknown(self, raw):
        assert youtube_early_retention(raw, 3.0) == UNKNOWN

    def test_a_video_shorter_than_the_second_reads_its_last_point(self):
        points = [{"elapsedVideoTimeRatio": 1.0, "audienceWatchRatio": 0.5}]
        out = youtube_early_retention(_curve(duration=2, points=points), 3.0)
        assert out["earlyWatchingPct"] == 50.0


class _Resource:
    def __init__(self, raw: Any = None, error: Exception | None = None) -> None:
        self.raw = _curve() if raw is None else raw
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def get_you_tube_video_retention(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.raw


def _read(resource, published, stored=None, quota=None, settings=None):
    measured = PostMetrics(post_id="p", platform_metrics={"youtube": {"views": 9.0}})
    return _with_youtube_early_retention(
        resource,
        measured,
        LEGS,
        published,
        stored or {},
        settings or AnalyticsConfig(),
        quota or _YouTubeQuota(),
    )


class TestSweepStep:
    def test_a_settled_video_gets_the_reading_beside_its_figures(self):
        resource = _Resource()
        out = _read(resource, _days_ago(8))

        assert out.platform_metrics["youtube"] == {
            "views": 9.0,
            "earlyWatchingPct": 110.0,
            "earlyRelativeRetention": 0.4,
        }
        call = resource.calls[0]
        assert call["video_id"] == "vid" and call["account_id"] == "acc"
        assert call["start_date"] == _days_ago(8)[:10]

    def test_a_young_video_is_not_read(self):
        resource = _Resource()
        out = _read(resource, _days_ago(6))

        assert resource.calls == []
        assert "earlyWatchingPct" not in out.platform_metrics["youtube"]

    def test_the_age_comes_from_the_config(self):
        resource = _Resource()
        _read(
            resource,
            _days_ago(3),
            settings=AnalyticsConfig(retention_curve_min_age_days=3),
        )
        assert len(resource.calls) == 1

    def test_a_stored_reading_is_not_read_again(self):
        resource = _Resource()
        stored = {
            "p": PostMetrics(
                post_id="p", platform_metrics={"youtube": {"earlyWatchingPct": 80.0}}
            )
        }
        out = _read(resource, _days_ago(10), stored=stored)

        assert resource.calls == []
        assert "earlyWatchingPct" not in out.platform_metrics["youtube"]

    def test_a_stored_unknown_is_read_again(self):
        resource = _Resource()
        stored = {
            "p": PostMetrics(
                post_id="p", platform_metrics={"youtube": {"earlyWatchingPct": None}}
            )
        }
        _read(resource, _days_ago(10), stored=stored)
        assert len(resource.calls) == 1

    def test_a_rate_limit_ends_the_youtube_calls(self):
        quota = _YouTubeQuota()
        resource = _Resource(error=LateRateLimitError("Rate limit exceeded"))
        out = _read(resource, _days_ago(8), quota=quota)

        assert quota.exhausted
        assert "earlyWatchingPct" not in out.platform_metrics["youtube"]
        _read(resource, _days_ago(8), quota=quota)
        assert len(resource.calls) == 1

    def test_another_failure_keeps_the_post(self):
        out = _read(_Resource(error=ValueError("bad json")), _days_ago(8))
        assert out.platform_metrics["youtube"] == {"views": 9.0}

    def test_no_youtube_leg_makes_no_call(self):
        resource = _Resource()
        measured = PostMetrics(post_id="p")
        out = _with_youtube_early_retention(
            resource,
            measured,
            [{"platform": "tiktok"}],
            _days_ago(9),
            {},
            AnalyticsConfig(),
            _YouTubeQuota(),
        )
        assert out is measured and resource.calls == []


def test_the_report_segments_the_youtube_readings():
    metrics = [
        PostMetrics(
            post_id="a",
            platform_metrics={
                "youtube": {
                    "avgViewPercentage": 60.0,
                    "earlyWatchingPct": 90.0,
                    "earlyRelativeRetention": 0.25,
                }
            },
        ),
        PostMetrics(
            post_id="b",
            platform_metrics={
                "youtube": {"earlyWatchingPct": 70.0, "earlyRelativeRetention": None}
            },
        ),
    ]
    lines = segment_quality(
        metrics,
        {"a": "A", "b": "B"},
        {"A": {"content_format": "x"}, "B": {"content_format": "x"}},
    )

    assert lines == [
        "content_format=x [2 post(s)]: youtube.avgViewPercentage=60.0(n=1) "
        "youtube.earlyWatchingPct=80.0(n=2) youtube.earlyRelativeRetention=0.25(n=1)"
    ]


@pytest.mark.parametrize(
    "field, value",
    [
        ("retention_curve_min_age_days", 0),
        ("retention_curve_min_age_days", 2.5),
        ("retention_curve_min_age_days", True),
        ("early_retention_sec", 0),
        ("early_retention_sec", "3"),
    ],
)
def test_bad_settings_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        AnalyticsConfig(**{field: value})
