"""Account-level audience figures: follower history, YouTube watch figures,
Instagram profile views, and the range report (REQ-PUB-162 to REQ-PUB-165).
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from src.publisher.analytics import PostMetrics
from src.publisher.audience import (
    audience_path,
    daily_totals,
    follower_history,
    instagram_profile_day,
    instagram_profile_views_per_1k,
    load_audience,
    median_view_percentage,
    merge_audience,
    net_follows,
    platform_rates,
    save_audience,
    summed_rates,
    sweep_audience,
    youtube_leg,
    youtube_retention,
)

pytestmark = pytest.mark.req("REQ-PUB-162", "REQ-PUB-163", "REQ-PUB-164", "REQ-PUB-165")

D = date(2026, 1, 10)


def _followers_raw() -> dict[str, Any]:
    return {
        "accounts": [
            {"_id": "a1", "platform": "tiktok"},
            {"_id": "a2", "platform": "instagram"},
        ],
        "stats": {
            "a1": [
                {"date": "2026-01-09", "followers": 100},
                {"date": "2026-01-12", "followers": 106},
            ],
            "a2": [
                {"date": "2026-01-09", "followers": 50},
                {"date": "2026-01-12", "followers": None},
            ],
            "zz": [{"date": "2026-01-09", "followers": 1}],
        },
    }


def _daily_raw() -> dict[str, Any]:
    return {
        "dailyData": [
            {
                "date": "2026-01-10",
                "platformMetrics": {
                    "tiktok": {"views": 1000, "comments": 2, "shares": 3, "saves": 1},
                    "instagram": {"views": 500, "comments": 1, "shares": 0, "saves": 0},
                },
            },
            {
                "date": "2026-01-12",
                "platformMetrics": {
                    "tiktok": {"views": 1000, "comments": 0, "shares": 0, "saves": 0}
                },
            },
        ]
    }


class TestParsers:
    def test_follower_history_maps_accounts_to_platforms(self):
        # REQ-PUB-162: a day with no count stays absent, not 0.
        assert follower_history(_followers_raw()) == {
            "tiktok": {"2026-01-09": 100, "2026-01-12": 106},
            "instagram": {"2026-01-09": 50},
        }

    def test_unexposed_fields_are_dropped_not_zero(self):
        # The provider zero-fills fields a platform doesn't expose.
        raw = {
            "dailyData": [
                {
                    "date": "2026-01-10",
                    "platformMetrics": {
                        "youtube": {"views": 9, "comments": 1, "shares": 0, "saves": 0},
                        "tiktok": {"views": 9, "shares": 2, "saves": 0},
                    },
                }
            ]
        }
        out = daily_totals(raw)
        assert out["youtube"]["2026-01-10"] == {"views": 9, "comments": 1}
        assert out["tiktok"]["2026-01-10"] == {"views": 9, "shares": 2}
        store = {"daily": out}
        assert platform_rates(store, "youtube", D, D).shares_saves_per_1k is None

    def test_a_platform_without_a_field_list_keeps_its_figures(self):
        raw = {
            "dailyData": [
                {
                    "date": "2026-01-10",
                    "platformMetrics": {
                        "facebook": {"views": 500, "comments": 2, "shares": 0}
                    },
                }
            ]
        }
        assert daily_totals(raw)["facebook"]["2026-01-10"] == {
            "views": 500,
            "comments": 2,
            "shares": 0,
        }

    def test_two_accounts_on_one_platform_add_up(self):
        raw = {
            "accounts": [
                {"_id": "a", "platform": "tiktok"},
                {"_id": "b", "platform": "tiktok"},
            ],
            "stats": {
                "a": [{"date": "2026-01-09", "followers": 10}],
                "b": [{"date": "2026-01-09", "followers": 5}],
            },
        }
        assert follower_history(raw) == {"tiktok": {"2026-01-09": 15}}

    def test_daily_totals_by_platform_and_day(self):
        out = daily_totals(_daily_raw())
        assert out["tiktok"]["2026-01-10"]["views"] == 1000
        assert out["instagram"]["2026-01-10"]["comments"] == 1
        assert "2026-01-12" not in out["instagram"]

    def test_instagram_profile_day(self):
        raw = {
            "metrics": {
                "profile_views": {"total": 4},
                "profile_links_taps": {"total": 0},
            }
        }
        assert instagram_profile_day(raw) == {
            "profile_views": 4,
            "profile_links_taps": 0,
        }
        assert instagram_profile_day({"metrics": {}}) == {}
        assert instagram_profile_day(None) == {}

    def test_youtube_leg(self):
        legs = [
            {"platform": "tiktok", "platformPostId": "t", "accountId": "x"},
            {
                "platform": "youtube",
                "platformPostId": "vid",
                "accountId": {"_id": "acc"},
            },
        ]
        assert youtube_leg(legs) == ("vid", "acc")
        assert youtube_leg([{"platform": "youtube", "platformPostId": None}]) is None
        assert youtube_leg(None) is None


class TestYoutubeRetention:
    def test_view_weighted(self):
        # REQ-PUB-163: weighted by each day's views, not a mean of the days.
        rows = [
            {
                "views": 30,
                "averageViewPercentage": 50,
                "averageViewDuration": 10,
                "subscribersGained": 1,
            },
            {
                "views": 10,
                "averageViewPercentage": 10,
                "averageViewDuration": 2,
                "subscribersGained": 0,
            },
            {
                "views": 0,
                "averageViewPercentage": 0,
                "averageViewDuration": 0,
                "subscribersGained": 0,
            },
        ]
        assert youtube_retention(rows) == {
            "avgViewPercentage": 40.0,
            "avgViewDurationSec": 8.0,
            "subscribersGained": 1.0,
        }

    @pytest.mark.parametrize(
        "rows", [[], None, [{"views": 0, "averageViewPercentage": 0}]]
    )
    def test_no_views_is_unknown_not_zero(self, rows):
        assert youtube_retention(rows) == {
            "avgViewPercentage": None,
            "avgViewDurationSec": None,
            "subscribersGained": None,
        }

    def test_a_viewed_day_without_the_figure_is_unknown(self):
        rows = [{"views": 5, "averageViewDuration": 3, "subscribersGained": 0}]
        assert youtube_retention(rows)["avgViewPercentage"] is None


class TestStore:
    def test_merge_overlays_and_keeps_stored_days(self):
        stored = {"followers": {"tiktok": {"2026-01-01": 90, "2026-01-02": 91}}}
        fresh = {"followers": {"tiktok": {"2026-01-02": 92, "2026-01-03": 93}}}
        assert merge_audience(stored, fresh) == {
            "followers": {
                "tiktok": {"2026-01-01": 90, "2026-01-02": 92, "2026-01-03": 93}
            }
        }

    def test_save_merges_with_the_file(self, tmp_path: Path):
        save_audience({"followers": {"tiktok": {"2026-01-01": 1}}}, tmp_path)
        save_audience({"followers": {"tiktok": {"2026-01-02": 2}}}, tmp_path)
        assert load_audience(tmp_path)["followers"]["tiktok"] == {
            "2026-01-01": 1,
            "2026-01-02": 2,
        }

    def test_unreadable_file_is_refused_not_overwritten(self, tmp_path: Path):
        path = audience_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{broken")
        with pytest.raises(OSError, match="Refusing"):
            save_audience({"followers": {}}, tmp_path)
        assert path.read_text() == "{broken"


class _Fake:
    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.insight_calls: list[str] = []

    def get_follower_stats(self, **_: Any) -> Any:
        if "followers" in self.fail:
            raise ValueError("down")
        return _followers_raw()

    def get_daily_metrics(self, **_: Any) -> Any:
        return _daily_raw()

    def get_instagram_account_insights(
        self, *, account_id: str, since: str, **_: Any
    ) -> Any:
        self.insight_calls.append(f"{account_id}:{since}")
        return {
            "metrics": {
                "profile_views": {"total": 2},
                "profile_links_taps": {"total": 0},
            }
        }


def _guard(label: str, call: Any) -> Any:
    try:
        return call()
    except ValueError:
        return None


class TestSweep:
    def test_reads_every_source(self):
        # REQ-PUB-162, REQ-PUB-164.
        fake = _Fake()
        fresh = sweep_audience(fake, fake, D, _guard)
        assert set(fresh) == {"followers", "daily", "instagram_profile"}
        assert len(fake.insight_calls) == 7
        assert fake.insight_calls[0] == "a2:2026-01-03"
        assert {c.split(":")[0] for c in fake.insight_calls} == {"a2"}
        assert fresh["instagram_profile"]["2026-01-09"]["profile_views"] == 2

    def test_one_source_failing_keeps_the_rest(self):
        # Without follower stats there is no Instagram account id to ask about.
        fake = _Fake(fail={"followers"})
        fresh = sweep_audience(fake, fake, D, _guard)
        assert set(fresh) == {"daily"}


def _store() -> dict[str, Any]:
    return {
        "followers": follower_history(_followers_raw()),
        "daily": daily_totals(_daily_raw()),
        "instagram_profile": {
            "2026-01-10": {"profile_views": 2},
            "2026-01-11": {"profile_views": 0},
            "2026-01-12": {"profile_views": 3},
        },
    }


class TestReport:
    def test_net_follows_needs_both_ends(self):
        history = {"2026-01-09": 100, "2026-01-12": 106}
        assert net_follows(history, D, date(2026, 1, 12)) == 6
        assert net_follows(history, D, date(2026, 1, 11)) is None

    def test_platform_rates(self):
        # REQ-PUB-165.
        r = platform_rates(_store(), "tiktok", D, date(2026, 1, 12))
        assert (r.views, r.net_follows, r.follows_per_1k) == (2000, 6, 3.0)
        # TikTok exposes shares but not saves, so its saves are dropped.
        assert (r.shares_saves_per_1k, r.comments_per_1k) == (1.5, 1.0)

    def test_unknown_inputs_give_unknown_rates(self):
        r = platform_rates(_store(), "instagram", D, date(2026, 1, 12))
        assert r.net_follows is None and r.follows_per_1k is None
        assert r.views == 500
        empty = platform_rates(_store(), "youtube", D, date(2026, 1, 12))
        assert empty.views is None and empty.comments_per_1k is None

    def test_summed_is_unknown_when_a_platform_is(self):
        store = _store()
        both = summed_rates(store, ["tiktok", "instagram"], D, date(2026, 1, 12))
        assert (
            both.views == 2500
            and both.net_follows is None
            and both.follows_per_1k is None
        )
        only = summed_rates(store, ["tiktok"], D, date(2026, 1, 12))
        assert only.follows_per_1k == 3.0

    def test_median_view_percentage_counts_posts_in_range(self):
        def post(day: str, pct: float | None) -> PostMetrics:
            return PostMetrics(
                post_id=day,
                published_at=f"{day}T10:00:00Z",
                platform_metrics={"youtube": {"avgViewPercentage": pct}},
            )

        posts = [
            post("2026-01-10", 20),
            post("2026-01-11", 40),
            post("2026-01-11", None),
            post("2026-02-01", 90),
        ]
        assert median_view_percentage(posts, D, date(2026, 1, 12)) == 30.0
        assert median_view_percentage(posts, date(2026, 3, 1), date(2026, 3, 2)) is None

    def test_profile_views_over_part_of_the_range_are_unknown(self):
        # Partial days over the whole range's views would read low.
        assert (
            instagram_profile_views_per_1k(
                _store(), date(2026, 1, 9), date(2026, 1, 12)
            )
            is None
        )

    def test_instagram_profile_views_per_1k(self):
        assert instagram_profile_views_per_1k(_store(), D, date(2026, 1, 12)) == 10.0
        assert (
            instagram_profile_views_per_1k(_store(), date(2026, 3, 1), date(2026, 3, 2))
            is None
        )


class TestCli:
    def test_youtube_retention_lands_in_platform_metrics(self):
        # REQ-PUB-163: merged into the post's youtube row beside the counts.
        from src.publisher.late.cli import _with_youtube_retention

        class Resource:
            def get_you_tube_daily_views(self, **kwargs: Any) -> Any:
                assert (
                    kwargs["video_id"] == "vid" and kwargs["start_date"] == "2026-01-10"
                )
                return {
                    "dailyViews": [
                        {
                            "views": 10,
                            "averageViewPercentage": 30,
                            "averageViewDuration": 6,
                            "subscribersGained": 1,
                        }
                    ]
                }

        measured = PostMetrics(
            post_id="p", platform_metrics={"youtube": {"views": 10.0}}
        )
        legs = [{"platform": "youtube", "platformPostId": "vid", "accountId": "acc"}]
        out = _with_youtube_retention(
            Resource(), measured, legs, "2026-01-10T08:00:00Z"
        )
        assert out.platform_metrics["youtube"] == {
            "views": 10.0,
            "avgViewPercentage": 30.0,
            "avgViewDurationSec": 6.0,
            "subscribersGained": 1.0,
        }

    def test_youtube_call_failing_keeps_the_post(self):
        from src.publisher.late.cli import _with_youtube_retention

        class Resource:
            def get_you_tube_daily_views(self, **_: Any) -> Any:
                raise ValueError("bad json")

        measured = PostMetrics(post_id="p")
        legs = [{"platform": "youtube", "platformPostId": "vid", "accountId": "acc"}]
        assert _with_youtube_retention(Resource(), measured, legs, "") is measured

    def test_audience_command_prints_rates(self, tmp_path: Path, caplog):
        from src.publisher.late.cli import cmd_audience

        save_audience(_store(), tmp_path)
        args = argparse.Namespace(
            since=datetime(2026, 1, 10),
            until=datetime(2026, 1, 12),
            outputs_dir=tmp_path,
        )
        with caplog.at_level("INFO", logger="src.publisher.late.cli"):
            cmd_audience(args)
        lines = [r.getMessage() for r in caplog.records]
        tiktok = next(line for line in lines if line.startswith("tiktok"))
        assert tiktok.split() == ["tiktok", "2000", "6", "3", "1.5", "1"]
        assert any("Instagram profile views per 1k views: 10" in line for line in lines)

        assert json.loads(audience_path(tmp_path).read_text())["followers"]

    def test_large_counts_print_whole(self, tmp_path: Path, caplog):
        from src.publisher.late.cli import cmd_audience

        store = {"daily": {"tiktok": {"2026-01-10": {"views": 1234567.0}}}}
        save_audience(store, tmp_path)
        args = argparse.Namespace(
            since=datetime(2026, 1, 10),
            until=datetime(2026, 1, 10),
            outputs_dir=tmp_path,
        )
        with caplog.at_level("INFO", logger="src.publisher.late.cli"):
            cmd_audience(args)
        assert any(
            r.getMessage().split()[:2] == ["tiktok", "1234567"] for r in caplog.records
        )
