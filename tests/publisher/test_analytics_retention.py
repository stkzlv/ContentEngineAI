"""Tests that a truncated re-read cannot erase figures already captured.

The provider's timeline does not reach back indefinitely. Measured against the
live API on 2026-08-23, posts aged 121, 157 and 188 days all returned rows
starting 2026-07-16/17 -- the same recent date -- and passing `from_date`
changed nothing. So re-reading an old post yields no day-2, no day-7, no ratio,
and a `views_total` counted from the middle of its life.

The stored row previously took the newer reading whenever it carried any figure
at all, and a truncated read carries `views_total`. So every sweep quietly
replaced captured launch figures with unmeasurable ones -- for exactly the
posts a durability comparison exists to read.
"""

import json

import pytest

from src.publisher.analytics import (
    PostMetrics,
    load_metrics,
    metrics_path,
    publish_time,
    save_metrics,
)


def _stored(tmp_path, **kw):
    save_metrics([PostMetrics(**kw)], tmp_path)


@pytest.mark.unit
class TestATruncatedRereadKeepsWhatWasMeasured:
    def test_launch_figures_survive_a_window_that_no_longer_reaches_them(
        self, tmp_path
    ):
        _stored(
            tmp_path,
            post_id="p1",
            published_at="2026-04-24",
            views_day_2=1200,
            views_day_7=3400,
            views_total=9000,
            durability_ratio=1.4,
            timeline_end="2026-06-01",
        )

        # What the API returns for the same post months later.
        save_metrics(
            [
                PostMetrics(
                    post_id="p1",
                    published_at="2026-04-24",
                    views_total=500,
                    timeline_end="2026-08-23",
                )
            ],
            tmp_path,
        )

        m = {x.post_id: x for x in load_metrics(tmp_path)}["p1"]
        assert m.views_day_2 == 1200
        assert m.views_day_7 == 3400
        assert m.durability_ratio == 1.4

    def test_a_truncated_total_does_not_replace_a_full_one(self, tmp_path):
        """Views only accumulate, so a smaller figure is evidence of a
        shorter window, not of a post losing views.
        """
        _stored(tmp_path, post_id="p1", views_total=9000)
        save_metrics([PostMetrics(post_id="p1", views_total=500)], tmp_path)

        assert load_metrics(tmp_path)[0].views_total == 9000

    def test_a_real_new_figure_still_lands(self, tmp_path):
        """The guard must not freeze the record."""
        _stored(tmp_path, post_id="p1", views_day_2=1200, views_total=9000)
        save_metrics(
            [PostMetrics(post_id="p1", views_total=11000, durability_ratio=0.8)],
            tmp_path,
        )

        m = load_metrics(tmp_path)[0]
        assert m.views_total == 11000
        assert m.durability_ratio == 0.8
        assert m.views_day_2 == 1200

    def test_timeline_end_stays_with_the_ratio_it_dates(self, tmp_path):
        """The field exists to say how mature a ratio is.

        Advancing it while keeping an older ratio reports a barely-mature
        reading as a fully-mature one -- which is precisely the comparison it
        was added to make possible.
        """
        _stored(
            tmp_path,
            post_id="p1",
            durability_ratio=1.4,
            timeline_end="2026-06-01",
        )
        save_metrics(
            [PostMetrics(post_id="p1", views_total=99, timeline_end="2026-08-23")],
            tmp_path,
        )

        m = load_metrics(tmp_path)[0]
        assert m.durability_ratio == 1.4
        assert m.timeline_end == "2026-06-01"

    def test_a_fresh_ratio_brings_its_own_date(self, tmp_path):
        _stored(
            tmp_path,
            post_id="p1",
            durability_ratio=1.4,
            timeline_end="2026-06-01",
        )
        save_metrics(
            [
                PostMetrics(
                    post_id="p1", durability_ratio=0.9, timeline_end="2026-08-23"
                )
            ],
            tmp_path,
        )

        m = load_metrics(tmp_path)[0]
        assert m.durability_ratio == 0.9
        assert m.timeline_end == "2026-08-23"

    def test_an_unseen_post_is_added(self, tmp_path):
        _stored(tmp_path, post_id="p1", views_total=10)
        save_metrics([PostMetrics(post_id="p2", views_total=20)], tmp_path)

        assert {m.post_id for m in load_metrics(tmp_path)} == {"p1", "p2"}


@pytest.mark.unit
class TestDayNIsClockedFromTheRealPublishTime:
    """A leg that failed and was retried goes live after its slot.

    Measuring from `scheduledFor` then starts the clock before the video
    existed. Retry rate is not random with respect to content format, so it
    tilts a format comparison rather than adding noise.
    """

    def test_a_legs_publish_time_wins_over_the_slot(self):
        post = {
            "scheduledFor": "2026-08-01T04:00:00Z",
            "platforms": [
                {"platform": "youtube", "publishedAt": "2026-08-04T09:12:00Z"}
            ],
        }
        assert publish_time(post) == "2026-08-04T09:12:00Z"

    def test_the_earliest_leg_wins_when_they_differ(self):
        """That is when the content first reached anyone."""
        post = {
            "scheduledFor": "2026-08-01T04:00:00Z",
            "platforms": [
                {"platform": "tiktok", "publishedAt": "2026-08-05T10:00:00Z"},
                {"platform": "youtube", "publishedAt": "2026-08-02T10:00:00Z"},
            ],
        }
        assert publish_time(post) == "2026-08-02T10:00:00Z"

    def test_the_slot_is_the_fallback(self):
        post = {
            "scheduledFor": "2026-08-01T04:00:00Z",
            "platforms": [{"platform": "x"}],
        }
        assert publish_time(post) == "2026-08-01T04:00:00Z"

    def test_no_information_at_all_is_empty_not_an_error(self):
        assert publish_time({}) == ""


@pytest.mark.unit
class TestTheRowsAreLifetimeCumulative:
    """The premise the merge rests on, pinned.

    Measured live on 2026-08-23: a post 121 days old read 308 views at both
    its first retained row and its last, and one 188 days old read 354 at
    both. Rows carry total views as of their date, not views within the
    window, so a truncated read's `views_total` is the true lifetime figure.

    If that ever stopped holding, a truncated read would start returning
    *wrong* numbers rather than absent ones, and the merge would be
    preserving the wrong thing.
    """

    def _rows(self, published, first_day, last_day):
        from datetime import timedelta

        return [
            {
                "date": (published + timedelta(days=d)).strftime("%Y-%m-%d"),
                "views": 900 if d < 5 else (1000 if d < 30 else 1000 + (d - 30) * 30),
            }
            for d in range(first_day, last_day + 1)
        ]

    def test_a_truncated_window_returns_absence_not_a_wrong_number(self):
        from datetime import datetime

        from src.publisher.analytics import summarize_post

        published = datetime(2026, 6, 1)
        full = summarize_post("p", published.isoformat(), self._rows(published, 0, 40))
        truncated = summarize_post(
            "p", published.isoformat(), self._rows(published, 35, 40)
        )

        assert (full.views_day_2, full.views_day_7) == (900, 1000)
        assert full.durability_ratio == pytest.approx(0.3)

        # The window starts past both launch marks, so they are unavailable --
        # not small, not stale, absent.
        assert truncated.views_day_2 is None
        assert truncated.views_day_7 is None
        assert truncated.durability_ratio is None
        # And the lifetime total is unaffected by where the window began.
        assert truncated.views_total == full.views_total

    def test_the_merge_recovers_the_full_row(self, tmp_path):
        from datetime import datetime

        from src.publisher.analytics import load_metrics, save_metrics, summarize_post

        tmp = tmp_path
        published = datetime(2026, 6, 1)
        save_metrics(
            [summarize_post("p", published.isoformat(), self._rows(published, 0, 40))],
            tmp,
        )
        save_metrics(
            [summarize_post("p", published.isoformat(), self._rows(published, 35, 40))],
            tmp,
        )

        m = load_metrics(tmp)[0]
        assert (m.views_day_2, m.views_day_7) == (900, 1000)
        assert m.durability_ratio == pytest.approx(0.3)


@pytest.mark.unit
class TestPublishTimeOrdersByInstant:
    def test_offsets_do_not_reorder_the_legs(self):
        """Lexical order on the printed string picks the wrong leg when the
        offsets differ; 09:12+02:00 is earlier than 08:30Z.
        """
        post = {
            "platforms": [
                {"publishedAt": "2026-08-04T09:12:00+02:00"},
                {"publishedAt": "2026-08-04T08:30:00+00:00"},
            ]
        }
        assert publish_time(post) == "2026-08-04T09:12:00+02:00"

    def test_datetime_values_are_accepted(self):
        """The SDK types this field as an aware datetime, not a string."""
        from datetime import UTC, datetime

        post = {
            "platforms": [
                {"publishedAt": datetime(2026, 8, 5, 10, tzinfo=UTC)},
                {"publishedAt": datetime(2026, 8, 2, 10, tzinfo=UTC)},
            ]
        }
        assert publish_time(post).startswith("2026-08-02")


@pytest.mark.unit
class TestTheClientCarriesTheLegPublishTime:
    """The normalizer is the other half of the day-N fix, and it fails quietly.

    `list_posts` builds its own dict from the SDK objects. It dropped
    `publishedAt` entirely, so nothing downstream could have used it and the
    scheduled slot was the only value left. If the accessor ever stops
    resolving -- the SDK aliasing the field, say -- every leg reads None, the
    clock silently reverts to the slot, and no line logs.
    """

    @pytest.mark.asyncio
    async def test_the_normalized_leg_carries_publishedat(self, monkeypatch):
        from datetime import UTC, datetime
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from src.publisher.late.client import LatePublisher

        leg = SimpleNamespace(
            platform="youtube",
            field_id="acc_1",
            publishedAt=datetime(2026, 8, 4, 9, 12, tzinfo=UTC),
        )
        post = SimpleNamespace(
            field_id="p1",
            status="published",
            scheduledFor="2026-08-01T04:00:00Z",
            platforms=[leg],
        )

        pub = LatePublisher(api_key="sk_live_test")
        monkeypatch.setattr(
            pub,
            "_posts_list_safe",
            AsyncMock(return_value=SimpleNamespace(posts=[post])),
        )

        posts = await pub.list_posts(status="published")

        assert posts, "no posts returned"
        legs = posts[0]["platforms"]
        assert (
            legs and legs[0]["publishedAt"] is not None
        ), "the normalizer dropped the leg publish time"
        assert publish_time(posts[0]).startswith("2026-08-04")
