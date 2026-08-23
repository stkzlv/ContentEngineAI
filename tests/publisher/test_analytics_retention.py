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

    def test_the_furthest_timeline_end_is_kept(self, tmp_path):
        _stored(tmp_path, post_id="p1", timeline_end="2026-08-23")
        save_metrics([PostMetrics(post_id="p1", timeline_end="2026-06-01")], tmp_path)

        assert load_metrics(tmp_path)[0].timeline_end == "2026-08-23"

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
