"""Tests for day-N views and the durability ratio.

The scheduler's timeline is cumulative, so every figure here is a lookup rather
than a sum. Reading it as a per-day delta understates everything after the first
row, which is the mistake these tests exist to prevent.
"""

from datetime import datetime, timedelta

import pytest

from src.publisher.analytics import (
    PostMetrics,
    durability_ratio,
    normalize_timeline,
    rank_by_durability,
    summarize_post,
    views_at_day,
)

PUBLISHED = datetime(2026, 1, 1)


def _rows(*pairs):
    """Build API-shaped rows from (day_offset, cumulative_views)."""
    return [
        {"date": (PUBLISHED + timedelta(days=d)).isoformat(), "views": v}
        for d, v in pairs
    ]


@pytest.mark.unit
class TestNormalizeTimeline:
    def test_sorts_by_date(self):
        rows = _rows((7, 300), (1, 100), (3, 200))
        assert [v for _, v in normalize_timeline(rows)] == [100, 200, 300]

    def test_drops_rows_without_a_usable_figure(self):
        """A missing count is not zero.

        Defaulting it would be indistinguishable from a real zero and would drag
        a durability ratio toward nothing.
        """
        rows = _rows((1, 100)) + [{"date": "2026-01-05", "views": None}]
        assert len(normalize_timeline(rows)) == 1

    def test_tolerates_a_trailing_z(self):
        rows = [{"date": "2026-01-02T00:00:00Z", "views": 10}]
        assert normalize_timeline(rows) == [(datetime(2026, 1, 2), 10)]

    def test_empty_input_is_empty(self):
        assert normalize_timeline(None) == []
        assert normalize_timeline([]) == []


@pytest.mark.unit
class TestViewsAtDay:
    def test_reads_the_cumulative_value_not_a_sum(self):
        """273 / 283 / 288 on consecutive days is a total, not three days."""
        timeline = normalize_timeline(_rows((1, 273), (2, 283), (3, 288)))
        assert views_at_day(timeline, PUBLISHED, 2) == 283

    def test_takes_the_last_row_at_or_before_the_cutoff(self):
        """The timeline may skip days; the figure still holds on a gap."""
        timeline = normalize_timeline(_rows((1, 100), (5, 500)))
        assert views_at_day(timeline, PUBLISHED, 2) == 100

    def test_a_window_the_post_has_not_reached_is_unknown(self):
        """Not zero, and not the latest figure.

        Reporting the running total as a day-7 number would make every young
        post look like it finished its launch curve.
        """
        timeline = normalize_timeline(_rows((1, 100), (2, 150)))
        assert views_at_day(timeline, PUBLISHED, 7) is None

    def test_empty_timeline_is_unknown(self):
        assert views_at_day([], PUBLISHED, 2) is None


@pytest.mark.unit
class TestDurabilityRatio:
    def test_a_post_that_kept_earning_scores_above_one(self):
        timeline = normalize_timeline(_rows((30, 100), (90, 250)))
        assert durability_ratio(timeline, PUBLISHED) == pytest.approx(1.5)

    def test_a_post_that_stopped_scores_zero(self):
        """Distinct from unmeasurable: this one was measured and did not earn."""
        timeline = normalize_timeline(_rows((30, 100), (90, 100)))
        assert durability_ratio(timeline, PUBLISHED) == 0.0

    def test_a_post_younger_than_the_window_is_unknown(self):
        timeline = normalize_timeline(_rows((1, 100), (7, 400)))
        assert durability_ratio(timeline, PUBLISHED) is None

    def test_no_views_in_the_window_is_unknown_not_zero(self):
        """A ratio against zero is undefined.

        Returning 0.0 would rank a post nothing can be said about alongside one
        measured and found dead.
        """
        timeline = normalize_timeline(_rows((30, 0), (90, 50)))
        assert durability_ratio(timeline, PUBLISHED) is None

    def test_the_window_is_configurable(self):
        timeline = normalize_timeline(_rows((7, 100), (30, 300)))
        assert durability_ratio(timeline, PUBLISHED, window_days=7) == pytest.approx(
            2.0
        )


@pytest.mark.unit
class TestSummarizePost:
    def test_reduces_a_timeline_to_the_stored_figures(self):
        rows = _rows((2, 200), (7, 260), (30, 300), (60, 450))
        m = summarize_post("p1", PUBLISHED.isoformat(), rows)
        assert m.post_id == "p1"
        assert m.views_day_2 == 200
        assert m.views_day_7 == 260
        assert m.views_total == 450
        assert m.durability_ratio == pytest.approx(0.5)

    def test_an_unparseable_publish_date_yields_an_empty_record(self):
        """Better than guessing: every figure is relative to that date."""
        m = summarize_post("p1", "not-a-date", _rows((2, 200)))
        assert m.views_day_2 is None
        assert m.durability_ratio is None

    def test_a_post_with_no_timeline_yields_an_empty_record(self):
        m = summarize_post("p1", PUBLISHED.isoformat(), [])
        assert m.views_total is None

    def test_round_trips_through_a_dict(self):
        m = summarize_post("p1", PUBLISHED.isoformat(), _rows((2, 5)))
        assert m.to_dict()["post_id"] == "p1"


@pytest.mark.unit
class TestRankByDurability:
    def test_most_durable_first(self):
        posts = [
            PostMetrics("a", durability_ratio=0.2),
            PostMetrics("b", durability_ratio=1.4),
            PostMetrics("c", durability_ratio=0.9),
        ]
        assert [m.post_id for m in rank_by_durability(posts)] == ["b", "c", "a"]

    def test_unmeasurable_posts_sort_last_rather_than_as_zero(self):
        """A post too young to score is not a post that scored badly."""
        posts = [
            PostMetrics("young", durability_ratio=None),
            PostMetrics("dead", durability_ratio=0.0),
            PostMetrics("good", durability_ratio=2.0),
        ]
        assert [m.post_id for m in rank_by_durability(posts)] == [
            "good",
            "dead",
            "young",
        ]


@pytest.mark.unit
class TestMetricsStorage:
    def test_round_trips(self, tmp_path):
        from src.publisher.analytics import load_metrics, save_metrics

        save_metrics([PostMetrics("p1", durability_ratio=1.2)], tmp_path)
        loaded = load_metrics(tmp_path)
        assert [m.post_id for m in loaded] == ["p1"]
        assert loaded[0].durability_ratio == pytest.approx(1.2)

    def test_a_second_reading_replaces_the_first(self, tmp_path):
        """A post measured again keeps one row, carrying the later reading.

        Appending would double-count it in any ranking, and the newer reading is
        the one with more history behind it.
        """
        from src.publisher.analytics import load_metrics, save_metrics

        save_metrics([PostMetrics("p1", views_total=100)], tmp_path)
        save_metrics([PostMetrics("p1", views_total=400)], tmp_path)
        loaded = load_metrics(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].views_total == 400

    def test_other_posts_survive_a_partial_write(self, tmp_path):
        from src.publisher.analytics import load_metrics, save_metrics

        save_metrics([PostMetrics("a"), PostMetrics("b")], tmp_path)
        save_metrics([PostMetrics("a", views_total=7)], tmp_path)
        assert {m.post_id for m in load_metrics(tmp_path)} == {"a", "b"}

    def test_missing_file_reads_as_empty(self, tmp_path):
        from src.publisher.analytics import load_metrics

        assert load_metrics(tmp_path) == []

    def test_malformed_file_reads_as_empty_rather_than_raising(self, tmp_path):
        from src.publisher.analytics import load_metrics, metrics_path

        metrics_path(tmp_path).write_text("{not json", encoding="utf-8")
        assert load_metrics(tmp_path) == []
