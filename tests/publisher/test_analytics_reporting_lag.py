"""A day-N figure counts every platform or none.

A platform's first timeline row carries its lifetime total to that date, not
that day's increment, and platforms start reporting on their own lag. A figure
taken before a leg started counts only part of the post while `views_total`
counts all of it, so the two describe different posts.

The reach test ranks arms on median day-7 views. A post understated by
reporting lag would rank below an identical one for a reason that is not
reach, so the figure is reported unknown rather than small — the same rule
`views_at_day` already applies to a window the timeline has not reached.
"""

from datetime import UTC, datetime, timedelta

import pytest

from src.publisher.analytics import (
    first_report_dates,
    summarize_post,
)

PUBLISHED = datetime(2026, 6, 29, 8, 0, tzinfo=UTC)


def _row(platform: str, day: int, views: int) -> dict:
    return {
        "platform": platform,
        "date": (PUBLISHED + timedelta(days=day)).isoformat(),
        "views": views,
    }


@pytest.mark.unit
class TestFirstReportDates:
    def test_earliest_date_per_platform(self):
        rows = [
            _row("youtube", 1, 10),
            _row("youtube", 5, 40),
            _row("tiktok", 17, 326),
        ]
        first = first_report_dates(rows)
        # Normalised to naive, the way every timeline date is.
        assert first["youtube"] == (PUBLISHED + timedelta(days=1)).replace(tzinfo=None)
        assert first["tiktok"] == (PUBLISHED + timedelta(days=17)).replace(tzinfo=None)

    def test_rows_without_a_usable_figure_are_ignored(self):
        rows = [_row("youtube", 1, 10), {"platform": "tiktok", "date": None}]
        assert set(first_report_dates(rows)) == {"youtube"}

    def test_no_rows_is_empty(self):
        assert first_report_dates([]) == {}
        assert first_report_dates(None) == {}


@pytest.mark.unit
class TestDayNBlanksOnALaggingLeg:
    def test_the_live_example_from_the_report(self):
        # Post 6a3c42e56c05e44a97f5072a: TikTok's first row is day 17,
        # carrying 326 lifetime views. Day-2 and day-7 previously read 701
        # and 702 — YouTube only — while views_total counted TikTok too.
        rows = [
            _row("youtube", 1, 600),
            _row("youtube", 2, 701),
            _row("youtube", 7, 702),
            _row("tiktok", 17, 326),
            _row("youtube", 17, 747),
        ]
        m = summarize_post("p", PUBLISHED, rows)
        assert m.views_day_2 is None
        assert m.views_day_7 is None
        # The total is still correct: it counts every leg.
        assert m.views_total == 1073

    def test_a_leg_reporting_before_the_cutoff_does_not_blank_it(self):
        rows = [
            _row("youtube", 1, 600),
            _row("tiktok", 1, 200),
            _row("youtube", 2, 700),
            _row("tiktok", 2, 300),
            _row("youtube", 7, 750),
        ]
        m = summarize_post("p", PUBLISHED, rows)
        # Day 2 has both legs: 700 + 300.
        assert m.views_day_2 == 1000
        # Day 7 carries TikTok forward at its last known figure.
        assert m.views_day_7 == 1050

    def test_a_leg_starting_exactly_on_the_cutoff_counts(self):
        rows = [
            _row("youtube", 1, 600),
            _row("tiktok", 2, 100),
            _row("youtube", 7, 700),
        ]
        m = summarize_post("p", PUBLISHED, rows)
        assert m.views_day_2 == 700
        assert m.views_day_7 == 800

    def test_a_single_platform_post_is_unaffected(self):
        rows = [_row("youtube", 1, 10), _row("youtube", 2, 20), _row("youtube", 7, 30)]
        m = summarize_post("p", PUBLISHED, rows)
        assert m.views_day_2 == 20
        assert m.views_day_7 == 30


@pytest.mark.unit
class TestDurabilityRatioBlanksToo:
    def test_a_leg_starting_after_the_window_blanks_the_ratio(self):
        # The late leg's whole lifetime figure lands in the "after" half
        # while contributing nothing to "within", which inflates the ratio
        # for a reason unrelated to earning attention later.
        rows = [
            _row("youtube", 1, 100),
            _row("youtube", 30, 200),
            _row("tiktok", 35, 5000),
            _row("youtube", 40, 210),
        ]
        m = summarize_post("p", PUBLISHED, rows)
        assert m.durability_ratio is None

    def test_a_leg_inside_the_window_still_yields_a_ratio(self):
        rows = [
            _row("youtube", 1, 100),
            _row("tiktok", 3, 50),
            _row("youtube", 30, 200),
            _row("youtube", 40, 400),
        ]
        m = summarize_post("p", PUBLISHED, rows)
        assert m.durability_ratio is not None
        assert m.durability_ratio > 0
