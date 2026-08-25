"""A day-N figure counts every platform or none.

A platform's first timeline row carries its lifetime total to that date, not
that day's increment, and platforms start reporting on their own lag. A figure
taken before a leg started counts only part of the post while `views_total`
counts all of it, so the two describe different posts.

A comparison that ranks by median day-7 views would place a post understated
by reporting lag below an identical one, for a reason that is not reach, so
the figure is reported unknown rather than small — the same rule
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


@pytest.mark.unit
class TestASweepBeforeTheLagAppears:
    """The sweep that first measures a post usually cannot see the lag.

    A daily run reaches a young post while the slow platform has no rows at
    all. One leg is visible, nothing looks incomplete, and the biased figure
    is stored. Only a later sweep knows better, and the per-field merge keeps
    a measured value over a missing one -- so without a stored marker the
    first sweep's number would stand forever.
    """

    @staticmethod
    def _sweep(tmp_path, rows):
        from src.publisher.analytics import load_metrics, save_metrics, summarize_post

        save_metrics([summarize_post("p", PUBLISHED.isoformat(), rows)], tmp_path)
        return next(m for m in load_metrics(tmp_path) if m.post_id == "p")

    def test_a_later_sweep_withdraws_the_biased_figure(self, tmp_path):
        # Day-2 sweep: only the fast leg has reported.
        first = self._sweep(
            tmp_path, [_row("youtube", 1, 600), _row("youtube", 2, 701)]
        )
        assert first.views_day_2 == 701

        # Day-10 sweep: the slow leg appears, having started on day 4.
        later = self._sweep(
            tmp_path,
            [
                _row("youtube", 1, 600),
                _row("youtube", 2, 701),
                _row("tiktok", 4, 326),
                _row("youtube", 10, 747),
            ],
        )
        assert later.views_day_2 is None, "the day-2 figure counted one leg of two"
        assert later.views_total == 1073

    def test_a_figure_taken_after_every_leg_reported_survives(self, tmp_path):
        rows = [
            _row("youtube", 1, 600),
            _row("tiktok", 1, 100),
            _row("youtube", 2, 700),
            _row("tiktok", 2, 200),
        ]
        first = self._sweep(tmp_path, rows)
        assert first.views_day_2 == 900
        later = self._sweep(tmp_path, [*rows, _row("youtube", 9, 800)])
        assert later.views_day_2 == 900

    def test_a_truncated_later_sweep_does_not_erase_a_good_figure(self, tmp_path):
        # The retention case: the merge must still prefer a measured value
        # over an absent one when the absence is only a shorter timeline.
        rows = [
            _row("youtube", 1, 600),
            _row("tiktok", 1, 100),
            _row("youtube", 2, 700),
            _row("tiktok", 2, 200),
        ]
        assert self._sweep(tmp_path, rows).views_day_2 == 900
        truncated = self._sweep(
            tmp_path, [_row("youtube", 40, 900), _row("tiktok", 40, 400)]
        )
        assert truncated.views_day_2 == 900

    def test_the_mark_survives_a_sweep_that_can_no_longer_see_it(self, tmp_path):
        # Three sweeps. The second is the only one that can observe the lag;
        # the third's rows all begin past the retention horizon, so it sees
        # no disagreement between legs. The figure must stay withdrawn: a
        # sweep that cannot see the lag says nothing about whether the number
        # was biased when it was taken.
        self._sweep(tmp_path, [_row("youtube", 1, 600), _row("youtube", 2, 701)])
        marked = self._sweep(
            tmp_path,
            [
                _row("youtube", 1, 600),
                _row("youtube", 2, 701),
                _row("tiktok", 4, 326),
                _row("youtube", 10, 747),
            ],
        )
        assert marked.views_day_2 is None

        truncated = self._sweep(
            tmp_path, [_row("youtube", 40, 900), _row("tiktok", 40, 400)]
        )
        assert truncated.views_day_2 is None
        assert 2 in truncated.lagged_cutoff_days

    def test_timeline_end_is_not_pinned_to_a_withdrawn_ratio(self, tmp_path):
        # `timeline_end` is frozen while it dates a preserved ratio. A ratio
        # the marker withdraws is not preserved, so the field must move with
        # the newer reading rather than stay frozen against nothing.
        early = [
            _row("youtube", 1, 100),
            _row("tiktok", 1, 10),
            _row("youtube", 30, 200),
            _row("youtube", 45, 400),
        ]
        first = self._sweep(tmp_path, early)
        assert first.durability_ratio is not None
        pinned = first.timeline_end

        # A later sweep reveals a third leg that only started after day 30.
        later = self._sweep(
            tmp_path,
            [*early, _row("instagram", 40, 500), _row("youtube", 60, 420)],
        )
        assert later.durability_ratio is None
        assert later.timeline_end != pinned


@pytest.mark.unit
class TestATruncatedSweepDoesNotWithdraw:
    """Past the retention horizon, a first row is not a start.

    Every leg's retained rows begin at the window edge, so a leg that happens
    to be absent from that first date looks exactly like one that started
    late. Marking there would withdraw a ratio an earlier, fuller sweep had
    measured correctly — and no later sweep could recompute it, because the
    days it needs are gone.
    """

    @staticmethod
    def _sweep(tmp_path, rows):
        from src.publisher.analytics import load_metrics, save_metrics, summarize_post

        save_metrics([summarize_post("p", PUBLISHED.isoformat(), rows)], tmp_path)
        return next(m for m in load_metrics(tmp_path) if m.post_id == "p")

    def test_a_ratio_measured_in_full_survives_a_truncated_sweep(self, tmp_path):
        full = [
            _row("youtube", 1, 100),
            _row("tiktok", 1, 50),
            _row("youtube", 30, 200),
            _row("youtube", 45, 400),
        ]
        first = self._sweep(tmp_path, full)
        assert first.durability_ratio is not None
        measured = first.durability_ratio

        # Aged out: rows retained only from day 25, and the slow leg happens
        # to be absent from that first retained date.
        later = self._sweep(
            tmp_path,
            [
                _row("youtube", 25, 190),
                _row("youtube", 30, 200),
                _row("tiktok", 33, 60),
                _row("youtube", 45, 400),
            ],
        )
        assert later.lagged_cutoff_days == []
        assert later.durability_ratio == measured

    def test_a_truncated_sweep_marks_nothing_at_all(self, tmp_path):
        later = self._sweep(
            tmp_path,
            [_row("youtube", 25, 190), _row("tiktok", 33, 60)],
        )
        assert later.lagged_cutoff_days == []


@pytest.mark.unit
class TestRowsWrittenBeforeThisRule:
    """A missing provenance key must not read as "measured from a stub".

    Every row already on disk lacks it. Defaulting to False would tell the
    merge each stored ratio came from a truncated record, so the first sweep
    after the upgrade would overwrite all of them -- the figures the field
    exists to protect, and the ones that cannot be recomputed.
    """

    def test_a_legacy_row_keeps_its_ratio_against_a_truncated_sweep(self, tmp_path):
        import json

        from src.publisher.analytics import (
            load_metrics,
            metrics_path,
            save_metrics,
            summarize_post,
        )

        # Written the way a pre-0.74 release wrote it: no new keys at all.
        metrics_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        metrics_path(tmp_path).write_text(
            json.dumps(
                [
                    {
                        "post_id": "p",
                        "published_at": PUBLISHED.replace(tzinfo=None).isoformat(),
                        "views_day_2": 900,
                        "views_day_7": 950,
                        "views_total": 1000,
                        "durability_ratio": 0.8,
                        "timeline_end": "2026-08-13T08:00:00",
                    }
                ]
            )
        )
        assert load_metrics(tmp_path)[0].covers_publication is None

        truncated = summarize_post(
            "p",
            PUBLISHED.isoformat(),
            [
                _row("youtube", 25, 190),
                _row("youtube", 30, 200),
                _row("tiktok", 33, 60),
                _row("youtube", 45, 400),
            ],
        )
        assert truncated.covers_publication is False
        save_metrics([truncated], tmp_path)
        assert load_metrics(tmp_path)[0].durability_ratio == 0.8

    def test_an_unknown_column_does_not_make_the_file_unreadable(self, tmp_path):
        import json

        from src.publisher.analytics import load_metrics, metrics_path

        metrics_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        metrics_path(tmp_path).write_text(
            json.dumps([{"post_id": "p", "views_total": 5, "a_future_column": 1}])
        )
        rows = load_metrics(tmp_path)
        assert len(rows) == 1
        assert rows[0].views_total == 5


@pytest.mark.unit
class TestATruncatedSweepStillWithholdsItsOwnFigure:
    """Marking and withholding answer different questions.

    Marking is persisted and withdraws a figure other sweeps stored, so it
    needs a record reaching publication to tell a late start from a truncated
    one. Withholding governs only this reading, and is right wherever the
    retained window actually covers the cutoff -- there the legs demonstrably
    disagree.
    """

    def test_a_lagging_leg_inside_a_truncated_window_is_not_reported(self):
        from src.publisher.analytics import summarize_post

        # Retained from day 3, so day-7 is inside the window; the slow leg
        # genuinely starts on day 20.
        m = summarize_post(
            "p",
            PUBLISHED.isoformat(),
            [
                _row("youtube", 3, 500),
                _row("youtube", 7, 700),
                _row("tiktok", 20, 4000),
                _row("youtube", 40, 1200),
            ],
        )
        assert m.covers_publication is False
        assert m.views_day_7 is None, "700 counted one leg of two"
        # Not marked: the record cannot tell a late start from truncation,
        # and a mark would withdraw other sweeps' figures too.
        assert m.lagged_cutoff_days == []
        assert m.views_total == 5200


@pytest.mark.unit
class TestProvenanceFollowsTheStoredRatio:
    """`covers_publication` records where the *stored ratio* came from.

    Unioning it instead let a young sweep that carried no ratio at all mark
    the row as whole. Every later reading was then rejected as truncated, so
    the first truncated ratio froze permanently — including against readings
    that are strictly more mature and agree with the full record.
    """

    @staticmethod
    def _sweep(tmp_path, rows):
        from src.publisher.analytics import load_metrics, save_metrics, summarize_post

        save_metrics([summarize_post("p", PUBLISHED.isoformat(), rows)], tmp_path)
        return next(m for m in load_metrics(tmp_path) if m.post_id == "p")

    def test_a_later_truncated_reading_still_updates_the_ratio(self, tmp_path):
        # Sweep 1: young and whole, so no ratio yet.
        young = self._sweep(tmp_path, [_row("youtube", 1, 100), _row("tiktok", 1, 20)])
        assert young.durability_ratio is None
        assert young.covers_publication is True

        # Sweep 2: aged out, first ratio arrives from a truncated record.
        second = self._sweep(
            tmp_path,
            [
                _row("youtube", 5, 150),
                _row("tiktok", 5, 30),
                _row("youtube", 30, 300),
                _row("youtube", 40, 400),
            ],
        )
        first_ratio = second.durability_ratio
        assert first_ratio is not None
        # The row must not claim a whole record just because sweep 1 saw one.
        assert second.covers_publication is False

        # Sweep 3: still truncated, but more mature. It must win.
        third = self._sweep(
            tmp_path,
            [
                _row("youtube", 5, 150),
                _row("tiktok", 5, 30),
                _row("youtube", 30, 300),
                _row("youtube", 55, 900),
            ],
        )
        assert third.durability_ratio != first_ratio
