"""A post that reported views recently and returns none is the usable signal.

A sweep where every timeline comes back empty is indistinguishable, in that
sweep alone, from an account whose posts are too young to have rows. Both
succeed, both store stubs, and failing on it would fail a new account daily.
Across sweeps they separate, but only after excluding the other reason a
timeline empties: past the retention horizon the provider returns no rows for
a post at all, so an aged-out post regresses on a completely healthy install.

Post ids here are deliberately unlike any word in the warning text. Using
``"a"`` made ``assert "a" in caplog.text`` pass against the message's own
prose, so the test held with the sample removed entirely.
"""

import logging
from datetime import UTC, datetime, timedelta

from src.publisher.analytics import (
    RETENTION_HORIZON_DAYS,
    PostMetrics,
    load_metrics,
    save_metrics,
)

RECENT = (datetime.now(UTC) - timedelta(days=2)).strftime("%Y-%m-%d")
AGED_OUT = (datetime.now(UTC) - timedelta(days=RETENTION_HORIZON_DAYS + 30)).strftime(
    "%Y-%m-%d"
)


def _reporting(post_id, total=400, end=RECENT):
    """A post with a stored view count whose timeline reached ``end``."""
    return PostMetrics(
        post_id=post_id,
        published_at="2026-07-01",
        views_total=total,
        timeline_end=end,
    )


def _stub(post_id):
    """What a sweep stores when the timeline comes back empty."""
    return PostMetrics(post_id=post_id, published_at="2026-07-01")


class TestARecentlyReportingPostThatGoesQuiet:
    def test_it_is_named_in_the_warning(self, tmp_path, caplog):
        save_metrics([_reporting("zulu_one")], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stub("zulu_one")], tmp_path)

        assert "zulu_one" in caplog.text

    def test_the_proportion_reports_the_whole_sweep(self, tmp_path, caplog):
        """`1 of 2`, not `1 of 1`.

        The denominator is what separates one odd post from a broken reader.
        Every-post-regresses cases cannot see it, because the two numbers are
        equal there whatever the code does.
        """
        save_metrics([_reporting("zulu_one"), _reporting("zulu_two")], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stub("zulu_one"), _reporting("zulu_two", 450)], tmp_path)

        assert "1 of 2" in caplog.text

    def test_a_healthy_post_is_not_named(self, tmp_path, caplog):
        save_metrics([_reporting("zulu_one"), _reporting("zulu_two")], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stub("zulu_one"), _reporting("zulu_two", 450)], tmp_path)

        assert "zulu_two" not in caplog.text

    def test_the_stored_figure_is_still_kept(self, tmp_path):
        """The warning reports a risk; it does not act on one."""
        save_metrics([_reporting("zulu_one", 400)], tmp_path)
        save_metrics([_stub("zulu_one")], tmp_path)

        assert load_metrics(tmp_path)[0].views_total == 400

    def test_a_long_list_is_truncated_with_a_count(self, tmp_path, caplog):
        ids = [f"zulu_{i}" for i in range(8)]
        save_metrics([_reporting(p) for p in ids], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stub(p) for p in ids], tmp_path)

        assert "8 of 8" in caplog.text
        assert "+3 more" in caplog.text
        assert "zulu_7" not in caplog.text

    def test_the_ids_are_returned_for_the_caller_to_record(self, tmp_path):
        """A log line alone reaches no surface an operator is told to check."""
        save_metrics([_reporting("zulu_one")], tmp_path)

        assert save_metrics([_stub("zulu_one")], tmp_path) == ["zulu_one"]


class TestTheQuietCases:
    def test_a_post_past_the_retention_horizon_does_not_warn(self, tmp_path, caplog):
        """The provider stops returning rows for old posts entirely.

        Measured on this project: posts at 188 days still returned rows, one at
        248 days returned none. Without this exclusion a healthy install warns
        on every sweep forever once its posts age, which is the cry-wolf
        outcome the whole design exists to avoid.
        """
        save_metrics([_reporting("zulu_old", end=AGED_OUT)], tmp_path)

        with caplog.at_level(logging.WARNING):
            regressed = save_metrics([_stub("zulu_old")], tmp_path)

        assert regressed == []
        assert "stopped reporting" not in caplog.text
        assert "returned none" not in caplog.text

    def test_a_first_reading_does_not_warn(self, tmp_path, caplog):
        """Nothing to regress from. Every new post's first sweep."""
        with caplog.at_level(logging.WARNING):
            assert save_metrics([_stub("zulu_new")], tmp_path) == []

        assert "returned none" not in caplog.text

    def test_a_young_post_staying_empty_does_not_warn(self, tmp_path, caplog):
        """A stub replacing a stub is a post still waiting for its first rows.

        This is the case that rules out failing the sweep, so it must stay out
        of the warning too: a new account would otherwise warn on every post,
        every day.
        """
        save_metrics([_stub("zulu_new")], tmp_path)

        with caplog.at_level(logging.WARNING):
            assert save_metrics([_stub("zulu_new")], tmp_path) == []

        assert "returned none" not in caplog.text

    def test_a_figure_arriving_does_not_warn(self, tmp_path, caplog):
        """The normal direction: a young post gains its first count."""
        save_metrics([_stub("zulu_new")], tmp_path)

        with caplog.at_level(logging.WARNING):
            assert save_metrics([_reporting("zulu_new", 250)], tmp_path) == []

        assert "returned none" not in caplog.text

    def test_a_post_absent_from_this_sweep_does_not_warn(self, tmp_path, caplog):
        """Beyond --limit is not a regression; it was never measured."""
        save_metrics([_reporting("zulu_one"), _reporting("zulu_two")], tmp_path)

        with caplog.at_level(logging.WARNING):
            assert save_metrics([_reporting("zulu_one", 450)], tmp_path) == []

        assert "returned none" not in caplog.text
