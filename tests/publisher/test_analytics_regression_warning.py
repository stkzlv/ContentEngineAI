"""A post that reported views recently and returns none is the usable signal.

A sweep where every timeline comes back empty is indistinguishable, in that
sweep alone, from an account whose posts are too young to have rows. Both
succeed, both store stubs, and failing on it would fail a new account daily.
Across sweeps they separate, but only in aggregate. Past the retention horizon
the provider stops returning a post's rows entirely, so an aged-out post loses
its figures on a healthy install -- one at a time. A reader that stopped
understanding the response takes every post with it in the same sweep.

Post ids here are deliberately unlike any word in the warning text. Using
``"a"`` made ``assert "a" in caplog.text`` pass against the message's own
prose, so the test held with the sample removed entirely.
"""

import logging

from src.publisher.analytics import PostMetrics, load_metrics, save_metrics


def _reporting(post_id, total=400):
    """A post with a stored view count."""
    return PostMetrics(post_id=post_id, published_at="2026-07-01", views_total=total)


def _stub(post_id):
    """What a sweep stores when the timeline comes back empty."""
    return PostMetrics(post_id=post_id, published_at="2026-07-01")


class TestARecentlyReportingPostThatGoesQuiet:
    def test_it_is_named_in_the_warning(self, tmp_path, caplog):
        save_metrics([_reporting("zulu_one")], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stub("zulu_one")], tmp_path)

        assert "zulu_one" in caplog.text

    def test_the_proportion_counts_the_whole_sweep(self, tmp_path, caplog):
        """`2 of 3`, not `2 of 2`.

        The denominator is the measured sweep, not the regressed set, so it
        says how much of the run went quiet. A case where the two sets are the
        same size cannot see the difference whatever the code does, so this
        one measures a third post that had no figure to lose.
        """
        save_metrics([_reporting("zulu_one"), _reporting("zulu_two")], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics(
                [_stub("zulu_one"), _stub("zulu_two"), _stub("zulu_fresh")], tmp_path
            )

        assert "2 of 3" in caplog.text

    def test_a_post_with_nothing_to_lose_is_not_named(self, tmp_path, caplog):
        """A post that never had a figure did not regress."""
        save_metrics([_reporting("zulu_one")], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stub("zulu_one"), _stub("zulu_fresh")], tmp_path)

        assert "zulu_fresh" not in caplog.text

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
    def test_one_post_of_many_going_quiet_does_not_warn(self, tmp_path, caplog):
        """A single post losing its figures is ageing, not breakage.

        The provider stops returning a post's rows once it is old enough, so
        this happens on a healthy install, one post at a time. Warning here
        would fire on every sweep forever as posts age -- the cry-wolf outcome
        the whole design exists to avoid.
        """
        save_metrics([_reporting("zulu_one"), _reporting("zulu_two")], tmp_path)

        with caplog.at_level(logging.WARNING):
            regressed = save_metrics(
                [_stub("zulu_one"), _reporting("zulu_two", 450)], tmp_path
            )

        assert regressed == []
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


class TestItFiresOnTheTransition:
    """Once, when the figures go away -- not on every sweep after.

    The merge keeps a stored figure through an empty reading, so "had a count
    and returned none" stays true forever once it becomes true. Without a
    marker the warning repeats daily, which is the cry-wolf outcome the
    all-or-nothing rule was chosen to avoid.
    """

    def test_a_second_quiet_sweep_is_silent(self, tmp_path, caplog):
        save_metrics([_reporting("zulu_one"), _reporting("zulu_two")], tmp_path)
        assert save_metrics([_stub("zulu_one"), _stub("zulu_two")], tmp_path) != []

        # The first sweep's warning is in caplog too; only the second matters.
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            again = save_metrics([_stub("zulu_one"), _stub("zulu_two")], tmp_path)

        assert again == []
        assert "returned none" not in caplog.text

    def test_a_fourth_quiet_sweep_is_still_silent(self, tmp_path):
        """A dormant account must not accumulate a daily entry forever."""
        save_metrics([_reporting("zulu_one")], tmp_path)
        save_metrics([_stub("zulu_one")], tmp_path)

        for _ in range(3):
            assert save_metrics([_stub("zulu_one")], tmp_path) == []

    def test_figures_returning_re_arms_the_check(self, tmp_path):
        """A fixed reader must be able to report the next break."""
        save_metrics([_reporting("zulu_one")], tmp_path)
        save_metrics([_stub("zulu_one")], tmp_path)
        save_metrics([_reporting("zulu_one", 500)], tmp_path)

        assert save_metrics([_stub("zulu_one")], tmp_path) == ["zulu_one"]

    def test_the_stored_figure_survives_every_quiet_sweep(self, tmp_path):
        save_metrics([_reporting("zulu_one", 400)], tmp_path)
        for _ in range(4):
            save_metrics([_stub("zulu_one")], tmp_path)

        assert load_metrics(tmp_path)[0].views_total == 400
