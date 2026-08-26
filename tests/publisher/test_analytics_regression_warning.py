"""A post that had a view count and returns none is the one usable signal.

A sweep where every timeline comes back empty is indistinguishable, in that
sweep alone, from an account whose posts are too young to have rows. Both
succeed, both store stubs, and failing on it would fail a new account daily.
Across sweeps they separate: a young post gains a figure within days, while a
post that already had one did not get younger.
"""

import logging

from src.publisher.analytics import PostMetrics, load_metrics, save_metrics


def _stored(post_id, total):
    return PostMetrics(post_id=post_id, published_at="2026-07-01", views_total=total)


def _stub(post_id):
    return PostMetrics(post_id=post_id, published_at="2026-07-01")


class TestARegressedFigureWarns:
    def test_a_post_that_loses_its_view_count_is_named(self, tmp_path, caplog):
        save_metrics([_stored("a", 400)], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stub("a")], tmp_path)

        assert "1 of 1" in caplog.text
        assert "a" in caplog.text

    def test_the_stored_figure_is_still_kept(self, tmp_path):
        """The warning reports a risk, it does not act on one.

        The merge keeps figures per field, so nothing is lost at the moment
        this fires. What is at risk is the capture continuing to look healthy
        while collecting nothing.
        """
        save_metrics([_stored("a", 400)], tmp_path)
        save_metrics([_stub("a")], tmp_path)

        assert load_metrics(tmp_path)[0].views_total == 400

    def test_every_post_regressing_is_reported_as_such(self, tmp_path, caplog):
        """A renamed timeline key regresses every mature post at once."""
        save_metrics([_stored(p, 100) for p in ("a", "b", "c")], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stub(p) for p in ("a", "b", "c")], tmp_path)

        assert "3 of 3" in caplog.text

    def test_a_long_list_is_truncated_with_a_count(self, tmp_path, caplog):
        ids = [f"p{i}" for i in range(8)]
        save_metrics([_stored(p, 100) for p in ids], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stub(p) for p in ids], tmp_path)

        assert "8 of 8" in caplog.text
        assert "+3 more" in caplog.text


class TestTheQuietCases:
    def test_a_first_reading_does_not_warn(self, tmp_path, caplog):
        """Nothing to regress from. This is every new post's first sweep."""
        with caplog.at_level(logging.WARNING):
            save_metrics([_stub("a")], tmp_path)

        assert "returned none" not in caplog.text

    def test_a_young_post_staying_empty_does_not_warn(self, tmp_path, caplog):
        """A stub replacing a stub is a post still waiting for its first rows.

        This is the case that rules out failing the sweep, so it must also
        stay out of the warning: an account in its first days would otherwise
        warn on every post, every day, and train the reader to ignore it.
        """
        save_metrics([_stub("a")], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stub("a")], tmp_path)

        assert "returned none" not in caplog.text

    def test_a_figure_arriving_does_not_warn(self, tmp_path, caplog):
        """The normal direction: a young post gains its first count."""
        save_metrics([_stub("a")], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stored("a", 250)], tmp_path)

        assert "returned none" not in caplog.text

    def test_a_post_absent_from_this_sweep_does_not_warn(self, tmp_path, caplog):
        """Beyond --limit is not a regression; it was never measured."""
        save_metrics([_stored("a", 400), _stored("b", 400)], tmp_path)

        with caplog.at_level(logging.WARNING):
            save_metrics([_stored("a", 450)], tmp_path)

        assert "returned none" not in caplog.text
