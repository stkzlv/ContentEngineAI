"""A sweep that captures nothing must fail, not exit quietly.

The scheduled setup detects trouble only through a failed systemd unit:
`OnFailure=`, the journal, the log file, the notification. A sweep that
exits 0 having stored nothing keeps the timer green, satisfies the
installer's proof-of-life mtime check, and lets the figures expire.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.publisher.late.cli import cmd_analytics
from src.publisher.models import AnalyticsConfig, PublisherConfig


def _config():
    return PublisherConfig(
        provider="late",
        api_key="sk_live_key_12345",
        analytics_config=AnalyticsConfig(limit=50),
    )


def _args(tmp_path):
    return argparse.Namespace(
        limit=None, rank_only=False, outputs_dir=Path(tmp_path), debug=False
    )


def _publisher(posts):
    pub = MagicMock()
    pub.authenticate = AsyncMock(return_value=True)
    pub.list_posts = AsyncMock(return_value=posts)
    pub.client = MagicMock()
    return pub


async def _run(tmp_path, posts, timeline_side_effect):
    resource = MagicMock()
    resource.get_post_timeline.side_effect = timeline_side_effect
    with (
        patch(
            "src.publisher.late.cli._create_publisher_from_config",
            return_value=_publisher(posts),
        ),
        patch("src.publisher.late.cli.timeline_resource", return_value=resource),
    ):
        await cmd_analytics(_args(tmp_path), _config(), MagicMock())


POSTS = [{"id": "a", "platforms": []}, {"id": "b", "platforms": []}]

# A post that can actually produce a figure: a real publish time, and a
# timeline whose rows carry the date, view count and platform the reducer
# reads. Without these `summarize_post` takes its early return and stores a
# stub, which satisfies "the file exists" while measuring nothing.
MEASURABLE_POST = {
    "id": "b",
    "platforms": [{"platform": "youtube", "publishedAt": "2026-07-01T08:00:00Z"}],
}
MEASURABLE_TIMELINE = {
    "timeline": [
        {"date": "2026-07-02", "views": 100, "platform": "youtube"},
        {"date": "2026-07-08", "views": 400, "platform": "youtube"},
    ]
}


class TestASweepThatCapturesNothing:
    @pytest.mark.asyncio
    async def test_every_timeline_failing_exits_non_zero(self, tmp_path):
        """The timer's whole failure story rests on this case."""
        with pytest.raises(SystemExit) as exc:
            await _run(tmp_path, POSTS, RuntimeError("endpoint gone"))

        assert exc.value.code != 0

    @pytest.mark.asyncio
    async def test_nothing_is_written_when_everything_failed(self, tmp_path):
        """An unchanged mtime is what the installer checks, so do not touch it."""
        with pytest.raises(SystemExit):
            await _run(tmp_path, POSTS, RuntimeError("endpoint gone"))

        assert not (tmp_path / "post_metrics.json").exists()

    @pytest.mark.asyncio
    async def test_posts_with_no_usable_id_exit_non_zero(self, tmp_path):
        """Listing 50 posts and identifying none is a break, not an empty run.

        The id field has moved once already, which is why two keys are read.
        If it moves again every post is skipped, and counting inside the loop
        would leave the tally at zero and slip past the guard the tally feeds:
        the sweep would exit 0, keep the timer green, and let the windows
        expire.
        """
        with pytest.raises(SystemExit) as exc:
            await _run(
                tmp_path,
                [{"platforms": []}, {"unexpected_key": "b"}],
                RuntimeError("never reached"),
            )

        assert exc.value.code != 0

    @pytest.mark.asyncio
    async def test_an_account_with_no_posts_is_not_an_error(self, tmp_path):
        """Nothing to measure differs from everything broke.

        Failing here would make a legitimately empty account noisy every day.
        """
        await _run(tmp_path, [], RuntimeError("never called"))

    @pytest.mark.asyncio
    async def test_a_partial_sweep_stores_the_surviving_figure(self, tmp_path):
        """One bad post must not lose the rest, and the rest must be real.

        Asserting only that the file exists is not enough: an empty timeline
        makes ``summarize_post`` store a stub, so a regression that captured
        no figures at all would still write a file and still pass. The stored
        row has to carry a view count.
        """
        await _run(
            tmp_path,
            [POSTS[0], MEASURABLE_POST],
            [RuntimeError("one bad post"), MEASURABLE_TIMELINE],
        )

        stored = json.loads((tmp_path / "post_metrics.json").read_text())
        assert [r["post_id"] for r in stored] == ["b"]
        assert stored[0]["views_total"] == 400
