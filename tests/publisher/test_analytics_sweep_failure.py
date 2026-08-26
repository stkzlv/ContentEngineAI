"""A sweep that captures nothing must fail, not exit quietly.

The scheduled setup detects trouble only through a failed systemd unit:
`OnFailure=`, the journal, the log file, the notification. A sweep that
exits 0 having stored nothing keeps the timer green, satisfies the
installer's proof-of-life mtime check, and lets the figures expire.
"""

import argparse
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
    async def test_an_account_with_no_posts_is_not_an_error(self, tmp_path):
        """Nothing to measure differs from everything broke.

        Failing here would make a legitimately empty account noisy every day.
        """
        await _run(tmp_path, [], RuntimeError("never called"))

    @pytest.mark.asyncio
    async def test_a_partial_sweep_still_succeeds(self, tmp_path):
        """One bad post must not lose the rest; a partial reading is usable."""
        await _run(tmp_path, POSTS, [RuntimeError("one bad post"), {"timeline": []}])

        assert (tmp_path / "post_metrics.json").exists()
