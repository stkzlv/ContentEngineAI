"""A series post joins its YouTube playlist on upload, and the groupings the
provider has no call for are logged as manual steps once it is live
(REQ-PUB-169).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.publisher.models import Platform, PublishMetadata
from src.publisher.series_queue import (
    manual_step,
    mark_live,
    series_groups,
    with_playlist,
)
from src.publisher.tracking import load_tracking, record_publish

pytestmark = pytest.mark.req("REQ-PUB-169")

PID = "B0TESTGRP1"
GROUPS = {"youtube_playlist_id": "PLtest", "tiktok_playlist": "Myths"}


def _render(outputs, groups=GROUPS, claim="the widget hums"):
    (outputs / PID).mkdir(parents=True, exist_ok=True)
    (outputs / PID / "series.json").write_text(
        json.dumps({"claim": claim, "groups": groups})
    )


class TestThePlaylist:
    def test_the_youtube_entry_gains_the_playlist(self):
        contents = with_playlist({"youtube": {"title": "T"}}, GROUPS)
        assert contents == {"youtube": {"title": "T", "playlist_id": "PLtest"}}

    def test_no_playlist_leaves_the_contents_alone(self):
        contents = {"tiktok": {"content": "c"}}
        assert with_playlist(contents, {"tiktok_playlist": "x"}) is contents
        assert with_playlist(None, {}) is None

    def test_the_render_record_gives_the_groups(self, tmp_path):
        _render(tmp_path)
        assert series_groups(PID, tmp_path) == GROUPS
        assert series_groups("B0TESTNONE", tmp_path) == {}

    def test_the_payload_carries_the_playlist_id(self):
        from src.publisher.late.client import LatePublisher

        publisher = LatePublisher(api_key="k" * 40)
        platforms, _ = publisher._build_sdk_platforms(
            [{"platform": "youtube", "account_id": "a"}],
            "media",
            platform_contents={"youtube": {"title": "T", "playlist_id": "PLtest"}},
        )
        psd = platforms[0]["platformSpecificData"]
        assert isinstance(psd, dict) and psd["playlistId"] == "PLtest"


def _meta() -> PublishMetadata:
    return PublishMetadata(
        platform=Platform.YOUTUBE,
        title="Title",
        description="Description",
        hashtags=["tag"],
        keywords=[],
        product_id=PID,
        disclosure="#ad",
        carries_affiliate_content=False,
    )


def _publisher():
    calls: list[dict] = []

    async def publish(**kwargs):
        calls.append(kwargs)
        return {"post_id": "p1"}

    publisher = MagicMock()
    publisher.publish = publish
    publisher.youtube_profile_pointer = False
    return publisher, calls


@pytest.mark.asyncio
class TestEveryPublishPathPassesIt:
    async def test_single_unified_and_platform_specific(self, tmp_path):
        from src.publisher import publish_modes

        _render(tmp_path)
        platforms = [{"platform": "youtube", "account_id": "a"}]
        with (
            patch.object(publish_modes, "load_platform_metadata", return_value=_meta()),
            patch.object(
                publish_modes, "_build_platform_contents_with_comments", return_value={}
            ),
            patch.object(
                publish_modes, "resolve_correction_links", AsyncMock(return_value={})
            ),
        ):
            for publish in (
                publish_modes._publish_unified,
                publish_modes._publish_platform_specific,
            ):
                publisher, calls = _publisher()
                await publish(publisher, "m", PID, platforms, tmp_path, None)
                contents = calls[0]["platform_contents"]
                assert contents["youtube"]["playlist_id"] == "PLtest"

    async def test_the_scheduler_paths(self, tmp_path):
        from src.publisher.schedule import ScheduleManager

        manager = ScheduleManager(schedule_path=tmp_path / "schedule.json")
        for post in (manager._post_per_platform, manager._post_unified):
            publisher, calls = _publisher()
            await post(
                publisher=publisher,
                platform_dicts=[{"platform": "youtube", "account_id": "a"}],
                platform_metas={"youtube": _meta()},
                titles={"youtube": "Title"},
                carries_affiliate={"youtube": False},
                first_comments={},
                media_id="m",
                product_id=PID,
                next_time=datetime(2026, 12, 1, 8, tzinfo=UTC),
                next_idx=0,
                groups=GROUPS,
            )
            assert calls[0]["platform_contents"]["youtube"]["playlist_id"] == "PLtest"


class TestTheManualSteps:
    def test_each_unsupported_grouping_is_a_step_with_the_url(self):
        groups = {**GROUPS, "instagram_link": True}
        assert manual_step(groups, "tiktok", "https://x/t") == (
            "add the TikTok post https://x/t to the playlist 'Myths'"
        )
        assert manual_step(groups, "instagram", "https://x/i") == (
            "link the Instagram post https://x/i to the previous episode"
        )
        assert manual_step(groups, "youtube", "https://x/y") is None

    def test_an_unconfigured_grouping_needs_no_step(self):
        assert manual_step({"youtube_playlist_id": "PLtest"}, "tiktok", "u") is None

    def test_a_post_logs_only_the_legs_it_has(self, tmp_path, caplog):
        _render(tmp_path, groups={**GROUPS, "instagram_link": True})
        record_publish(PID, "youtube", "p_yt", tmp_path)
        record_publish(PID, "tiktok", "p_tt", tmp_path)
        posts = [
            {
                "id": "p_yt",
                "status": "published",
                "platforms": [{"platform": "youtube", "url": "https://x/y"}],
            },
            {
                "id": "p_tt",
                "status": "published",
                "platforms": [{"platform": "tiktok", "url": "https://x/t"}],
            },
        ]
        with caplog.at_level(logging.INFO, logger="src.publisher.series_queue"):
            mark_live(posts, tmp_path)
        steps = [
            r.getMessage() for r in caplog.records if "manual step" in r.getMessage()
        ]
        assert steps == [
            "Series grouping, manual step for post p_tt: add the TikTok post "
            "https://x/t to the playlist 'Myths'"
        ]

    def test_a_leg_that_gets_its_url_later_is_logged_then(self, tmp_path, caplog):
        _render(tmp_path)
        record_publish(PID, "all", "post1", tmp_path)
        waiting = {
            "id": "post1",
            "status": "partial",
            "platforms": [{"platform": "tiktok", "url": None}],
        }
        live = {
            "id": "post1",
            "status": "published",
            "platforms": [{"platform": "tiktok", "url": "https://x/t"}],
        }
        with caplog.at_level(logging.INFO, logger="src.publisher.series_queue"):
            mark_live([waiting], tmp_path)
            assert "manual step" not in caplog.text
            mark_live([live], tmp_path)
        assert "https://x/t to the playlist 'Myths'" in caplog.text

    def test_history_keeps_the_groups_and_the_live_post_is_logged_once(
        self, tmp_path, caplog
    ):
        _render(tmp_path)
        record_publish(PID, "all", "post1", tmp_path)
        row = load_tracking(tmp_path)["posts"][f"{PID}:all"]
        assert row["series"]["groups"] == GROUPS
        post = {
            "id": "post1",
            "status": "published",
            "platforms": [{"platform": "tiktok", "url": "https://example.invalid/t"}],
        }
        with caplog.at_level(logging.INFO, logger="src.publisher.series_queue"):
            mark_live([post], tmp_path)
            first = caplog.text
            caplog.clear()
            mark_live([post], tmp_path)
        assert "https://example.invalid/t to the playlist 'Myths'" in first
        assert "manual step" not in caplog.text

    def test_a_scheduled_post_is_not_logged(self, tmp_path, caplog):
        _render(tmp_path)
        record_publish(PID, "all", "post1", tmp_path)
        with caplog.at_level(logging.INFO, logger="src.publisher.series_queue"):
            mark_live([{"id": "post1", "status": "scheduled"}], tmp_path)
        assert "manual step" not in caplog.text


def test_the_render_leaves_its_groups_without_a_queue(tmp_path):
    from src.video.config.series_models import SeriesConfig, SeriesGroups
    from src.video.series import write_series_record

    series = SeriesConfig(groups=SeriesGroups(youtube_playlist_id="PLtest"))
    write_series_record(series, MagicMock(claim="c"), tmp_path)
    record = json.loads((tmp_path / "series.json").read_text())
    assert record == {"claim": "c", "groups": {"youtube_playlist_id": "PLtest"}}
    (tmp_path / "series.json").unlink()
    write_series_record(SeriesConfig(), MagicMock(claim="c"), tmp_path)
    assert not (tmp_path / "series.json").exists()


@pytest.mark.asyncio
async def test_a_listed_leg_carries_its_url(monkeypatch):
    from types import SimpleNamespace

    from src.publisher.late.client import LatePublisher

    legs = [
        SimpleNamespace(platform="tiktok", field_id="a", platformPostUrl="https://x/t"),
        SimpleNamespace(platform="youtube", field_id="b", platformPostUrl=""),
    ]
    post = SimpleNamespace(
        field_id="p1", status="published", scheduledFor=None, platforms=legs
    )
    pub = LatePublisher(api_key="k" * 40)
    monkeypatch.setattr(
        pub, "_posts_list_safe", AsyncMock(return_value=SimpleNamespace(posts=[post]))
    )
    posts = await pub.list_posts()
    assert [leg["url"] for leg in posts[0]["platforms"]] == ["https://x/t", None]


def test_the_scheduler_hands_the_series_groups_to_its_post_helper():
    """The helpers take the groups as an argument; this pins the caller."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/publisher/schedule.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "post"
    ]
    assert calls
    for call in calls:
        groups = next(k for k in call.keywords if k.arg == "groups")
        assert "series_groups" in ast.unparse(groups.value)
