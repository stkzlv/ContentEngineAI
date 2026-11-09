"""Listing viewer comments on a published record (REQ-PUB-167)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.publisher.viewer_comments import (
    NO_COMMENT_API,
    list_viewer_comments,
    record_posts,
)

RECORD = "B0TEST0001"


def _history(outputs: Path, rows: dict[str, dict]) -> None:
    state = outputs / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "publish_history.json").write_text(json.dumps({"posts": rows}))


def _reader(comments: dict[str, list[dict]]) -> AsyncMock:
    reader = AsyncMock()
    reader.get_post_platforms.return_value = [
        {"platform": p, "platform_post_id": f"{p}-1", "account_id": f"acc-{p}"}
        for p in ("youtube", "instagram", "tiktok")
    ]
    reader.get_post_comments.side_effect = lambda pid, acc, limit=25: comments[pid]
    return reader


OWN = {"id": "c0", "message": "Link below.", "from": {"isOwner": True, "name": "me"}}
YT = {"id": "c1", "message": "Does it work offline?", "from": {"name": "Sam"}}
IG = {
    "id": "c2",
    "message": "Mine broke in a week",
    "from": {"name": "Alex B", "username": "alexb"},
}


@pytest.mark.req("REQ-PUB-167")
class TestTheHistory:
    def test_it_maps_each_platform_to_its_post(self, tmp_path) -> None:
        _history(
            tmp_path,
            {
                f"{RECORD}:youtube": {"post_id": "p1", "product_id": RECORD},
                # A row backfilled from the provider carries no product_id.
                f"{RECORD}:instagram": {"post_id": "p1"},
                "B0TEST0002:youtube": {"post_id": "p2"},
            },
        )

        assert record_posts(RECORD, tmp_path) == {"youtube": "p1", "instagram": "p1"}
        assert record_posts(RECORD, tmp_path, ["Instagram"]) == {"instagram": "p1"}

    def test_an_unpublished_record_maps_to_nothing(self, tmp_path) -> None:
        assert record_posts(RECORD, tmp_path) == {}


@pytest.mark.req("REQ-PUB-167")
class TestTheListing:
    @pytest.mark.asyncio
    async def test_it_lists_viewer_comments_without_the_accounts_own(self) -> None:
        reader = _reader({"youtube-1": [OWN, YT], "instagram-1": [IG]})

        results = await list_viewer_comments(
            reader, {"youtube": "p1", "instagram": "p1"}
        )

        by_platform = {r.platform: r for r in results}
        (yt,) = by_platform["youtube"].comments
        (ig,) = by_platform["instagram"].comments
        assert (yt.comment_id, yt.author, yt.text) == (
            "c1",
            "Sam",
            "Does it work offline?",
        )
        assert (ig.comment_id, ig.author) == ("c2", "alexb")
        reader.get_post_platforms.assert_awaited_once_with("p1")

    @pytest.mark.asyncio
    async def test_tiktok_is_reported_not_called(self) -> None:
        reader = _reader({})

        (result,) = await list_viewer_comments(reader, {"tiktok": "p1"})

        assert result.skipped == NO_COMMENT_API
        reader.get_post_comments.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_leg_not_live_yet_is_reported(self) -> None:
        reader = AsyncMock()
        reader.get_post_platforms.return_value = [
            {"platform": "youtube", "platform_post_id": None, "account_id": "a"}
        ]

        (result,) = await list_viewer_comments(reader, {"youtube": "p1"})

        assert result.skipped == "not live yet"
        reader.get_post_comments.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_limit_counts_viewer_comments(self) -> None:
        many = [OWN] + [
            {"id": f"v{n}", "message": "x", "from": {"name": "v"}} for n in range(5)
        ]
        reader = _reader({"youtube-1": many})

        (result,) = await list_viewer_comments(reader, {"youtube": "p1"}, limit=2)

        assert [c.comment_id for c in result.comments] == ["v0", "v1"]


@pytest.mark.req("REQ-PUB-167")
@pytest.mark.asyncio
async def test_one_failing_platform_leaves_the_others_listed() -> None:
    from src.publisher.base import PublishError

    def comments(pid, acc, limit=25):
        if pid == "youtube-1":
            raise PublishError("video removed")
        return [IG]

    reader = _reader({})
    reader.get_post_comments.side_effect = comments

    results = await list_viewer_comments(reader, {"youtube": "p1", "instagram": "p1"})

    by_platform = {r.platform: r for r in results}
    assert [c.comment_id for c in by_platform["instagram"].comments] == ["c2"]
    assert by_platform["youtube"].skipped.startswith("read failed")


@pytest.mark.req("REQ-PUB-167")
def test_the_command_refuses_a_record_with_no_post(tmp_path, monkeypatch) -> None:
    import argparse
    import asyncio

    from src.publisher.late import cli

    args = argparse.Namespace(
        record_id=RECORD, platform=None, limit=20, outputs_dir=tmp_path
    )
    with pytest.raises(SystemExit) as exc:
        asyncio.run(cli.cmd_comments(args, None, None))

    assert exc.value.code == 1
