"""Tests for first-comment delivery verification."""

from unittest.mock import AsyncMock

import pytest

from src.publisher.comment_verify import (
    FIRST_COMMENT_PLATFORMS,
    verify_post_first_comments,
)


def _owner_comment() -> dict:
    return {"message": "Get it here: ...", "from": {"name": "@chan", "isOwner": True}}


def _viewer_comment() -> dict:
    return {"message": "nice", "from": {"name": "@viewer", "isOwner": False}}


def _client(platforms: list[dict], comments_by_id: dict[str, list]) -> AsyncMock:
    client = AsyncMock()
    client.get_post_platforms = AsyncMock(return_value=platforms)

    async def _get_comments(platform_post_id, account_id, limit=25):
        return comments_by_id.get(platform_post_id, [])

    client.get_post_comments = AsyncMock(side_effect=_get_comments)
    return client


@pytest.mark.asyncio
async def test_owner_comment_present():
    client = _client(
        [{"platform": "youtube", "platform_post_id": "vid1", "account_id": "acc"}],
        {"vid1": [_owner_comment()]},
    )
    checks = await verify_post_first_comments(client, "post1")
    assert len(checks) == 1
    assert checks[0].platform == "youtube"
    assert checks[0].present is True


@pytest.mark.asyncio
async def test_owner_comment_absent():
    # Only a viewer comment exists, no owner first comment.
    client = _client(
        [{"platform": "instagram", "platform_post_id": "ig1", "account_id": "acc"}],
        {"ig1": [_viewer_comment()]},
    )
    checks = await verify_post_first_comments(client, "post1")
    assert len(checks) == 1
    assert checks[0].present is False


@pytest.mark.asyncio
async def test_tiktok_skipped():
    client = _client(
        [{"platform": "tiktok", "platform_post_id": "tt1", "account_id": "acc"}],
        {},
    )
    checks = await verify_post_first_comments(client, "post1")
    assert checks == []
    client.get_post_comments.assert_not_called()


@pytest.mark.asyncio
async def test_not_live_skipped():
    # No platform_post_id yet -> the post isn't live, skip without an API call.
    client = _client(
        [{"platform": "youtube", "platform_post_id": None, "account_id": "acc"}],
        {},
    )
    checks = await verify_post_first_comments(client, "post1")
    assert checks == []
    client.get_post_comments.assert_not_called()


def test_first_comment_platforms_excludes_tiktok():
    assert "youtube" in FIRST_COMMENT_PLATFORMS
    assert "instagram" in FIRST_COMMENT_PLATFORMS
    assert "tiktok" not in FIRST_COMMENT_PLATFORMS
