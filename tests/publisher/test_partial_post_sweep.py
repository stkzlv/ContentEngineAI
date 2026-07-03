"""Tests for the partial-post delivery sweep."""

from unittest.mock import AsyncMock

import pytest

from src.publisher.partial_post_sweep import sweep_partial_posts


def _leg(platform, status, error_message=None, error_category=None):
    return {
        "platform": platform,
        "platform_post_id": None,
        "account_id": "acc",
        "status": status,
        "error_message": error_message,
        "error_category": error_category,
    }


@pytest.mark.asyncio
async def test_sweep_flags_partial_and_failed_only():
    client = AsyncMock()
    client.list_posts = AsyncMock(
        return_value=[
            {"id": "p_pub", "status": "published", "scheduledFor": "t"},
            {"id": "p_part", "status": "partial", "scheduledFor": "t"},
            {"id": "p_fail", "status": "failed", "scheduledFor": "t"},
            {"id": "p_sched", "status": "scheduled", "scheduledFor": "t"},
        ]
    )

    async def platforms(post_id):
        if post_id == "p_part":
            return [
                _leg("youtube", "published"),
                _leg("tiktok", "failed", "upload timed out", "platform_rejected"),
            ]
        if post_id == "p_fail":
            return [_leg("tiktok", "failed", "disclosure error", "account_issue")]
        return []

    client.get_post_platforms = AsyncMock(side_effect=platforms)

    results = await sweep_partial_posts(client, limit=25)

    ids = [r.post_id for r in results]
    assert ids == ["p_part", "p_fail"]  # published/scheduled skipped
    part = results[0]
    assert part.top_status == "partial"
    assert [leg.platform for leg in part.failed_legs] == ["tiktok"]  # youtube ok
    assert part.failed_legs[0].error_message == "upload timed out"
    assert part.failed_legs[0].error_category == "platform_rejected"


@pytest.mark.asyncio
async def test_sweep_partial_with_no_failed_leg_still_flagged():
    # A partial post whose legs don't report a failed status still gets flagged
    # on the top-level status, with an empty failed_legs list.
    client = AsyncMock()
    client.list_posts = AsyncMock(
        return_value=[{"id": "p", "status": "partial", "scheduledFor": "t"}]
    )
    client.get_post_platforms = AsyncMock(
        return_value=[_leg("youtube", "published"), _leg("tiktok", "pending")]
    )

    results = await sweep_partial_posts(client, limit=25)

    assert len(results) == 1
    assert results[0].failed_legs == []


@pytest.mark.asyncio
async def test_sweep_respects_limit():
    client = AsyncMock()
    client.list_posts = AsyncMock(
        return_value=[
            {"id": f"p{i}", "status": "partial", "scheduledFor": "t"} for i in range(10)
        ]
    )
    client.get_post_platforms = AsyncMock(return_value=[])

    results = await sweep_partial_posts(client, limit=3)

    assert len(results) == 3  # only the first 3 of 10 swept


@pytest.mark.asyncio
async def test_sweep_empty_when_all_delivered():
    client = AsyncMock()
    client.list_posts = AsyncMock(
        return_value=[{"id": "p", "status": "published", "scheduledFor": "t"}]
    )
    client.get_post_platforms = AsyncMock(return_value=[])

    results = await sweep_partial_posts(client, limit=25)

    assert results == []
    client.get_post_platforms.assert_not_called()  # never inspected a delivered post
