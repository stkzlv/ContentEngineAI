"""Tests for Vercel Blob retention policy."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from src.publisher import blob_retention
from src.publisher.blob_retention import (
    run_blob_retention,
    select_blobs_to_delete,
)
from src.publisher.models import BlobRetentionConfig

NOW = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
MB = 1024 * 1024


def _blob(url: str, age_days: int, size_mb: int = 5) -> dict:
    uploaded = (NOW - timedelta(days=age_days)).isoformat()
    return {"url": url, "size": size_mb * MB, "uploadedAt": uploaded}


def test_age_policy_deletes_old_unprotected():
    policy = BlobRetentionConfig(enabled=True, max_age_days=30, max_total_mb=10_000)
    blobs = [_blob("old", 45), _blob("fresh", 5)]
    selected = select_blobs_to_delete(blobs, policy, set(), NOW)
    assert [b["url"] for b in selected] == ["old"]


def test_protected_blobs_survive_both_policies():
    # Old AND over-size, but referenced by an unpublished post: never deleted.
    policy = BlobRetentionConfig(enabled=True, max_age_days=30, max_total_mb=1)
    blobs = [_blob("scheduled", 90, size_mb=50)]
    selected = select_blobs_to_delete(blobs, policy, {"scheduled"}, NOW)
    assert selected == []


def test_size_policy_trims_oldest_first():
    # All within age, store over the cap: oldest unprotected go first.
    policy = BlobRetentionConfig(enabled=True, max_age_days=365, max_total_mb=10)
    blobs = [_blob("d10", 10), _blob("d20", 20), _blob("d5", 5)]  # 15 MB total
    selected = select_blobs_to_delete(blobs, policy, set(), NOW)
    assert [b["url"] for b in selected] == ["d20"]  # 10 MB left, under cap


def test_protected_size_counts_toward_total():
    # Protected blob pushes the store over the cap; unprotected oldest pay.
    policy = BlobRetentionConfig(enabled=True, max_age_days=365, max_total_mb=10)
    blobs = [_blob("pinned", 1, size_mb=8), _blob("d20", 20), _blob("d5", 5)]
    selected = select_blobs_to_delete(blobs, policy, {"pinned"}, NOW)
    assert [b["url"] for b in selected] == ["d20", "d5"]


def test_nothing_to_delete_under_policy():
    policy = BlobRetentionConfig(enabled=True, max_age_days=30, max_total_mb=100)
    blobs = [_blob("fresh", 1)]
    assert select_blobs_to_delete(blobs, policy, set(), NOW) == []


def test_unparseable_timestamp_treated_as_new():
    policy = BlobRetentionConfig(enabled=True, max_age_days=30, max_total_mb=100)
    blobs = [{"url": "weird", "size": MB, "uploadedAt": "not-a-date"}]
    assert select_blobs_to_delete(blobs, policy, set(), NOW) == []


def _publisher(token: str | None = "tok") -> AsyncMock:  # noqa: S107 - test stub
    pub = AsyncMock()
    pub.vercel_token = token
    pub.get_unpublished_media_urls = AsyncMock(return_value=set())
    return pub


@pytest.mark.asyncio
async def test_run_skips_when_disabled():
    pub = _publisher()
    await run_blob_retention(pub, BlobRetentionConfig(enabled=False))
    pub.get_unpublished_media_urls.assert_not_called()


@pytest.mark.asyncio
async def test_run_skips_without_token():
    pub = _publisher(token=None)
    await run_blob_retention(pub, BlobRetentionConfig(enabled=True))
    pub.get_unpublished_media_urls.assert_not_called()


@pytest.mark.asyncio
async def test_run_deletes_selected_blobs(monkeypatch):
    pub = _publisher()
    pub.get_unpublished_media_urls = AsyncMock(return_value={"keep"})
    blobs = [_blob("old", 90), _blob("keep", 90), _blob("fresh", 1)]
    deleted: list[list[str]] = []

    async def fake_list(session, token):
        return blobs

    async def fake_delete(session, token, urls):
        deleted.append(urls)

    monkeypatch.setattr(blob_retention, "list_blobs", fake_list)
    monkeypatch.setattr(blob_retention, "delete_blobs", fake_delete)

    policy = BlobRetentionConfig(enabled=True, max_age_days=30, max_total_mb=10_000)
    await run_blob_retention(pub, policy, now=NOW)

    assert deleted == [["old"]]


@pytest.mark.asyncio
async def test_run_swallows_errors(monkeypatch):
    # Retention failures must never propagate into the publish flow.
    pub = _publisher()
    pub.get_unpublished_media_urls = AsyncMock(side_effect=RuntimeError("api down"))
    await run_blob_retention(pub, BlobRetentionConfig(enabled=True))
