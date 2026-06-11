"""Vercel Blob retention: trim the upload store after publishing.

Videos over the direct-upload limit are staged in the user's Vercel Blob
store and fetched by the scheduling service when a post goes live. Nothing
upstream deletes them, so the store grows until the free tier pauses access
and large uploads start failing. This module applies a config-driven
retention policy (age + total size) after a publish run.

Safety rule (unconditional, not configurable): a blob referenced by any post
that isn't fully published yet is never deleted, regardless of policy.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from src.publisher.late.client import LatePublisher
    from src.publisher.models import BlobRetentionConfig

logger = logging.getLogger(__name__)

BLOB_API_BASE = "https://blob.vercel-storage.com"
LIST_PAGE_SIZE = 1000
DELETE_BATCH_SIZE = 100


async def list_blobs(
    session: aiohttp.ClientSession, token: str
) -> list[dict[str, Any]]:
    """Return every blob in the store (paginated list API)."""
    blobs: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        url = f"{BLOB_API_BASE}/?limit={LIST_PAGE_SIZE}"
        if cursor:
            url += f"&cursor={cursor}"
        async with session.get(  # type: ignore[attr-defined]
            url, headers={"Authorization": f"Bearer {token}"}
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        blobs.extend(data.get("blobs") or [])
        cursor = data.get("cursor")
        if not data.get("hasMore"):
            return blobs


async def delete_blobs(
    session: aiohttp.ClientSession, token: str, urls: list[str]
) -> None:
    """Delete blobs by URL, batched to the API's tolerance."""
    for start in range(0, len(urls), DELETE_BATCH_SIZE):
        batch = urls[start : start + DELETE_BATCH_SIZE]
        async with session.post(
            f"{BLOB_API_BASE}/delete",
            headers={"Authorization": f"Bearer {token}"},
            json={"urls": batch},
        ) as resp:
            resp.raise_for_status()


def _uploaded_at(blob: dict[str, Any], fallback: datetime) -> datetime:
    """Parse a blob's upload time; unparseable timestamps count as new."""
    raw = blob.get("uploadedAt")
    if not raw:
        return fallback
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return fallback


def select_blobs_to_delete(
    blobs: list[dict[str, Any]],
    policy: BlobRetentionConfig,
    protected_urls: set[str],
    now: datetime,
) -> list[dict[str, Any]]:
    """Pick blobs to delete under the age + size policy.

    Protected URLs (posts not yet fully published) are never selected, but
    their size still counts toward the store total the size policy trims to.
    """
    candidates = [b for b in blobs if b.get("url") not in protected_urls]

    cutoff = now - timedelta(days=policy.max_age_days)
    selected: dict[str, dict[str, Any]] = {
        b["url"]: b for b in candidates if _uploaded_at(b, now) < cutoff
    }

    max_bytes = policy.max_total_mb * 1024 * 1024
    remaining_total = sum(int(b.get("size") or 0) for b in blobs) - sum(
        int(b.get("size") or 0) for b in selected.values()
    )
    if remaining_total > max_bytes:
        rest = sorted(
            (b for b in candidates if b.get("url") not in selected),
            key=lambda b: _uploaded_at(b, now),
        )
        for blob in rest:
            if remaining_total <= max_bytes:
                break
            selected[blob["url"]] = blob
            remaining_total -= int(blob.get("size") or 0)

    return list(selected.values())


async def run_blob_retention(
    publisher: LatePublisher, policy: BlobRetentionConfig | None
) -> None:
    """Apply the retention policy once after a publish run.

    Never raises: any failure logs a WARNING and returns, so retention can't
    break a publish that already succeeded.
    """
    if policy is None or not policy.enabled:
        logger.debug("Blob retention disabled - skipping")
        return
    token = publisher.vercel_token
    if not token:
        logger.debug("No Vercel token configured - skipping blob retention")
        return

    try:
        protected = await publisher.get_unpublished_media_urls()
        async with aiohttp.ClientSession() as session:
            blobs = await list_blobs(session, token)
            to_delete = select_blobs_to_delete(
                blobs, policy, protected, datetime.now(UTC)
            )
            if not to_delete:
                logger.info(
                    "Blob retention: nothing to delete (%d blobs, %d protected)",
                    len(blobs),
                    len(protected),
                )
                return
            await delete_blobs(session, token, [b["url"] for b in to_delete])
        freed = sum(int(b.get("size") or 0) for b in to_delete)
        logger.info(
            "Blob retention: deleted %d blob(s), freed %.1f MB "
            "(%d blobs kept, %d protected)",
            len(to_delete),
            freed / (1024 * 1024),
            len(blobs) - len(to_delete),
            len(protected),
        )
    except Exception as exc:  # noqa: BLE001 - retention must never break publishing
        logger.warning("Blob retention failed (publish unaffected): %s", exc)
