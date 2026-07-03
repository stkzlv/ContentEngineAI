"""Sweep delivered posts for silently-failed platform legs.

Zernio reports a multi-platform post ``partial`` when some platforms went live
and one leg failed at publish time; a fully-failed post is ``failed``. Neither
is surfaced anywhere, so a dead leg costs reach with no alert. This module
scans recent posts and returns the ones whose delivery is incomplete, with the
failing platform and error, so the miss stops being invisible.

It reads Zernio's per-platform status (``platforms[*].status``), not
``publish_history.json``, which records queue time, not live delivery. The fix
for a flagged post is ``posts.retry(post_id)``: it re-publishes the failed leg
from Zernio's CDN with no re-render and leaves already-published platforms
untouched.

The sweep function does not log; the caller chooses the severity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.publisher.late.client import LatePublisher

logger = logging.getLogger(__name__)

# Top-level post statuses that mean at least one leg didn't deliver.
INCOMPLETE_STATUSES = frozenset({"partial", "failed"})
# Per-leg statuses that mean the leg didn't publish.
FAILED_LEG_STATUSES = frozenset({"failed", "error"})


@dataclass
class FailedLeg:
    """One platform leg of a post that didn't publish."""

    platform: str
    status: str | None
    error_message: str | None
    error_category: str | None


@dataclass
class PartialPost:
    """A recent post whose delivery is incomplete, with its failed legs."""

    post_id: str
    top_status: str
    scheduled_for: str | None
    failed_legs: list[FailedLeg] = field(default_factory=list)


async def sweep_partial_posts(
    client: LatePublisher, limit: int = 25
) -> list[PartialPost]:
    """Return recent posts whose delivery is incomplete.

    Scans the most recent ``limit`` posts, keeps those whose top status is
    ``partial`` or ``failed``, and attaches each failed platform leg with its
    error. Does not log; the caller decides what a hit means.
    """
    posts = (await client.list_posts())[:limit]
    results: list[PartialPost] = []
    for post in posts:
        top = str(post.get("status") or "").lower()
        if top not in INCOMPLETE_STATUSES:
            continue
        post_id = post.get("id")
        if not post_id:
            continue
        legs = await client.get_post_platforms(post_id)
        failed = [
            FailedLeg(
                platform=str(leg.get("platform") or ""),
                status=leg.get("status"),
                error_message=leg.get("error_message"),
                error_category=leg.get("error_category"),
            )
            for leg in legs
            if str(leg.get("status") or "").lower() in FAILED_LEG_STATUSES
        ]
        results.append(
            PartialPost(
                post_id=str(post_id),
                top_status=top,
                scheduled_for=post.get("scheduledFor"),
                failed_legs=failed,
            )
        )
    return results
