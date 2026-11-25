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
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.publisher.late.client import LatePublisher
    from src.publisher.models import DeliverySweepConfig

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


# Rejections a bare retry resubmits unchanged; the payload has to change first.
_NEEDS_PAYLOAD_CHANGE = ("commercial content disclosure", "description is invalid")


def _hint(post: PartialPost) -> str:
    """What fixes this post, from what the legs report.

    A top status of ``partial`` with no failed leg is a stale top status
    (seen live on two posts whose legs all read ``published``); there is
    nothing to retry, so the hint says to read the post rather than name
    ``posts.retry`` for a call that would do nothing.
    """
    if not post.failed_legs:
        return (
            f"no failed leg; top status may be stale, check posts.get({post.post_id!r})"
        )
    messages = " ".join((leg.error_message or "").lower() for leg in post.failed_legs)
    if any(marker in messages for marker in _NEEDS_PAYLOAD_CHANGE):
        return f"posts.update({post.post_id!r}, ...) then posts.retry({post.post_id!r})"
    return f"posts.retry({post.post_id!r})"


async def run_delivery_sweep(
    publisher: LatePublisher,
    config: DeliverySweepConfig | None,
    outputs_dir: Path,
) -> None:
    """Sweep a trailing window of posts after a publish run and WARN per miss.

    Never raises: the publish that Zernio already accepted must not be
    reported failed because a status read failed afterwards. Runs on every
    publish path, including one that published nothing, because its value is
    in the previous runs' posts, which have fired since; the post this run
    created is still pending and cannot be judged.
    """
    if config is None or not config.enabled:
        logger.debug("Delivery sweep disabled - skipping")
        return
    try:
        results = await sweep_partial_posts(publisher, config.limit)
    except Exception as exc:  # noqa: BLE001 - a hook must not fail the publish
        logger.warning("Delivery sweep failed: %s", exc)
        return
    for post in results:
        legs = (
            ", ".join(
                f"{leg.platform} ({leg.error_category or leg.status})"
                for leg in post.failed_legs
            )
            if post.failed_legs
            else "no per-leg detail"
        )
        logger.warning(
            "Delivery incomplete: post %s status=%s failing: %s; fix: %s",
            post.post_id,
            post.top_status,
            legs,
            _hint(post),
        )
        for leg in post.failed_legs:
            if leg.error_message:
                logger.warning("  %s error: %s", leg.platform, leg.error_message)
    logger.info(
        "Delivery sweep: %d of the last %d post(s) with failed/partial delivery",
        len(results),
        config.limit,
    )
    await _mark_live_series(publisher, outputs_dir)


async def _mark_live_series(publisher: LatePublisher, outputs_dir: Path) -> None:
    """Move series posts live on any platform to `live` (REQ-PUB-168).

    Only when publish history holds a series post, so other installations
    make no extra call. Never raises, like the sweep.
    """
    from src.publisher.series_queue import has_series_posts, mark_live

    root = outputs_dir
    if not has_series_posts(root):
        return
    try:
        # The whole list the call returns: a series post can sit past the
        # sweep's window once newer posts are scheduled.
        posts = await publisher.list_posts()
        marked = mark_live(posts, root)
    except Exception as exc:  # noqa: BLE001 - a hook must not fail the publish
        logger.warning("Claim queue live marks skipped: %s", exc)
        return
    if marked:
        logger.info("Claim queue: %d series post(s) marked live", marked)
