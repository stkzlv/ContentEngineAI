"""Viewer comments on a published record's posts (REQ-PUB-167).

The operator picks a comment to answer from this list; nothing here writes.
TikTok's API does not let the provider read comments, so its leg is reported
as unsupported rather than called.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.publisher.tracking import load_tracking

logger = logging.getLogger(__name__)

# The provider answers 400 for these: their APIs expose no comment reads.
UNREADABLE_PLATFORMS = frozenset({"tiktok"})
NO_COMMENT_API = "the platform's API exposes no comments"


class CommentReader(Protocol):
    """The two provider calls the listing needs."""

    async def get_post_platforms(self, post_id: str) -> list[dict[str, Any]]: ...

    async def get_post_comments(
        self, platform_post_id: str, account_id: str, limit: int = 25
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ViewerComment:
    """One comment left by someone other than the account."""

    platform: str
    comment_id: str
    author: str
    text: str


@dataclass(frozen=True)
class PlatformComments:
    """The comments read on one platform, or why none were read."""

    platform: str
    comments: list[ViewerComment]
    skipped: str = ""


def record_posts(
    record_id: str, outputs_dir: Path, platforms: list[str] | None = None
) -> dict[str, str]:
    """Map each platform the record went out on to its provider post id.

    Keyed by the history key (`<record id>:<platform>`), so a row backfilled
    from the provider without a `product_id` field still counts.
    """
    wanted = {p.lower() for p in platforms} if platforms else None
    out: dict[str, str] = {}
    for key, row in load_tracking(outputs_dir).get("posts", {}).items():
        record, _, platform = key.rpartition(":")
        if record != record_id or not row.get("post_id"):
            continue
        if wanted is None or platform in wanted:
            out[platform] = str(row["post_id"])
    return out


def _author(comment: dict[str, Any]) -> str:
    author = comment.get("from")
    if not isinstance(author, dict):
        return ""
    # Instagram names the handle `username`; YouTube gives only `name`.
    return str(author.get("username") or author.get("name") or "")


def _viewer_comments(
    platform: str, raw: list[dict[str, Any]], limit: int
) -> list[ViewerComment]:
    out: list[ViewerComment] = []
    for comment in raw:
        author = comment.get("from")
        if isinstance(author, dict) and author.get("isOwner"):
            continue
        out.append(
            ViewerComment(
                platform=platform,
                comment_id=str(comment.get("id") or ""),
                author=_author(comment),
                text=str(comment.get("message") or ""),
            )
        )
    return out[:limit]


async def list_viewer_comments(
    reader: CommentReader, posts: dict[str, str], limit: int = 20
) -> list[PlatformComments]:
    """Read the viewer comments on each platform's post.

    One provider post can carry several platforms, so its legs are read once.
    """
    legs_by_post: dict[str, list[dict[str, Any]]] = {}
    results: list[PlatformComments] = []
    for platform, post_id in sorted(posts.items()):
        if platform in UNREADABLE_PLATFORMS:
            results.append(PlatformComments(platform, [], NO_COMMENT_API))
            continue
        if post_id not in legs_by_post:
            legs_by_post[post_id] = await reader.get_post_platforms(post_id)
        leg = next(
            (
                item
                for item in legs_by_post[post_id]
                if str(item.get("platform") or "").lower() == platform
            ),
            None,
        )
        if not leg or not leg.get("platform_post_id") or not leg.get("account_id"):
            results.append(PlatformComments(platform, [], "not live yet"))
            continue
        # The account's own comments share the page, so ask for room past them.
        raw = await reader.get_post_comments(
            str(leg["platform_post_id"]), str(leg["account_id"]), limit=limit + 5
        )
        results.append(
            PlatformComments(platform, _viewer_comments(platform, raw, limit))
        )
    return results
