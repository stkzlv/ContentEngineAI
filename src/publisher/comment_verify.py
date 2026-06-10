"""Verify that first comments actually landed on published posts.

Zernio accepts ``platformSpecificData.firstComment`` and reports the post
published with no error, but the comment can silently fail to post on-platform.
The ``posts.get`` response has no comment-delivery field. The platform inbox
does: our first comment is the only comment authored by the account owner
(``from.isOwner == true``). This module checks that signal per platform so a
missing affiliate first comment stops being invisible.

The verify function does not log; callers choose the severity. A sweep over
already-live posts treats a miss as a WARNING; an inline check right after
publish should be softer, because the comment can lag the video.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.publisher.late.client import LatePublisher

logger = logging.getLogger(__name__)

# Platforms that carry a first comment. Matches first_comment.py, which builds
# comments for YouTube and Instagram and returns None for TikTok.
FIRST_COMMENT_PLATFORMS = frozenset({"youtube", "instagram"})


@dataclass
class CommentCheck:
    """Result of checking one platform's first comment on a post."""

    platform: str
    present: bool
    platform_post_id: str


async def verify_post_first_comments(
    client: LatePublisher, post_id: str
) -> list[CommentCheck]:
    """Check each first-comment platform of a post for the owner comment.

    Returns one ``CommentCheck`` per checked platform. Platforms that don't
    carry a first comment (TikTok), or aren't live yet (no ``platform_post_id``),
    are skipped. Does not log; the caller decides what a miss means.
    """
    checks: list[CommentCheck] = []
    platforms = await client.get_post_platforms(post_id)
    for platform in platforms:
        name = str(platform.get("platform") or "").lower()
        if name not in FIRST_COMMENT_PLATFORMS:
            continue
        platform_post_id = platform.get("platform_post_id")
        account_id = platform.get("account_id")
        if not platform_post_id or not account_id:
            continue  # not live yet, nothing to check
        comments = await client.get_post_comments(platform_post_id, account_id)
        checks.append(
            CommentCheck(
                platform=name,
                present=_has_owner_comment(comments),
                platform_post_id=platform_post_id,
            )
        )
    return checks


def _has_owner_comment(comments: list[dict[str, Any]]) -> bool:
    """True when any comment is authored by the account owner (our first comment)."""
    for comment in comments:
        author = comment.get("from")
        if isinstance(author, dict) and author.get("isOwner"):
            return True
    return False
