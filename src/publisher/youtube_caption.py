"""YouTube's own copy of a post's metadata (REQ-PUB-148, REQ-CNT-150).

Shared by every caption builder: `publish_modes` (unified and
platform-specific publishing, `single`, the batches) and the scheduler.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.publisher.models import (
    _TOPIC_ID_PREFIX,
    Platform,
    _trim_on_word_boundary,
    description_budget,
)
from src.utils.youtube_description import (
    PROFILE_POINTER,
    drop_urls,
    with_profile_pointer,
    without_profile_pointer,
)


def youtube_metadata(meta: Any, product_id: str, *, pointer: bool) -> Any:
    """The metadata as YouTube posts it: no URL, and a profile pointer.

    A link in a Shorts description can't be clicked (REQ-PUB-148), so URLs
    go, and a product video ends on "Link in bio." when `pointer` is on and
    the description names no profile link already (REQ-CNT-150); with it
    off, a line that is only a bio-link pointer goes, since metadata written
    while it was on still carries one. Unified
    metadata shares one description across platforms, so this builds a copy
    for the YouTube entry only. The copy fits YouTube's caps with the pointer
    kept: the body is trimmed, never the pointer.
    """
    if not isinstance(getattr(meta, "description", None), str):
        return meta
    body = drop_urls(meta.description)
    if not pointer:
        body = without_profile_pointer(body)
        if not body:
            # Nothing but a pointer: the title stands in, since an empty
            # description is refused and the pointer may not stay.
            body = drop_urls(str(getattr(meta, "title", "") or ""))
    wants_pointer = bool(
        pointer and product_id and not product_id.startswith(_TOPIC_ID_PREFIX)
    )
    tail = ""
    if wants_pointer and (not body or with_profile_pointer(body) != body):
        tail = f"\n\n{PROFILE_POINTER}" if body else PROFILE_POINTER
    if not body and not tail:
        # Nothing but a URL, and no pointer to stand in: an empty description
        # is refused, so the text stays as written.
        body = meta.description

    youtube = replace(meta, description=body + tail)
    youtube.character_counts = dict(meta.character_counts)
    youtube.clamp_for_platforms([Platform.YOUTUBE])
    if tail and not youtube.description.endswith(tail):
        wrapper = len(youtube.format_content()) - len(youtube.description)
        budget = description_budget([Platform.YOUTUBE], wrapper) or 0
        youtube.description = (
            _trim_on_word_boundary(body, max(budget - len(tail), 0)) + tail
        )
        youtube.character_counts["description"] = len(youtube.description)
    return youtube
