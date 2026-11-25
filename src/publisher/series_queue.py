"""Keep a series claim queue in step with publishing (REQ-PUB-168).

A series render leaves `series.json` beside its video (the claim and the
queue file, written by the producer), so the publisher needs no video
config. The claim row goes to `scheduled` with its post id when the
publisher records the post, and to `live` when the delivery sweep sees the
post live on any platform. Bookkeeping: nothing here raises into a publish.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.claim_queue import set_published

logger = logging.getLogger(__name__)

SERIES_FILE = "series.json"


def series_record(product_id: str, outputs_dir: Path) -> tuple[Path, str] | None:
    """The render's queue file and claim, or None for a non-series render."""
    try:
        data = json.loads((outputs_dir / product_id / SERIES_FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    queue, claim = data.get("queue_file"), data.get("claim")
    if not isinstance(queue, str) or not isinstance(claim, str) or not claim:
        return None
    return Path(queue), claim


def mark_scheduled(found: tuple[Path, str], post_id: str) -> None:
    """Set a series render's claim row to `scheduled` with its post id."""
    set_published(found[0], found[1], "scheduled", post_id)


# A partial post is live on at least one platform.
LIVE_STATUSES = frozenset({"published", "partial"})


def _history_series(outputs_dir: Path) -> dict[str, dict[str, str]]:
    """Post id -> {claim, queue_file} from publish history's series rows."""
    from src.publisher.tracking import load_tracking

    found: dict[str, dict[str, str]] = {}
    for record in load_tracking(outputs_dir).get("posts", {}).values():
        series = record.get("series") if isinstance(record, dict) else None
        if (
            isinstance(series, dict)
            and series.get("claim")
            and series.get("queue_file")
        ):
            found[str(record.get("post_id"))] = series
    return found


def has_series_posts(outputs_dir: Path) -> bool:
    """Whether publish history holds any series post."""
    return bool(_history_series(outputs_dir))


def mark_live(posts: list[dict[str, Any]], outputs_dir: Path) -> int:
    """Set each live series post's claim row to `live`; returns how many.

    Read from publish history, not the render directory, which cleanup
    removes after publishing.
    """
    series_by_post = _history_series(outputs_dir)
    marked = 0
    for post in posts:
        post_id = str(post.get("id"))
        series = series_by_post.get(post_id)
        live = str(post.get("status") or "").lower() in LIVE_STATUSES
        if (
            series
            and live
            and set_published(Path(series["queue_file"]), series["claim"], "live")
        ):
            marked += 1
    return marked
