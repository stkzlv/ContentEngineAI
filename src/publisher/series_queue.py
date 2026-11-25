"""Keep a series claim queue in step with publishing (REQ-PUB-168).

A series render leaves `series.json` beside its video (the claim and the
queue file, written by the producer), so the publisher needs no video
config. The claim row goes to `scheduled` with its post id when the
publisher records the post, and to `live` when the delivery sweep sees the
post published. Bookkeeping: nothing here raises into a publish.
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


def mark_scheduled(product_id: str, post_id: str, outputs_dir: Path) -> None:
    """Set a series render's claim row to `scheduled` with its post id."""
    found = series_record(product_id, outputs_dir)
    if found:
        set_published(found[0], found[1], "scheduled", post_id)


def has_series_renders(outputs_dir: Path) -> bool:
    """Whether any render under the outputs root is a series episode."""
    try:
        return any(outputs_dir.glob(f"*/{SERIES_FILE}"))
    except OSError:
        return False


def mark_live(posts: list[dict[str, Any]], outputs_dir: Path) -> int:
    """Set each published series post's claim row to `live`; returns how many."""
    from src.publisher.tracking import load_tracking

    published = {
        str(p.get("id"))
        for p in posts
        if str(p.get("status") or "").lower() == "published"
    }
    marked = set()
    for key, record in load_tracking(outputs_dir).get("posts", {}).items():
        product_id = key.rpartition(":")[0]
        if str(record.get("post_id")) not in published or product_id in marked:
            continue
        found = series_record(product_id, outputs_dir)
        if found:
            set_published(found[0], found[1], "live")
            marked.add(product_id)
    return len(marked)
