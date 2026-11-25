"""The claim queue CSV a series keeps (REQ-CNT-195, REQ-PUB-168).

Shared by the producer, which numbers and renders episodes, and the
publisher, which schedules and publishes them, so neither imports the other.
"""

from __future__ import annotations

import csv
import fcntl
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

QUEUE_COLUMNS = [
    "id",
    "claim",
    "verdict",
    "sources",
    "status",
    "episode",
    "output_id",
    "post_id",
]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    # utf-8-sig: a spreadsheet's "CSV UTF-8" export starts with a BOM, which
    # would otherwise become part of the first column's name.
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(QUEUE_COLUMNS)
    for row in rows:
        columns += [k for k in row if k not in columns]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def update_row(path: Path, claim: str, update: Any) -> dict[str, str] | None:
    """Apply `update(row, rows)` to the claim's row under a file lock.

    Bookkeeping, so a queue it can't read or write logs a warning and never
    fails the render that called it.
    """
    if not path.exists():
        logger.warning("Claim queue %s not found", path)
        return None
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        with lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            rows = read_rows(path)
            if any(None in row for row in rows):
                # A row with more cells than the header (an unquoted comma):
                # rewriting it would shift every cell after the extra one.
                logger.warning("Claim queue %s has a row with extra cells", path)
                return None
            row = next(
                (r for r in rows if norm(r.get("claim") or "") == norm(claim)), None
            )
            if row is None:
                logger.warning("Claim queue %s has no row for claim %r", path, claim)
                return None
            update(row, rows)
            write_rows(path, rows)
            return row
    except (OSError, ValueError, csv.Error) as e:
        logger.warning("Claim queue %s not updated: %s", path, e)
        return None


def set_published(
    queue_file: Path, claim: str, status: str, post_id: str | None = None
) -> bool:
    """Set the claim's row to `scheduled` (with its post id) or `live`
    (REQ-PUB-168); a live row is never moved back to scheduled. Returns
    whether the row changed.
    """
    changed = False

    def set_status(row: dict[str, str], rows: list[dict[str, str]]) -> None:
        nonlocal changed
        current = (row.get("status") or "").strip()
        if status == "scheduled" and current == "live":
            return
        new_id = bool(post_id) and row.get("post_id") != post_id
        changed = current != status or new_id
        row["status"] = status
        if post_id:
            row["post_id"] = post_id

    update_row(queue_file, claim, set_status)
    return changed
