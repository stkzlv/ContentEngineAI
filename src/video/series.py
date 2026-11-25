"""A series episode's opening line, title prefix and claim queue.

REQ-CNT-194: the script opens on the series' opening line word for word, and
the title and platform captions start with the prefix, the episode number
under `numbering: episode`, then the search phrase. REQ-CNT-195: the claim
queue's row gets its episode number before the script and description steps
and `rendered` with the output id when the render succeeds.
"""

from __future__ import annotations

import csv
import fcntl
import logging
import re
from pathlib import Path
from typing import Any

from src.video.config.series_models import SeriesConfig

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


def series_for(config: Any, product: Any) -> SeriesConfig | None:
    """The series a record belongs to, or None."""
    series_id = getattr(product, "series", None)
    return config.series.get(series_id) if series_id else None


def opening_line(series: SeriesConfig, product: Any) -> str:
    """The opening line with the claim filled in; empty when none is set."""
    if not series.opening_line:
        return ""
    return series.opening_line.replace("{claim}", getattr(product, "claim", "") or "")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def place_opening(script: str, line: str) -> str:
    """The script starting with `line` word for word, prepended if it doesn't."""
    if not line:
        return script
    if _norm(script).startswith(_norm(line)):
        return script
    return f"{line} {script.lstrip()}"


def prefix(series: SeriesConfig, episode: int | None) -> str:
    """The title and caption prefix, with the episode number when numbered."""
    if series.numbering == "episode" and episode is not None:
        return f"{series.title_prefix}{episode} "
    return series.title_prefix


def with_prefix(text: str | None, pre: str) -> str | None:
    """`text` starting with `pre`, unless it already does."""
    if not pre or text is None:
        return text
    return text if text.startswith(pre) else f"{pre}{text}"


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    # utf-8-sig: a spreadsheet's "CSV UTF-8" export starts with a BOM, which
    # would otherwise become part of the first column's name.
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(QUEUE_COLUMNS)
    for row in rows:
        columns += [k for k in row if k not in columns]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _update_row(path: Path, claim: str, update: Any) -> dict[str, str] | None:
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
            rows = _rows(path)
            if any(None in row for row in rows):
                # A row with more cells than the header (an unquoted comma):
                # rewriting it would shift every cell after the extra one.
                logger.warning("Claim queue %s has a row with extra cells", path)
                return None
            row = next(
                (r for r in rows if _norm(r.get("claim") or "") == _norm(claim)), None
            )
            if row is None:
                logger.warning("Claim queue %s has no row for claim %r", path, claim)
                return None
            update(row, rows)
            _write(path, rows)
            return row
    except (OSError, ValueError, csv.Error) as e:
        logger.warning("Claim queue %s not updated: %s", path, e)
        return None


def assign_episode(series: SeriesConfig, product: Any) -> int | None:
    """The claim's episode number, filled in when empty (REQ-CNT-195)."""
    if series.numbering != "episode" or series.queue_file is None:
        return None

    def fill(row: dict[str, str], rows: list[dict[str, str]]) -> None:
        if not (row.get("episode") or "").strip().isdigit():
            values = [(r.get("episode") or "").strip() for r in rows]
            numbers = [int(v) for v in values if v.isdigit()]
            row["episode"] = str(max(numbers, default=0) + 1)

    row = _update_row(series.queue_file, getattr(product, "claim", "") or "", fill)
    return int(row["episode"]) if row else None


def mark_rendered(series: SeriesConfig, product: Any, output_id: str) -> None:
    """Set the claim's row to `rendered` with its output id (REQ-CNT-195)."""
    if series.queue_file is None:
        return

    def done(row: dict[str, str], rows: list[dict[str, str]]) -> None:
        # A knowing re-render of an episode already scheduled or live leaves
        # the row where the publisher put it.
        if (row.get("status") or "").strip() in ("scheduled", "live"):
            return
        row["status"] = "rendered"
        row["output_id"] = output_id

    _update_row(series.queue_file, getattr(product, "claim", "") or "", done)
