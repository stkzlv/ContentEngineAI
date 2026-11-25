"""A series episode's opening line, title prefix and claim queue.

REQ-CNT-194: the script opens on the series' opening line word for word, and
the title and platform captions start with the prefix, the episode number
under `numbering: episode`, then the search phrase. REQ-CNT-195: the claim
queue's row gets its episode number before the script and description steps
and `rendered` with the output id when the render succeeds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.claim_queue import QUEUE_COLUMNS, norm, update_row
from src.video.config.series_models import SeriesConfig

__all__ = ["QUEUE_COLUMNS"]

logger = logging.getLogger(__name__)


def series_for(config: Any, product: Any) -> SeriesConfig | None:
    """The series a record belongs to, or None."""
    series_id = getattr(product, "series", None)
    return config.series.get(series_id) if series_id else None


def opening_line(series: SeriesConfig, product: Any) -> str:
    """The opening line with the claim filled in; empty when none is set."""
    if not series.opening_line:
        return ""
    return series.opening_line.replace("{claim}", getattr(product, "claim", "") or "")


def place_opening(script: str, line: str) -> str:
    """The script starting with `line` word for word, prepended if it doesn't."""
    if not line:
        return script
    if norm(script).startswith(norm(line)):
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


def assign_episode(series: SeriesConfig, product: Any) -> int | None:
    """The claim's episode number, filled in when empty (REQ-CNT-195)."""
    if series.numbering != "episode" or series.queue_file is None:
        return None

    def fill(row: dict[str, str], rows: list[dict[str, str]]) -> None:
        if not (row.get("episode") or "").strip().isdigit():
            values = [(r.get("episode") or "").strip() for r in rows]
            numbers = [int(v) for v in values if v.isdigit()]
            row["episode"] = str(max(numbers, default=0) + 1)

    row = update_row(series.queue_file, getattr(product, "claim", "") or "", fill)
    return int(row["episode"]) if row else None


def write_series_record(series: SeriesConfig, product: Any, run_root: Path) -> None:
    """Leave the claim and queue file beside the render for the publisher,
    which doesn't read the video config (REQ-PUB-168).
    """
    if series.queue_file is None:
        return
    record = {
        "claim": getattr(product, "claim", "") or "",
        "queue_file": str(series.queue_file),
    }
    try:
        (run_root / "series.json").write_text(json.dumps(record), encoding="utf-8")
    except OSError as e:
        logger.warning("series.json not written: %s", e)


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

    update_row(series.queue_file, getattr(product, "claim", "") or "", done)
