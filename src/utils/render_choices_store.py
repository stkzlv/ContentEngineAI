"""Where render choices are kept, and how they are read back.

The producer appends one row per finished render; the variety report and the
analytics segment report read them. Kept apart from `src.video` so the
publisher can read the rows without importing the video package.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.utils.outputs_paths import durable_state_path

CHOICES_FILENAME = "render_choices.jsonl"


def choices_path(outputs_dir: Path) -> Path:
    return durable_state_path(outputs_dir, CHOICES_FILENAME)


def load_recent(outputs_dir: Path, last: int) -> list[dict[str, Any]]:
    """The newest `last` rows, skipping lines that are not valid JSON.

    Never raises: a write cut off mid-character (an out-of-memory kill, a full
    disk) leaves bytes that are not UTF-8, and the script step reads this file,
    so a broken store must not fail a render.
    """
    path = choices_path(outputs_dir)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows[-last:] if last > 0 else rows


def latest_per_product(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each product's newest row, in order. A resume of a finished product
    appends a second row for the same render, which would count twice.
    """
    newest: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        # A row with no id is its own product, not one shared "None".
        key = str(row["product_id"]) if row.get("product_id") else f"#{i}"
        newest.pop(key, None)
        newest[key] = row
    return list(newest.values())
