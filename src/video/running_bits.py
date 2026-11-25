"""Which running bit, if any, a render's script is offered (REQ-CNT-188).

A running bit builds recognition until it wears out, so it is offered only
while the cap allows: at most `per_batch` uses in one process (a producer or
batch run), and only after `min_gap` renders of the kind since the last use.
Uses are read back from the render-choices rows, which record the id of a bit
the script used.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.utils.render_choices_store import latest_per_product, load_recent
from src.video.config.identity_models import NarratorIdentity, RunningBit

# Rows recorded since this time belong to the current run.
RUN_STARTED = datetime.now(UTC).isoformat(timespec="seconds")


def select_running_bit(
    identity: NarratorIdentity,
    outputs_dir: Path,
    kind: str,
    run_started: str = RUN_STARTED,
) -> RunningBit | None:
    """The bit to offer this render, or None when the cap holds them back."""
    if not identity.running_bits:
        return None
    rows = [
        row
        for row in latest_per_product(load_recent(outputs_dir, 0))
        if row.get("content_kind") == kind
    ]
    cap = identity.running_bit_cap
    used = [i for i, row in enumerate(rows) if row.get("running_bit")]
    this_run = sum(
        1 for i in used if str(rows[i].get("recorded_at", "")) >= run_started
    )
    if this_run >= cap.per_batch:
        return None
    if used and len(rows) - 1 - used[-1] < cap.min_gap:
        return None
    # The bit used longest ago (or never) goes next.
    last_seen = {str(rows[i]["running_bit"]): i for i in used}
    return min(identity.running_bits, key=lambda b: last_seen.get(b.id, -1))


def used_bit(script: str, bit: RunningBit | None) -> str | None:
    """The bit's id when the script carries one of its cues."""
    if bit is None or not script:
        return None
    text = script.lower()
    return bit.id if any(cue.lower() in text for cue in bit.cues) else None
