"""CLAUDE.md stays a rules file, and every note it moved out stays reachable.

It had grown to 224KB -- roughly 50k tokens read at the start of every assisted
session, most of it per-entry postmortems rather than rules that have to be in
front of you at all times. The entries are worth keeping: each one records a
defect that a plausible change had already caused. What they are not is
something to re-read on the way to a one-line fix.

Two failure modes follow the split, and neither shows up in review:

The file grows back. Every entry arrives one at a time, each small, each
plainly belonging where the rule is; the size is the only thing that says the
sum has stopped being a rules file.

A notes file is orphaned. Nothing links `docs/` from the code, so a file no
pointer names is invisible -- it stays correct, stays current, and is read by
nobody.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLAUDE = REPO / "CLAUDE.md"
NOTES = REPO / "docs" / "notes"

# It came out of the split at about 24,000 bytes, from 224,616. The bound is
# close on purpose: reaching it means moving the next section out, not raising
# the number.
MAX_BYTES = 30_000

LINK = re.compile(r"\(docs/notes/([A-Za-z0-9_.-]+\.md)\)")


def test_claude_md_is_still_a_rules_file() -> None:
    size = CLAUDE.stat().st_size

    assert size <= MAX_BYTES, (
        f"CLAUDE.md is {size} bytes, over the {MAX_BYTES} bound. Move a "
        "section's entries into docs/notes/ and leave a pointer, the way the "
        "module notes were moved; do not raise the bound"
    )


def test_every_notes_file_is_linked_from_claude_md() -> None:
    linked = set(LINK.findall(CLAUDE.read_text(encoding="utf-8")))
    present = {p.name for p in NOTES.glob("*.md")}

    assert present, "docs/notes/ is empty"
    assert present <= linked, (
        f"notes nothing points at: {sorted(present - linked)}. A file no "
        "pointer names is read by nobody"
    )
    assert (
        linked <= present
    ), f"CLAUDE.md points at notes that do not exist: {sorted(linked - present)}"


def test_each_notes_file_says_where_it_came_from() -> None:
    """The header is what tells a reader these are entries, not a manual."""
    missing = [
        p.name
        for p in sorted(NOTES.glob("*.md"))
        if "Moved out of CLAUDE.md" not in p.read_text(encoding="utf-8")
    ]

    assert not missing, f"notes files with no provenance header: {missing}"
