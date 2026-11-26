"""The commenter credit line of a series episode (REQ-VID-168).

The claim is spoken inside the opening line, at its `{claim}` slot, so the
line is on screen from the claim's first transcript word to its last.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.video.config.series_models import CreditLine as CreditLineSettings
from src.video.script_structure import _words


@dataclass(frozen=True)
class CreditLine:
    """The line's text, when it is on screen in seconds, and its look."""

    text: str
    start: float
    end: float
    settings: CreditLineSettings


def _time(word: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = word.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def claim_window(
    opening_template: str, claim: str, words: list[dict[str, Any]]
) -> tuple[float, float] | None:
    """When the claim is spoken, from the opening line's `{claim}` slot.

    Transcript words are matched to the opening line by count, as the
    script-shape starts are; None without a slot, a claim or enough words.
    """
    if "{claim}" not in opening_template or not claim.strip():
        return None
    # Counted as the transcript groups words: a stray quote or dash is no word.
    before = _words(opening_template.split("{claim}", 1)[0])
    count = _words(claim)
    if count == 0 or len(words) < before + count:
        return None
    start = _time(words[before], "start_time", "start")
    end = _time(words[before + count - 1], "end_time", "end")
    if start is None or end is None or end <= start:
        return None
    return start, end
