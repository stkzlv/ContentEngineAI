"""The verdict stamp card of a series episode (REQ-VID-167).

The script step names the sentence that states the verdict; the card shows
the topic's verdict label from that sentence's first spoken word. Without a
named sentence, it lands on the label's first spoken mention after the
opening line.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.video.config.series_models import VerdictBeat

VERDICT_PROMPT = (
    "Below is a short video script, one numbered sentence per line. The "
    "script checks a claim and reaches the verdict {label}. Reply with JSON "
    'only, {{"verdict": N}}: N is the number of the sentence that states the '
    "verdict. Use null when no sentence does.\n\n{sentences}"
)

_TOKEN = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class VerdictCard:
    """The label, when the card is on screen in seconds, and its look."""

    label: str
    start: float
    end: float
    beat: VerdictBeat


def verdict_prompt(sentences: list[str], label: str) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    return VERDICT_PROMPT.format(label=json.dumps(label), sentences=numbered)


def parse_verdict(reply: str, count: int) -> int | None:
    """The 0-based verdict sentence index from the reply, or None."""
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    value = data.get("verdict") if isinstance(data, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= count:
        return value - 1
    return None


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _start(word: dict[str, Any]) -> float | None:
    value = word.get("start_time", word.get("start"))
    return float(value) if isinstance(value, int | float) else None


def mention_start(
    words: list[dict[str, Any]], label: str, skip_words: int
) -> float | None:
    """When the label is first spoken after the first `skip_words` words.

    Each transcript word counts as its first token, so a word Whisper splits
    ("it's") still counts once, as the opening line's words do.
    """
    target = _tokens(label)
    if not target:
        return None
    flat = [
        (toks[0], _start(w)) for w in words if (toks := _tokens(str(w.get("word", ""))))
    ]
    for i in range(skip_words, len(flat) - len(target) + 1):
        if [t for t, _ in flat[i : i + len(target)]] == target:
            return flat[i][1]
    return None
