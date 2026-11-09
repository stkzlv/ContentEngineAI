"""First-person claims of a physical act, for a virtual narrator (REQ-CNT-173).

A synthetic narrator can't buy, hold, try on or test anything, and a script
that has it say so makes a claim it can't have made true. The patterns match a
first-person subject followed by a physical verb; opinion, reading and
checking ("I think", "I read", "I checked") and "tried to" don't match.
"""

from __future__ import annotations

import re

_SUBJECT = (
    r"\b(?:i|i've|i have|i'd|we|we've|we have)\s+(?:just\s+|finally\s+|actually\s+)?"
)

DEFAULT_PATTERNS: tuple[str, ...] = (
    _SUBJECT + r"(?:bought|ordered|purchased|unboxed|held|wore|worn)\b",
    _SUBJECT + r"(?:tested|used)\s+(?:it|them|this|these|mine|one)\b",
    _SUBJECT + r"tried\s+(?:it|them|this|these|mine|one)\b",
    _SUBJECT + r"(?:picked|plugged)\s+(?:it|them|one)\s+(?:up|in)\b",
    _SUBJECT + r"took\s+(?:it|them)\s+(?:to|on|out)\b",
    _SUBJECT + r"spent\s+(?:\$|\d|a\s+week|a\s+month|weeks|months)",
)


def compile_patterns(patterns: list[str] | tuple[str, ...]) -> list[re.Pattern[str]]:
    """Case-insensitive patterns; a typo in a configured one raises at load."""
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def physical_claim(sentences: list[str], patterns: list[re.Pattern[str]]) -> str | None:
    """The first sentence that claims a first-person physical act."""
    for sentence in sentences:
        # A curly apostrophe reads as a straight one.
        text = sentence.replace("\u2019", "'")
        if any(p.search(text) for p in patterns):
            return sentence
    return None
