"""First-person claims of a physical act, for a virtual narrator (REQ-CNT-173).

A synthetic narrator can't buy, hold, try on or test anything, and a script
that has it say so makes a claim it can't have made true. The patterns match a
first-person subject followed by a physical verb; opinion, reading and
checking ("I think", "I read", "I checked") and "tried to" don't match.
"""

from __future__ import annotations

import re

# A first-person subject, not inside an "if" clause ("If I bought this, ...").
_SUBJECT = (
    r"(?<!\bif )\b(?:i|i've|i have|i'd|we|we've|we have)\s+"
    r"(?:just\s+|finally\s+|actually\s+|already\s+|recently\s+)?"
)
# The product or a part of it, never a claim, a figure or a view.
_OBJECT = r"(?:it|them|one|mine|a pair|two)\b"
_NOT_RESEARCH = r"(?!\s+(?:against|on the|claim|theory|data|number|figure|stud|report))"

DEFAULT_PATTERNS: tuple[str, ...] = (
    _SUBJECT + r"(?:bought|ordered|purchased|got|picked up)\s+" + _OBJECT,
    _SUBJECT + r"(?:unboxed|wore|worn)\b",
    _SUBJECT + r"held\s+" + _OBJECT + r"(?!\s+(?:view|idea|belief))",
    _SUBJECT + r"(?:tested|used|tried)\s+" + _OBJECT + _NOT_RESEARCH,
    _SUBJECT + r"(?:been\s+)?(?:using|testing|wearing|trying)\s+" + _OBJECT,
    _SUBJECT + r"(?:use|wear)\s+(?:it|them|mine)\s+(?:every|daily|all)",
    _SUBJECT + r"(?:had|owned)\s+(?:it|them|one|mine)\s+for\b",
    _SUBJECT + r"(?:picked|plugged)\s+(?:it|them|one)\s+(?:up|in)\b",
    _SUBJECT + r"(?:took|brought)\s+(?:it|them)\s+(?:to|on|out|along)\b",
    _SUBJECT + r"went\s+[^.,;]{0,30}?\bwith\s+(?:it|them)\b",
    _SUBJECT + r"spent\s+[^.,;]{0,30}?\b(?:on|with)\s+(?:it|them|mine|one)\b",
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
