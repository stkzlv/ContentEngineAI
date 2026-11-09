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
    r"(?<!\bif )\b(?:i|i've|i have|i'd|i'm|i am|we|we've|we have|we're|we are)\s+"
    r"(?:just\s+|finally\s+|actually\s+|already\s+|recently\s+)?"
)
# What may follow a product word: the end of the clause, or a word that keeps
# the product as the thing acted on. "One question" or "these figures" is not.
_AFTER = (
    r"(?=\s*(?:[.,;!?]|$)|\s+(?:for|on|in|at|last|every|since|myself|and|when"
    r"|after|before|now|yesterday|today|daily|from|of|out|around|outside)\b)"
)
# The product or a part of it, never a claim, a figure, a count or a view.
_PRODUCT = r"(?:(?:it|them|mine|a pair)\b|(?:one|two|this|these)" + _AFTER + ")"
_NOT_RESEARCH = (
    r"(?!\s+(?:against|on the (?:data|numbers|figures|benchmark)|as\b|in my\b"
    r"|to compare|claim|theory|data|number|figure|stud|report|source))"
)

DEFAULT_PATTERNS: tuple[str, ...] = (
    _SUBJECT + r"(?:bought|ordered|purchased|picked up)\s+" + _PRODUCT,
    # "Got it" is also "understood", so `got` takes a counted product only.
    _SUBJECT + r"got\s+(?:(?:mine|a pair)\b|(?:one|two)" + _AFTER + ")",
    _SUBJECT + r"(?:unboxed|wore|worn)\b(?!\s+(?:out|down|thin))",
    _SUBJECT + r"held\s+" + _PRODUCT + r"(?!\s+(?:view|idea|belief))",
    _SUBJECT + r"(?:tested|used|tried)\s+" + _PRODUCT + _NOT_RESEARCH,
    _SUBJECT + r"(?:been\s+)?(?:using|testing|wearing|trying)\s+" + _PRODUCT,
    _SUBJECT + r"(?:use|wear|own)\s+(?:it|them|mine|one)\b",
    _SUBJECT + r"(?:had|owned)\s+(?:it|them|one|mine)\s+(?:for|since)\b",
    _SUBJECT + r"(?:picked|plugged)\s+(?:it|them|one)\s+(?:up|in)\b",
    _SUBJECT + r"(?:took|brought)\s+(?:it|them)\s+(?:to|on|out|along)\b",
    # At least one word between: "went with it" is choosing, not travelling.
    _SUBJECT + r"went\s+(?:[^\s.,;]+\s+){1,5}?with\s+(?:it|them)\b",
    _SUBJECT
    + r"spent\s+(?:(?!\w+ing\b)[^.,;\s]+\s+){0,5}?(?:on|with)\s+(?:it|them|mine|one)\b",
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
