"""Where a caption segment ends: at a phrase boundary, within the char limits.

pycaps' `limit_by_chars` splitter fills a segment greedily up to its
character limit and ends it wherever that falls, so a caption can end on
"because it" or split "iPhone | 15" across two screens (REQ-VID-157). This
module picks the end among the same candidates the greedy fill allows,
preferring a break after punctuation or before a word that opens a phrase,
and avoiding one after a word that needs the next one, before the particle
of a phrasal verb, or inside a name. It has no
pycaps import, so it is tested without the optional group; the renderer
wraps it in a splitter class at render time.
"""

from __future__ import annotations

import re

# A break before one of these starts a new phrase.
_PHRASE_OPENERS = frozenset(
    "and but or so because if when while then to for with from into onto "
    "on in at by about after before until unless which who where like "
    "instead without".split()
)
# A segment must not end on one of these: each needs the word after it.
_BINDERS = frozenset(
    "a an the your my our their his her its this these those to of in on "
    "at for with from by into onto and or but than as is are was were "
    "be been being has have had can will would should could do does did not "
    "no very more most under over about through between across".split()
)
# "turn on", "plug it in", "log out": after one of these verbs (or its
# pronoun object) a particle belongs to the verb, so a segment may end on it
# and must not start with it.
_PARTICLE_VERBS = frozenset(
    "turn switch plug log sign check set put pick power zoom opt back hold "
    "swipe scroll slide pull push shut start top clean fill wipe tap".split()
)
_PARTICLES = frozenset("on in off up out down".split())
_PRONOUN_OBJECTS = frozenset("it them this that everything".split())
_PUNCTUATED = re.compile(r"[,.;:!?)\]\"']$")
_SENTENCE_END = re.compile(r"[.!?]$")

_PUNCTUATION_SCORE = 3
_OPENER_SCORE = 2
_FORBIDDEN = -1


def _bare(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word).lower()


def _is_name_part(words: list[str], index: int) -> bool:
    """A model, version or brand token: a digit, or a capital mid-sentence."""
    word = words[index].strip("\"'([")
    if any(ch.isdigit() for ch in word):
        return True
    if not any(ch.isupper() for ch in word):
        return False
    return index > 0 and not _SENTENCE_END.search(words[index - 1])


def _is_particle(words: list[str], index: int) -> bool:
    """`words[index]` is the particle of a phrasal verb such as "turn on"."""
    if _bare(words[index]) not in _PARTICLES or index == 0:
        return False
    previous = _bare(words[index - 1])
    if previous in _PARTICLE_VERBS:
        return True
    return (
        previous in _PRONOUN_OBJECTS
        and index > 1
        and _bare(words[index - 2]) in _PARTICLE_VERBS
    )


def break_score(words: list[str], end: int) -> int:
    """Score ending a segment before `words[end]`; negative is avoided."""
    before, after = words[end - 1], words[end]
    if _PUNCTUATED.search(before):
        return _PUNCTUATION_SCORE
    if _is_particle(words, end):
        return _FORBIDDEN
    if _bare(before) in _BINDERS and not _is_particle(words, end - 1):
        return _FORBIDDEN
    if _is_name_part(words, end - 1) and _is_name_part(words, end):
        return _FORBIDDEN
    if _bare(after) in _PHRASE_OPENERS:
        return _OPENER_SCORE
    return 0


def segment_end(words: list[str], start: int, max_chars: int, min_chars: int) -> int:
    """Index one past the last word of the segment starting at `start`.

    Characters are counted without spaces, as pycaps counts them. The
    greedy end (the most words within `max_chars`) is pycaps' own choice,
    and its rule that a remainder under `min_chars` joins this segment is
    kept. Any earlier end that leaves the segment at `min_chars` or more is
    a candidate; the best-scoring one wins, the later on a tie, and when
    every candidate is one to avoid the greedy end stands.
    """
    greedy = start
    chars = 0
    while greedy < len(words) and chars + len(words[greedy]) <= max_chars:
        chars += len(words[greedy])
        greedy += 1
    if greedy == start:
        return start + 1
    if sum(len(w) for w in words[greedy:]) < min_chars:
        return len(words)
    if greedy == len(words):
        return greedy

    best, best_score = greedy, _FORBIDDEN
    chars = 0
    for end in range(start + 1, greedy + 1):
        chars += len(words[end - 1])
        if chars < min_chars and end < greedy:
            continue
        score = break_score(words, end)
        if score >= best_score:
            best, best_score = end, score
    return best


def split_words(words: list[str], max_chars: int, min_chars: int) -> list[int]:
    """The end index of every segment over `words`."""
    ends = []
    start = 0
    while start < len(words):
        start = segment_end(words, start, max_chars, min_chars)
        ends.append(start)
    return ends
