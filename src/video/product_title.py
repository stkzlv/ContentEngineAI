"""A short written title for a product video (design 0009, REQ-PUB-008).

The store listing title runs past 100 characters ("Smart Watch for Men Women,
1.83" HD Touch Screen with Bluetooth Call, IP68 ..."); trending Shorts titles
run about 20-40 characters with the keyword first. The title is built from the
render's hook headline, so the title and the burned-in hook make the same
promise (REQ-CNT-149).
"""

from __future__ import annotations

import re

# The listing title's first clause ends at a comma, a bar, a bracket or a dash.
_CLAUSE_END = re.compile(r"\s*[,|(\[–—]\s*|\s+-\s+")


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _trim(text: str, max_len: int) -> str:
    """Cut on a word boundary, never mid-word."""
    if len(text) <= max_len:
        return text
    cut = text[: max_len + 1].rsplit(" ", 1)[0].rstrip(" ,:;-")
    return cut or text[:max_len]


def short_title(
    listing_title: str, keyword: str | None, headline: str | None, max_len: int
) -> str:
    """The shortest-path title: keyword and headline, the headline, or the
    listing title's first clause, whichever first fits within `max_len`.
    """
    headline = (headline or "").strip().rstrip(".")
    keyword = (keyword or "").strip()
    candidates = []
    if headline:
        if keyword and _compact(keyword) not in _compact(headline):
            candidates.append(f"{keyword[:1].upper()}{keyword[1:]}: {headline}")
        candidates.append(headline)
    clause = _CLAUSE_END.split(listing_title.strip(), maxsplit=1)[0].strip()
    candidates.append(clause or listing_title.strip())
    for title in candidates:
        if title and len(title) <= max_len:
            return title
    return _trim(candidates[-1], max_len)
