"""Where a render's search phrase appears (design 0007, REQ-CNT-055).

Platforms read spoken words, on-screen text and the caption for search, so
the phrase belongs in the first spoken sentence, the hook headline and the
first 60 characters of each platform caption. This measures where it is;
nothing here changes a script, a headline or a caption.

The phrase is the product's search keyword, or a topic's title, which is
already phrased the way people type it. A place counts when every word of the
phrase of three or more letters appears there in any case and order (a
topic's title leaves its question and filler words aside), or the phrase
appears with its spaces closed up ("smart watch" in "smartwatch"). Captions
are measured before the publisher puts its disclosure in front of them.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.utils.script_sanitizer import split_sentences

logger = logging.getLogger(__name__)

CAPTION_PREFIX_CHARS = 60
PLATFORMS = ("youtube", "tiktok", "instagram")
# Question and filler words a spoken answer drops: "Why your laptop fan
# runs" is answered by "Your laptop fan runs when idle because...".
STOP_WORDS = frozenset(
    "why how what when where which who whose does did can could should would "
    "will your you yours the and for with this that these those its are was "
    "were has have not".split()
)


def _terms(phrase: str, *, question: bool = False) -> set[str]:
    """The words a place must carry. A question (a topic's title) drops its
    question and filler words; a product keyword keeps every word, or
    "can opener" would match any opener.
    """
    return {
        w
        for w in re.findall(r"[\w']+", phrase.lower())
        if len(w) >= 3 and not (question and w in STOP_WORDS)
    }


def key_words(phrase: str, *, question: bool = False) -> list[str]:
    """The words a place must carry, in the phrase's order."""
    terms = _terms(phrase, question=question)
    seen: list[str] = []
    for word in re.findall(r"[\w']+", phrase.lower()):
        if word in terms and word not in seen:
            seen.append(word)
    return seen


def contains(text: str | None, phrase: str, *, question: bool = False) -> bool:
    """Whether every significant word of `phrase` appears in `text`."""
    terms = _terms(phrase, question=question)
    if not text or not terms:
        return False
    if terms <= set(re.findall(r"[\w']+", text.lower())):
        return True
    compact = re.sub(r"[^a-z0-9]", "", phrase.lower())
    return compact in re.sub(r"[^a-z0-9]", "", text.lower())


def search_phrase(product: Any) -> str | None:
    """A topic's title, else the product's search keyword."""
    if getattr(product, "topic", None):
        return getattr(product, "title", None) or None
    return (getattr(product, "keyword", "") or "").strip() or None


def captions(run_root: Path) -> dict[str, str]:
    """Each platform's caption text, read the way the publisher reads it:
    the unified file first, else the platform's own.

    This is the generated caption, before the publisher puts a disclosure
    and the affiliate line in front of it, so the measure is where the
    phrase sits in the text the pipeline wrote.
    """
    found: dict[str, str] = {}
    unified = _description(run_root / "metadata.json")
    for platform in PLATFORMS:
        text = (
            unified
            if unified is not None
            else _description(run_root / f"metadata_{platform}.json")
        )
        if text is not None:
            found[platform] = text
    return found


def _description(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        logger.debug("Unreadable caption file %s: %s", path.name, e)
        return None
    value = data.get("description") if isinstance(data, dict) else None
    return value if isinstance(value, str) else None


def placement(
    phrase: str | None,
    script: str | None,
    headline: str | None,
    platform_captions: dict[str, str],
    *,
    question: bool = False,
) -> dict[str, Any] | None:
    """Where the phrase appears; None without a phrase to look for.

    `question` marks a topic's title, whose question words a spoken answer
    drops.
    """
    if not phrase or not _terms(phrase, question=question):
        return None
    sentences = split_sentences(script or "")
    return {
        "phrase": phrase,
        "spoken": contains(
            sentences[0] if sentences else "", phrase, question=question
        ),
        "headline": contains(headline, phrase, question=question),
        "captions": {
            platform: contains(text[:CAPTION_PREFIX_CHARS], phrase, question=question)
            for platform, text in platform_captions.items()
        },
    }


def coverage_lines(rows: list[dict[str, Any]]) -> list[str]:
    """The report's lines: the share of renders with the phrase per place."""
    placed = [r["search_phrase"] for r in rows if r.get("search_phrase")]
    if not placed:
        return []
    total = len(placed)

    def share(count: int) -> str:
        return f"{count}/{total}"

    caption_ok = [bool(p["captions"]) and all(p["captions"].values()) for p in placed]
    all_three = sum(
        1
        for p, cap in zip(placed, caption_ok, strict=True)
        if p["spoken"] and p["headline"] and cap
    )
    return [
        "Search phrase placement:",
        f"  first spoken sentence {share(sum(p['spoken'] for p in placed))}, "
        f"hook headline {share(sum(p['headline'] for p in placed))}, "
        f"every caption opening {share(sum(caption_ok))}, "
        f"all three {share(all_three)}",
    ]
