"""Step cards for a tutorial render: which card shows when (REQ-VID-124).

A topic render with a step list gets one card per step: a counter
("Step 2 of 4") over the step's menu path ("Settings > iCloud"), shown from
the moment the narration reaches that step until the next one starts. The
moment is found in Whisper's word timings by matching the step's words;
a step whose words can't be found gets no card, since a card shown at the
wrong time teaches the wrong step (design 0019: a failed graphic is skipped).

No FFmpeg here: the assembler turns the cards into drawtext filters.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PATH_SPLIT = re.compile(r"\s*(?:>|›|→|->)\s*")
_PLACEHOLDER = re.compile(r"\[([^\]]+)\]")
_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")
# A step's target is usually spoken right after its verb ("Tap General");
# the same name also turns up in the previous step's result ("The General
# menu opens"), so a match after one of these wins over an earlier one.
_ACTION_VERBS = frozenset(
    "tap open select choose go press click hit pick turn toggle enable disable "
    "find scroll swipe head".split()
)
_STOPWORDS = frozenset(
    "a an the to on in of your you and then tap open go select turn".split()
)


@dataclass(frozen=True)
class StepCard:
    """One card: its two lines and when it is on screen, in seconds."""

    counter: str
    path: str
    start: float
    end: float


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def display_path(ui_path: str, max_words: int) -> str:
    """The menu path as the card shows it: 'Settings > Your name > iCloud'.

    A bracketed placeholder ("[your name]") reads as its words, capitalised,
    and a segment repeated back to back shows once. A path longer than
    `max_words` keeps its last segments, after a leading "...".
    """
    cleaned: list[str] = []
    for segment in _PATH_SPLIT.split(ui_path.strip()):
        text = _PLACEHOLDER.sub(
            lambda m: m.group(1)[:1].upper() + m.group(1)[1:], segment
        ).strip()
        if text and (not cleaned or cleaned[-1].lower() != text.lower()):
            cleaned.append(text)
    kept: list[str] = []
    for segment in reversed(cleaned):
        if kept and sum(len(s.split()) for s in [*kept, segment]) > max_words:
            break
        kept.insert(0, segment)
    prefix = "... > " if len(kept) < len(cleaned) else ""
    return prefix + " > ".join(kept)


def _key(step: dict[str, Any]) -> str:
    """The letters that mark a step's narration, spaces and punctuation dropped.

    The path's last segment ("Back Tap" -> "backtap"), matched against the
    spoken words joined the same way, so Whisper's "BackTap" or "Wi-Fi"
    still match. Falls back to the action's first content words when the
    path is empty ("Restart the router" -> "restart").
    """
    segments = [s for s in _PATH_SPLIT.split(str(step.get("ui_path") or "")) if s]
    if segments:
        last = _PLACEHOLDER.sub(lambda m: m.group(1), segments[-1])
        key = "".join(_tokens(last))
        if key:
            return key
    action = [t for t in _tokens(str(step.get("action") or "")) if t not in _STOPWORDS]
    return action[0] if action else ""


def _matches(words: list[str], key: str, after: int) -> list[tuple[int, int]]:
    """Every (index, count) of spoken words from `after` whose letters are `key`."""
    found = []
    for i in range(after, len(words)):
        joined = ""
        for j in range(i, len(words)):
            joined += words[j]
            if joined == key:
                found.append((i, j - i + 1))
                break
            if not key.startswith(joined):
                break
    return found


def _find(words: list[str], key: str, after: int) -> tuple[int, int] | None:
    """The match a step is spoken at.

    The first one within two words of an action verb, else the first one.
    """
    matches = _matches(words, key, after)
    for index, count in matches:
        if any(w in _ACTION_VERBS for w in words[max(after, index - 2) : index]):
            return index, count
    return matches[0] if matches else None


def plan_step_cards(
    steps: list[dict[str, Any]],
    spoken: list[dict[str, Any]],
    *,
    not_before: float,
    min_sec: float,
    max_sec: float,
    max_path_words: int = 4,
) -> list[StepCard]:
    """One card per step whose narration is found, in order.

    The search starts after the first sentence. `spoken` is Whisper's word
    list (`word`, `start_time`, `end_time`). A card starts at its step's
    first matched word, no earlier than `not_before` (the hook overlay's
    end), and ends when the next card starts, after at most `max_sec`; one
    that would be on screen less than `min_sec` is dropped. A step found
    before the previous one is skipped.
    """
    raw = [str(w.get("word", "")) for w in spoken]
    words = ["".join(_tokens(w)) for w in raw]
    total = len(steps)
    found: list[tuple[int, int, str]] = []
    # The first sentence names the topic ("Here's how to set up Back Tap"),
    # so it would match the last step's target before any step is spoken.
    cursor = next((i + 1 for i, w in enumerate(raw) if _SENTENCE_END.search(w)), 0)
    for number, step in enumerate(steps, start=1):
        key = _key(step)
        match = _find(words, key, cursor) if key else None
        if match is None:
            logger.info(
                "Step card %d of %d skipped: narration not found", number, total
            )
            continue
        index, count = match
        path = display_path(str(step.get("ui_path") or ""), max_path_words)
        found.append((number, index, path))
        cursor = index + count

    speech_end = float(spoken[-1]["end_time"]) if spoken else 0.0
    cards: list[StepCard] = []
    for position, (number, index, path) in enumerate(found):
        start = max(float(spoken[index]["start_time"]), not_before)
        if position + 1 < len(found):
            next_start = float(spoken[found[position + 1][1]]["start_time"])
        else:
            next_start = speech_end
        end = min(next_start, start + max_sec)
        if end - start < min_sec:
            logger.info(
                "Step card %d of %d skipped: on screen too briefly", number, total
            )
            continue
        cards.append(StepCard(f"Step {number} of {total}", path, start, end))
    return cards


def load_step_cards(
    step_list_file: Path,
    spoken: list[dict[str, Any]] | None,
    *,
    not_before: float,
    min_sec: float,
    max_sec: float,
    max_path_words: int = 4,
) -> list[StepCard]:
    """Cards for a render, or none when it has no step list or no timings."""
    if not spoken or not step_list_file.exists():
        return []
    try:
        data = json.loads(step_list_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Step cards skipped: unreadable %s: %s", step_list_file, exc)
        return []
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list) or not steps:
        return []
    steps = [s for s in steps if isinstance(s, dict)]
    return plan_step_cards(
        steps,
        spoken,
        not_before=not_before,
        min_sec=min_sec,
        max_sec=max_sec,
        max_path_words=max_path_words,
    )
