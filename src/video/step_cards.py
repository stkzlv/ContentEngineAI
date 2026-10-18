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
# How far past the last step found a step's narration may be, in words, per
# step since then: real scripts put up to 42 words between two steps, and a
# step not found widens the next one's window. Without a bound, a step whose
# words don't match its own narration matches the closing path recap, and
# the steps in between are lost.
_SEARCH_WINDOW_WORDS = 60
_VERB_LOOKBACK_WORDS = 3
_STOPWORDS = frozenset(
    "a an the to on in of your you and then tap open go select turn".split()
)


@dataclass(frozen=True)
class StepCard:
    """One card: its two lines and when it is on screen, in seconds.

    A `checklist` card shows the step's common mistake (`dont`) and the fix
    (`do`) instead of its counter and path, which it keeps for when the
    checklist can't fit the frame.
    """

    counter: str
    path: str
    start: float
    end: float
    kind: str = "step"
    dont: str = ""
    do: str = ""


def add_checklist(
    cards: list[StepCard],
    *,
    mistake_step: int | None,
    dont: str | None,
    do: str | None,
    total: int,
    min_sec: float,
) -> list[StepCard]:
    """The cards with the mistake step's second half given to a checklist.

    A step's card shows first and the checklist ("Don't" / "Do") for the
    rest of that step's time, one graphic at a time (REQ-VID-164). A step
    too short to hold both for `min_sec` each shows only the checklist; a
    step with no card, or no "don't" and "do", leaves the cards as they are.
    """
    if not (mistake_step and dont and do):
        return cards
    counter = f"Step {mistake_step} of {total}"
    out: list[StepCard] = []
    for card in cards:
        if card.counter != counter or card.kind != "step":
            out.append(card)
            continue
        middle = (card.start + card.end) / 2
        start = middle if middle - card.start >= min_sec else card.start
        if start > card.start:
            out.append(StepCard(card.counter, card.path, card.start, start))
        out.append(
            StepCard(
                card.counter,
                card.path,
                start,
                card.end,
                kind="checklist",
                dont=dont,
                do=do,
            )
        )
    return out


def _tokens(text: str) -> list[str]:
    # Whisper writes a spoken "&" as "and".
    return _TOKEN.findall(text.lower().replace("&", " and "))


def path_segments(ui_path: str) -> list[str]:
    """A menu path's segments as shown: ['Settings', 'Your name', 'iCloud'].

    A bracketed placeholder ("[your name]") reads as its words, capitalised,
    and a segment repeated back to back shows once.
    """
    cleaned: list[str] = []
    for segment in _PATH_SPLIT.split(ui_path.strip()):
        text = _PLACEHOLDER.sub(
            lambda m: m.group(1)[:1].upper() + m.group(1)[1:], segment
        ).strip()
        if text and (not cleaned or cleaned[-1].lower() != text.lower()):
            cleaned.append(text)
    return cleaned


def display_path(ui_path: str, max_words: int) -> str:
    """The menu path as the card shows it: 'Settings > Your name > iCloud'.

    Segments as `path_segments` gives them. A path longer than `max_words`
    keeps its last segments, after a leading "...".
    """
    cleaned = path_segments(ui_path)
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
    # A segment that is only a placeholder ("[App Name]") is never spoken as
    # written, so the step is found by its action instead.
    if segments and not _PLACEHOLDER.fullmatch(segments[-1].strip()):
        key = "".join(_tokens(segments[-1]))
        if key:
            return key
    action = [t for t in _tokens(str(step.get("action") or "")) if t not in _STOPWORDS]
    return action[0] if action else ""


def _matches(
    words: list[str], key: str, after: int, window: int
) -> list[tuple[int, int]]:
    """Every (index, count) of words in the window after `after` spelling `key`."""
    found = []
    for i in range(after, min(len(words), after + window)):
        joined = ""
        for j in range(i, len(words)):
            joined += words[j]
            if joined == key:
                found.append((i, j - i + 1))
                break
            if not key.startswith(joined):
                break
    return found


def _find(
    words: list[str], key: str, after: int, window: int
) -> tuple[int, int] | None:
    """The match a step is spoken at.

    The first one within three words of an action verb ("Tap the word
    Password"), else the first one.
    """
    matches = _matches(words, key, after, window)
    for index, count in matches:
        lookback = words[max(after, index - _VERB_LOOKBACK_WORDS) : index]
        if any(w in _ACTION_VERBS for w in lookback):
            return index, count
    return matches[0] if matches else None


def find_step_words(
    steps: list[dict[str, Any]], spoken: list[dict[str, Any]]
) -> dict[int, int]:
    """Where each step's narration starts: step number -> index in `spoken`.

    The search starts after the first sentence and moves forward, so a step
    is found only after the one before it. `spoken` is Whisper's word list
    (`word`, `start_time`, `end_time`). A step whose words can't be found is
    missing from the result.
    """
    raw = [str(w.get("word", "")) for w in spoken]
    words = ["".join(_tokens(w)) for w in raw]
    found: dict[int, int] = {}
    # The first sentence names the topic ("Here's how to set up Back Tap"),
    # so it would match the last step's target before any step is spoken.
    cursor = next((i + 1 for i, w in enumerate(raw) if _SENTENCE_END.search(w)), 0)
    missed = 0
    for number, step in enumerate(steps, start=1):
        key = _key(step)
        window = _SEARCH_WINDOW_WORDS * (1 + missed)
        match = _find(words, key, cursor, window) if key else None
        if match is None:
            missed += 1
            continue
        missed = 0
        index, count = match
        found[number] = index
        cursor = index + count
    return found


def sentence_starts(spoken: list[dict[str, Any]]) -> list[int]:
    """The index of each sentence's first word in Whisper's word list."""
    starts = [0] if spoken else []
    for i, w in enumerate(spoken[:-1]):
        if _SENTENCE_END.search(str(w.get("word", ""))):
            starts.append(i + 1)
    return starts


def first_segment_word(
    step: dict[str, Any], spoken: list[dict[str, Any]], after: int, before: int
) -> int | None:
    """The first word in (`after`, `before`] that starts any of the step's path
    segments, so "find Auto-Lock" opens a step whose key is a later "30".
    """
    words = ["".join(_tokens(str(w.get("word", "")))) for w in spoken]
    keys = ["".join(_tokens(s)) for s in path_segments(str(step.get("ui_path") or ""))]
    hits = [
        i
        for key in keys
        if key
        for i, _ in _matches(words, key, after + 1, before - after)
    ]
    return min(hits) if hits else None


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

    A card starts at its step's first matched word (`find_step_words`), no
    earlier than `not_before` (the hook overlay's end), and ends when the
    next card starts, after at most `max_sec`; one that would be on screen
    less than `min_sec` is dropped.
    """
    total = len(steps)
    starts = find_step_words(steps, spoken)
    found: list[tuple[int, int, str]] = []
    for number, step in enumerate(steps, start=1):
        if number not in starts:
            logger.info(
                "Step card %d of %d skipped: narration not found", number, total
            )
            continue
        path = display_path(str(step.get("ui_path") or ""), max_path_words)
        found.append((number, starts[number], path))

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
    checklist: bool = False,
) -> list[StepCard]:
    """Cards for a render, or none when it has no step list or no timings.

    With `checklist`, the step the list names a common mistake at also
    shows the mistake and its fix (`add_checklist`).
    """
    if not step_list_file.exists():
        return []
    if not spoken:
        logger.info(
            "Step cards skipped: no word timings (the pycaps caption engine "
            "writes them)"
        )
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
    cards = plan_step_cards(
        steps,
        spoken,
        not_before=not_before,
        min_sec=min_sec,
        max_sec=max_sec,
        max_path_words=max_path_words,
    )
    if not checklist:
        return cards
    mistake_step = data.get("mistake_step")
    return add_checklist(
        cards,
        mistake_step=mistake_step if isinstance(mistake_step, int) else None,
        dont=data.get("mistake_dont") or None,
        do=data.get("mistake_do") or None,
        total=len(steps),
        min_sec=min_sec,
    )
