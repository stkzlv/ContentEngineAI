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
# How many of the action's content words a step is looked for by, at most.
_ACTION_KEY_WORDS = 5
_STOPWORDS = frozenset(
    "a an the to on in of your you and then tap open go select turn "
    "it its this that for from with at or is".split()
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


def _action_words(step: dict[str, Any], *, from_verb: bool = False) -> list[str]:
    """The action's content words in order: "Restart the router" -> restart, router.

    Action verbs are passed over too ("Click the trash icon" -> "trash"): a
    script says "click" for every step, so the verb finds the wrong one.
    With `from_verb`, the words start after the first action verb, so a
    leading purpose clause ("To finish the capture, press ...") is skipped.
    """
    # Whole words joined, as `_matches` joins the spoken ones: "Wi-Fi" is
    # "wifi", not "wi". A slash separates words ("Power/Side"), and a
    # possessive is dropped ("app's" is spoken "app").
    text = re.sub(r"['\u2019]s\b", "", str(step.get("action") or ""))
    words = ["".join(_tokens(w)) for w in re.split(r"[\s/]+", text)]
    if from_verb:
        verb = next((i for i, w in enumerate(words) if w in _ACTION_VERBS), None)
        words = words[verb + 1 :] if verb is not None else words
    action = [w for w in words if w and w not in _STOPWORDS | _ACTION_VERBS]
    return list(dict.fromkeys(action))


def _action_key(step: dict[str, Any]) -> str:
    """The action's first content word."""
    words = _action_words(step)
    return words[0] if words else ""


def _segment_keys(ui_path: str) -> list[str]:
    """Each segment's letters, placeholders ("[App Name]") left out."""
    segments = [s for s in _PATH_SPLIT.split(ui_path) if s]
    return [
        "".join(_tokens(s)) for s in segments if not _PLACEHOLDER.fullmatch(s.strip())
    ]


def _primary_key(step: dict[str, Any]) -> str:
    """The path's last segment ("Back Tap" -> "backtap"), unless a placeholder.

    Matched against the spoken words joined the same way, so Whisper's
    "BackTap" or "Wi-Fi" still match.
    """
    segments = [s for s in _PATH_SPLIT.split(str(step.get("ui_path") or "")) if s]
    if not segments or _PLACEHOLDER.fullmatch(segments[-1].strip()):
        return ""
    return "".join(_tokens(segments[-1]))


def _fallback_keys(
    step: dict[str, Any], next_step: dict[str, Any] | None = None
) -> list[str]:
    """What else marks a step when its last segment isn't spoken.

    Scripts often name an earlier segment or describe the control instead
    ("Go back to Display" for "Display > Night Shift", "the small i" for
    "Details"): the earlier segments, last first, then the action's first
    content word. A segment the next step's path also has is left out,
    since the next step's narration names it too.
    """
    primary = _primary_key(step)
    shared = set(_segment_keys(str((next_step or {}).get("ui_path") or "")))
    keys: list[str] = []
    for key in reversed(_segment_keys(str(step.get("ui_path") or ""))):
        if key and key != primary and key not in shared and key not in keys:
            keys.append(key)
    action = _action_key(step)
    if action and action != primary and action not in shared and action not in keys:
        keys.append(action)
    return keys


def _last_resort_keys(
    step: dict[str, Any], next_step: dict[str, Any] | None = None
) -> list[str]:
    """The action's other content words, for when nothing else is spoken.

    A step list's wording often differs from the script's in one word
    ("any area outside the toolbar" spoken as "anywhere outside the
    toolbar"). Counted from the first action verb, up to
    `_ACTION_KEY_WORDS`, and tried only after every other key misses,
    because a later word is more often in the previous step's result
    sentence too.
    """
    taken = {_primary_key(step), *_fallback_keys(step, next_step)}
    shared = set(_segment_keys(str((next_step or {}).get("ui_path") or "")))
    words = _action_words(step, from_verb=True)[:_ACTION_KEY_WORDS]
    return [w for w in words if w not in taken and w not in shared]


def step_keys(
    step: dict[str, Any], next_step: dict[str, Any] | None = None
) -> list[str]:
    """Every key a step is looked for by, for the log: the last segment first."""
    keys = [_primary_key(step), *_fallback_keys(step, next_step)]
    return [k for k in keys if k] + _last_resort_keys(step, next_step)


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


def _verb_led(words: list[str], after: int, index: int) -> bool:
    """Whether an action verb is within three words before `index`."""
    lookback = words[max(after, index - _VERB_LOOKBACK_WORDS) : index]
    return any(w in _ACTION_VERBS for w in lookback)


def _find(
    words: list[str], key: str, after: int, window: int
) -> tuple[int, int] | None:
    """The match a step is spoken at.

    The first one within three words of an action verb ("Tap the word
    Password"), else the first one.
    """
    matches = _matches(words, key, after, window)
    for index, count in matches:
        if _verb_led(words, after, index):
            return index, count
    return matches[0] if matches else None


def _earliest(
    words: list[str], keys: list[str], after: int, window: int
) -> tuple[int, int] | None:
    """The earliest verb-led match of any key, else the earliest match."""
    hits = [m for key in keys if (m := _find(words, key, after, window)) is not None]
    led = [m for m in hits if _verb_led(words, after, m[0])]
    return min(led or hits) if hits else None


def _next_step_word(
    words: list[str],
    step: dict[str, Any] | None,
    next_step: dict[str, Any] | None,
    after: int,
    window: int,
) -> int | None:
    """Where `step` would be found by its labels or first action word."""
    if not step:
        return None
    primary = _primary_key(step)
    keys = [primary] if primary else []
    hit = _earliest(words, [*keys, *_fallback_keys(step, next_step)], after, window)
    return hit[0] if hit else None


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
        window = _SEARCH_WINDOW_WORDS * (1 + missed)
        primary = _primary_key(step)
        match = _find(words, primary, cursor, window) if primary else None
        if match is None:
            # The earliest verb-led fallback, else the earliest: a later key
            # can be spoken well before an earlier one turns up again, and a
            # label with no verb before it is often the last step's result
            # ("The Display screen opens").
            next_step = steps[number] if number < len(steps) else None
            match = _earliest(words, _fallback_keys(step, next_step), cursor, window)
            if match is None and (keys := _last_resort_keys(step, next_step)):
                # Generic words ("button", "screen") are spoken in the next
                # step's sentence too, so the last tier stops where the next
                # step is found by its own labels.
                after = steps[number + 1] if number + 1 < len(steps) else None
                ahead = _next_step_word(words, next_step, after, cursor, window)
                limit = window if ahead is None else min(window, ahead - cursor)
                match = _earliest(words, keys, cursor, limit)
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
                "Step card %d of %d skipped: narration not found (searched %s)",
                number,
                total,
                ", ".join(step_keys(step, steps[number] if number < total else None))
                or "-",
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
