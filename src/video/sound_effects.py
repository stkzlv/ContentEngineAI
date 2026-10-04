"""Where sparse sound effects land and which file each one plays (design 0003).

Three events: each crossfade between visuals, the reveal (the start of the
sentence after the hook) and the call to action (the start of the last
sentence). The plan is capped per 10 seconds, keeping the call to action and
the reveal before any transition, and an effect never lands in a spoken
word's first 100 ms, where it would mask the consonant.

Nothing here raises: a missing file is skipped with a warning, and a render
with no word timings gets transition effects only.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lower sorts first: what the cap keeps when it must drop something.
PRIORITY = {"cta": 0, "reveal": 1, "transition": 2}
WORD_ONSET_SEC = 0.1
WINDOW_SEC = 10.0
_SENTENCE_END = (".", "!", "?")


@dataclass(frozen=True)
class SoundEvent:
    kind: str
    time: float


def transition_times(durations: list[float], transition_sec: float) -> list[float]:
    """The midpoint of each crossfade, from the timeline's segment durations."""
    times = []
    offset = 0.0
    for duration in durations[:-1]:
        offset += duration - transition_sec
        times.append(offset + transition_sec / 2)
    return times


def sentence_starts(words: list[dict[str, Any]]) -> list[float]:
    """The start time of each sentence in Whisper's word timings."""
    starts = []
    new_sentence = True
    for word in words:
        if new_sentence:
            starts.append(float(word["start_time"]))
        new_sentence = str(word["word"]).strip().endswith(_SENTENCE_END)
    return starts


def _clear_of_onsets(time: float, words: list[dict[str, Any]]) -> float:
    """`time` moved past any word's first 100 ms it falls in."""
    moved = True
    while moved:
        moved = False
        for word in words:
            start = float(word["start_time"])
            if start <= time < start + WORD_ONSET_SEC:
                time = start + WORD_ONSET_SEC
                moved = True
    return time


def plan_events(
    transitions: list[float],
    words: list[dict[str, Any]],
    duration: float,
    max_per_10_sec: int,
) -> list[SoundEvent]:
    """The events to play, in time order, capped per 10 seconds."""
    events = [SoundEvent("transition", t) for t in transitions]
    starts = sentence_starts(words)
    if len(starts) >= 2:
        events.append(SoundEvent("reveal", starts[1]))
    if len(starts) >= 3:
        events.append(SoundEvent("cta", starts[-1]))
    events = [SoundEvent(e.kind, _clear_of_onsets(e.time, words)) for e in events]
    kept: list[SoundEvent] = []
    for event in sorted(events, key=lambda e: (PRIORITY[e.kind], e.time)):
        if not 0 <= event.time < duration:
            continue
        if _fits(kept, event, max_per_10_sec):
            kept.append(event)
    return sorted(kept, key=lambda e: e.time)


def _fits(kept: list[SoundEvent], event: SoundEvent, cap: int) -> bool:
    """Whether every 10-second window still holds at most `cap` events."""
    times = sorted([k.time for k in kept] + [event.time])
    return all(
        sum(1 for t in times if start <= t < start + WINDOW_SEC) <= cap
        for start in times
    )


def choose_file(
    pool: list[Path], product_id: str, kind: str, index: int
) -> Path | None:
    """The pool file for this event, the same for a product on every run."""
    present = [p for p in pool if Path(p).is_file()]
    for missing in (p for p in pool if not Path(p).is_file()):
        logger.warning("Sound effect file missing, skipped: %s", missing)
    if not present:
        return None
    digest = hashlib.md5(
        f"{product_id}:sfx:{kind}:{index}".encode(), usedforsecurity=False
    ).hexdigest()
    return present[int(digest[:8], 16) % len(present)]


def resolve_effects(
    settings: Any, events: list[SoundEvent], product_id: str
) -> list[tuple[Path, float]]:
    """(file, start second) for each planned event whose pool has a file."""
    chosen = []
    counts: dict[str, int] = {}
    for event in events:
        index = counts.get(event.kind, 0)
        counts[event.kind] = index + 1
        path = choose_file(getattr(settings, event.kind), product_id, event.kind, index)
        if path is not None:
            chosen.append((path, event.time))
    return chosen
