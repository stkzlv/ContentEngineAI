"""A delivery plan: the script in segments, each with a style and events.

Design 0040. The plan adds delivery only: its segments' text is the script,
and its tags exist only in the text sent to the Gemini API speech model
(REQ-CNT-184), never in the script file, the captions or the metadata.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError


class DeliveryEvent(BaseModel):
    """A vocal event placed before a word of its segment, or at its end."""

    tag: str
    before_word: int | None = Field(default=None, ge=0)
    after_last: bool = False


class DeliverySegment(BaseModel):
    """Whole sentences of the script, voiced in one request."""

    text: str
    style: str = ""
    events: list[DeliveryEvent] = Field(default_factory=list)
    gap_after_ms: int = Field(default=0, ge=0, le=5000)


class DeliveryPlan(BaseModel):
    version: int = 1
    segments: list[DeliverySegment] = Field(default_factory=list)


def text_with_tags(segment: DeliverySegment) -> str:
    """The segment's text with its events inserted as inline tags.

    The text's own whitespace is kept, paragraph breaks included.
    """
    before: dict[int, list[str]] = {}
    tail: list[str] = []
    for event in segment.events:
        if event.after_last or event.before_word is None:
            tail.append(event.tag)
        else:
            before.setdefault(event.before_word, []).append(event.tag)
    out: list[str] = []
    index = 0
    for token in re.split(r"(\s+)", segment.text.strip()):
        if not token or token.isspace():
            out.append(token)
            continue
        out.extend(tag + " " for tag in before.get(index, []))
        out.append(token)
        index += 1
    text = "".join(out)
    return " ".join([text, *tail]) if tail else text


PLAN_FILE = "delivery_plan.json"

PLAN_PROMPT = (
    "Below is a short video script, one numbered sentence per line. Plan how "
    "to perform it. Group consecutive whole sentences into segments, starting "
    "a new segment only where the meaning or mood changes. For each segment "
    "write a short style describing how to read those lines (pace, mood, "
    "energy), never quoting or restating them. Add a vocal event only where a "
    "line earns one (an absurd number, a turn, an aside): at most one per "
    "segment and at most {max_events} in all, most segments none. Events must "
    "be one of: {tags}. Reply with JSON only:\n"
    '{{"segments": [{{"sentences": [1, 2], "style": "...", "event": '
    '{{"tag": "<sigh>", "before_word": 0}} or null, "gap_after_ms": 300}}]}}\n'
    "`sentences` lists the sentence numbers in order; every sentence appears "
    "exactly once. `before_word` is the 0-based index of a word in the "
    'segment\'s text; use "after_last": true instead to place the event at the '
    "segment's end. `gap_after_ms` is the silence before the next segment, "
    "0 to 1200.\n\n{sentences}"
)


def plan_prompt(sentences: list[str], tags: list[str], max_events: int) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    return PLAN_PROMPT.format(
        sentences=numbered, tags=", ".join(sorted(tags)), max_events=max_events
    )


def parse_plan(reply: str, sentences: list[str]) -> DeliveryPlan | None:
    """The plan in a model reply, its segments' text taken from the script's
    own sentences; None when the reply holds no usable plan.
    """
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        segments = []
        for raw in data["segments"]:
            numbers = [int(n) for n in raw["sentences"]]
            if not numbers or any(n < 1 or n > len(sentences) for n in numbers):
                return None
            event = raw.get("event")
            segments.append(
                DeliverySegment(
                    text=" ".join(sentences[n - 1] for n in numbers),
                    style=str(raw.get("style") or ""),
                    events=[DeliveryEvent.model_validate(event)] if event else [],
                    gap_after_ms=int(raw.get("gap_after_ms") or 0),
                )
            )
    except (ValueError, KeyError, TypeError, ValidationError):
        return None
    return DeliveryPlan(segments=segments)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _words(text: str) -> list[str]:
    return [w.lower().strip(".,!?;:\"'()") for w in text.split()]


def validate_plan(
    plan: DeliveryPlan,
    script: str,
    sentences: list[str],
    allowed_tags: frozenset[str],
    max_events: int,
) -> list[str]:
    """Each check the plan fails (REQ-CNT-180); empty when it passes."""
    failures: list[str] = []
    if not plan.segments:
        return ["the plan has no segments"]
    if _norm(" ".join(s.text for s in plan.segments)) != _norm(script):
        failures.append("the joined segment text differs from the script")
    # Word counts at sentence ends; every segment must end on one.
    ends, count = set(), 0
    for sentence in sentences:
        count += len(sentence.split())
        ends.add(count)
    count = 0
    for i, segment in enumerate(plan.segments):
        count += len(segment.text.split())
        if count not in ends:
            failures.append(f"segment {i + 1} ends inside a sentence")
    total = 0
    script_words = _words(script)
    for i, segment in enumerate(plan.segments):
        n_words = len(segment.text.split())
        if len(segment.events) > 1:
            failures.append(f"segment {i + 1} has more than one event")
        total += len(segment.events)
        for event in segment.events:
            if event.tag not in allowed_tags:
                failures.append(f"segment {i + 1} uses {event.tag!r}, not allowed")
            if not event.after_last and (
                event.before_word is None or event.before_word >= n_words
            ):
                failures.append(f"segment {i + 1} has an event outside its words")
        if _quotes_script(segment.style, script, script_words):
            failures.append(f"segment {i + 1}'s style quotes the script")
    if total > max_events:
        failures.append(f"{total} events, more than {max_events}")
    return failures


def _quotes_script(style: str, script: str, script_words: list[str]) -> bool:
    """Whether a style quotes the script: a quoted span found in it, or four
    of its words in a row.
    """
    for span in re.findall(r"[\"“']([^\"”']{3,})[\"”']", style):
        if _norm(span).lower() in _norm(script).lower():
            return True
    style_words = _words(style)
    runs = {tuple(script_words[i : i + 4]) for i in range(len(script_words) - 3)}
    return any(
        tuple(style_words[i : i + 4]) in runs for i in range(len(style_words) - 3)
    )
