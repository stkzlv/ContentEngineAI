"""A delivery plan: the script in segments, each with a style and events.

Design 0040. The plan adds delivery only: its segments' text is the script,
and its tags exist only in the text sent to the Gemini API speech model
(REQ-CNT-184), never in the script file, the captions or the metadata.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


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
