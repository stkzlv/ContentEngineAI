"""A YouTube Shorts description: no URL, and a pointer to the profile link.

A link in a Shorts description can't be clicked, so the description carries
none and a product video ends on "Link in bio." instead (REQ-PUB-148,
REQ-CNT-150). Shared by the optimized metadata generator and the publisher,
which applies it to every YouTube post whatever the metadata mode.
"""

from __future__ import annotations

import re

PROFILE_POINTER = "Link in bio."

_URL_LINE = re.compile(r"^[^\n]*?(?:https?://|www\.)\S+[^\n]*$", re.MULTILINE)
_URL = re.compile(r"(?:https?://|www\.)\S+")
_POINTER = re.compile(r"\b(?:link|links)\s+in\s+(?:my\s+|our\s+|the\s+)?bio\b", re.I)


def drop_urls(description: str) -> str:
    """The description without URLs (REQ-PUB-148).

    A line that is only a lead-in and a URL goes whole; a URL inside a
    sentence goes alone.
    """

    def line(match: re.Match[str]) -> str:
        rest = _URL.sub("", match.group(0)).strip(" :-")
        if len(rest.split()) <= 3:
            return ""
        return re.sub(r" {2,}", " ", _URL.sub("", match.group(0))).rstrip()

    cleaned = _URL_LINE.sub(line, description)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def with_profile_pointer(description: str) -> str:
    """The description ending on the profile pointer, unless it names one."""
    if _POINTER.search(description):
        return description
    return f"{description.rstrip()}\n\n{PROFILE_POINTER}"
