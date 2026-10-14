"""A YouTube Shorts description: no URL, and a pointer to the profile link.

A link in a Shorts description can't be clicked, so the description carries
none and, when the installation has a bio link (`youtube_profile_pointer`),
a product video ends on "Link in bio." instead (REQ-PUB-148, REQ-CNT-150).
Shared by the optimized metadata generator and the publisher, which applies
it to every YouTube post whatever the metadata mode.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.utils.config_overlay import read_config_yaml
from src.utils.outputs_paths import get_project_root

PROFILE_POINTER = "Link in bio."

PUBLISHER_CONFIG = get_project_root() / "config" / "publisher.yaml"

_URL_LINE = re.compile(r"^[^\n]*?(?:https?://|www\.)\S+[^\n]*$", re.MULTILINE)
_URL = re.compile(r"(?:https?://|www\.)\S+")
_POINTER = re.compile(r"\b(?:link|links)\s+in\s+(?:my\s+|our\s+|the\s+)?bio\b", re.I)
_POINTER_LINE = re.compile(
    r"^[ \t]*(?:the\s+)?(?:link|links)\s+(?:is\s+|are\s+)?in\s+"
    r"(?:my\s+|our\s+|the\s+)?bio[ \t]*[.!]?[ \t]*(?:\n|$)",
    re.I | re.MULTILINE,
)


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


def without_profile_pointer(description: str) -> str:
    """The description without a line that is only a bio-link pointer.

    For an installation with the pointer off: metadata written while it was
    on still ends on "Link in bio." (REQ-PUB-148).
    """
    if not _POINTER_LINE.search(description):
        return description
    cleaned = _POINTER_LINE.sub("", description)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def profile_pointer_on(path: Path | None = None) -> bool:
    """Whether the installation's YouTube descriptions point at a bio link.

    Read from the publisher config, overlay applied, so the render's metadata
    prompts and the publisher agree. Only a YAML `true` turns it on.
    """
    try:
        settings = read_config_yaml(path or PUBLISHER_CONFIG)
    except FileNotFoundError:
        return False
    if not isinstance(settings, dict):
        return False
    return settings.get("youtube_profile_pointer", False) is True


# What the caption and metadata prompts say about the bio link, by whether
# the installation has one. With it on, each value is the prompt's original
# text, so the rendered prompts are unchanged.
_PROMPT_TEXT_ON = {
    "CTA_EXAMPLE": '"Link in bio"',
    "PP_MIRROR_BEFORE": " before the profile pointer",
    "PP_MIRROR_END": " and end\nwith the profile pointer",
    "PP_NO_URL": "; the clickable link is on the profile",
    "PP_FORMAT": '; then a blank line; then "Link in bio." on the final line',
    "PP_FORMAT_EMPTY": ' and end with "Link in bio."',
    "PP_EXAMPLE": f"\n\n{PROFILE_POINTER}",
    "PP_IMPORTANT": f', blank line, "{PROFILE_POINTER}" on the final line',
}
_PROMPT_TEXT_OFF = {
    "CTA_EXAMPLE": '"Follow for more"',
    "PP_MIRROR_BEFORE": "",
    "PP_MIRROR_END": "",
    "PP_NO_URL": "",
    "PP_FORMAT": "",
    "PP_FORMAT_EMPTY": ".",
    "PP_EXAMPLE": "",
    "PP_IMPORTANT": "",
}


def pointer_placeholders(on: bool | None = None) -> dict[str, str]:
    """The prompt placeholders that name the bio link, or drop it.

    `on` defaults to the installation's `youtube_profile_pointer`.
    """
    if on is None:
        on = profile_pointer_on()
    return dict(_PROMPT_TEXT_ON if on else _PROMPT_TEXT_OFF)
