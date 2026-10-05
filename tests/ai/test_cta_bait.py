"""No configured call to action or closing-line example is engagement bait.

Design 0008. Meta demotes asking for shares, tags, votes or a specific word or
emoji, and TikTok's feed standards exclude false incentives for following.
Asking for a choice or an experience is genuine and passes. The pools are
edited after the reach-test readout (#540), so the lines found today are
listed as known exceptions, and the edit removes them from here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.video.config import config

SCRIPTS = Path(__file__).parents[2] / "src" / "ai" / "prompts" / "scripts"

BAIT = [
    re.compile(r"\bshare\b", re.IGNORECASE),
    re.compile(r"\btag (a|your|someone|a friend)\b", re.IGNORECASE),
    re.compile(r"\bvote\b", re.IGNORECASE),
    # "Comment YES", "type 1 below": a set word, not an opinion.
    re.compile(r"\b(?i:comment|type|reply)(?i: with)? [\"']?[A-Z0-9]{1,8}\b[\"']?"),
    re.compile(r"\b(drop|leave|comment)( an?)? emoji\b", re.IGNORECASE),
    re.compile(r"\bfollow (for|to (see|get|unlock)) (part|the rest|the answer)", re.I),
]

# Removed by the pool edit after the readout (#549).
KNOWN_EXCEPTIONS = {
    "Share with someone who needs this.",
    "Share it with whoever needs it.",
}


def is_bait(line: str) -> bool:
    return any(p.search(line) for p in BAIT)


def closing_examples() -> list[str]:
    """Quoted examples on each script template's closing-line rule."""
    found = []
    for path in sorted(SCRIPTS.glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "close with" in line.lower() or "closing line" in line.lower():
                found += re.findall(r'"([^"]{6,})"', line)
    return found


@pytest.mark.parametrize(
    "line",
    [
        "Share this with a friend.",
        "Tag someone who needs this.",
        "Vote in the comments.",
        "Comment YES if you agree.",
        "Drop an emoji if this helped.",
        "Follow for part 2.",
    ],
)
def test_bait_is_recognised(line: str) -> None:
    assert is_bait(line)


@pytest.mark.parametrize(
    "line",
    [
        "Link in bio if you want one.",
        "Drop a comment if you've tried it.",
        "Team magnetic or team plug-in?",
        "Save this for the next time it happens.",
        "Follow for more fixes like this.",
    ],
)
def test_a_genuine_line_passes(line: str) -> None:
    assert not is_bait(line)


def test_no_configured_line_is_bait_beyond_the_known_ones() -> None:
    templates = config.llm_settings.script_templates
    lines = [*templates.cta_options, *templates.cta_options_topic, *closing_examples()]
    assert closing_examples(), "no closing-line examples found in the templates"

    flagged = {line for line in lines if is_bait(line)}

    # Equal, not a subset: a known exception that is gone must leave the list.
    assert flagged == KNOWN_EXCEPTIONS
