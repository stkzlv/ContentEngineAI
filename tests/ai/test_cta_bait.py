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

_I = re.IGNORECASE
# Bait is an imperative, so each pattern matches the start of a sentence:
# "Share this", "Comment YES", "Vote A or B". A question or a description
# that only uses the word ("gets your vote", "both share a flaw") passes.
BAIT = [
    # Pass the video on; "share your setup" asks for an experience.
    re.compile(r"^(share|send)\b(?! your)", _I),
    re.compile(r"^tag\b", _I),
    re.compile(r"^vote\b|\byour vote (below|in the comments)", _I),
    # A set word: "Comment YES for the link", "type 1 below".
    re.compile(
        r"^(comment|type|reply)( with)? [\"']?(?!(below|here|now|if)\b)\w{1,8}[\"']?"
        r"( (if|below|for|and)\b|[.!]?$)",
        _I,
    ),
    # An emoji, not ASCII punctuation such as a dash.
    re.compile(r"^(drop|leave|comment)( an?)? (emoji|[^\w\s\x00-\x7f])", _I),
    re.compile(r"^follow (for|to (see|get|unlock)) (part|the rest|the answer)", _I),
]

# Removed by the pool edit after the readout (#549).
KNOWN_EXCEPTIONS = {
    "Share with someone who needs this.",
    "Share it with whoever needs it.",
}


def is_bait(line: str) -> bool:
    sentences = re.split(r"(?<=[.!?])\s+", line.strip())
    return any(p.search(sentence) for sentence in sentences for p in BAIT)


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
        "Drop a \U0001f525 if this helped.",
        "comment yes if you agree",
        "Send this to someone who needs it.",
        "Comment YES for the link.",
        "Type YES if you agree.",
        "Type 1 below.",
        "Vote A or B in the comments.",
        "Cast your vote below.",
        "Love it? Share with your friends.",
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
        "Which one gets your vote?",
        "Reply A or B.",
        "Share your setup in the comments.",
        "Type C beats micro-USB for every cable.",
        "Drop a comment below if this helped.",
        "Comment below if you've tried it.",
        "Reply here if this worked.",
        "Which one gets your vote for bedside use?",
        "Both mounts share this weakness.",
        "Comment - which one would you pick?",
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
