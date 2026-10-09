"""No configured call to action or closing-line example is engagement bait.

Design 0008. Meta demotes asking for shares, tags, votes or a specific word or
emoji, and TikTok's feed standards exclude false incentives for following.
Asking for a choice or an experience is genuine and passes.
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


@pytest.mark.req("REQ-CNT-148")
def test_no_configured_line_is_bait() -> None:
    templates = config.llm_settings.script_templates
    lines = [*templates.cta_options, *templates.cta_options_topic, *closing_examples()]
    assert closing_examples(), "no closing-line examples found in the templates"

    flagged = {line for line in lines if is_bait(line)}

    assert flagged == set()


# Verbs a call to action may open on; each names where the viewer goes or what
# they get (the link, the comments, the next video, the saved post).
IMPERATIVES = {"check", "follow", "drop", "save", "comment", "try", "grab"}


@pytest.mark.req("REQ-CNT-148")
def test_every_configured_cta_opens_on_an_imperative() -> None:
    templates = config.llm_settings.script_templates
    for line in [*templates.cta_options, *templates.cta_options_topic]:
        assert line.split()[0].lower() in IMPERATIVES, line
