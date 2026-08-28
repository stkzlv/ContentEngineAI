"""The stock-search prompt must not hand the model phrases it can copy.

Measured on a real topic script (wifi channel congestion), the prompt's own
worked examples came back verbatim in 2 of 10 runs -- the phone-battery block,
for a video about wifi. The render then searched a stock library for a phone
at night and got one, which is what put an unrelated sunset photo under
narration about wifi channels.

This is the project's recorded lesson about examples applied to a second
prompt: a worked example teaches its subject, not just its shape, so the fix
is to delete it rather than warn against it. The bad-example block is left
alone, since copying an example labelled bad is self-defeating.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROMPT = Path("src/ai/prompts/visual_search_terms.md")

# The exact phrases that came back copied. Asserting their absence rather than
# the absence of an "Examples" heading, because the failure was the phrases
# being available, not the section being titled.
COPIED_PHRASES = [
    "hand holding smartphone at night",
    "phone charging on bedside table",
    "person adjusting phone settings",
    "wifi router on a shelf",
    "frustrated person with laptop",
    "person resetting router cables",
]


@pytest.fixture(scope="module")
def prompt() -> str:
    return PROMPT.read_text(encoding="utf-8")


@pytest.mark.parametrize("phrase", COPIED_PHRASES)
def test_no_copyable_search_phrase_is_offered(prompt, phrase):
    assert phrase not in prompt, (
        f"{PROMPT} offers '{phrase}' as a worked example; measured at 2 in 10 "
        "runs, the model returns such a block verbatim for an unrelated script"
    )


def test_the_bad_examples_are_kept(prompt):
    """They teach without being copyable, so removing them costs accuracy."""
    assert "Bad, and why:" in prompt
    assert "background app refresh" in prompt


def test_the_shape_is_still_taught(prompt):
    """Deleting the examples alone made the phrases bare categories.

    Three of ten runs came back with `wifi router` and nothing else, which the
    prompt's own rules call a catalogue search. The template and the
    both-halves rule are what recovered the shape.
    """
    assert "<object>" in prompt
    assert "Every phrase needs both halves" in prompt


def test_a_minimum_length_is_required(prompt):
    """A two-word phrase is an object with no place and no actor."""
    assert "3 to {MAX_WORDS} words" in prompt
