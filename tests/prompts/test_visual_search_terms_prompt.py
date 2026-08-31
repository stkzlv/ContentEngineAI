"""The stock-search prompt must not hand the model phrases it can copy.

Measured on a real topic script (wifi channel congestion), the prompt's own
worked examples came back verbatim in 2 of 10 runs -- the phone-battery block,
for a video about wifi. The render then searched a stock library for a phone
at night and got one, which is what put an unrelated sunset photo under
narration about wifi channels.

This is the project's recorded lesson about examples applied to a second
prompt: a worked example teaches its subject, not just its shape, so the fix
is to delete it rather than warn against it. The bad-example block is kept and
extended, since copying an example labelled bad is self-defeating.
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
    # Not from an examples block: this one illustrated a rule inline, which
    # made it exactly as copyable and put a phone in a bedroom on the page of
    # a prompt that a wifi script also reads.
    "hand holding smartphone in bed",
]


# The reformulations that motivated the rule change. Each is correct for the
# video it was measured on and wrong for any other, so they are exactly what a
# future editor would reach for as an illustration -- which is the failure this
# file already exists to prevent.
REFORMULATIONS = [
    "canned air duster",
    "laptop cooling fan",
    "laptop repair screwdriver",
    "phone battery health screen",
    "compressed air keyboard cleaning",
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


def test_no_worked_example_block_is_reintroduced(prompt):
    """The phrase list above pins six strings; this pins the shape.

    A fresh block with different phrases would satisfy every assertion above
    while reintroducing exactly the defect, since what gets copied is any
    good-example block, not those six lines in particular.

    Matching any example heading rather than the literal `## Examples`: the
    first version of this test passed against a block titled `## Worked
    examples`, which is the same defect under another name.
    """
    import re

    headings = re.findall(r"^#+\s*(.+)$", prompt, re.M)
    offending = [
        h
        for h in headings
        if "example" in h.lower() and not h.lower().startswith("bad")
    ]

    assert not offending, (
        f"{PROMPT} has a worked-example section ({offending}); measured at "
        "2 in 10 runs, the model returns such a block verbatim for an "
        "unrelated script. Only the bad-example block is safe, because "
        "copying an example labelled bad is self-defeating."
    )


def test_the_bad_examples_are_kept(prompt):
    """They teach without being copyable, so removing them costs accuracy."""
    assert "Bad, and why:" in prompt
    assert "background app refresh" in prompt


def test_the_shape_is_still_taught(prompt):
    """Deleting the examples alone made the phrases bare categories.

    Three of ten runs came back with `wifi router` and nothing else, which the
    prompt's own rules call a catalogue search -- and which the sanitizer then
    discards for being under the floor, leaving the render one generic search
    for the whole video. A template is what recovered the shape.

    The template no longer reaches the floor by attaching a person. That was
    measured to be the cause of a different failure: `hand holding compressed
    air` returns a hand holding a hotdog, because `hand` and `holding` are in
    a large share of the library's captions and consume two of the words. The
    length is now reached with qualifiers, which say something about the shot.
    """
    assert "<object>" in prompt
    assert "<qualifier>" in prompt

    # The mandate that produced the diluted phrases, asserted absent. A prompt
    # test that keeps asserting the old contract is what blocks the fix.
    assert "Every phrase needs both halves" not in prompt
    assert "<body part> <verb-ing> <object>" not in prompt

    # The floor still has to be taught, or the bare-category regression above
    # returns by a different route.
    assert "thrown away before it is searched" in prompt


def test_the_floor_is_not_restated_in_the_prompt(prompt):
    """The number lives in `MIN_PHRASE_WORDS` and renders in.

    Stated separately, the instruction and the filter drift: the prompt asked
    for three words while the sanitizer accepted two, so the rule held only
    for as long as the model chose to follow it.
    """
    from src.ai.script_generator import MIN_PHRASE_WORDS

    assert "{MIN_WORDS} to {MAX_WORDS} words" in prompt
    assert MIN_PHRASE_WORDS == 3


def test_the_sanitizer_enforces_the_floor():
    """A prompt rule is model compliance; the filter is the constraint.

    Measured 0 of 30 phrases dropped, so this was latent rather than live --
    which is the reason to close it now rather than after it bites.
    """
    from src.ai.script_generator import sanitize_visual_search_phrases

    kept = sanitize_visual_search_phrases(
        "wifi router\nhand typing on laptop\nperson at a desk",
        max_phrases=3,
        max_words=5,
    )

    assert kept == ["hand typing on laptop", "person at a desk"]


def test_the_rendered_prompt_carries_the_floor():
    """The template names `{MIN_WORDS}`; exactly one call site supplies it.

    Dropping it is not a loud failure. `format_prompt` raises, the caller
    catches `ValueError`, logs one warning and returns no phrases, which the
    render reads as "keep the terms you had" -- a stock profile then searches
    on the topic title alone. Nothing else renders this template, so nothing
    else would notice.
    """
    from src.ai.description_generator import format_prompt
    from src.ai.script_generator import MIN_PHRASE_WORDS
    from src.scraper.amazon.models import ProductData
    from src.scraper.base.models import Platform

    product = ProductData(
        title="Why your wifi keeps dropping",
        price="",
        url="",
        platform=Platform.AMAZON,
        description="Router placement and channel congestion.",
        asin="topic-x",
        topic="Why your wifi keeps dropping",
    )

    rendered = format_prompt(
        PROMPT.read_text(encoding="utf-8"),
        product,
        video_script="Change the channel to 1, 6 or 11.",
        extra_placeholders={
            "MAX_PHRASES": "3",
            "MAX_WORDS": "5",
            "MIN_WORDS": str(MIN_PHRASE_WORDS),
        },
    )

    assert f"Each phrase is {MIN_PHRASE_WORDS} to 5 words" in rendered
    assert "{MIN_WORDS}" not in rendered


def test_the_call_site_supplies_the_floor():
    """The render test above builds its own placeholder dict.

    So it proves the template is renderable, not that production renders it.
    Dropping `MIN_WORDS` from the real call site left that test green while
    every stock render fell back to a title-only search.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/ai/script_generator.py").read_text())
    supplied = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and {
            k.value
            for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        >= {"MAX_PHRASES", "MAX_WORDS", "MIN_WORDS"}
    ]

    assert supplied, (
        "no call site builds a placeholder dict carrying MIN_WORDS; the "
        "visual-search prompt names it, so rendering raises and the render "
        "falls back to searching on the topic title alone"
    )


@pytest.mark.parametrize("phrase", REFORMULATIONS)
def test_no_reformulation_became_a_worked_example(prompt, phrase):
    """The rules are stated abstractly; the evidence for them is not in here.

    These phrases are why the rule changed, and they belong in the changelog
    and the issue. In the prompt they would be copied, the way the previous
    good-example block was copied verbatim into a video about something else.
    """
    assert phrase not in prompt


def test_the_generic_human_words_are_named_as_the_problem(prompt):
    """The rule has to say which words dilute, or it is only a preference."""
    for word in ("hand", "person", "holding"):
        assert word in prompt


def test_the_taught_template_reaches_the_floor(prompt):
    """Read from the prompt, not from a list beside it.

    An earlier version of this test fed a hardcoded list to the sanitizer and
    ignored the `prompt` fixture entirely, so it stayed green when the Shape
    block was mutated to a two-word template -- the one change it existed to
    catch. A test that names the prompt in its signature and never reads it is
    the failure this file keeps finding in the prompt itself.

    A template under the floor is not a weaker prompt, it is a silent one: the
    sanitizer drops every phrase, and with none left `steps.py` falls back to a
    single generic search for the whole video.
    """
    import re

    from src.ai.script_generator import MIN_PHRASE_WORDS

    templates = [line for line in prompt.splitlines() if line.startswith("    <")]
    assert templates, "no shape template found; the block moved or was removed"

    for line in templates:
        slots = re.findall(r"<[^>]+>", line)
        assert len(slots) >= MIN_PHRASE_WORDS, (
            f"template {line.strip()!r} teaches {len(slots)} words, below the "
            f"sanitizer floor of {MIN_PHRASE_WORDS}; every phrase in this shape "
            "is dropped before it is searched"
        )


def test_a_qualified_phrase_survives_the_sanitizer():
    """The second half: a phrase in that shape actually passes the filter."""
    from src.ai.script_generator import sanitize_visual_search_phrases

    survived = sanitize_visual_search_phrases(
        "\n".join(REFORMULATIONS), max_phrases=len(REFORMULATIONS), max_words=5
    )

    assert survived == REFORMULATIONS, f"the sanitizer dropped some: {survived}"
