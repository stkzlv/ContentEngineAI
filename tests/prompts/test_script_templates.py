"""Sanity tests for the script templates in src/ai/prompts/scripts/.

Asserts that the cross-template rules have not drifted out of any template.
These rules are repeated across every file; a single template losing one is a
silent regression that won't surface until a render lands wrong.

The directory holds two families with different contracts. Product templates
pitch a scraped listing and carry the hook, anti-setup and closing-line rules.
Topic templates answer a question and must not mention a product at all, so the
product rules are not merely absent from them, they would be wrong. Each family
is collected separately, and a new template has to be added to one of the two
sets deliberately rather than inheriting the other's contract by being dropped
in the same folder.
"""

from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "src" / "ai" / "prompts" / "scripts"

EXPECTED_TEMPLATES = {
    "before_after",
    "challenge_dare",
    "classic_promo",
    "comparison",
    "curiosity_hook",
    "lifestyle_flex",
    "myth_buster",
    "problem_solution",
    "question_driven",
    "rapid_fire",
    "secret_reveal",
    "skeptic_converted",
    "social_proof",
    "story_driven",
    "unboxing_reaction",
}

# Templates that use the analytical closing line (spec-or-material claim).
# The remaining 7 templates use the comment-fork close. Keeping the split
# explicit here so a future template gets categorised intentionally rather
# than silently inheriting a default.
ANALYTICAL_TEMPLATES = {
    "before_after",
    "challenge_dare",
    "classic_promo",
    "comparison",
    "myth_buster",
    "problem_solution",
    "question_driven",
    "rapid_fire",
}


# Templates for topic renders, which have no product. They carry the answer-first
# rules instead of the product ones.
TOPIC_TEMPLATES = {
    "topic_answer_first",
    "topic_mistake_fix",
    "topic_symptom_cause",
}


def _all_templates() -> list[Path]:
    return sorted(SCRIPTS_DIR.glob("*.md"))


def _product_templates() -> list[Path]:
    return [p for p in _all_templates() if p.stem not in TOPIC_TEMPLATES]


def _topic_templates() -> list[Path]:
    return [p for p in _all_templates() if p.stem in TOPIC_TEMPLATES]


def test_topic_family_matches_the_config() -> None:
    """The runtime family is the config list, not this constant.

    A template added here but not to `script_templates.topic_templates` is drawn
    by product renders, and every test still passes: the topic-pool test asserts
    against the same config list that is missing the entry.
    """
    from src.video.config import config

    assert set(config.llm_settings.script_templates.topic_templates) == TOPIC_TEMPLATES


def test_every_template_is_categorised() -> None:
    """A new template must join a family, not sit between them.

    Dropping a file in the directory makes it selectable, so one that belongs to
    neither set would ship with none of either family's rules checked.
    """
    names = {p.stem for p in _all_templates()}
    assert names == EXPECTED_TEMPLATES | TOPIC_TEMPLATES


@pytest.mark.parametrize("template", _product_templates(), ids=lambda p: p.stem)
def test_phase_1_1_audio_keyword_rule_present(template: Path) -> None:
    """Phase 1.1: every template carries the audio-keyword hook rule."""
    text = template.read_text()
    assert (
        "Open with a natural conversational hook that carries the audio keyword" in text
    )


@pytest.mark.parametrize("template", _product_templates(), ids=lambda p: p.stem)
def test_phase_1_2_anti_setup_clause_present(template: Path) -> None:
    """Phase 1.2: every template carries the anti-setup clause on line 1."""
    text = template.read_text()
    assert "Anti-setup" in text
    assert 'Avoid "Today I\'ll"' in text


@pytest.mark.parametrize("template", _product_templates(), ids=lambda p: p.stem)
def test_phase_1_5_closing_line_rule_present(template: Path) -> None:
    """Phase 1.5: every template carries a comment-fork OR debatable-claim close."""
    text = template.read_text()
    has_fork = "two-option opinion question right before the CTA" in text
    has_claim = "debatable claim right before the CTA" in text
    assert has_fork or has_claim, "Template missing both fork and debatable-claim"


@pytest.mark.parametrize(
    "template",
    [p for p in _product_templates() if p.stem in ANALYTICAL_TEMPLATES],
    ids=lambda p: p.stem,
)
def test_analytical_close_has_passive_product_branch(template: Path) -> None:
    """Every analytical template carries the spec-vs-passive conditional.

    Bug class: the spec-correction close on a passive product (phone holder,
    bracket, organizer) made the LLM fabricate a spec (e.g. "eight hours of
    battery for phone holders") then walk it back. The rule branches on a
    self-check: spec claim only when the description states a measurement,
    material-or-use claim otherwise.
    """
    text = template.read_text()
    # Branch condition is grep-able, and demands a verbatim quote rather than
    # a loose "does the description mention X" scan.
    assert "written out with its unit attached" in text
    assert "quote the measurement verbatim" in text
    # Anchored against both production canaries.
    assert "a tracker tag has no ports" in text
    assert "a phone holder has no battery life" in text
    # The passive branch keeps worked examples; they are non-numeric.
    assert "Steel beats plastic for any clamp-style mount" in text


@pytest.mark.parametrize(
    "template",
    [p for p in _product_templates() if p.stem in ANALYTICAL_TEMPLATES],
    ids=lambda p: p.stem,
)
def test_analytical_close_has_no_worked_spec_example(template: Path) -> None:
    """The spec branch must not ship a worked closing-line example.

    Regression guard for the second occurrence of the fabrication bug. The
    rule previously demonstrated its spec branch with "Most people only need
    two ports, but three is usually better." On a Bluetooth tracker tag the
    model reproduced that line's *subject* almost verbatim ("four ports is
    the minimum you need, but honestly, three is usually enough") for a
    product with no ports. Examples outrank rules, so the spec branch now
    carries no closing-line example to copy: the model has to derive the
    subject from the measurement it quoted.
    """
    text = template.read_text()
    assert "only need two ports" not in text
    assert "65W is the sweet spot for laptop charging" not in text
    assert "Eight hours is the right battery target" not in text


@pytest.mark.parametrize(
    "template",
    [p for p in _product_templates() if p.stem in ANALYTICAL_TEMPLATES],
    ids=lambda p: p.stem,
)
def test_analytical_close_warns_against_substring_units(template: Path) -> None:
    """The self-check must reject a unit found inside a longer word.

    The tracker's description contained no standalone "port" — only
    "supports", "Portable", and "Important". A substring scan finds "port"
    in all three and concludes the product has ports.
    """
    text = template.read_text()
    assert "as a whole word" in text
    assert '"supports" and "Portable" are not ports' in text


@pytest.mark.parametrize("template", _product_templates(), ids=lambda p: p.stem)
def test_tradeoff_rule_present(template: Path) -> None:
    """Trade-off honesty rule (Phase 0.43.1) on every template."""
    text = template.read_text()
    assert "one short trade-off or limitation" in text


@pytest.mark.parametrize("template", _product_templates(), ids=lambda p: p.stem)
def test_product_data_block_present(template: Path) -> None:
    """Every template ends with the {AUDIENCE} placeholder block."""
    text = template.read_text()
    assert "{FULL_PRODUCT_NAME}" in text
    assert "{SHORT_PRODUCT_NAME}" in text
    assert "{PRODUCT_DESCRIPTION}" in text
    assert "{AUDIENCE}" in text


@pytest.mark.parametrize("template", _topic_templates(), ids=lambda p: p.stem)
def test_topic_template_uses_neutral_placeholders(template: Path) -> None:
    """A topic template must not ask for product fields.

    `{SHORT_PRODUCT_NAME}` is documented to the model as the thing's name, so a
    topic template carrying it tells the model to speak the question as a
    product.
    """
    text = template.read_text()
    assert "{TOPIC_TITLE}" in text
    assert "{TOPIC_DETAIL}" in text
    assert "{PRODUCT_DESCRIPTION}" not in text
    assert "{SHORT_PRODUCT_NAME}" not in text
    assert "{FULL_PRODUCT_NAME}" not in text


@pytest.mark.parametrize("template", _topic_templates(), ids=lambda p: p.stem)
def test_topic_template_forbids_inventing_a_product(template: Path) -> None:
    """The observed failure was the model inventing something to sell.

    Two real renders produced "I just got this thing" and closed on "Link in bio
    if you want one" for topics with no product anywhere in the input.
    """
    text = template.read_text()
    assert "Do not invent a product" in text
    assert "this thing" in text


@pytest.mark.parametrize("template", _topic_templates(), ids=lambda p: p.stem)
def test_topic_template_states_the_answer_first(template: Path) -> None:
    """Withholding the fix loses the viewer who came for it.

    The answer-first shape is the main thing separating tutorial content from
    the promo structure the product templates use.
    """
    text = template.read_text().lower()
    assert "first three seconds" in text or "same breath" in text


@pytest.mark.parametrize("template", _topic_templates(), ids=lambda p: p.stem)
def test_topic_template_places_the_honest_limit_inside_the_method(
    template: Path,
) -> None:
    """Topic scripts closed on the caveat instead of the result.

    Each template's body asks the script to finish on the result, and the
    Rules block asked for an honest limit without saying where. The limit
    bullet sat directly above `{CTA_RULE}`, so it was the nearest imperative
    to the ending and won: the last spoken line before the call to action was
    "if it's still loud, you might have more serious issues". That is the
    body-versus-Rules placement class, with the two instructions competing for
    the same final beat.

    The best-practice guidance puts the payoff last and caveats in the method,
    because a video ending on a caveat sends the viewer away unsure it worked.
    """
    text = " ".join(template.read_text().split())
    assert "One honest limit, inside the method rather than after it" in text
    assert "placed among the steps" in text
    # And the body has to name the position too, or the Rules bullet is the
    # only thing saying where the close goes and the body still competes.
    assert "immediately before the call to action" in text


@pytest.mark.parametrize("template", _topic_templates(), ids=lambda p: p.stem)
def test_topic_template_does_not_end_its_rules_on_the_limit(
    template: Path,
) -> None:
    """Position, not wording, is what displaced the closing beat.

    A rule stating where the limit goes while itself sitting last before
    `{CTA_RULE}` still puts a caveat in the model's path to the ending, which
    is how it shipped on two of the three by position. It is not the whole mechanism --
    `topic_answer_first` had a delivery-format bullet in between and still
    closed on the caveat -- so the wording assertion above carries the rest.
    This one pins the half a re-ordering can silently undo.
    """
    body, _, _ = template.read_text().partition("{CTA_RULE}")
    last_bullet = [ln for ln in body.splitlines() if ln.startswith("- ")][-1]

    assert "honest limit" not in last_bullet


@pytest.mark.parametrize("template", _topic_templates(), ids=lambda p: p.stem)
def test_topic_template_requires_the_spoken_search_phrase(template: Path) -> None:
    """Platforms index the transcript, so the phrase has to be said aloud."""
    text = template.read_text()
    assert "first five seconds" in text


@pytest.mark.parametrize("template", _topic_templates(), ids=lambda p: p.stem)
def test_topic_template_branches_on_whether_the_path_is_known(
    template: Path,
) -> None:
    """A shipped script sent viewers to a Windows menu node that does not exist.

    The narrator profile used to demand "name the actual setting, menu, or
    number wherever there is one", a quantifier the model cannot evaluate,
    offering three shapes that were all the same shape: something read off an
    interface. A topic with no nameable path had no satisfiable branch, so the
    model invented one. Four distinct invented methods turned up in seventeen
    scripts from the pool.

    The demand is gone from the profile and this branch replaces it: a
    mechanical self-check, and an alternative shape that is followable without
    knowing a path.
    """
    # Collapsed, because the bullet is hard-wrapped like its neighbours and a
    # phrase that spans a line break is not a literal substring of the file.
    text = " ".join(template.read_text().split())
    assert "only if you can state it exactly" in text
    # The referent per shape. "Labels in order" alone gives a URL nothing to
    # satisfy, so an invented address passed the gate on the platform anchor
    # alone; and gating "an exact setting" contradicted the profile, which
    # holds up "channel 1, 6, or 11" as the answer two paragraphs earlier.
    assert "the labels in\n  order, or the address in full" in template.read_text()
    # The platform anchor is what makes the self-check mechanical, and the
    # abstention arm is what stops the rule demanding a path it cannot supply.
    # Without these two the bullet reads the same and pins nothing.
    assert "for a platform you name" in text
    assert "say plainly that it differs by device" in text
    assert "name what the viewer can see or feel on the device instead" in text
    assert "is not the same as it naming one" in text


@pytest.mark.parametrize("template", _topic_templates(), ids=lambda p: p.stem)
def test_topic_template_names_no_worked_path(template: Path) -> None:
    """The branch above must not carry an example, not even a negative one.

    The model produced "System Information / Components / Power" from a prompt
    that never contained those words. Naming them to warn against them hands it
    the subject back, which is how a worked example inside a conditional branch
    was previously copied verbatim onto the wrong product.
    """
    text = template.read_text()
    assert "System Information" not in text
    assert "Components" not in text


def test_the_topic_profile_no_longer_demands_a_menu_path() -> None:
    """The upstream half of the same fix.

    The absences are the load-bearing assertions: leaving the demand in place
    makes the prompt argue with itself, an unconditional "name the actual menu"
    forty lines above a conditional "only if you can state the labels".
    """
    from src.video.config import config

    profile = config.llm_settings.script_templates.narrator_profile_topic
    assert "wherever there is one" not in profile
    assert "A vague instruction cannot be followed" not in profile
    # The condition here is the same mechanical one the templates carry, not a
    # self-assessment. Dropping it entirely was measured and lost: on 50 canary
    # scripts the account-deletion topic went from zero invented paths to
    # three, and the impossible "turn the VPN on before you join the wifi"
    # ordering came back. The clause earns its place; it just has to be
    # checkable rather than ask the model what it knows.
    assert "state it exactly for a platform you name" in profile
    assert "a wrong path costs more than a general one" in profile.lower()
