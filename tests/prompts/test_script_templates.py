"""Sanity tests for the 15 script templates in src/ai/prompts/scripts/.

Asserts that the cross-template rules introduced by Phase 1.1 / 1.2 / 1.5
have not drifted out of any template. These rules are repeated across all
15 files; a single template losing one is a silent regression that won't
surface until a render lands wrong.
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


def _all_templates() -> list[Path]:
    return sorted(SCRIPTS_DIR.glob("*.md"))


def test_expected_15_templates_present() -> None:
    names = {p.stem for p in _all_templates()}
    assert names == EXPECTED_TEMPLATES


@pytest.mark.parametrize("template", _all_templates(), ids=lambda p: p.stem)
def test_phase_1_1_audio_keyword_rule_present(template: Path) -> None:
    """Phase 1.1: every template carries the audio-keyword hook rule."""
    text = template.read_text()
    assert (
        "Open with a natural conversational hook that carries the audio keyword" in text
    )


@pytest.mark.parametrize("template", _all_templates(), ids=lambda p: p.stem)
def test_phase_1_2_anti_setup_clause_present(template: Path) -> None:
    """Phase 1.2: every template carries the anti-setup clause on line 1."""
    text = template.read_text()
    assert "Anti-setup" in text
    assert 'Avoid "Today I\'ll"' in text


@pytest.mark.parametrize("template", _all_templates(), ids=lambda p: p.stem)
def test_phase_1_5_closing_line_rule_present(template: Path) -> None:
    """Phase 1.5: every template carries a comment-fork OR debatable-claim close."""
    text = template.read_text()
    has_fork = "two-option opinion question right before the CTA" in text
    has_claim = "debatable claim right before the CTA" in text
    assert has_fork or has_claim, "Template missing both fork and debatable-claim"


@pytest.mark.parametrize(
    "template",
    [p for p in _all_templates() if p.stem in ANALYTICAL_TEMPLATES],
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
    [p for p in _all_templates() if p.stem in ANALYTICAL_TEMPLATES],
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
    [p for p in _all_templates() if p.stem in ANALYTICAL_TEMPLATES],
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


@pytest.mark.parametrize("template", _all_templates(), ids=lambda p: p.stem)
def test_tradeoff_rule_present(template: Path) -> None:
    """Trade-off honesty rule (Phase 0.43.1) on every template."""
    text = template.read_text()
    assert "one short trade-off or limitation" in text


@pytest.mark.parametrize("template", _all_templates(), ids=lambda p: p.stem)
def test_product_data_block_present(template: Path) -> None:
    """Every template ends with the {AUDIENCE} placeholder block."""
    text = template.read_text()
    assert "{FULL_PRODUCT_NAME}" in text
    assert "{SHORT_PRODUCT_NAME}" in text
    assert "{PRODUCT_DESCRIPTION}" in text
    assert "{AUDIENCE}" in text
