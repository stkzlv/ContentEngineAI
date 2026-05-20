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
    battery for phone holders") then walk it back. The rule now branches on
    a keyword self-check: spec claim if the description has a contestable
    number, material-or-use claim otherwise.
    """
    text = template.read_text()
    # Branch condition is grep-able (token list).
    assert "contestable performance number" in text
    # Anchor against the canary case from production.
    assert "don't claim battery life for a phone holder" in text
    # Both branches' example sets present.
    assert "65W is the sweet spot for laptop charging" in text
    assert "Steel beats plastic for any clamp-style mount" in text


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
