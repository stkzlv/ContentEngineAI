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
    """Phase 1.5: every template carries a comment-fork OR spec-correction close."""
    text = template.read_text()
    has_fork = "two-option opinion question right before the CTA" in text
    has_spec = "debatable spec claim right before the CTA" in text
    assert has_fork or has_spec, "Template missing both fork and spec-correction"


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
