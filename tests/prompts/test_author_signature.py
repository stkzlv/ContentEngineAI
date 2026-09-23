"""A recurring author signature, off by default (#440).

Rotating hooks, templates, CTAs and music is the defence against templated
sameness; what no layer gave was authored recurrence, the opener, transition
and sign-off a viewer recognises the channel by. Each element is drawn per
product below 1.0, so it recurs without appearing in every render.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.ai import script_generator
from src.ai.llm_settings import SignatureConfig
from src.ai.script_generator import (
    SignatureChoice,
    render_cta_rule,
    render_ending_rules,
    render_signature_rules,
    select_signature,
    validate_script_completeness,
)
from src.publisher.first_comment import build_first_comment, extract_closing_line
from src.publisher.models import FirstCommentConfig
from src.scraper.amazon.models import ProductData
from src.video.config import load_video_config_modular

CTA = "Link in bio if you want one."
POOLS = {
    "openers": ["Quick one", "Real talk"],
    "transitions": ["here's the thing"],
    "signoffs": ["That's the find for today."],
}
CLOSING = "Team magnetic or team plug-in?"
SCRIPT = (
    "Quick one, this magnetic phone mount grips hard and fits in a pocket. "
    "I took it on a hike and it never slipped. Here's the thing, the magnet "
    "is strong enough for a big phone in a case. The battery lasts about six "
    f"hours under load, which covers a day out. {CLOSING} "
    f"That's the find for today. {CTA}"
)


class TestSelection:
    def test_unconfigured_draws_nothing(self) -> None:
        assert select_signature(SignatureConfig(), "B0X") == SignatureChoice()

    def test_no_product_draws_nothing(self) -> None:
        config = SignatureConfig(use_rate=1.0, **POOLS)
        assert select_signature(config, None) == SignatureChoice()

    def test_rate_one_always_draws_every_element(self) -> None:
        choice = select_signature(SignatureConfig(use_rate=1.0, **POOLS), "B0X")
        assert choice.opener in POOLS["openers"]
        assert choice.transition == "here's the thing"
        assert choice.signoff == "That's the find for today."

    def test_rate_zero_never_draws(self) -> None:
        config = SignatureConfig(use_rate=0.0, **POOLS)
        assert not config.configured
        assert select_signature(config, "B0X") == SignatureChoice()

    def test_the_same_product_gets_the_same_choice(self) -> None:
        config = SignatureConfig(use_rate=0.5, **POOLS)
        assert select_signature(config, "B0A") == select_signature(config, "B0A")

    def test_a_batch_recurs_at_about_the_configured_rate(self) -> None:
        config = SignatureConfig(use_rate=0.5, **POOLS)
        draws = [select_signature(config, f"B0{i:05d}") for i in range(400)]
        for element in ("opener", "transition", "signoff"):
            share = sum(bool(getattr(d, element)) for d in draws) / len(draws)
            assert 0.4 < share < 0.6, (element, share)

    def test_elements_are_drawn_independently(self) -> None:
        config = SignatureConfig(use_rate=0.5, **POOLS)
        draws = [select_signature(config, f"B0{i:05d}") for i in range(400)]
        both = sum(bool(d.opener and d.signoff) for d in draws) / len(draws)
        assert 0.15 < both < 0.35

    def test_an_empty_entry_is_refused_at_load(self) -> None:
        with pytest.raises(ValidationError):
            SignatureConfig(signoffs=["..."])


class TestTheRules:
    def test_an_empty_choice_renders_nothing(self) -> None:
        assert render_signature_rules(SignatureChoice()) == ""

    def test_unconfigured_leaves_the_ending_rules_byte_identical(self) -> None:
        assert render_ending_rules(CTA, False, 0, SignatureChoice()) == (
            render_cta_rule(CTA)
        )

    def test_the_cta_rule_stays_next_to_the_closing_beat(self) -> None:
        choice = SignatureChoice(opener="Quick one", signoff="That's it for today.")
        lines = render_ending_rules(CTA, False, 0, choice).split("\n")
        assert lines[0] == render_cta_rule(CTA)
        assert any('"Quick one,"' in line for line in lines[1:])
        assert any("directly before the call to action" in line for line in lines)

    def test_the_opener_joins_the_first_sentence(self) -> None:
        rule = render_signature_rules(SignatureChoice(opener="Quick one"))
        assert "continue that same sentence" in rule

    def test_a_signed_off_script_still_validates(self) -> None:
        ok, reason = validate_script_completeness(
            SCRIPT, min_chars=200, min_words=50, cta_options=[CTA]
        )
        assert ok, reason


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_the_drawn_signature_reaches_the_composed_prompt(
    monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    settings = load_video_config_modular().llm_settings
    settings.script_templates.fixed_cta = CTA
    settings.script_templates.signature = (
        SignatureConfig(use_rate=1.0, **POOLS) if configured else SignatureConfig()
    )
    seen: list[str] = []

    async def capture(prompt, *a, **k):
        seen.append(prompt)
        return SCRIPT

    monkeypatch.setattr(script_generator, "_call_llm_api_with_retry", capture)
    monkeypatch.setattr(
        script_generator, "fetch_and_select_model", AsyncMock(return_value=[])
    )
    await script_generator.generate_script(
        ProductData(title="Magnetic phone mount", price="", url="", platform="t"),
        settings,
        {settings.api_key_env_var: "k"},
        None,
        {},
        False,
        product_id="B0TEST0001",
    )
    assert seen
    assert ('"That\'s the find for today."' in seen[0]) is configured


class TestTheFirstCommentSkipsTheSignoff:
    def test_the_recorded_signoff_is_stripped(self) -> None:
        assert (
            extract_closing_line(SCRIPT, signoff="That's the find for today.")
            == CLOSING
        )

    def test_without_the_record_the_signoff_would_be_the_comment(self) -> None:
        """The reason the step records it."""
        assert extract_closing_line(SCRIPT) == "That's the find for today."

    def test_a_script_without_the_signoff_is_unaffected(self) -> None:
        script = SCRIPT.replace("That's the find for today. ", "")
        assert extract_closing_line(script, signoff="That's the find for today.") == (
            CLOSING
        )

    @pytest.mark.parametrize("where", ["top", "step"])
    def test_build_reads_the_signoff_from_the_pipeline_state(
        self, tmp_path: Path, where: str
    ) -> None:
        temp = tmp_path / "B0X" / "temp"
        temp.mkdir(parents=True)
        (temp / "script.txt").write_text(SCRIPT)
        signoff = "That's the find for today."
        state = (
            {"signoff": signoff}
            if where == "top"
            else {"generate_script": {"status": "done", "signoff": signoff}}
        )
        (temp / "pipeline_state.json").write_text(json.dumps(state))
        config = FirstCommentConfig(
            enabled=True, platforms={"youtube": "{closing_line}"}
        )
        assert build_first_comment(config, "youtube", "B0X", tmp_path) == CLOSING
