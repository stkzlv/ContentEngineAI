"""Conversational delivery written into the script, off by default (#438).

Generated scripts are grammatically flawless, which reads as machine output.
The naturalism rule asks for contractions, spoken fillers and one emotional
beat, and keeps them out of the places other checks depend on: the first
sentence, any factual claim, and the closing call to action.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.ai import script_generator
from src.ai.llm_settings import NaturalismConfig
from src.ai.script_generator import (
    NATURALISM_FILLERS,
    render_cta_rule,
    render_ending_rules,
    render_naturalism_rule,
    validate_script_completeness,
)
from src.scraper.amazon.models import ProductData
from src.video.config import load_video_config_modular

CTA = "Link in bio if you want one."
RULE_HEAD = "Sound like someone talking, not reading."

NATURALISED = (
    "This magnetic phone mount fits in a jacket pocket and grips hard. So I "
    "took it on a hike last month, fully expecting it to slide off. Okay, it "
    "didn't. It's, well, it's smaller than I expected, but the magnet holds "
    "on rough trails. The battery lasts about six hours under load, which "
    "covers a day out. Look, I was tired of phones flying off the dash, and "
    "this one just stays put. Team magnetic or team plug-in? " + CTA
)


class TestOffIsTheOldPrompt:
    def test_zero_renders_nothing(self) -> None:
        assert render_naturalism_rule(0) == ""

    @pytest.mark.parametrize("is_topic", [False, True])
    def test_zero_leaves_the_ending_rules_byte_identical(self, is_topic) -> None:
        assert render_ending_rules(CTA, is_topic, 0) == render_cta_rule(
            CTA, is_topic=is_topic
        )

    def test_the_default_is_off(self) -> None:
        assert NaturalismConfig().intensity == 0

    @pytest.mark.parametrize("value", [-1, 3])
    def test_out_of_range_is_refused_at_load(self, value) -> None:
        with pytest.raises(ValidationError):
            NaturalismConfig(intensity=value)


class TestTheRule:
    def test_one_asks_for_fillers_without_a_correction(self) -> None:
        rule = render_naturalism_rule(1)
        assert "one or two spoken fillers" in rule
        assert "self-correction" not in rule

    def test_two_adds_a_correction_that_never_touches_a_fact(self) -> None:
        rule = render_naturalism_rule(2)
        assert "two or three spoken fillers" in rule
        assert "self-correction" in rule
        assert "never changing a fact" in rule

    def test_the_correction_carries_no_worked_example(self) -> None:
        """An example teaches its subject with its shape; the first draft's
        was a first-person size claim that topic prompts forbid.
        """
        assert (
            '"'
            not in render_naturalism_rule(2).split("self-correction")[1].split(".")[0]
        )

    @pytest.mark.parametrize("intensity", [1, 2])
    def test_it_names_every_protected_position(self, intensity) -> None:
        rule = render_naturalism_rule(intensity)
        assert "first sentence" in rule
        assert "number, name or claim" in rule
        assert "call to action" in rule
        assert "emotional beat" in rule

    def test_it_offers_no_filler_the_narrator_profiles_ban(self) -> None:
        """The profiles list "honestly" and "literally" as empty intensifiers.
        A rule that offered them would contradict the profile it sits under.
        """
        templates = load_video_config_modular().llm_settings.script_templates
        for profile in (templates.narrator_profile, templates.narrator_profile_topic):
            banned = next(
                line for line in profile.splitlines() if "intensifiers" in line
            )
            for filler in NATURALISM_FILLERS.replace('"', "").split(", "):
                assert f'"{filler}"' not in banned

    def test_the_cta_rule_stays_next_to_the_closing_beat(self) -> None:
        """The CTA rule says "after the closing beat above", meaning the
        template bullet directly over the placeholder. The naturalism rule
        goes after it so it cannot take that referent.
        """
        lines = render_ending_rules(CTA, False, 2).split("\n")
        assert lines[0] == render_cta_rule(CTA)
        assert RULE_HEAD in lines[1]


class TestANaturalisedScriptStillValidates:
    def test_fillers_and_a_correction_pass_the_completeness_check(self) -> None:
        ok, reason = validate_script_completeness(
            NATURALISED, min_chars=200, min_words=50, cta_options=[CTA]
        )
        assert ok, reason


@pytest.mark.asyncio
@pytest.mark.parametrize("intensity", [0, 2])
async def test_the_configured_intensity_reaches_the_composed_prompt(
    monkeypatch: pytest.MonkeyPatch, intensity: int
) -> None:
    settings = load_video_config_modular().llm_settings
    settings.script_templates.fixed_cta = CTA
    settings.script_templates.naturalism = NaturalismConfig(intensity=intensity)
    seen: list[str] = []

    async def capture(prompt, *a, **k):
        seen.append(prompt)
        return NATURALISED

    monkeypatch.setattr(script_generator, "_call_llm_api_with_retry", capture)
    monkeypatch.setattr(
        script_generator, "fetch_and_select_model", AsyncMock(return_value=[])
    )
    script, _, _ = await script_generator.generate_script(
        ProductData(title="Magnetic phone mount", price="", url="", platform="t"),
        settings,
        {settings.api_key_env_var: "k"},
        None,
        {},
        False,
        product_id="B0TEST0001",
    )
    assert seen, "the generator never called the model"
    assert (RULE_HEAD in seen[0]) is (intensity > 0)
    assert script == NATURALISED
