"""The script lint (design 0007, REQ-CNT-053), held off."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from src.ai.llm_settings import ScriptLintConfig
from src.ai.script_lint import lint_script
from src.video.config import config

CLEAN = "This lamp clips on. It folds flat. It charges by USB."
LINT = ScriptLintConfig(enabled=True)


@pytest.mark.req("REQ-CNT-053")
@pytest.mark.parametrize(
    "tell",
    [
        "It's not a lamp, it's a lifestyle.",
        "This is a game-changer for desks.",
        "Say goodbye to clutter.",
        "It will elevate your desk.",
        "The fold is seamless.",
        "Let's delve into the specs.",
        "Whether you're a student or not, it works.",
        "In today's video we look at a lamp.",
        "You won't believe the price.",
    ],
)
def test_each_tell_fails_and_the_clean_script_passes(tell: str) -> None:
    assert lint_script(CLEAN, LINT) is None

    failure = lint_script(f"{CLEAN} {tell}", LINT)

    assert failure is not None and failure.startswith("uses the phrase")


@pytest.mark.req("REQ-CNT-053")
def test_curly_apostrophes_are_caught_too() -> None:
    assert lint_script(f"{CLEAN} It\u2019s not a toy, it\u2019s a tool.", LINT)
    assert lint_script(f"{CLEAN} Whether you\u2019re at home or not.", LINT)


@pytest.mark.parametrize(
    "ordinary",
    [
        "It's not cheap, but it lasts for years.",
        "The elevated stand lifts your screen.",
    ],
)
def test_ordinary_lines_pass(ordinary: str) -> None:
    assert lint_script(f"{CLEAN} {ordinary}", LINT) is None


def test_the_products_own_name_is_no_tell() -> None:
    script = f"{CLEAN} Seamless leggings stay put."

    assert lint_script(script, LINT) is not None
    assert lint_script(script, LINT, exempt="Seamless Leggings, high waist") is None


@pytest.mark.req("REQ-CNT-053")
def test_a_long_sentence_fails() -> None:
    long = "This lamp " + "really " * 14 + "clips on."

    assert lint_script(f"{CLEAN} {long}", LINT) == ("a sentence runs 18 words, over 16")


@pytest.mark.req("REQ-CNT-053")
def test_the_word_cap_follows_the_target_duration() -> None:
    script = "It clips on. " * 38  # 114 words, over 2.8 x 40 = 112

    assert lint_script(script, LINT) == "114 words, over 112 for 40 s"
    assert lint_script(script, LINT, word_cap=False) is None
    assert lint_script("It clips on. " * 37, LINT) is None


def test_an_invalid_pattern_is_refused_at_load() -> None:
    with pytest.raises(ValidationError):
        ScriptLintConfig(banned_phrases=["(unclosed"])


def _product():
    from src.video.producer.topic_input import TopicSpec, build_topic_product

    return build_topic_product(TopicSpec(title="How to fix it", description="x"))


async def _generate(replies: list[str], lint_on: bool, step_list=None):
    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_validation.lint = ScriptLintConfig(enabled=lint_on)
    call = AsyncMock(side_effect=replies * 10)
    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        script, _, _ = await script_generator.generate_script(
            _product(),
            settings,
            {settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
            step_list=step_list,
        )
    return script, call.await_count


def _script(body: str) -> str:
    cta = config.llm_settings.script_templates.cta_options_for(True)[0]
    filler = "Open the settings and turn the switch off. " * 8
    return f"{body} {filler}{cta}"


@pytest.mark.req("REQ-CNT-053")
@pytest.mark.asyncio
async def test_a_tell_is_retried_when_on_and_accepted_when_off() -> None:
    tell = _script("It's not a bug, it's a setting.")
    clean = _script("The fix is one setting.")

    off, off_calls = await _generate([tell], lint_on=False)
    on, on_calls = await _generate([tell, clean], lint_on=True)

    assert off == tell and off_calls == 1
    assert on == clean and on_calls == 2


@pytest.mark.req("REQ-CNT-053")
@pytest.mark.asyncio
async def test_a_script_failing_only_the_lint_ships_as_a_last_resort() -> None:
    tell = _script("It's not a bug, it's a setting.")

    script, calls = await _generate([tell], lint_on=True)

    assert script == tell and calls > 1


@pytest.mark.req("REQ-CNT-053")
@pytest.mark.asyncio
async def test_the_lint_last_resort_beats_a_script_missing_its_cta() -> None:
    no_cta = "Open the settings and turn the switch off. " * 9
    tell = _script("It's not a bug, it's a setting.")

    script, _ = await _generate([no_cta, tell], lint_on=True)

    # Complete and closing on its CTA, only the lint objected: kept as is.
    assert script == tell


@pytest.mark.req("REQ-CNT-053")
@pytest.mark.asyncio
async def test_a_tutorial_is_not_held_to_the_duration_cap() -> None:
    from src.ai.step_list import Step, StepList

    steps = [Step(f"Do {i}", "A > B", "done", "https://x/") for i in range(5)]
    sl = StepList("Settings", "iOS", False, steps, topic_failures=[])
    long_ok = _script("Open the settings and turn the switch off. " * 14)
    assert lint_script(long_ok, LINT) is not None  # over 112 words

    script, calls = await _generate([long_ok], lint_on=True, step_list=sl)

    assert script == long_ok and calls == 1
