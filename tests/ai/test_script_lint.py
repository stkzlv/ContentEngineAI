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


async def _generate(replies: list[str], lint_on: bool):
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
