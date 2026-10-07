"""A script that borrows a quoted prompt example is retried (#672)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.ai.script_generator import copied_example
from src.video.config import config

CTA = "Follow for more finds like this."
PROMPT = (
    "Close with a material, shape, or use claim. Examples of (c): "
    '"Steel beats plastic for any clamp-style mount." / "Gooseneck arms win '
    'over ball joints for bedside use." End on the CTA: "Follow for more '
    'finds like this."'
)
WATCH = (
    "Smartwatch with a 1.43 inch AMOLED display, stainless steel case, "
    "heart rate and sleep tracking, seven-day battery."
)


def script(closing: str) -> str:
    return f"This watch tracks sleep all week. It lasts seven days. {closing} {CTA}"


@pytest.mark.req("REQ-CNT-155")
def test_the_prompts_example_is_caught_on_a_product_it_does_not_fit() -> None:
    closing = "Steel beats plastic for any clamp-style mount."

    assert copied_example(script(closing), PROMPT, WATCH, [CTA]) == closing


@pytest.mark.req("REQ-CNT-155")
def test_a_reworded_copy_is_caught_too() -> None:
    closing = "Steel always beats plastic on a clamp mount."

    assert copied_example(script(closing), PROMPT, WATCH, [CTA]) == closing


@pytest.mark.req("REQ-CNT-155")
def test_an_example_the_listing_backs_is_not_borrowed() -> None:
    mount = "Steel clamp-style mount for any desk, beats plastic arms."
    closing = "Steel beats plastic for any clamp-style mount."

    assert copied_example(script(closing), PROMPT, mount, [CTA]) is None


@pytest.mark.req("REQ-CNT-155")
def test_a_listing_backs_an_example_in_the_plural() -> None:
    mounts = "Steel clamp-style mounts beat plastic arms on any desk."
    closing = "Steel beats plastic for any clamp-style mount."

    assert copied_example(script(closing), PROMPT, mounts, [CTA]) is None


@pytest.mark.req("REQ-CNT-155")
@pytest.mark.parametrize(
    "closing",
    [
        # Two of an example's words are a coincidence, not a copy.
        "Steel feels better than plastic on the wrist.",
        "Seven days beats charging every night.",
    ],
)
def test_an_ordinary_closing_line_passes(closing: str) -> None:
    assert copied_example(script(closing), PROMPT, WATCH, [CTA]) is None


@pytest.mark.req("REQ-CNT-155")
def test_the_quoted_call_to_action_is_never_a_copy() -> None:
    assert copied_example(script("It is light."), PROMPT, "", [CTA]) is None


@pytest.mark.req("REQ-CNT-155")
def test_a_call_to_action_of_two_sentences_is_never_a_copy() -> None:
    from src.ai.script_generator import render_cta_rule

    cta = "Tap the link in bio. Follow for more finds like this."
    prompt = "Rules.\n" + render_cta_rule(cta)
    body = script("It is light.").replace(CTA, cta)

    assert copied_example(body, prompt, "", [cta]) is None


@pytest.mark.req("REQ-CNT-155")
def test_a_two_option_question_is_never_a_copy() -> None:
    prompt = 'Close with a fork, for example "Team magnetic or team plug-in?"'
    closing = "Team magnetic or team plug-in?"

    assert copied_example(script(closing), prompt, "magnetic mount", [CTA]) is None


CLAIM = "Steel beats plastic for any clamp-style mount."


def _product():
    from src.scraper.amazon.models import ProductData
    from src.scraper.base.models import Platform

    return ProductData(
        title="Smartwatch",
        price="$40",
        url="https://example.com",
        platform=Platform.AMAZON,
        description=WATCH,
    )


def _product_script(closing: str) -> str:
    cta = config.llm_settings.script_templates.cta_options_for(False)[0]
    filler = "This watch tracks your sleep every night of the week. " * 8
    return f"{filler}{closing} {cta}".replace("  ", " ")


async def _generate(replies: list[str], on: bool = True):
    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_validation.reject_copied_examples = on
    # The template whose closing-claim rule quotes the example.
    settings.script_templates.template_pool = ["before_after"]
    call = AsyncMock(side_effect=replies * 10)
    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        out, _, _ = await script_generator.generate_script(
            _product(),
            settings,
            {settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
        )
    return out, call.await_count, call.await_args_list[0].args[0]


@pytest.mark.req("REQ-CNT-155")
@pytest.mark.asyncio
async def test_a_borrowed_line_is_retried() -> None:
    borrowed = _product_script(CLAIM)
    clean = _product_script("A week per charge beats charging nightly.")

    out, calls, prompt = await _generate([borrowed, clean])

    assert CLAIM in prompt  # the fixture relies on the template quoting it
    assert out == clean and calls == 2


@pytest.mark.req("REQ-CNT-155")
@pytest.mark.asyncio
async def test_when_every_attempt_borrows_the_line_is_dropped() -> None:
    out, calls, _ = await _generate([_product_script(CLAIM)])

    assert out == _product_script("")
    assert calls > 1


@pytest.mark.req("REQ-CNT-155")
def test_a_short_quote_earlier_on_the_line_hides_no_example() -> None:
    prompt = (
        'Describe the "before" state first. Example: "Before this $18 magnetic '
        'phone mount, my GPS lived in a cup holder."'
    )
    hook = "Before this $18 magnetic phone mount, my GPS lived in a cup holder."

    assert copied_example(script(hook), prompt, WATCH, [CTA]) == hook


@pytest.mark.req("REQ-CNT-155")
@pytest.mark.asyncio
async def test_no_last_resort_ships_the_borrowed_line() -> None:
    # Every attempt borrows and ends on a paraphrased CTA, so only the
    # fallback that appends the configured CTA is left.
    cta = config.llm_settings.script_templates.cta_options_for(False)[0]
    paraphrase = _product_script(CLAIM).replace(cta, "Grab one if you like it.")

    out, _, _ = await _generate([paraphrase])

    assert out is not None and CLAIM not in out
    assert out.endswith(cta)


@pytest.mark.req("REQ-CNT-155")
@pytest.mark.asyncio
async def test_off_the_first_script_ships_as_before() -> None:
    borrowed = _product_script(CLAIM)

    out, calls, _ = await _generate([borrowed], on=False)

    assert out == borrowed and calls == 1


@pytest.mark.req("REQ-CNT-155")
def test_the_question_a_borrowed_line_answers_goes_with_it() -> None:
    from src.ai.script_generator import drop_borrowed

    line = "I charged it Sunday, forgot about it until Friday."
    text = f"It tracks sleep. And the battery? {line} Is it worth it? {CTA}"

    assert drop_borrowed(text, line) == f"It tracks sleep. Is it worth it? {CTA}"
