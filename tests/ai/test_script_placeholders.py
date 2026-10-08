"""Placeholders and backticks never reach the voice (#702)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.utils.script_sanitizer import sanitize_script
from src.video.config import config


@pytest.mark.req("REQ-CNT-052")
def test_backticks_round_a_search_operator_are_removed() -> None:
    out = sanitize_script("Type `from:` followed by the sender's address.")

    assert out == "Type from: followed by the sender's address."


@pytest.mark.req("REQ-CNT-052")
def test_a_bracketed_placeholder_keeps_its_words() -> None:
    out = sanitize_script(
        'A message says "All [number] conversations on this page are selected".'
    )

    assert out == 'A message says "All number conversations on this page are selected".'


@pytest.mark.req("REQ-CNT-052")
def test_a_markdown_link_keeps_its_text_only() -> None:
    out = sanitize_script("Open [Settings](https://example.com/settings) first.")

    assert out == "Open Settings first."


@pytest.mark.req("REQ-CNT-052")
def test_a_bracketed_ui_label_survives_as_words() -> None:
    out = sanitize_script("Tap [Unrecognised Device or Session] and confirm.")

    assert out == "Tap Unrecognised Device or Session and confirm."


def _product():
    from src.scraper.amazon.models import ProductData
    from src.scraper.base.models import Platform

    return ProductData(
        title="Smartwatch",
        price="$40",
        url="https://example.com",
        platform=Platform.AMAZON,
        description="Smartwatch with sleep tracking and a seven-day battery.",
    )


def _script(middle: str) -> str:
    cta = config.llm_settings.script_templates.cta_options_for(False)[0]
    filler = "This watch tracks your sleep every night of the week. " * 8
    return f"{filler}{middle} {cta}".replace("  ", " ")


async def _generate(replies: list[str]):
    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_validation.reject_copied_examples = False
    settings.script_validation.lint.enabled = False
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
    return out, call.await_count


@pytest.mark.req("REQ-CNT-052")
@pytest.mark.asyncio
async def test_a_script_with_a_placeholder_is_retried() -> None:
    clean = _script("It shows how many days are left.")

    out, calls = await _generate([_script("It shows [number] days left."), clean])

    assert out == clean and calls == 2


@pytest.mark.req("REQ-CNT-052")
@pytest.mark.asyncio
async def test_when_every_attempt_has_a_placeholder_one_still_ships() -> None:
    held = _script("It shows [number] days left.")

    out, calls = await _generate([held])

    assert out == held and calls > 1
    assert "[" not in sanitize_script(out)


@pytest.mark.req("REQ-CNT-052")
@pytest.mark.asyncio
async def test_a_script_without_brackets_ships_on_the_first_call() -> None:
    clean = _script("It shows how many days are left.")

    out, calls = await _generate([clean])

    assert out == clean and calls == 1


@pytest.mark.req("REQ-CNT-052")
@pytest.mark.asyncio
async def test_brackets_the_listing_carries_are_not_retried() -> None:
    from src.ai import script_generator

    product = _product()
    product.title = "[Apple MFi Certified] Smartwatch"
    quoted = _script("It is [Apple MFi Certified] and tracks sleep.")
    settings = config.llm_settings.model_copy(deep=True)
    settings.script_validation.reject_copied_examples = False
    settings.script_validation.lint.enabled = False
    call = AsyncMock(side_effect=[quoted] * 10)
    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        out, _, _ = await script_generator.generate_script(
            product, settings, {settings.api_key_env_var: "k"}, AsyncMock(), {}, False
        )

    assert out == quoted and call.await_count == 1


@pytest.mark.req("REQ-CNT-052")
@pytest.mark.asyncio
async def test_brackets_a_step_lists_ui_path_carries_are_not_retried() -> None:
    from src.ai import script_generator
    from src.ai.step_list import Step, StepList

    steps = [Step("Open the app", "Settings > [App Name]", "the app page", "x")]
    quoted = _script("Open Settings, then [App Name], and clear its cache.")
    settings = config.llm_settings.model_copy(deep=True)
    settings.script_validation.reject_copied_examples = False
    settings.script_validation.lint.enabled = False
    call = AsyncMock(side_effect=[quoted] * 10)
    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        out, _, _ = await script_generator.generate_script(
            _product(),
            settings,
            {settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
            step_list=StepList("Settings", "Android", False, steps, topic_failures=[]),
        )

    assert out == quoted and call.await_count == 1
