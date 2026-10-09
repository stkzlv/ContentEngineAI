"""A draft that fails only a soft check is kept, not sent to a fallback model.

REQ-CNT-164.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.video.config import config

LONG = (
    "This watch tracks your sleep and your steps and your heart rate every "
    "single night of the week without fail. "
)


def _settings(lint: bool = True):
    settings = config.llm_settings.model_copy(deep=True)
    settings.provider = "gemini"
    settings.models = ["primary-m"]
    settings.script_validation.lint.enabled = lint
    settings.script_validation.reject_copied_examples = False
    settings.script_templates.signature.enabled = False
    fallback = settings.model_copy(deep=True)
    fallback.models = ["fallback-m"]
    fallback.fallback_provider = None
    fallback.api_key_env_var = "FALLBACK_KEY"
    settings.fallback_provider = fallback
    return settings


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


def _script(body: str) -> str:
    cta = config.llm_settings.script_templates.cta_options_for(False)[0]
    filler = "It tracks sleep. The battery lasts a week. " * 6
    return f"{body}{filler}{cta}"


async def _generate(settings, side_effect):
    from src.ai import script_generator

    written_by: dict[str, str] = {}
    call = AsyncMock(side_effect=side_effect)
    with (
        patch.object(script_generator, "_call_llm_api_with_retry", call),
        patch.object(script_generator, "configured_live", lambda s: list(s.models)),
        patch.object(
            script_generator, "fetch_and_select_model", AsyncMock(return_value=[])
        ),
    ):
        out, _, _ = await script_generator.generate_script(
            _product(),
            settings,
            {settings.api_key_env_var: "k", "FALLBACK_KEY": "k2"},
            AsyncMock(),
            {},
            False,
            written_by=written_by,
        )
    return out, call, written_by


@pytest.mark.req("REQ-CNT-164")
@pytest.mark.asyncio
async def test_a_lint_near_miss_is_kept_without_a_fallback_call() -> None:
    draft = _script(LONG)

    out, call, written_by = await _generate(_settings(), [draft, draft, _script("")])

    assert [c.args[1] for c in call.await_args_list] == ["primary-m", "primary-m"]
    assert out == draft and written_by == {"model": "primary-m"}


@pytest.mark.req("REQ-CNT-164")
@pytest.mark.asyncio
async def test_a_placeholder_near_miss_is_kept_without_a_fallback_call() -> None:
    draft = _script("It shows [n] days. ")

    out, call, _ = await _generate(_settings(lint=False), [draft, draft, _script("")])

    assert call.await_count == 2 and out == draft


@pytest.mark.req("REQ-CNT-164")
@pytest.mark.asyncio
async def test_a_hard_failure_still_reaches_the_fallback() -> None:
    from src.ai.script_generator import ScriptGenerationError

    failure = ScriptGenerationError("400")
    good = _script("")

    out, call, written_by = await _generate(_settings(), [failure, failure, good])

    assert call.await_args_list[-1].args[1] == "fallback-m"
    assert out == good and written_by == {"model": "fallback-m"}


@pytest.mark.req("REQ-CNT-164")
@pytest.mark.asyncio
async def test_an_openrouter_primary_discovers_no_free_model_after_a_near_miss() -> (
    None
):
    from src.ai import script_generator

    settings = _settings()
    settings.provider = "openrouter"
    settings.fallback_discover_any_free = True
    settings.fallback_provider = None
    draft = _script(LONG)
    discover = AsyncMock(return_value=["free-m"])
    call = AsyncMock(side_effect=[draft, draft, _script("")])
    with (
        patch.object(script_generator, "_call_llm_api_with_retry", call),
        patch.object(script_generator, "configured_live", lambda s: list(s.models)),
        patch.object(
            script_generator, "fetch_and_select_model", AsyncMock(return_value=[])
        ),
        patch.object(script_generator, "discover_any_free_model", discover),
    ):
        out, _, _ = await script_generator.generate_script(
            _product(),
            settings,
            {settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
        )

    assert discover.await_count == 0 and call.await_count == 2 and out == draft
