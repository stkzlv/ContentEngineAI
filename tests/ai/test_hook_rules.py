"""Hook and search-phrase prompt rules (design 0007, REQ-CNT-054), held off."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.ai.description_generator import search_phrase_rule
from src.ai.script_generator import HOOK_RULES, render_ending_rules
from src.video.config import config


def _settings(on: bool):
    settings = config.llm_settings.model_copy(deep=True)
    settings.script_templates.hook_rules.enabled = on
    return settings


def test_the_ending_rules_carry_the_hook_rules_only_when_on() -> None:
    off = render_ending_rules("Follow for more.", False, 0)

    assert HOOK_RULES not in off
    assert render_ending_rules("Follow for more.", False, 0, hook_rules=True) == (
        off + "\n" + HOOK_RULES
    )


@pytest.mark.req("REQ-CNT-054")
def test_the_search_phrase_rule_names_the_keyword_or_title() -> None:
    product = SimpleNamespace(keyword="smart watch", title="X")
    topic = SimpleNamespace(topic="t", title="Why your wifi drops", keyword="")

    assert search_phrase_rule(product, _settings(False)) == ""
    assert '"smart watch"' in search_phrase_rule(product, _settings(True))
    # A topic names its title's key words, which a short headline can hold.
    rule = search_phrase_rule(topic, _settings(True))
    assert "Search words: wifi, drops." in rule and "Why" not in rule
    assert search_phrase_rule(SimpleNamespace(keyword=""), _settings(True)) == ""


def _product():
    from src.video.producer.topic_input import TopicSpec, build_topic_product

    return build_topic_product(TopicSpec(title="How to fix it", description="x"))


@pytest.mark.req("REQ-CNT-054")
@pytest.mark.asyncio
async def test_the_script_prompt_carries_the_rules_only_when_on() -> None:
    from src.ai import script_generator

    prompts = {}
    for on in (False, True):
        call = AsyncMock(return_value="")
        with patch.object(script_generator, "_call_llm_api_with_retry", call):
            await script_generator.generate_script(
                _product(),
                _settings(on),
                {config.llm_settings.api_key_env_var: "k"},
                AsyncMock(),
                {},
                False,
            )
        prompts[on] = call.call_args_list[0].args[0] if call.call_args_list else ""

    assert HOOK_RULES not in prompts[False]
    assert HOOK_RULES in prompts[True]


@pytest.mark.req("REQ-CNT-054")
@pytest.mark.asyncio
async def test_generate_with_llm_appends_the_rule_only_for_its_callers(
    tmp_path,
) -> None:
    from src.ai.platform_metadata import utilities

    template = tmp_path / "t.md"
    template.write_text("Write about {FULL_PRODUCT_NAME}.")
    product = SimpleNamespace(
        title="Lamp", description="d", keyword="desk lamp", topic=None
    )
    seen = []

    async def fake_call(prompt, *a, **k):
        seen.append(prompt)
        return "ok"

    with patch.object(utilities, "call_llm_api_with_retry", fake_call):
        for flag in (False, True):
            await utilities.generate_with_llm(
                template,
                product,
                _settings(True),
                "k",
                AsyncMock(),
                lead_with_search_phrase=flag,
            )

    assert '"desk lamp"' not in seen[0]
    assert seen[1].startswith("Write about Lamp.") and '"desk lamp"' in seen[1]


@pytest.mark.req("REQ-CNT-054")
def test_every_caption_title_and_headline_site_asks() -> None:
    """Each site that writes text a viewer reads first passes the rule."""
    from src.ai import description_generator, script_generator
    from src.ai.platform_metadata import instagram, tiktok, youtube

    for module in (tiktok, youtube):
        assert "lead_with_search_phrase=True" in inspect.getsource(module)
    assert "lead_with_search_phrase=True" in inspect.getsource(
        script_generator.generate_hook_headline
    )
    assert "search_phrase_rule(product, settings)" in inspect.getsource(instagram)
    assert "search_phrase_rule(" in inspect.getsource(
        description_generator.generate_description
    )
