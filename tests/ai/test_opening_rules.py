"""A narrator identity's opening rules (REQ-CNT-186, REQ-CNT-187, REQ-VID-166)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from src.ai.llm_settings import CLAIM_FIRST_RULE
from src.video.config import config
from src.video.config.identity_models import NarratorIdentity

PATTERNS = [r"^(hi|hey|hello)\b", r"\bmy name is\b"]


def _product():
    from src.scraper.amazon.models import ProductData
    from src.scraper.base.models import Platform

    return ProductData(
        title="A Phone Stand",
        price="$20",
        url="https://www.amazon.com/dp/B0TEST0001",
        platform=Platform.AMAZON,
        asin="B0TEST0001",
        description="An adjustable aluminium phone stand for desks.",
    )


def _script(opening: str) -> str:
    cta = config.llm_settings.script_templates.cta_options_for(False)[0]
    body = "The stand tilts back far enough for a video call at a desk. " * 6
    return f"{opening} {body}{cta}"


async def _generate(replies: list[str], patterns: list[str]):
    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_templates.self_intro_patterns = patterns
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
    return out, call


@pytest.mark.req("REQ-CNT-186")
def test_claim_first_adds_the_rule_only_when_set() -> None:
    templates = config.llm_settings.script_templates.model_copy(deep=True)
    plain = {kind: templates.narrator_for(kind) for kind in (False, True)}
    templates.claim_first_opening = True
    for is_topic in (False, True):
        ruled = templates.narrator_for(is_topic)
        assert CLAIM_FIRST_RULE in ruled
        assert CLAIM_FIRST_RULE not in plain[is_topic]


@pytest.mark.req("REQ-CNT-186")
@pytest.mark.req("REQ-CNT-187")
@pytest.mark.req("REQ-VID-166")
def test_the_identity_reaches_the_records_config() -> None:
    from src.video.producer.orchestration import config_for_record

    cfg = config.model_copy(deep=True)
    cfg.narrator_identity.product = NarratorIdentity(
        opening="claim_first", self_intro_patterns=PATTERNS, title_card=False
    )
    record = config_for_record(cfg, _product())
    plain = config_for_record(config, _product())

    assert record.llm_settings.script_templates.claim_first_opening
    assert record.llm_settings.script_templates.self_intro_patterns == PATTERNS
    assert record.video_settings.hook_overlay.enabled is False
    assert not plain.llm_settings.script_templates.claim_first_opening
    assert plain.llm_settings.script_templates.self_intro_patterns == []
    assert (
        plain.video_settings.hook_overlay.enabled
        == config.video_settings.hook_overlay.enabled
    )
    assert config.video_settings.hook_overlay.enabled is True


@pytest.mark.req("REQ-VID-166")
def test_title_card_unset_leaves_the_hook_overlay() -> None:
    from src.video.producer.orchestration import config_for_record

    cfg = config.model_copy(deep=True)
    cfg.narrator_identity.product = NarratorIdentity(title_card=None)
    assert config_for_record(cfg, _product()).video_settings.hook_overlay.enabled


@pytest.mark.req("REQ-CNT-187")
def test_a_bad_pattern_fails_validation() -> None:
    with pytest.raises(ValidationError, match="not a regex"):
        NarratorIdentity(self_intro_patterns=["(unclosed"])


@pytest.mark.req("REQ-CNT-187")
@pytest.mark.asyncio
async def test_a_self_introduction_is_retried() -> None:
    greeting, clean = _script("Hey, my name is Sam."), _script("This stand tilts.")
    out, call = await _generate([greeting, clean], PATTERNS)
    assert out == clean and call.await_count == 2


@pytest.mark.req("REQ-CNT-187")
@pytest.mark.asyncio
async def test_when_every_attempt_introduces_one_ships_as_a_near_miss() -> None:
    greeting = _script("Hey, my name is Sam.")
    out, call = await _generate([greeting], PATTERNS)
    assert out == greeting and call.await_count > 1


@pytest.mark.req("REQ-CNT-187")
@pytest.mark.asyncio
async def test_without_patterns_a_greeting_ships_first_time() -> None:
    greeting = _script("Hey, my name is Sam.")
    out, call = await _generate([greeting], [])
    assert out == greeting and call.await_count == 1


@pytest.mark.req("REQ-CNT-187")
def test_the_lint_near_miss_is_kept_before_a_self_introduction() -> None:
    import inspect

    from src.ai import script_generator

    src = inspect.getsource(script_generator.generate_script)
    assert src.index('if "lint" in near_miss') < src.index(
        'if "self_intro" in near_miss'
    )


@pytest.mark.req("REQ-CNT-187")
@pytest.mark.asyncio
async def test_the_cta_fallback_prefers_a_draft_that_does_not_greet() -> None:
    body = "The stand tilts back far enough for a video call at a desk. " * 6
    greeting, clean = f"Hey, my name is Sam. {body}", f"This stand tilts. {body}"

    out, _ = await _generate([greeting, clean], PATTERNS)

    assert out is not None and out.startswith("This stand tilts.")


@pytest.mark.req("REQ-CNT-187")
def test_the_internal_key_validates_its_patterns_too() -> None:
    templates = config.llm_settings.script_templates
    with pytest.raises(ValidationError, match="not a regex"):
        templates.model_validate(
            templates.model_dump() | {"self_intro_patterns": ["(unclosed"]}
        )
