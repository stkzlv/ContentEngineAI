"""A virtual narrator claims no physical act; commentary asks for two devices.

REQ-CNT-173 (design 0033): a narrator identity marked `virtual` gets a prompt
rule, and the script step refuses a first-person claim of a physical act,
whatever the lint setting. REQ-CNT-174 (design 0034): a switch adds a
comparison rule and a contrast rule to a commentary prompt only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.ai.llm_settings import VIRTUAL_NARRATOR_RULE
from src.ai.physical_claims import DEFAULT_PATTERNS, compile_patterns, physical_claim
from src.ai.script_generator import COMMENTARY_DEVICES_RULE, render_ending_rules
from src.video.config import config

PATTERNS = compile_patterns(DEFAULT_PATTERNS)


@pytest.mark.req("REQ-CNT-173")
@pytest.mark.parametrize(
    "sentence",
    [
        "I bought one last week.",
        "I've tested it for a month.",
        "We ordered two.",
        "I just unboxed it.",
        "I tried it on and it fit.",
        "I picked it up at the store.",
        "I took it to the gym.",
        "I spent $40 on mine.",
        "I\u2019ve worn them every day.",
        "I've been using it for a week.",
        "I use mine every day.",
        "I've had mine for a month.",
        "I got one last week.",
        "I went to the store with it.",
        "I bought this.",
        "I tested this.",
        "I've been using this for a month.",
        "We tested it on the beach.",
        "I'm using it every day.",
        "I own one.",
        "I've owned one since May.",
        "I got one last week.",
    ],
)
def test_a_physical_claim_is_named(sentence: str) -> None:
    assert physical_claim(["A fact first.", sentence], PATTERNS) == sentence


@pytest.mark.req("REQ-CNT-173")
@pytest.mark.parametrize(
    "sentence",
    [
        "I think it's overpriced.",
        "I read the reviews.",
        "I checked the numbers.",
        "I compared three of them.",
        "I tried to find a cheaper one.",
        "People who bought it say it lasts.",
        "Last year we spent $40 billion on gadgets.",
        "In 2017 we bought 200 million fidget spinners.",
        "I tested this claim against the data.",
        "I used these figures from the report.",
        "I spent weeks reading reviews.",
        "I held off on buying one for years.",
        "I held that view until I saw the data.",
        "We ordered the list by price.",
        "If I bought this, I'd return it.",
        "I used to think so.",
        "I'm worn out by these launches.",
        "I got it, the battery is the weak point.",
        "I've got one question about the price.",
        "We got two reasons to skip it.",
        "I used one source for this.",
        "I tried one thing: reading the manual.",
        "We went with it because the reviews were better.",
        "I spent an hour reading up on it.",
        "I used it as an example of the trend.",
    ],
)
def test_opinion_research_and_third_person_pass(sentence: str) -> None:
    assert physical_claim([sentence], PATTERNS) is None


@pytest.mark.req("REQ-CNT-173")
def test_a_virtual_narrator_gets_the_rule_and_others_do_not() -> None:
    templates = config.llm_settings.script_templates.model_copy(deep=True)
    plain = {kind: templates.narrator_for(kind) for kind in (False, True)}

    templates.virtual_narrator = True

    for is_topic in (False, True):
        virtual = templates.narrator_for(is_topic)
        assert VIRTUAL_NARRATOR_RULE in virtual
        assert VIRTUAL_NARRATOR_RULE not in plain[is_topic]
        assert virtual.startswith(plain[is_topic])


@pytest.mark.req("REQ-CNT-173")
def test_the_identity_flag_reaches_the_records_config() -> None:
    from src.video.producer.orchestration import config_for_record

    cfg = config.model_copy(deep=True)
    cfg.narrator_identity.product.virtual = True
    record = _product()

    assert config_for_record(cfg, record).llm_settings.script_templates.virtual_narrator
    assert not config_for_record(
        config, record
    ).llm_settings.script_templates.virtual_narrator


@pytest.mark.req("REQ-CNT-173")
def test_the_flag_is_not_a_pinned_dimension() -> None:
    identity = config.narrator_identity.product.model_copy(update={"virtual": True})

    assert "virtual" not in identity.pinned_dimensions()


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


def _script(claim: str) -> str:
    cta = config.llm_settings.script_templates.cta_options_for(False)[0]
    body = "The stand tilts back far enough for a video call at a desk. " * 6
    return f"{claim} {body}{cta}"


async def _generate(replies: list[str], virtual: bool = True):
    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_templates.virtual_narrator = virtual
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


@pytest.mark.req("REQ-CNT-173")
@pytest.mark.asyncio
async def test_a_claim_is_retried() -> None:
    claiming, clean = _script("I bought one last week."), _script("Here's the stand.")

    out, call = await _generate([claiming, clean])

    assert out == clean and call.await_count == 2


@pytest.mark.req("REQ-CNT-173")
@pytest.mark.asyncio
async def test_when_every_attempt_claims_the_sentence_is_dropped() -> None:
    out, call = await _generate([_script("I bought one last week.")])

    assert out is not None and "bought" not in out and call.await_count > 1


@pytest.mark.req("REQ-CNT-173")
@pytest.mark.asyncio
async def test_without_the_flag_a_claim_ships() -> None:
    claiming = _script("I bought one last week.")

    out, call = await _generate([claiming], virtual=False)

    assert out == claiming and call.await_count == 1


@pytest.mark.req("REQ-CNT-174")
def test_the_devices_rule_is_added_only_when_asked() -> None:
    base = render_ending_rules("Follow for more.", True, 0)

    assert (
        render_ending_rules("Follow for more.", True, 0, commentary_devices=False)
        == base
    )
    with_devices = render_ending_rules(
        "Follow for more.", True, 0, commentary_devices=True
    )
    assert with_devices == f"{base}\n{COMMENTARY_DEVICES_RULE}"
    assert '"' not in COMMENTARY_DEVICES_RULE


async def _prompt(kind: str | None, on: bool) -> str:
    from src.ai import script_generator
    from tests.ai.test_commentary_checks import QUESTION, _take, _topic

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_templates.commentary_devices.enabled = on
    if config.llm_settings.script_templates.commentary_templates:
        settings.script_templates.topic_templates = list(
            settings.script_templates.commentary_templates
        )
    record = _topic()
    record.topic_kind = kind
    call = AsyncMock(return_value=_take("1,016", QUESTION))
    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        await script_generator.generate_script(
            record, settings, {settings.api_key_env_var: "k"}, AsyncMock(), {}, False
        )
    return str(call.await_args_list[0])


@pytest.mark.req("REQ-CNT-174")
@pytest.mark.asyncio
async def test_a_commentary_prompt_carries_the_devices_when_on() -> None:
    assert "one comparison built only from numbers" in await _prompt("commentary", True)
    assert "one comparison built only from numbers" not in await _prompt(
        "commentary", False
    )


@pytest.mark.req("REQ-CNT-174")
@pytest.mark.asyncio
async def test_another_topic_never_carries_them() -> None:
    assert "one comparison built only from numbers" not in await _prompt(None, True)


@pytest.mark.req("REQ-CNT-173")
def test_a_revision_cannot_bring_a_claim_back() -> None:
    from src.ai.script_fact_check import FactCheckClaim, accept_revision

    original = _script("The stand is aluminium.")
    revised = original.replace("The stand is aluminium.", "I tested it at my desk.")
    flagged = [
        FactCheckClaim(
            claim="The stand is aluminium.", fix="It is steel.", reason="wrong"
        )
    ]
    cta = config.llm_settings.script_templates.cta_options_for(False)
    kwargs = {
        "cta_options": cta,
        "min_chars": 50,
        "min_words": 10,
        "max_length_drift": 1.0,
    }

    refused, reason = accept_revision(
        original, revised, flagged, claim_patterns=PATTERNS, **kwargs
    )
    allowed, _ = accept_revision(original, revised, flagged, **kwargs)

    assert refused is None and "physical act" in reason
    assert allowed is not None, "only the claim guard refuses this revision"
