"""A commentary script keeps to its detail's figures and closes on a question (#810)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.ai.script_figures import figures, has_closing_question, unbacked_figure
from src.video.config import config

DETAIL = (
    "A survey counted 1,016 answers last month, down from 207 thousand in "
    "twenty fourteen. Source: the site's public data dump."
)


@pytest.mark.req("REQ-CNT-177")
def test_figures_read_digits_words_scales_and_spoken_years() -> None:
    assert figures("1,016 answers and 22.3 percent") == {1016.0, 22.3}
    assert figures("two hundred and seven thousand") == {207000.0}
    assert figures("62 billion kilos") == {62e9}
    assert figures("In twenty twenty-four, nineteen eighty-four") == {2024.0, 1984.0}


@pytest.mark.req("REQ-CNT-177")
def test_a_lone_one_is_a_pronoun_not_a_figure() -> None:
    assert figures("The one next to it isn't funny.") == set()


@pytest.mark.req("REQ-CNT-177")
def test_a_changed_or_invented_figure_is_named() -> None:
    sentences = ["Last month: about a thousand answers.", "Down from 207 thousand."]

    assert unbacked_figure(sentences, DETAIL) == sentences[0]
    assert unbacked_figure(["Last month: 1,016 answers."], DETAIL) is None
    assert unbacked_figure(["Last month: one thousand and sixteen."], DETAIL) is None


@pytest.mark.req("REQ-CNT-177")
def test_the_closing_question_is_one_of_the_last_two_sentences() -> None:
    assert has_closing_question(["A claim.", "Who answers now?", "Tell me."])
    assert not has_closing_question(["Who answers now?", "A claim.", "Tell me."])


def _topic():
    from src.scraper.amazon.models import ProductData
    from src.scraper.base.models import Platform

    return ProductData(
        title="The forum went quiet",
        price="",
        url="",
        platform=Platform.AMAZON,
        description=DETAIL,
        topic="The forum went quiet",
        topic_kind="commentary",
    )


def _take(figure: str, closing: str) -> str:
    cta = config.llm_settings.script_templates.cta_options_for(True)[0]
    body = "People stopped asking strangers for help and asked a chatbot. " * 6
    return f"Last month it got {figure} answers. {body}{closing} {cta}"


QUESTION = "Who answers the next question?"


async def _generate(replies: list[str], on: bool = True, kind="commentary"):
    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_validation.check_commentary_facts = on
    settings.script_validation.lint.enabled = False
    record = _topic()
    record.topic_kind = kind
    call = AsyncMock(side_effect=replies * 10)
    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        out, _, _ = await script_generator.generate_script(
            record,
            settings,
            {settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
        )
    return out, call.await_count


@pytest.mark.req("REQ-CNT-177")
@pytest.mark.asyncio
async def test_a_changed_figure_is_retried() -> None:
    rounded, exact = _take("about a thousand", QUESTION), _take("1,016", QUESTION)

    out, calls = await _generate([rounded, exact])

    assert out == exact and calls == 2


@pytest.mark.req("REQ-CNT-177")
@pytest.mark.asyncio
async def test_when_every_attempt_changes_it_the_line_is_dropped() -> None:
    out, calls = await _generate([_take("about a thousand", QUESTION)])

    assert out is not None and "thousand" not in out and calls > 1
    assert out.startswith("People stopped asking")


@pytest.mark.req("REQ-CNT-177")
@pytest.mark.asyncio
async def test_a_script_without_a_closing_question_is_retried() -> None:
    flat, asked = _take("1,016", "That is the trade."), _take("1,016", QUESTION)

    out, calls = await _generate([flat, asked])

    assert out == asked and calls == 2


@pytest.mark.req("REQ-CNT-177")
@pytest.mark.asyncio
async def test_off_or_for_a_tutorial_the_first_script_ships() -> None:
    rounded = _take("about a thousand", "That is the trade.")

    assert await _generate([rounded], on=False) == (rounded, 1)
    assert await _generate([rounded], kind=None) == (rounded, 1)
