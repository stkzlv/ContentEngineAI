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
def test_a_lone_count_up_to_ten_is_a_way_of_talking() -> None:
    assert figures("The one next to it isn't funny.") == set()
    assert figures("Two things bother me about that.") == set()
    assert figures("Eleven reasons.") == {11.0}


@pytest.mark.req("REQ-CNT-177")
def test_abbreviated_scales_halves_and_ranges_read_as_spoken() -> None:
    assert figures("Revenue hit $62B, 10K users, 1.5M views") == {62e9, 1e4, 1.5e6}
    assert figures("one and a half million") == {1.5e6}
    assert figures("half a million, 1 and a half million") == {5e5, 1.5e6}
    assert figures("a 401k plan") == figures("a 401(k) plan")
    assert figures("sixty-two and a half percent") == {62.5}
    assert figures("between twenty and thirty years") == {20.0, 30.0}
    assert unbacked_figure(["It hit sixty-two billion dollars."], "$62B") is None


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


async def _generate(
    replies: list[str],
    on: bool = True,
    kind="commentary",
    signoff: str = "",
    ctas: list[str] | None = None,
    lint: bool = False,
):
    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_validation.check_commentary_facts = on
    settings.script_validation.lint.enabled = lint
    settings.script_templates.signature.topic.pinned_signoff = signoff
    if ctas:
        settings.script_templates.cta_options_topic = ctas
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


@pytest.mark.req("REQ-CNT-177")
@pytest.mark.asyncio
async def test_the_sign_off_and_a_two_sentence_cta_are_not_the_ending() -> None:
    cta = "Follow along. More takes soon."
    take = _take("1,016", QUESTION)
    take = take[: take.index(QUESTION) + len(QUESTION)] + f" Prove me wrong. {cta}"

    out, calls = await _generate([take], signoff="Prove me wrong.", ctas=[cta])

    assert out == take and calls == 1


@pytest.mark.req("REQ-CNT-177")
@pytest.mark.asyncio
async def test_the_cta_append_last_resort_prefers_a_draft_that_asked() -> None:
    cta = config.llm_settings.script_templates.cta_options_for(True)[0]
    flat = _take("1,016", "That is the trade.").replace(f" {cta}", "")
    asked = _take("1,016", QUESTION).replace(f" {cta}", "")

    out, _ = await _generate([flat, asked])

    assert out is not None and QUESTION in out and out.endswith(cta)


@pytest.mark.req("REQ-CNT-177")
@pytest.mark.asyncio
async def test_a_draft_without_its_question_still_has_to_pass_the_lint() -> None:
    tell = "It is a game changer."
    asked = _take("1,016", f"{tell} {QUESTION}")
    flat = _take("1,016", f"That is the trade. {tell}")

    out, _ = await _generate([asked, flat], lint=True)

    assert out == asked
