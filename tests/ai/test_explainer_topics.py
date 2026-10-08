"""A "Why ..." topic is written as a concept explainer (REQ-VID-161, #657)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.ai.step_list import (
    EXPLAINER_PROMPT_PATH,
    SCRIPT_PROMPT_PATH,
    Step,
    StepList,
    is_explainer_title,
    render_steps,
    script_prompt_path,
    too_short,
    word_range,
)
from src.video.config import config


def _sl(n: int, explainer: bool) -> StepList:
    steps = [
        Step(f"Check {i}", f"Settings > {i}", f"see {i}", "https://x/")
        for i in range(n)
    ]
    return StepList(
        "Settings", "iOS 18", False, steps, topic_failures=[], explainer=explainer
    )


@pytest.mark.req("REQ-VID-161")
@pytest.mark.parametrize(
    ("title", "explainer"),
    [
        ("Why wifi drops at night", True),
        ("  why your laptop fan runs when idle", True),
        ("How to clear app cache", False),
        ("Whyte keyboard review", False),
        (None, False),
    ],
)
def test_a_why_title_is_an_explainer(title: str | None, explainer: bool) -> None:
    assert is_explainer_title(title) is explainer


@pytest.mark.req("REQ-VID-161")
@pytest.mark.parametrize("count", [1, 2, 4])
def test_an_explainer_takes_its_own_band_whatever_the_step_count(count: int) -> None:
    assert word_range(count, explainer=True) == (100, 160)
    assert render_steps(_sl(count, True))["WORD_RANGE"] == "100-160"
    assert render_steps(_sl(count, False))["WORD_RANGE"] != "100-160"


@pytest.mark.req("REQ-VID-161")
def test_short_is_judged_against_the_explainer_band() -> None:
    # 70 words clears a one-step procedure's floor and not an explainer's.
    script = "word " * 70
    assert not too_short(script, 1)
    assert too_short(script, 1, explainer=True)
    assert not too_short("word " * 85, 1, explainer=True)


@pytest.mark.req("REQ-VID-161")
def test_an_explainer_is_written_from_its_own_prompt() -> None:
    assert script_prompt_path(_sl(2, True)) == EXPLAINER_PROMPT_PATH
    assert script_prompt_path(_sl(2, False)) == SCRIPT_PROMPT_PATH
    text = EXPLAINER_PROMPT_PATH.read_text(encoding="utf-8")
    for marker in (
        "<<STEP_LIST>>",
        "<<PLATFORM>>",
        "<<START_SCREEN>>",
        "<<MISTAKE_RULE>>",
        "<<WORD_RANGE>>",
        "{CTA_RULE}",
    ):
        assert marker in text
    assert "cause" in text


def _product(title: str):
    from src.scraper.amazon.models import ProductData
    from src.scraper.base.models import Platform

    topic = ProductData(
        title=title, price="", url="https://e.com", platform=Platform.AMAZON
    )
    topic.topic = title
    return topic


@pytest.mark.req("REQ-VID-161")
@pytest.mark.asyncio
async def test_the_script_call_sends_the_explainer_prompt_and_band() -> None:
    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_validation.lint.enabled = False
    call = AsyncMock(return_value="")
    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        _, template, _ = await script_generator.generate_script(
            _product("Why wifi drops at night"),
            settings,
            {settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
            product_id="t1",
            step_list=_sl(2, True),
        )

    prompt = call.await_args_list[0].args[0]
    assert "Explain the usual cause" in prompt
    assert "**Length: 100-160 words.**" in prompt


@pytest.mark.req("REQ-VID-161")
@pytest.mark.asyncio
async def test_the_step_marks_a_why_topic_as_an_explainer(
    tmp_path: Path, monkeypatch
) -> None:
    from src.ai import step_list as module
    from src.video.producer import steps

    cfg = config.model_copy(deep=True)
    cfg.llm_settings.topic_scripts.step_list.enabled = True
    monkeypatch.setattr(
        module, "build_step_list", AsyncMock(return_value=_sl(2, False))
    )
    ctx = SimpleNamespace(
        config=cfg,
        product=SimpleNamespace(
            topic="Why wifi drops at night",
            title="Why wifi drops at night",
            description="d",
        ),
        secrets={cfg.llm_settings.api_key_env_var: "k"},
        run_paths={"script_file": tmp_path / "text" / "script.txt"},
        state={},
    )

    step_list = await steps._topic_step_list(ctx)

    assert step_list.explainer is True
    record = json.loads((tmp_path / "text" / "step_list.json").read_text())
    assert record["explainer"] is True


@pytest.mark.req("REQ-VID-161")
@pytest.mark.asyncio
async def test_a_short_explainer_draft_is_retried() -> None:
    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_validation.lint.enabled = False
    settings.script_validation.reject_copied_examples = False
    cta = settings.script_templates.cta_options_for(True)[0]
    # About 70 words: past a two-step procedure's floor, under an explainer's.
    short = "Wifi drops at night because the router changes channel. " * 7 + cta
    full = "Wifi drops at night because the router changes channel. " * 11 + cta
    call = AsyncMock(side_effect=[short, full] * 5)
    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        script, _, _ = await script_generator.generate_script(
            _product("Why wifi drops at night"),
            settings,
            {settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
            product_id="t1",
            step_list=_sl(2, True),
        )

    assert script == full and call.await_count == 2


@pytest.mark.req("REQ-VID-161")
def test_an_explainers_sign_off_follows_its_closing_test_not_a_recap() -> None:
    from src.ai.script_generator import SignatureChoice, render_ending_rules

    choice = SignatureChoice(signoff="That's the real picture.")
    cta = config.llm_settings.script_templates.cta_options_for(True)[0]

    procedure = render_ending_rules(cta, True, 0, choice, tutorial=True)
    explainer = render_ending_rules(cta, True, 0, choice, tutorial=True, recap=False)

    assert "recap of the whole path" in procedure
    assert "recap" not in explainer and "the closing beat" in explainer
    # Still a tutorial for the lint: no duration word cap.
    assert explainer.count("Keep the whole script") == procedure.count(
        "Keep the whole script"
    )


@pytest.mark.req("REQ-VID-161")
def test_the_script_call_passes_the_explainer_to_the_sign_off() -> None:
    import inspect

    from src.ai import script_generator

    source = inspect.getsource(script_generator.generate_script)
    assert "recap=step_list is not None and not step_list.explainer" in source
