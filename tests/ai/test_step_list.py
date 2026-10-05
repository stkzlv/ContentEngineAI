"""Topic scripts written from a sourced step list (design 0017)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.ai.step_list import (
    Step,
    StepList,
    drop_reason,
    fill,
    parse_step_list,
    render_steps,
    too_short,
    word_range,
)
from src.video.config import config

ANSWER = {
    "start_screen": "the Settings app",
    "platform": "iOS 18",
    "forks": False,
    "steps": [
        {
            "action": "Tap General",
            "ui_path": "Settings > General",
            "expected": "General opens",
            "source": "https://support.apple.com/102551",
        },
        {
            "action": "Tap Background App Refresh",
            "ui_path": "General > Background App Refresh",
            "expected": "the refresh screen opens",
            "source": "",
        },
        {
            "action": "Choose Off",
            "ui_path": "Background App Refresh > Off",
            "expected": "refresh is off",
            "source": "support.apple.com/102551",
        },
        {
            "action": "Check the list",
            "ui_path": "",
            "expected": "every app is grey",
            "source": "http://example.com/x",
        },
    ],
    "common_mistake": {"step": 3, "mistake": "turning off one app only"},
}


def _sl(n: int, *, forks: bool = False) -> StepList:
    steps = [Step(f"Do {i}", f"A > {i}", f"see {i}", "https://x/") for i in range(n)]
    return StepList("Settings", "iOS", forks, steps)


@pytest.mark.req("REQ-VID-122")
def test_a_step_without_a_web_source_is_refused() -> None:
    parsed = parse_step_list("```json\n" + json.dumps(ANSWER) + "\n```")

    assert parsed is not None
    assert [s.action for s in parsed.steps] == ["Tap General", "Check the list"]
    assert [s.action for s in parsed.refused] == [
        "Tap Background App Refresh",
        "Choose Off",
    ]
    assert parsed.mistake_step == 3
    assert parsed.start_screen == "the Settings app"


@pytest.mark.parametrize(
    "answer", [None, "", "no json here", '{"steps": "none"}', "[1, 2]", "{bad"]
)
def test_an_unreadable_answer_is_no_list(answer) -> None:
    assert parse_step_list(answer) is None


@pytest.mark.req("REQ-VID-121", "REQ-VID-122")
def test_drop_reasons() -> None:
    assert drop_reason(None, 6) == "no step list came back"
    assert drop_reason(_sl(0), 6) == "no step could be sourced"
    assert "series" in (drop_reason(_sl(3, forks=True), 6) or "")
    assert "series" in (drop_reason(_sl(7), 6) or "")
    assert drop_reason(_sl(6), 6) is None


@pytest.mark.req("REQ-VID-121")
def test_length_follows_the_step_count() -> None:
    assert word_range(1) == word_range(2) == (50, 75)
    assert word_range(3) == word_range(6) == (100, 185)
    assert too_short("word " * 70, 3)
    assert not too_short("word " * 85, 3)
    assert not too_short("word " * 41, 2)


@pytest.mark.req("REQ-CNT-146", "REQ-CNT-147")
def test_the_prompt_names_the_start_and_the_mistake_at_its_step() -> None:
    sl = _sl(3)
    sl.mistake, sl.mistake_step = "skipping the restart", 2

    fills = render_steps(sl)

    assert fills["START_SCREEN"] == "Settings"
    assert fills["MISTAKE_RULE"].startswith("At step 2,")
    assert fills["STEP_LIST"].splitlines()[0].startswith("1. Do 0 (A > 0).")
    assert fills["WORD_RANGE"] == "100-185"
    sl.mistake = None
    assert "no mistake" in render_steps(sl)["MISTAKE_RULE"]


def test_fill_keeps_braces_in_a_step() -> None:
    assert fill("<<STEP_LIST>> {x}", {"STEP_LIST": "Type {name}"}) == (
        "Type {name} {x}"
    )


def _product():
    from src.video.producer.topic_input import TopicSpec, build_topic_product

    return build_topic_product(
        TopicSpec(title="How to turn off refresh", description="Save battery.")
    )


@pytest.mark.req("REQ-VID-121", "REQ-CNT-146")
@pytest.mark.asyncio
async def test_the_script_is_written_from_the_steps() -> None:
    from src.ai import script_generator

    cta = config.llm_settings.script_templates.cta_options_for(True)[0]
    reply = ("Settings first. " + "Tap the option and look. " * 25) + cta
    call = AsyncMock(return_value=reply)
    sl = _sl(3)
    sl.start_screen = "the Settings app"

    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        script, template, _ = await script_generator.generate_script(
            _product(),
            config.llm_settings,
            {config.llm_settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
            step_list=sl,
        )

    prompt = call.call_args.args[0]
    assert template == "topic_from_steps"
    assert "1. Do 0 (A > 0)." in prompt and "the Settings app" in prompt
    assert "100-185 words" in prompt and "<<" not in prompt
    assert script is not None


@pytest.mark.asyncio
async def test_a_short_draft_is_retried_then_used_as_a_last_resort() -> None:
    from src.ai import script_generator

    cta = config.llm_settings.script_templates.cta_options_for(True)[0]
    short = ("Tap the option and look closely. " * 12) + cta
    call = AsyncMock(return_value=short)

    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        script, _, _ = await script_generator.generate_script(
            _product(),
            config.llm_settings,
            {config.llm_settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
            step_list=_sl(4),
        )

    assert call.await_count >= 2
    assert script is not None and script.startswith("Tap the option")


def _ctx(tmp_path: Path, enabled: bool, topic: bool = True) -> SimpleNamespace:
    cfg = config.model_copy(deep=True)
    cfg.llm_settings.topic_scripts.step_list.enabled = enabled
    return SimpleNamespace(
        config=cfg,
        product=SimpleNamespace(
            topic="t" if topic else None, title="How to x", description="d"
        ),
        secrets={cfg.llm_settings.api_key_env_var: "k"},
        run_paths={"script_file": tmp_path / "text" / "script.txt"},
        state={},
    )


@pytest.mark.req("REQ-VID-121", "REQ-VID-122")
@pytest.mark.asyncio
async def test_the_step_records_the_list_or_drops_the_topic(tmp_path: Path) -> None:
    from src.video.producer import steps
    from src.video.producer.context import InsufficientMediaError

    built = AsyncMock(return_value=_sl(3))
    with patch("src.ai.step_list.build_step_list", built):
        assert await steps._topic_step_list(_ctx(tmp_path, False)) is None
        assert await steps._topic_step_list(_ctx(tmp_path, True, topic=False)) is None
        built.assert_not_called()

        ctx = _ctx(tmp_path, True)
        listed = await steps._topic_step_list(ctx)

    assert listed is not None and len(listed.steps) == 3
    assert ctx.state["step_list"] == "steps=3 refused=0"
    record = json.loads((tmp_path / "text" / "step_list.json").read_text())
    assert len(record["steps"]) == 3

    for result in (None, _sl(0), _sl(3, forks=True)):
        with (
            patch("src.ai.step_list.build_step_list", AsyncMock(return_value=result)),
            pytest.raises(steps.TopicNotSourcedError) as err,
        ):
            await steps._topic_step_list(_ctx(tmp_path, True))
        # A skip, handled like a listing with too little media.
        assert isinstance(err.value, InsufficientMediaError)


@pytest.mark.req("REQ-VID-121")
def test_the_script_step_builds_the_list_and_passes_it_on() -> None:
    import inspect

    from src.video.producer import steps

    source = inspect.getsource(steps.step_generate_script)
    assert source.index("await _topic_step_list(ctx)") < source.index(
        "step_list=step_list"
    )
