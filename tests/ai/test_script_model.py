"""Script generation can use its own model, and records which model wrote it.

REQ-CNT-162, #703.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.ai.llm_settings import ScriptModelConfig
from src.video.config import config


def _settings(script_model: ScriptModelConfig | None = None):
    settings = config.llm_settings.model_copy(deep=True)
    settings.provider = "gemini"
    settings.models = ["shared-a", "shared-b"]
    settings.thinking_budget = 0
    settings.max_tokens = 600
    settings.fallback_provider = None
    settings.script_validation.lint.enabled = False
    settings.script_validation.reject_copied_examples = False
    settings.script_model = script_model
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


def _script(closing: str = "It lasts a week.") -> str:
    cta = config.llm_settings.script_templates.cta_options_for(False)[0]
    filler = "This watch tracks your sleep every night of the week. " * 8
    return f"{filler}{closing} {cta}".replace("  ", " ")


async def _generate(settings, side_effect):
    from src.ai import script_generator

    written_by: dict[str, str] = {}
    call = AsyncMock(side_effect=side_effect)
    with (
        patch.object(script_generator, "_call_llm_api_with_retry", call),
        patch.object(script_generator, "configured_live", lambda s: list(s.models)),
    ):
        out, _, _ = await script_generator.generate_script(
            _product(),
            settings,
            {settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
            written_by=written_by,
        )
    return out, call, written_by


@pytest.mark.req("REQ-CNT-162")
def test_unset_the_script_call_shares_the_text_settings() -> None:
    settings = _settings()

    assert settings.for_scripts() is settings


@pytest.mark.req("REQ-CNT-162")
def test_set_the_script_call_takes_its_own_model_budget_and_limit() -> None:
    settings = _settings(
        ScriptModelConfig(model="script-m", thinking_budget=None, max_tokens=2000)
    )

    scripts = settings.for_scripts()

    assert scripts.models == ["script-m"]
    assert scripts.thinking_budget is None and scripts.max_tokens == 2000
    assert scripts.auto_select_free_model is False
    # The other text calls keep theirs.
    assert settings.models == ["shared-a", "shared-b"]
    assert settings.thinking_budget == 0 and settings.max_tokens == 600


@pytest.mark.req("REQ-CNT-162")
def test_the_shipped_config_sets_no_script_model() -> None:
    assert config.llm_settings.script_model is None


@pytest.mark.req("REQ-CNT-162")
@pytest.mark.parametrize("bad", [{"model": ""}, {"model": "m", "max_tokens": 0}])
def test_a_script_model_needs_a_name_and_room(bad: dict) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ScriptModelConfig(**bad)


@pytest.mark.req("REQ-CNT-162")
@pytest.mark.asyncio
async def test_the_script_call_sends_the_script_models_settings() -> None:
    settings = _settings(
        ScriptModelConfig(model="script-m", thinking_budget=None, max_tokens=2000)
    )

    out, call, written_by = await _generate(settings, [_script()])

    _, model, sent = call.await_args_list[0].args[:3]
    assert model == "script-m"
    assert sent.max_tokens == 2000 and sent.thinking_budget is None
    assert out == _script() and written_by == {"model": "script-m"}


@pytest.mark.req("REQ-CNT-162")
@pytest.mark.asyncio
async def test_a_fallback_model_is_the_one_recorded() -> None:
    from src.ai.script_generator import ScriptGenerationError

    failure = ScriptGenerationError("400")
    out, call, written_by = await _generate(_settings(), [failure, failure, _script()])

    assert [c.args[1] for c in call.await_args_list] == [
        "shared-a",
        "shared-a",
        "shared-b",
    ]
    assert written_by == {"model": "shared-b"}


@pytest.mark.req("REQ-CNT-162")
@pytest.mark.asyncio
async def test_a_last_resort_names_the_model_that_wrote_it() -> None:
    # Only shared-b's draft carries a placeholder near miss; shared-a's fails.
    out, _, written_by = await _generate(
        _settings(), ["too short.", "too short.", _script("It shows [n] days.")] * 2
    )

    assert out == _script("It shows [n] days.")
    assert written_by == {"model": "shared-b"}


@pytest.mark.req("REQ-CNT-162")
def test_the_script_step_records_the_model() -> None:
    from src.video.producer import steps

    source = inspect.getsource(steps.step_generate_script)
    assert "written_by=written_by" in source
    assert 'ctx.state["script_model"] = written_by["model"]' in source


@pytest.mark.req("REQ-CNT-162")
@pytest.mark.asyncio
async def test_the_step_entry_keeps_the_model(monkeypatch) -> None:
    from src.video.producer import state

    ctx = SimpleNamespace(
        state={"script_model": "script-m"},
        run_paths={"script_file": Path("script.txt")},
    )
    monkeypatch.setattr(state, "_drop_dependents", lambda ctx, step: None)

    await state._update_state_after_step(ctx, state.STEP_GENERATE_SCRIPT)

    assert ctx.state["generate_script"]["script_model"] == "script-m"


@pytest.mark.req("REQ-CNT-162")
@pytest.mark.parametrize(
    "state",
    [
        {"script_model": "script-m"},
        {"generate_script": {"status": "done", "script_model": "script-m"}},
    ],
)
def test_the_render_row_carries_the_model(tmp_path: Path, state: dict) -> None:
    from src.video.render_choices import choices_from_context

    ctx = SimpleNamespace(
        state=state,
        config=SimpleNamespace(
            video_settings=SimpleNamespace(
                first_frame_pre_motion=False, video_transition_duration=0.3
            )
        ),
        profile_name="p",
        profile=SimpleNamespace(
            video_assembly_mode="sequential",
            first_frame_pre_motion=None,
            video_transition_duration=0.5,
        ),
        run_paths={"music_info_file": None, "run_root": tmp_path / "B0X"},
        script="The script.",
    )

    assert choices_from_context(ctx)["script_model"] == "script-m"
