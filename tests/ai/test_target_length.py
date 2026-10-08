"""The spoken-length target comes from config (REQ-CNT-163, design 0025)."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from src.ai.llm_settings import TargetLength
from src.video.config import config

SHIPPED_LINE = "- Target 30-40 seconds at normal speaking pace (roughly 75-100 words)."


@pytest.mark.req("REQ-CNT-163")
@pytest.mark.parametrize("is_topic", [False, True])
def test_the_defaults_render_the_shipped_line(is_topic: bool) -> None:
    templates = config.llm_settings.script_templates
    raw = templates.narrator_profile_topic if is_topic else templates.narrator_profile

    rendered = templates.narrator_for(is_topic)

    assert "{TARGET_SECONDS}" in raw and "{TARGET_WORDS}" in raw
    assert SHIPPED_LINE in rendered
    assert "{TARGET" not in rendered
    # Only the placeholders change: the rest of the profile is the text as written.
    assert rendered == raw.replace("{TARGET_SECONDS}", "30-40").replace(
        "{TARGET_WORDS}", "75-100"
    )


@pytest.mark.req("REQ-CNT-163")
def test_the_shipped_targets_are_thirty_to_forty_seconds() -> None:
    lengths = config.llm_settings.script_templates.target_length
    for target in (lengths.product, lengths.topic):
        assert target.seconds == (30, 40) and target.words == (75, 100)


@pytest.mark.req("REQ-CNT-163")
def test_each_content_type_takes_its_own_target() -> None:
    templates = config.llm_settings.script_templates.model_copy(deep=True)
    templates.target_length.topic = TargetLength(seconds=(20, 30), words=(50, 75))

    assert "Target 20-30 seconds" in templates.narrator_for(True)
    assert "roughly 50-75 words" in templates.narrator_for(True)
    assert SHIPPED_LINE in templates.narrator_for(False)


@pytest.mark.req("REQ-CNT-163")
def test_an_override_wins_over_the_content_type() -> None:
    templates = config.llm_settings.script_templates
    short = TargetLength(seconds=(15, 30), words=(50, 60))

    assert "Target 15-30 seconds" in templates.narrator_for(False, short)
    assert "roughly 50-60 words" in templates.narrator_for(True, short)


@pytest.mark.req("REQ-CNT-163")
@pytest.mark.parametrize("bad", [(40, 30), (0, 30), (-5, 10)])
def test_a_range_must_rise_above_zero(bad: tuple[int, int]) -> None:
    assert TargetLength(seconds=(30, 30)).seconds == (30, 30)
    with pytest.raises(ValidationError):
        TargetLength(seconds=bad)


@pytest.mark.req("REQ-CNT-163")
def test_a_video_profile_accepts_an_override() -> None:
    from src.video.config.visual_models import VideoProfile

    profile = VideoProfile(
        description="short", target_length={"seconds": [15, 30], "words": [50, 60]}
    )

    assert profile.target_length == TargetLength(seconds=(15, 30), words=(50, 60))
    assert VideoProfile(description="plain").target_length is None


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


async def _prompt(target: TargetLength | None, lint: bool = False) -> str:
    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_validation.lint.enabled = lint
    call = AsyncMock(return_value="")
    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        await script_generator.generate_script(
            _product(),
            settings,
            {settings.api_key_env_var: "k"},
            AsyncMock(),
            {},
            False,
            target_length=target,
        )
    return str(call.await_args_list[0].args[0])


@pytest.mark.req("REQ-CNT-163")
@pytest.mark.asyncio
async def test_a_profile_override_reaches_the_script_prompt() -> None:
    prompt = await _prompt(TargetLength(seconds=(15, 30), words=(50, 60)))

    assert "Target 15-30 seconds at normal speaking pace (roughly 50-60 words)." in (
        prompt
    )
    assert SHIPPED_LINE in await _prompt(None)


@pytest.mark.req("REQ-CNT-163")
@pytest.mark.asyncio
async def test_an_override_sets_the_lint_word_cap() -> None:
    lint = config.llm_settings.script_validation.lint
    cap = int(lint.max_words_per_sec * 30)
    default_cap = int(lint.max_words_per_sec * lint.target_duration_sec)

    prompt = await _prompt(TargetLength(seconds=(15, 30), words=(50, 60)), lint=True)

    assert f"Keep the whole script to {cap} words or fewer" in prompt
    assert f"Keep the whole script to {default_cap} words or fewer" in (
        await _prompt(None, lint=True)
    )
    # The caller's settings are not changed by the override.
    assert (
        lint.target_duration_sec
        == config.llm_settings.script_validation.lint.target_duration_sec
    )


@pytest.mark.req("REQ-CNT-163")
def test_the_script_step_passes_the_profiles_override() -> None:
    from src.video.producer import steps

    source = inspect.getsource(steps.step_generate_script)
    assert "target_length=_target_length(ctx)" in source
    assert 'getattr(ctx, "profile", None), "target_length", None)' in inspect.getsource(
        steps._target_length
    )


@pytest.mark.req("REQ-CNT-163")
def test_every_narrator_prompt_in_the_steps_takes_the_override() -> None:
    """The reviser, hook, phrase and caption prompts read the same target."""
    from src.video.producer import steps

    # The text after each call's opening parenthesis, up to the next keyword.
    calls = [
        part.split("=", 1)[0]
        for part in inspect.getsource(steps).split("narrator_for(")[1:]
    ]
    assert len(calls) == 4
    assert all("_target_length(ctx)" in call for call in calls)
