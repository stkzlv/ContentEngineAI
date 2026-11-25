"""The delivery plan: written beside the script (REQ-CNT-179) and checked
against it before TTS (REQ-CNT-180).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.config.audio_models import PERFORMED_TTS_TAGS
from src.video.delivery_plan import (
    PLAN_FILE,
    DeliveryEvent,
    DeliveryPlan,
    DeliverySegment,
    parse_plan,
    validate_plan,
)

SENTENCES = ["The clocks stop.", "On purpose.", "Here is the twist."]
SCRIPT = " ".join(SENTENCES)


def _good() -> DeliveryPlan:
    return DeliveryPlan(
        segments=[
            DeliverySegment(
                text="The clocks stop. On purpose.",
                style="dry, amused",
                events=[DeliveryEvent(tag="<sigh>", before_word=3)],
                gap_after_ms=300,
            ),
            DeliverySegment(text="Here is the twist.", style="leaning in"),
        ]
    )


def _check(plan: DeliveryPlan, max_events: int = 4) -> list[str]:
    return validate_plan(plan, SCRIPT, SENTENCES, PERFORMED_TTS_TAGS, max_events)


@pytest.mark.req("REQ-CNT-180")
class TestTheCheck:
    def test_a_good_plan_passes_and_joins_back_to_the_script(self):
        assert _check(_good()) == []
        assert " ".join(s.text for s in _good().segments) == SCRIPT

    def test_text_that_differs_from_the_script(self):
        plan = _good()
        plan.segments[1].text = "Here is a twist."
        assert "the joined segment text differs from the script" in _check(plan)

    def test_a_segment_that_splits_a_sentence(self):
        plan = DeliveryPlan(
            segments=[
                DeliverySegment(text="The clocks"),
                DeliverySegment(text="stop. On purpose. Here is the twist."),
            ]
        )
        assert "segment 1 ends inside a sentence" in _check(plan)

    def test_an_event_outside_its_segment(self):
        plan = _good()
        plan.segments[1].events = [DeliveryEvent(tag="<sigh>", before_word=4)]
        assert "segment 2 has an event outside its words" in _check(plan)

    def test_a_tag_off_the_allowlist(self):
        plan = _good()
        plan.segments[1].events = [DeliveryEvent(tag="<scream>", before_word=0)]
        assert any("not allowed" in f for f in _check(plan))

    def test_more_than_one_event_in_a_segment(self):
        plan = _good()
        plan.segments[0].events.append(DeliveryEvent(tag="<laugh>", after_last=True))
        assert "segment 1 has more than one event" in _check(plan)

    def test_more_events_than_the_cap(self):
        plan = _good()
        plan.segments[1].events = [DeliveryEvent(tag="<laugh>", after_last=True)]
        assert "2 events, more than 1" in _check(plan, max_events=1)

    @pytest.mark.parametrize(
        "style", ['read "on purpose" slowly', "slow, stop on purpose here is the end"]
    )
    def test_a_style_that_quotes_the_script(self, style):
        plan = _good()
        plan.segments[1].style = style
        assert "segment 2's style quotes the script" in _check(plan)

    @pytest.mark.parametrize(
        ("style", "script"),
        [
            ("Deadpan, it's what's expected", "That is what you get for ten bucks."),
            ("Build up, one of the best moments", "This is one of the best mounts."),
            ("Slow down at the end of the line", "At the end of the day it holds."),
            ('read "is wh" slowly', "That is what you get."),
        ],
    )
    def test_a_style_that_quotes_nothing_passes(self, style, script):
        sentences = [script]
        plan = DeliveryPlan(segments=[DeliverySegment(text=script, style=style)])
        assert validate_plan(plan, script, sentences, PERFORMED_TTS_TAGS, 4) == []


@pytest.mark.req("REQ-CNT-179")
class TestTheReply:
    def test_segments_take_the_scripts_own_sentences(self):
        reply = json.dumps(
            {
                "segments": [
                    {
                        "sentences": [1, 2],
                        "style": "dry, amused",
                        "event": {"tag": "<sigh>", "before_word": 3},
                        "gap_after_ms": 300,
                    },
                    {"sentences": [3], "style": "leaning in", "event": None},
                ]
            }
        )
        plan = parse_plan("Here you go:\n" + reply, SENTENCES)
        assert plan == _good()

    @pytest.mark.parametrize(
        "reply", ["no json", '{"segments": [{"sentences": [9]}]}', '{"x": 1}']
    )
    def test_an_unusable_reply_is_no_plan(self, reply):
        assert parse_plan(reply, SENTENCES) is None


GOOD_REPLY = json.dumps(
    {
        "segments": [
            {"sentences": [1, 2], "style": "dry", "event": None},
            {"sentences": [3], "style": "leaning in", "event": None},
        ]
    }
)
BAD_REPLY = json.dumps(
    {"segments": [{"sentences": [1], "style": "x"}, {"sentences": [3]}]}
)


@pytest.fixture
def ctx(tmp_path):
    from src.video.config import config
    from src.video.producer.context import PipelineContext
    from src.video.producer.state import get_video_run_paths

    cfg = config.model_copy(deep=True)
    cfg.global_output_root_path = tmp_path
    cfg.tts_config.performed_delivery.enabled = True
    paths = get_video_run_paths(cfg, "B0TEST0179", "slideshow_images1")
    paths["script_file"].parent.mkdir(parents=True, exist_ok=True)
    context = PipelineContext(
        product=MagicMock(asin="B0TEST0179", topic=None, title="A mount"),
        profile=cfg.video_profiles["slideshow_images1"],
        profile_name="slideshow_images1",
        config=cfg,
        secrets={cfg.llm_settings.api_key_env_var: "k"},
        session=MagicMock(),
        run_paths=paths,
        debug_mode=False,
    )
    context.script = SCRIPT
    return context


def _llm(*replies):
    from src.ai import script_generator

    return patch.object(
        script_generator, "_call_llm_api_with_retry", AsyncMock(side_effect=replies)
    )


def _plan_path(ctx):
    return ctx.run_paths["script_file"].with_name(PLAN_FILE)


@pytest.mark.req("REQ-CNT-179")
@pytest.mark.asyncio
class TestWriting:
    async def test_a_plan_is_written_beside_the_script(self, ctx):
        from src.video.producer import steps

        with _llm(GOOD_REPLY):
            plan = await steps._write_delivery_plan(ctx)
        assert plan is not None and _plan_path(ctx).exists()
        assert ctx.state["delivery_plan"] == steps._plan_digest(
            _plan_path(ctx).read_text()
        )

    async def test_a_rejected_plan_is_retried_once_then_dropped(self, ctx):
        from src.video.producer import steps

        with _llm(BAD_REPLY, GOOD_REPLY) as call:
            assert await steps._write_delivery_plan(ctx) is not None
        assert call.await_count == 2
        _plan_path(ctx).unlink()
        with _llm(BAD_REPLY, BAD_REPLY, GOOD_REPLY) as call:
            assert await steps._write_delivery_plan(ctx) is None
        assert call.await_count == 2 and not _plan_path(ctx).exists()

    async def test_the_script_step_writes_it_only_when_enabled(self, ctx):
        from src.video.producer import steps

        async def fake(*a, **k):
            return SCRIPT, "classic_promo", ""

        for enabled, written in ((False, False), (True, True)):
            ctx.config.tts_config.performed_delivery.enabled = enabled
            ctx.run_paths["script_file"].unlink(missing_ok=True)
            _plan_path(ctx).unlink(missing_ok=True)
            with (
                patch.object(steps, "generate_ai_script", fake),
                patch.object(steps, "_ensure_fact_checked", AsyncMock()),
                patch.object(steps, "_ensure_hook_headline", AsyncMock()),
                _llm(GOOD_REPLY, GOOD_REPLY),
            ):
                await steps.step_generate_script(ctx)
            assert _plan_path(ctx).exists() is written


@pytest.mark.req("REQ-CNT-180")
@pytest.mark.asyncio
class TestTheVoiceGetsACheckedPlan:
    async def test_off_means_no_plan(self, ctx):
        from src.video.producer import steps

        ctx.config.tts_config.performed_delivery.enabled = False
        assert await steps._delivery_plan_for_voice(ctx) is None

    async def test_a_generated_plan_that_passes_is_voiced(self, ctx):
        from src.video.producer import steps

        with _llm(GOOD_REPLY):
            written = await steps._write_delivery_plan(ctx)
        assert await steps._delivery_plan_for_voice(ctx) == written

    async def test_a_stale_generated_plan_is_written_again(self, ctx):
        from src.video.producer import steps

        with _llm(GOOD_REPLY):
            await steps._write_delivery_plan(ctx)
        ctx.script = "The clocks stop. On purpose. Here is a twist."
        new = json.dumps(
            {"segments": [{"sentences": [1, 2, 3], "style": "dry", "event": None}]}
        )
        with _llm(new):
            plan = await steps._delivery_plan_for_voice(ctx)
        assert plan is not None and plan.segments[0].text == ctx.script

    async def test_an_edited_plan_that_fails_stops_naming_the_check(self, ctx):
        from src.video.producer import steps
        from src.video.producer.context import PipelineError

        with _llm(GOOD_REPLY):
            await steps._write_delivery_plan(ctx)
        plan = json.loads(_plan_path(ctx).read_text())
        plan["segments"][1]["events"] = [{"tag": "<scream>", "before_word": 0}]
        _plan_path(ctx).write_text(json.dumps(plan))
        with pytest.raises(PipelineError, match="not allowed"):
            await steps._delivery_plan_for_voice(ctx)

    async def test_the_plan_reaches_the_voice(self, ctx):
        from src.video.producer import steps

        with _llm(GOOD_REPLY):
            written = await steps._write_delivery_plan(ctx)
        ctx.run_paths["script_file"].write_text(SCRIPT)
        voice = AsyncMock(return_value=None)
        with (
            patch.object(steps.TTSManager, "generate_speech", voice),
            pytest.raises(steps.PipelineError),
        ):
            await steps.step_create_voiceover(ctx)
        assert voice.call_args.args[2] == written


@pytest.mark.req("REQ-CNT-179")
@pytest.mark.asyncio
async def test_the_digest_survives_a_truncating_resume(ctx):
    from src.video.producer.state import STEP_GENERATE_SCRIPT, _update_state_after_step

    ctx.run_paths["script_file"].write_text(SCRIPT)
    ctx.state["delivery_plan"] = "abc"
    await _update_state_after_step(ctx, STEP_GENERATE_SCRIPT)
    assert ctx.state[STEP_GENERATE_SCRIPT]["delivery_plan"] == "abc"


@pytest.mark.req("REQ-CNT-180")
@pytest.mark.asyncio
async def test_a_stale_generated_plan_after_a_truncating_resume_is_rewritten(ctx):
    from src.video.producer import steps
    from src.video.producer.state import STEP_GENERATE_SCRIPT

    with _llm(GOOD_REPLY):
        await steps._write_delivery_plan(ctx)
    # A truncating resume keeps only step entries.
    ctx.state = {STEP_GENERATE_SCRIPT: {"delivery_plan": ctx.state["delivery_plan"]}}
    ctx.script = "The clocks stop. On purpose. Here is a twist."
    new = json.dumps({"segments": [{"sentences": [1, 2, 3], "style": "dry"}]})
    with _llm(new):
        plan = await steps._delivery_plan_for_voice(ctx)
    assert plan is not None and plan.segments[0].text == ctx.script
    # The rewritten plan is recorded in the entry that survives the next cut.
    assert ctx.state[STEP_GENERATE_SCRIPT]["delivery_plan"] == steps._plan_digest(
        _plan_path(ctx).read_text()
    )


@pytest.mark.req("REQ-CNT-180")
@pytest.mark.asyncio
async def test_a_failed_write_leaves_no_old_plan_behind(ctx):
    from src.video.producer import steps

    with _llm(GOOD_REPLY):
        await steps._write_delivery_plan(ctx)
    ctx.state = {}
    ctx.script = "The clocks stop. On purpose. Here is a twist."
    with _llm(BAD_REPLY, BAD_REPLY):
        assert await steps._write_delivery_plan(ctx) is None
    assert not _plan_path(ctx).exists()
    with _llm(BAD_REPLY, BAD_REPLY):
        assert await steps._delivery_plan_for_voice(ctx) is None
