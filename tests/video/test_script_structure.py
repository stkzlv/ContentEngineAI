"""Script-shape targets (REQ-CNT-189 to REQ-CNT-191)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.script_generator import _with_running_bit
from src.video.config.structure_models import ScriptStructure
from src.video.script_structure import (
    MISCONCEPTION_RULE,
    NO_RECAP_RULE,
    draw_variant,
    estimate_starts,
    measure_starts,
    misses,
    parse_labels,
    prompt_rules,
    signoff_length,
    targets_for,
)

SCRIPT = (
    "Dark mode saves battery. Most people think it always does. "
    "But it only helps on bright OLED screens. The study found up to 47 percent. "
    "Tell me I'm wrong."
)


def _on(**kw) -> ScriptStructure:
    return ScriptStructure(enabled=True, **kw)


@pytest.mark.req("REQ-CNT-189")
def test_rules_follow_the_settings():
    assert prompt_rules(ScriptStructure()) == ""
    rules = prompt_rules(_on(setup="misconception", no_recap=True))
    assert MISCONCEPTION_RULE in rules and NO_RECAP_RULE in rules
    assert "Narrator." in _with_running_bit("Narrator.", "", rules)
    assert MISCONCEPTION_RULE in _with_running_bit("Narrator.", "", rules)


@pytest.mark.req("REQ-CNT-189")
def test_labels_parse_and_reject_out_of_range():
    assert parse_labels('Sure: {"turn": 3, "payoff": 4}', 5) == {"turn": 2, "payoff": 3}
    assert parse_labels('{"turn": 9, "payoff": null}', 5) == {
        "turn": None,
        "payoff": None,
    }
    assert parse_labels("no json", 5) == {"turn": None, "payoff": None}


@pytest.mark.req("REQ-CNT-189")
def test_estimate_counts_words_before_the_sentence():
    sentences = ["One two three four.", "Five six.", "Seven."]
    est = estimate_starts(sentences, {"turn": 1, "payoff": 2}, wpm=60)
    assert est == {"turn": 4.0, "payoff": 6.0}


@pytest.mark.req("REQ-CNT-189")
def test_a_late_beat_is_a_miss():
    targets = targets_for(_on(turn_by_sec=5, payoff_by_sec=10), None)
    assert misses({"turn": 6.0, "payoff": 9.0}, targets) == [
        "turn at 6.0 s, target 5.0 s"
    ]


@pytest.mark.req("REQ-CNT-190")
def test_measure_reads_the_word_timings():
    sentences = ["One two three four.", "Five six."]
    words = [
        {"word": w, "start": float(i), "end": i + 0.5} for i, w in enumerate("abcdef")
    ]
    assert measure_starts(sentences, {"turn": 1, "payoff": None}, words) == {
        "turn": 4.0,
        "payoff": None,
    }
    assert signoff_length(words, "Five six.") == 1.5
    extracted = [
        {"word": w["word"], "start_time": w["start"], "end_time": w["end"]}
        for w in words
    ]
    assert (
        measure_starts(sentences, {"turn": 1, "payoff": None}, extracted)["turn"] == 4.0
    )


@pytest.mark.req("REQ-CNT-191")
def test_a_variant_is_drawn_per_product_and_overrides_the_targets():
    structure = _on(
        payoff_by_sec=16,
        variants={"early": {"payoff_by_sec": 12}, "late": {"payoff_by_sec": 25}},
    )
    first = draw_variant(structure, "B0TEST0191")
    assert first == draw_variant(structure, "B0TEST0191")
    assert {draw_variant(structure, f"B0TEST{i:04d}") for i in range(40)} == {
        "early",
        "late",
    }
    assert targets_for(structure, "early").payoff_by_sec == 12
    assert draw_variant(_on(), "B0TEST0191") is None


@pytest.fixture
def step_ctx(tmp_path):
    from src.video.config import config
    from src.video.producer.context import PipelineContext
    from src.video.producer.state import get_video_run_paths

    cfg = config.model_copy(deep=True)
    cfg.global_output_root_path = tmp_path
    cfg.script_structure.product = _on(
        turn_by_sec=2, setup="misconception", variants={"v": {"payoff_by_sec": 30}}
    )
    paths = get_video_run_paths(cfg, "B0TEST0189", "slideshow_images1")
    product = MagicMock(
        asin="B0TEST0189", topic=None, topic_kind=None, title="A mount", keyword=""
    )
    return PipelineContext(
        product=product,
        profile=cfg.video_profiles["slideshow_images1"],
        profile_name="slideshow_images1",
        config=cfg,
        secrets={cfg.llm_settings.api_key_env_var: "k"},
        session=MagicMock(),
        run_paths=paths,
        debug_mode=False,
    )


@pytest.mark.req("REQ-CNT-189")
@pytest.mark.req("REQ-CNT-191")
@pytest.mark.asyncio
async def test_the_step_labels_estimates_and_records(step_ctx, caplog):
    from src.ai import script_generator
    from src.video.producer import steps
    from src.video.producer.state import STEP_GENERATE_SCRIPT, _update_state_after_step
    from src.video.render_choices import choices_from_context

    seen = {}

    async def fake(product, settings, *a, **k):
        seen["rules"] = settings.script_templates.structure_rules
        return SCRIPT, "classic_promo", ""

    label = AsyncMock(return_value='{"turn": 3, "payoff": 4}')
    with (
        patch.object(steps, "generate_ai_script", fake),
        patch.object(steps, "_ensure_fact_checked", AsyncMock()),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
        patch.object(script_generator, "_call_llm_api_with_retry", label),
    ):
        await steps.step_generate_script(step_ctx)

    assert MISCONCEPTION_RULE in seen["rules"]
    assert step_ctx.config.llm_settings.script_templates.structure_rules == ""
    record = json.loads(step_ctx.state["structure"])
    assert record["labels"] == {"turn": 2, "payoff": 3}
    assert (
        record["estimated"]["turn"] > 2
        and "Script shape, estimated: turn" in caplog.text
    )
    assert step_ctx.state["structure_variant"] == "v"
    await _update_state_after_step(step_ctx, STEP_GENERATE_SCRIPT)
    step_ctx.state = {
        k: v for k, v in step_ctx.state.items() if k == STEP_GENERATE_SCRIPT
    }
    row = choices_from_context(step_ctx)
    assert row["structure_variant"] == "v"
    assert row["structure"]["labels"] == {"turn": 2, "payoff": 3}


@pytest.mark.req("REQ-CNT-190")
def test_the_row_measures_from_the_transcript(step_ctx):
    from src.video.render_choices import choices_from_context

    step_ctx.script = "One two three four. Five six."
    step_ctx.state["structure"] = json.dumps(
        {"labels": {"turn": 1, "payoff": None}, "targets": {"turn_by_sec": 3}}
    )
    path = step_ctx.run_paths["whisper_transcript_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    words = [
        {"word": w, "start": float(i), "end": i + 0.5} for i, w in enumerate("abcdef")
    ]
    path.write_text(json.dumps({"segments": [{"words": words}]}))

    record = choices_from_context(step_ctx)["structure"]

    assert record["measured"] == {"turn": 4.0, "payoff": None}
    assert record["misses"] == ["turn at 4.0 s, target 3.0 s"]


def test_a_disabled_kind_records_nothing(step_ctx):
    from src.video.render_choices import choices_from_context

    row = choices_from_context(step_ctx)
    assert row["structure"] is None and row["structure_variant"] is None


@pytest.mark.req("REQ-CNT-189")
@pytest.mark.asyncio
async def test_a_failed_labelling_call_records_no_beats_and_the_step_completes(
    step_ctx,
):
    from src.ai import script_generator
    from src.ai.script_generator import ScriptGenerationError
    from src.video.producer import steps

    async def fake(product, settings, *a, **k):
        return SCRIPT, "classic_promo", ""

    failing = AsyncMock(side_effect=ScriptGenerationError("Gemini API error: 503"))
    with (
        patch.object(steps, "generate_ai_script", fake),
        patch.object(steps, "_ensure_fact_checked", AsyncMock()),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
        patch.object(script_generator, "_call_llm_api_with_retry", failing),
    ):
        await steps.step_generate_script(step_ctx)

    record = json.loads(step_ctx.state["structure"])
    assert record["labels"] == {"turn": None, "payoff": None}
    assert step_ctx.state["script_template"] == "classic_promo"


@pytest.mark.req("REQ-CNT-189")
@pytest.mark.asyncio
async def test_labels_index_the_script_after_the_fact_check(step_ctx):
    from src.ai import script_generator
    from src.video.producer import steps

    revised = "Dark mode saves battery. But only on bright OLED screens."
    seen = {}

    async def fake(product, settings, *a, **k):
        return SCRIPT, "classic_promo", ""

    async def fact_check(ctx, pillar):
        ctx.script = revised

    async def label(prompt, *a, **k):
        seen["prompt"] = prompt
        return '{"turn": 2, "payoff": null}'

    with (
        patch.object(steps, "generate_ai_script", fake),
        patch.object(steps, "_ensure_fact_checked", fact_check),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
        patch.object(script_generator, "_call_llm_api_with_retry", label),
    ):
        await steps.step_generate_script(step_ctx)

    assert "2. But only on bright OLED screens." in seen["prompt"]
    assert "Most people think" not in seen["prompt"]


@pytest.mark.req("REQ-CNT-190")
def test_an_edited_script_makes_the_record_stale(step_ctx):
    from src.video.producer.steps import script_digest
    from src.video.render_choices import choices_from_context

    step_ctx.script = "One two three four. Five six."
    step_ctx.state["structure"] = json.dumps(
        {"labels": {"turn": 1}, "targets": {}, "script": script_digest("Older text.")}
    )
    path = step_ctx.run_paths["whisper_transcript_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    words = [
        {"word": w, "start": float(i), "end": i + 0.5} for i, w in enumerate("abcdef")
    ]
    path.write_text(json.dumps({"segments": [{"words": words}]}))

    record = choices_from_context(step_ctx)["structure"]

    assert record["stale"] is True and "measured" not in record


@pytest.mark.req("REQ-CNT-190")
def test_the_signoff_is_measured_before_the_call_to_action():
    # "That's the take." (0.3 s a word) then "Would you buy it?" (0.6 s a word).
    words, t = [], 0.0
    for w, d in [("That's", 0.3), ("the", 0.3), ("take.", 0.3)] + [
        (w, 0.6) for w in ("Would", "you", "buy", "it?")
    ]:
        words.append({"word": w, "start_time": t, "end_time": t + d})
        t += d
    assert signoff_length(words, "That's the take.", "Would you buy it?") == 0.9


@pytest.mark.req("REQ-CNT-190")
def test_numbers_count_as_one_word_like_the_transcript():
    sentences = ["It costs $29.99 for 5,000 mAh.", "But the cable is short."]
    words = [
        {"word": w, "start": float(i)}
        for i, w in enumerate(
            "It costs $29.99 for 5,000 mAh. But the cable is short.".split()
        )
    ]
    assert measure_starts(sentences, {"turn": 1}, words) == {"turn": 6.0}
