"""A series episode's verdict stamp card (REQ-VID-167)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.video.assembler.overlay_builder import (
    apply_verdict_card,
    build_verdict_drawtext,
)
from src.video.config.series_models import SeriesConfig, VerdictBeat
from src.video.config.visual_models import HookOverlaySettings
from src.video.verdict_card import (
    VerdictCard,
    mention_start,
    parse_verdict,
    verdict_prompt,
)

pytestmark = pytest.mark.req("REQ-VID-167")

SCRIPT = (
    "Myth check: the widget hums tonight. The claim sounds right. "
    "The test found no hum anywhere. The verdict is false. Follow for more checks."
)


def _words(text: str, start: float = 0.0, step: float = 0.3) -> list[dict]:
    return [
        {"word": w, "start_time": start + i * step, "end_time": start + i * step + 0.2}
        for i, w in enumerate(text.split())
    ]


class TestTheLabel:
    def test_the_prompt_names_the_label_and_numbers_the_sentences(self):
        prompt = verdict_prompt(["One.", "Two."], "NOT QUITE")
        assert '"NOT QUITE"' in prompt and "1. One.\n2. Two." in prompt

    @pytest.mark.parametrize(
        ("reply", "index"),
        [
            ('{"verdict": 4}', 3),
            ('Sure: {"verdict": 1}', 0),
            ('{"verdict": null}', None),
            ('{"verdict": 9}', None),
            ('{"verdict": true}', None),
            ("no json", None),
        ],
    )
    def test_the_reply_gives_a_zero_based_index_or_none(self, reply, index):
        assert parse_verdict(reply, 5) == index


class TestTheSpokenMention:
    def test_the_first_mention_after_the_opening_counts(self):
        words = _words("myth check false claim the test says false")
        assert mention_start(words, "FALSE", 3) == pytest.approx(7 * 0.3)

    def test_a_two_word_label_matches_in_order(self):
        words = _words("it is fine not quite right")
        assert mention_start(words, "NOT QUITE", 0) == pytest.approx(3 * 0.3)

    def test_no_mention_is_none(self):
        assert mention_start(_words("nothing here"), "TRUE", 0) is None


class TestTheConfig:
    def test_defaults(self):
        beat = VerdictBeat()
        assert (beat.enabled, beat.colours, beat.duration_sec) == (False, None, 1.5)

    def test_an_unknown_colour_key_is_named(self):
        with pytest.raises(ValidationError, match="border"):
            VerdictBeat(colours={"border": "red"})


HOOK = HookOverlaySettings()


class TestTheDrawtext:
    def _card(self, **beat) -> VerdictCard:
        return VerdictCard("FALSE", 2.0, 3.5, VerdictBeat(enabled=True, **beat))

    def test_the_label_is_boxed_and_gated_to_its_time(self, tmp_path):
        out = build_verdict_drawtext(
            self._card(colours={"text": "white", "background": "red@0.9"}),
            HOOK,
            60,
            1080,
            tmp_path,
            "[in]",
            "[out]",
        )
        assert "fontcolor=white:" in out and "boxcolor=red@0.9:" in out
        assert "enable=between(t\\,2.000\\,3.500)[out]" in out
        assert "y=h*0.35:" in out

    def test_the_position_follows_the_config(self, tmp_path):
        out = build_verdict_drawtext(
            self._card(y_percent=0.6), HOOK, 60, 1080, tmp_path, "[a]", "[b]"
        )
        assert "y=h*0.6:" in out
        assert (tmp_path / "verdict_label.txt").read_text() == "FALSE"

    def test_missing_colours_take_the_hook_overlays(self, tmp_path):
        out = build_verdict_drawtext(
            self._card(colours={"text": "yellow"}),
            HOOK,
            60,
            1080,
            tmp_path,
            "[in]",
            "[out]",
        )
        assert "fontcolor=yellow:" in out
        assert f"boxcolor={HOOK.background_color}:" in out

    def test_the_size_follows_the_caption_font(self, tmp_path):
        out = build_verdict_drawtext(
            self._card(size_factor=2.0), HOOK, 50, 1080, tmp_path, "[a]", "[b]"
        )
        assert "fontsize=100:" in out

    def test_no_card_leaves_the_chain_alone(self, tmp_path):
        chain = ["[v0]copy[v_out]"]
        assert apply_verdict_card(chain, None, HOOK, 60, 1080, tmp_path) == chain

    def test_a_card_keeps_the_terminal_copy(self, tmp_path):
        out = apply_verdict_card(
            ["[v0]copy[v_out]"], self._card(), HOOK, 60, 1080, tmp_path
        )
        assert (
            out[0].startswith("[v0]drawtext=") and out[-1] == "[v_verdict]copy[v_out]"
        )


@pytest.fixture
def ctx(tmp_path):
    from src.video.config import config
    from src.video.producer.context import PipelineContext
    from src.video.producer.state import get_video_run_paths

    cfg = config.model_copy(deep=True)
    cfg.global_output_root_path = tmp_path
    cfg.series = {
        "myth": SeriesConfig(
            opening_line="Myth check: {claim}.",
            verdicts=["TRUE", "FALSE"],
            verdict_beat=VerdictBeat(enabled=True),
        )
    }
    paths = get_video_run_paths(cfg, "B0TEST0167", "slideshow_images1")
    paths["script_file"].parent.mkdir(parents=True, exist_ok=True)
    product = MagicMock(
        asin="B0TEST0167",
        topic="t",
        title="t",
        series="myth",
        claim="the widget hums tonight",
        verdict="FALSE",
    )
    context = PipelineContext(
        product=product,
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


@pytest.mark.asyncio
class TestTheProducer:
    async def test_the_verdict_sentence_is_labelled_with_the_script(self, ctx):
        from src.video.producer import steps

        with _llm('{"verdict": 4}'):
            await steps._label_verdict(ctx)
        record = json.loads(ctx.state["verdict_label"])
        assert record == {"index": 3, "script": steps.script_digest(SCRIPT)}

    async def test_a_failed_call_labels_nothing(self, ctx):
        from src.video.producer import steps

        with patch(
            "src.ai.script_generator._call_llm_api_with_retry",
            AsyncMock(side_effect=TimeoutError()),
        ):
            await steps._label_verdict(ctx)
        assert json.loads(ctx.state["verdict_label"])["index"] is None

    async def test_off_means_no_call(self, ctx):
        from src.video.producer import steps

        ctx.config.series["myth"].verdict_beat.enabled = False
        with _llm() as call:
            await steps._label_verdict(ctx)
        assert call.await_count == 0 and "verdict_label" not in ctx.state

    async def test_the_script_step_labels_the_verdict(self, ctx):
        from src.video.producer import steps

        async def fake(*a, **k):
            return SCRIPT, "classic_promo", ""

        ctx.product.topic = None  # the topic path also asks for a step list
        label = AsyncMock()
        with (
            patch.object(steps, "generate_ai_script", fake),
            patch.object(steps, "_ensure_fact_checked", AsyncMock()),
            patch.object(steps, "_ensure_hook_headline", AsyncMock()),
            patch.object(steps, "_label_verdict", label),
        ):
            await steps.step_generate_script(ctx)
        label.assert_awaited_once()


class TestTheCard:
    def _with_words(self, ctx):
        from src.video.producer import steps

        return patch.object(steps, "_word_timings", return_value=_words(SCRIPT))

    def test_it_starts_on_the_named_sentence(self, ctx):
        from src.video.producer import steps

        ctx.state["verdict_label"] = json.dumps(
            {"index": 3, "script": steps.script_digest(SCRIPT)}
        )
        with self._with_words(ctx):
            card = steps._verdict_card(ctx)
        # 16 words come before the fourth sentence.
        assert card is not None and card.label == "FALSE"
        assert card.start == pytest.approx(16 * 0.3)
        assert card.end == pytest.approx(card.start + 1.5)

    def test_a_stale_label_falls_back_to_the_spoken_label(self, ctx):
        from src.video.producer import steps

        ctx.state["verdict_label"] = json.dumps({"index": 1, "script": "old"})
        with self._with_words(ctx):
            card = steps._verdict_card(ctx)
        # "false." is word 20 (0-based 19): "The verdict is false."
        assert card is not None and card.start == pytest.approx(19 * 0.3)

    def test_the_label_survives_a_truncating_resume(self, ctx):
        from src.video.producer import steps

        ctx.state = {
            steps.STEP_GENERATE_SCRIPT: {
                "verdict_label": json.dumps(
                    {"index": 3, "script": steps.script_digest(SCRIPT)}
                )
            }
        }
        with self._with_words(ctx):
            card = steps._verdict_card(ctx)
        assert card is not None and card.start == pytest.approx(16 * 0.3)

    def test_the_opening_lines_mention_does_not_count(self, ctx):
        from src.video.producer import steps

        ctx.product.claim = "false alarms are rare"
        script = "Myth check: false alarms are rare. They are common. So, false."
        ctx.script = script
        ctx.config.series["myth"].verdict_beat.duration_sec = 2.5
        with patch.object(steps, "_word_timings", return_value=_words(script)):
            card = steps._verdict_card(ctx)
        # The opening's "false" is word 3; the verdict's is word 11 (0-based 10).
        assert card is not None and card.start == pytest.approx(10 * 0.3)
        assert card.end == pytest.approx(card.start + 2.5)

    def test_neither_skips_the_card(self, ctx, caplog):
        from src.video.producer import steps

        ctx.product.verdict = "TRUE"
        with self._with_words(ctx), caplog.at_level("INFO"):
            assert steps._verdict_card(ctx) is None
        assert "Verdict card skipped" in caplog.text

    def test_without_word_timings_there_is_no_card(self, ctx):
        from src.video.producer import steps

        with patch.object(steps, "_word_timings", return_value=None):
            assert steps._verdict_card(ctx) is None

    def test_a_topic_outside_a_series_has_no_card(self, ctx):
        from src.video.producer import steps

        ctx.product.series = None
        with self._with_words(ctx):
            assert steps._verdict_card(ctx) is None


@pytest.mark.asyncio
async def test_the_label_is_mirrored_into_the_script_entry(ctx):
    from src.video.producer.state import STEP_GENERATE_SCRIPT, _update_state_after_step

    ctx.run_paths["script_file"].write_text(SCRIPT)
    ctx.state["verdict_label"] = "abc"
    await _update_state_after_step(ctx, STEP_GENERATE_SCRIPT)
    assert ctx.state[STEP_GENERATE_SCRIPT]["verdict_label"] == "abc"
