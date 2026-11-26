"""A series episode's commenter credit line (REQ-VID-168)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.video.assembler.overlay_builder import apply_credit_line, build_credit_drawtext
from src.video.config.series_models import CreditLine as CreditLineSettings
from src.video.config.series_models import SeriesConfig
from src.video.config.visual_models import HookOverlaySettings
from src.video.credit_line import CreditLine, claim_window

pytestmark = pytest.mark.req("REQ-VID-168")

OPENING = "Myth check: {claim}."
CLAIM = "the widget hums tonight"
SCRIPT = "Myth check: the widget hums tonight. The test found no hum."


def _words(text: str, step: float = 0.3) -> list[dict]:
    return [
        {"word": w, "start_time": i * step, "end_time": i * step + 0.2}
        for i, w in enumerate(text.split())
    ]


class TestTheWindow:
    def test_the_claim_slot_sets_the_window(self):
        # Two words before the slot, four claim words: words 2 to 5.
        assert claim_window(OPENING, CLAIM, _words(SCRIPT)) == pytest.approx((0.6, 1.7))

    def test_a_claim_opening_the_line(self):
        assert claim_window("{claim}?", "it hums", _words("it hums? no")) == (
            pytest.approx((0.0, 0.5))
        )

    @pytest.mark.parametrize(
        ("opening", "claim", "text"),
        [
            ("Myth check.", CLAIM, SCRIPT),
            (OPENING, "", SCRIPT),
            (OPENING, CLAIM, "Myth check: the"),
        ],
    )
    def test_no_slot_claim_or_words_is_none(self, opening, claim, text):
        assert claim_window(opening, claim, _words(text)) is None


class TestTheConfig:
    def test_defaults(self):
        line = CreditLineSettings()
        assert line.format == "Suggested by {credit}" and line.y_percent == 0.62

    def test_a_format_without_the_credit_is_rejected(self):
        with pytest.raises(ValidationError, match="credit"):
            CreditLineSettings(format="Thanks!")

    def test_an_unknown_colour_key_is_named(self):
        with pytest.raises(ValidationError, match="border"):
            CreditLineSettings(colours={"border": "red"})


HOOK = HookOverlaySettings()


class TestTheDrawtext:
    def test_the_line_is_boxed_low_and_gated_to_its_time(self, tmp_path):
        settings = CreditLineSettings(
            colours={"text": "white", "background": "black@0.6"}, y_percent=0.7
        )
        out = build_credit_drawtext(
            CreditLine("Suggested by @someone", 0.6, 1.7, settings),
            HOOK,
            60,
            1080,
            tmp_path,
            "[in]",
            "[out]",
        )
        assert "fontcolor=white:" in out and "boxcolor=black@0.6:" in out
        assert "y=h*0.7:" in out
        assert "enable=between(t\\,0.600\\,1.700)[out]" in out
        assert (tmp_path / "credit_line.txt").read_text() == "Suggested by @someone"

    def test_the_size_follows_the_caption_font(self, tmp_path):
        out = build_credit_drawtext(
            CreditLine("x", 0, 1, CreditLineSettings(size_factor=0.5)),
            HOOK,
            60,
            1080,
            tmp_path,
            "[a]",
            "[b]",
        )
        assert "fontsize=30:" in out

    def test_no_line_leaves_the_chain_alone(self, tmp_path):
        chain = ["[v0]copy[v_out]"]
        assert apply_credit_line(chain, None, HOOK, 60, 1080, tmp_path) == chain

    def test_a_line_keeps_the_terminal_copy(self, tmp_path):
        line = CreditLine("x", 0, 1, CreditLineSettings())
        out = apply_credit_line(["[v0]copy[v_out]"], line, HOOK, 60, 1080, tmp_path)
        assert out[0].startswith("[v0]drawtext=") and out[-1] == "[v_credit]copy[v_out]"


@pytest.fixture
def ctx(tmp_path):
    from src.video.config import config
    from src.video.producer.context import PipelineContext
    from src.video.producer.state import get_video_run_paths

    cfg = config.model_copy(deep=True)
    cfg.global_output_root_path = tmp_path
    cfg.series = {
        "myth": SeriesConfig(
            opening_line=OPENING,
            credit_commenter=True,
            credit_line=CreditLineSettings(format="Asked by {credit}"),
        )
    }
    paths = get_video_run_paths(cfg, "B0TEST0168", "slideshow_images1")
    product = MagicMock(
        asin="B0TEST0168",
        topic="t",
        title="t",
        series="myth",
        claim=CLAIM,
        credit="@someone",
    )
    context = PipelineContext(
        product=product,
        profile=cfg.video_profiles["slideshow_images1"],
        profile_name="slideshow_images1",
        config=cfg,
        secrets={},
        session=MagicMock(),
        run_paths=paths,
        debug_mode=False,
    )
    context.script = SCRIPT
    return context


class TestTheProducer:
    def _words(self):
        from src.video.producer import steps

        return patch.object(steps, "_word_timings", return_value=_words(SCRIPT))

    def test_the_line_names_the_commenter_while_the_claim_is_spoken(self, ctx):
        from src.video.producer import steps

        with self._words():
            line = steps._credit_line(ctx)
        assert line is not None and line.text == "Asked by @someone"
        assert (line.start, line.end) == pytest.approx((0.6, 1.7))

    def test_off_means_no_line(self, ctx):
        from src.video.producer import steps

        ctx.config.series["myth"].credit_commenter = False
        with self._words():
            assert steps._credit_line(ctx) is None

    def test_a_topic_without_a_credit_has_no_line(self, ctx):
        from src.video.producer import steps

        ctx.product.credit = None
        with self._words():
            assert steps._credit_line(ctx) is None

    def test_no_word_timings_skip_the_line_and_log_it(self, ctx, caplog):
        from src.video.producer import steps

        with (
            patch.object(steps, "_word_timings", return_value=None),
            caplog.at_level("INFO"),
        ):
            assert steps._credit_line(ctx) is None
        assert "Credit line skipped" in caplog.text
