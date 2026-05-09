"""Unit tests for the on-frame disclosure overlay builder."""

from src.video.assembler.overlay_builder import (
    apply_disclosure_overlay,
    build_disclosure_drawtext,
)
from src.video.config.visual_models import DisclosureSettings


class TestBuildDisclosureDrawtext:
    """Test the single-filter drawtext builder."""

    def test_default_settings_produce_drawtext_filter(self):
        settings = DisclosureSettings()
        filter_str = build_disclosure_drawtext(
            settings,
            subtitle_font_size_pixels=80,
            input_stream="[v_in]",
            output_stream="[v_out]",
        )

        assert filter_str.startswith("[v_in]drawtext=")
        assert filter_str.endswith("[v_out]")
        assert "text='#ad'" in filter_str
        # 80 * 0.55 = 44
        assert "fontsize=44" in filter_str
        assert "fontcolor=white" in filter_str
        assert "bordercolor=black" in filter_str

    def test_custom_text_appears_in_filter(self):
        settings = DisclosureSettings(text="#publi")
        filter_str = build_disclosure_drawtext(settings, 80, "[v_in]", "[v_out]")
        assert "text='#publi'" in filter_str

    def test_size_factor_scales_font(self):
        settings = DisclosureSettings(size_factor=0.5)
        filter_str = build_disclosure_drawtext(settings, 100, "[v_in]", "[v_out]")
        assert "fontsize=50" in filter_str

    def test_top_right_position(self):
        settings = DisclosureSettings(
            position="top-right",
            margin_x_percent=0.04,
            margin_y_percent=0.12,
        )
        filter_str = build_disclosure_drawtext(settings, 80, "[v_in]", "[v_out]")
        assert "x=w-text_w-w*0.04" in filter_str
        assert "y=h*0.12" in filter_str

    def test_top_left_position(self):
        settings = DisclosureSettings(position="top-left")
        filter_str = build_disclosure_drawtext(settings, 80, "[v_in]", "[v_out]")
        assert "x=w*" in filter_str  # left margin
        assert "y=h*" in filter_str  # top margin
        assert "w-text_w" not in filter_str

    def test_bottom_right_position(self):
        settings = DisclosureSettings(position="bottom-right")
        filter_str = build_disclosure_drawtext(settings, 80, "[v_in]", "[v_out]")
        assert "x=w-text_w-" in filter_str
        assert "y=h-text_h-" in filter_str

    def test_bottom_left_position(self):
        settings = DisclosureSettings(position="bottom-left")
        filter_str = build_disclosure_drawtext(settings, 80, "[v_in]", "[v_out]")
        # x is plain margin, y subtracts text_h
        assert "y=h-text_h-" in filter_str

    def test_background_disabled_omits_box(self):
        settings = DisclosureSettings(background_enabled=False)
        filter_str = build_disclosure_drawtext(settings, 80, "[v_in]", "[v_out]")
        assert "box=1" not in filter_str

    def test_background_enabled_includes_box(self):
        settings = DisclosureSettings(
            background_enabled=True,
            background_color="black@0.5",
        )
        filter_str = build_disclosure_drawtext(settings, 80, "[v_in]", "[v_out]")
        assert "box=1" in filter_str
        assert "boxcolor=black@0.5" in filter_str

    def test_drawtext_special_chars_escaped(self):
        # Colons, single quotes, and percent signs are FFmpeg drawtext
        # metacharacters that must be backslash-escaped.
        settings = DisclosureSettings(text="A: it's 50% off")
        filter_str = build_disclosure_drawtext(settings, 80, "[v_in]", "[v_out]")
        # Each metachar is escaped exactly once.
        assert r"\:" in filter_str
        assert r"\'" in filter_str
        assert r"\%" in filter_str

    def test_minimum_font_size_floor(self):
        # Even tiny subtitle base sizes shouldn't produce illegible fonts.
        # 30px subtitle * 0.2 size_factor = 6px without the floor; floor lifts
        # to 8px so the disclosure stays readable on phone screens.
        settings = DisclosureSettings(size_factor=0.2)
        filter_str = build_disclosure_drawtext(settings, 30, "[v_in]", "[v_out]")
        assert "fontsize=8" in filter_str


class TestApplyDisclosureOverlay:
    """Test injection into the existing video filter chain."""

    def test_replaces_terminal_copy_with_drawtext(self):
        # Subtitle builder always emits a final "...copy[v_out]" no-op.
        filters = [
            "[0:v]scale=1080:1920[v0]",
            "[v0]copy[v_sub_1]",
            "[v_sub_1]copy[v_out]",
        ]
        out = apply_disclosure_overlay(filters, DisclosureSettings(), 80)

        assert len(out) == len(filters)
        assert out[:-1] == filters[:-1]  # earlier filters unchanged
        assert out[-1].endswith("[v_out]")
        assert "drawtext=" in out[-1]
        assert "copy[v_out]" not in out[-1]

    def test_disabled_returns_unchanged_chain(self):
        filters = ["[v_sub_1]copy[v_out]"]
        out = apply_disclosure_overlay(filters, DisclosureSettings(enabled=False), 80)
        assert out is filters or out == filters
        assert out[-1] == "[v_sub_1]copy[v_out]"

    def test_unexpected_terminal_filter_is_logged_and_skipped(self):
        # If the terminal filter doesn't match the expected shape, we leave the
        # chain alone rather than corrupting the graph.
        filters = ["[v0]something_else[v_out]"]
        out = apply_disclosure_overlay(filters, DisclosureSettings(), 80)
        assert out == filters

    def test_empty_filter_list_returns_unchanged(self):
        out = apply_disclosure_overlay([], DisclosureSettings(), 80)
        assert out == []

    def test_input_stream_is_carried_through(self):
        filters = ["[v_subtitle_3]copy[v_out]"]
        out = apply_disclosure_overlay(filters, DisclosureSettings(), 80)
        # The disclosure drawtext takes the same input stream that the copy did.
        assert out[-1].startswith("[v_subtitle_3]drawtext=")

    def test_custom_disclosure_text_is_burned(self):
        filters = ["[v_sub_1]copy[v_out]"]
        out = apply_disclosure_overlay(filters, DisclosureSettings(text="#publi"), 80)
        assert "text='#publi'" in out[-1]
