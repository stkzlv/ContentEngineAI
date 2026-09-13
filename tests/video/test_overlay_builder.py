"""Unit tests for the on-frame disclosure overlay builder."""

from pathlib import Path

from src.video.assembler.overlay_builder import (
    apply_disclosure_overlay,
    build_disclosure_drawtext,
)
from src.video.config.visual_models import DisclosureSettings


def _disclosure_text(temp_dir: Path) -> str:
    """Read back the text the builder wrote for the drawtext to render."""
    return (temp_dir / "disclosure_text.txt").read_text()


class TestBuildDisclosureDrawtext:
    """Test the single-filter drawtext builder."""

    def test_default_settings_produce_drawtext_filter(self, tmp_path):
        settings = DisclosureSettings()
        filter_str = build_disclosure_drawtext(
            settings,
            subtitle_font_size_pixels=80,
            temp_dir=tmp_path,
            input_stream="[v_in]",
            output_stream="[v_out]",
        )

        assert filter_str.startswith("[v_in]drawtext=")
        assert filter_str.endswith("[v_out]")
        assert "textfile=" in filter_str
        assert _disclosure_text(tmp_path) == "#ad"
        # 80 * 0.45 = 36
        assert "fontsize=36" in filter_str
        assert "fontcolor=white" in filter_str
        assert "bordercolor=black" in filter_str

    def test_custom_text_appears_in_filter(self, tmp_path):
        settings = DisclosureSettings(text="#publi")
        build_disclosure_drawtext(settings, 80, tmp_path, "[v_in]", "[v_out]")
        assert _disclosure_text(tmp_path) == "#publi"

    def test_size_factor_scales_font(self, tmp_path):
        settings = DisclosureSettings(size_factor=0.5)
        filter_str = build_disclosure_drawtext(
            settings, 100, tmp_path, "[v_in]", "[v_out]"
        )
        assert "fontsize=50" in filter_str

    def test_top_right_position(self, tmp_path):
        settings = DisclosureSettings(
            position="top-right",
            margin_x_percent=0.04,
            margin_y_percent=0.12,
        )
        filter_str = build_disclosure_drawtext(
            settings, 80, tmp_path, "[v_in]", "[v_out]"
        )
        assert "x=w-text_w-w*0.04" in filter_str
        assert "y=h*0.12" in filter_str

    def test_top_left_position(self, tmp_path):
        settings = DisclosureSettings(position="top-left")
        filter_str = build_disclosure_drawtext(
            settings, 80, tmp_path, "[v_in]", "[v_out]"
        )
        assert "x=w*" in filter_str  # left margin
        assert "y=h*" in filter_str  # top margin
        assert "w-text_w" not in filter_str

    def test_bottom_right_position(self, tmp_path):
        settings = DisclosureSettings(position="bottom-right")
        filter_str = build_disclosure_drawtext(
            settings, 80, tmp_path, "[v_in]", "[v_out]"
        )
        assert "x=w-text_w-" in filter_str
        assert "y=h-text_h-" in filter_str

    def test_bottom_left_position(self, tmp_path):
        settings = DisclosureSettings(position="bottom-left")
        filter_str = build_disclosure_drawtext(
            settings, 80, tmp_path, "[v_in]", "[v_out]"
        )
        # x is plain margin, y subtracts text_h
        assert "y=h-text_h-" in filter_str

    def test_background_disabled_omits_box(self, tmp_path):
        settings = DisclosureSettings(background_enabled=False)
        filter_str = build_disclosure_drawtext(
            settings, 80, tmp_path, "[v_in]", "[v_out]"
        )
        assert "box=1" not in filter_str

    def test_background_enabled_includes_box(self, tmp_path):
        settings = DisclosureSettings(
            background_enabled=True,
            background_color="black@0.5",
        )
        filter_str = build_disclosure_drawtext(
            settings, 80, tmp_path, "[v_in]", "[v_out]"
        )
        assert "box=1" in filter_str
        assert "boxcolor=black@0.5" in filter_str

    def test_expansion_metachars_escaped_in_textfile(self, tmp_path):
        # textfile= removes the filtergraph quoting problem (colons and
        # apostrophes pass through untouched), but drawtext still runs text
        # expansion over the file, so percent and backslash stay special. An
        # unescaped % drops the whole line from the render.
        settings = DisclosureSettings(text="A: it's 50% off")
        filter_str = build_disclosure_drawtext(
            settings, 80, tmp_path, "[v_in]", "[v_out]"
        )
        assert "text='" not in filter_str
        assert _disclosure_text(tmp_path) == r"A: it's 50\% off"

    def test_minimum_font_size_floor(self, tmp_path):
        # Even tiny subtitle base sizes shouldn't produce illegible fonts.
        # 30px subtitle * 0.2 size_factor = 6px without the floor; floor lifts
        # to 8px so the disclosure stays readable on phone screens.
        settings = DisclosureSettings(size_factor=0.2)
        filter_str = build_disclosure_drawtext(
            settings, 30, tmp_path, "[v_in]", "[v_out]"
        )
        assert "fontsize=8" in filter_str


class TestApplyDisclosureOverlay:
    """Test injection into the existing video filter chain."""

    def test_replaces_terminal_copy_with_drawtext(self, tmp_path):
        # Subtitle builder always emits a final "...copy[v_out]" no-op.
        filters = [
            "[0:v]scale=1080:1920[v0]",
            "[v0]copy[v_sub_1]",
            "[v_sub_1]copy[v_out]",
        ]
        out = apply_disclosure_overlay(filters, DisclosureSettings(), 80, tmp_path)

        assert len(out) == len(filters)
        assert out[:-1] == filters[:-1]  # earlier filters unchanged
        assert out[-1].endswith("[v_out]")
        assert "drawtext=" in out[-1]
        assert "copy[v_out]" not in out[-1]

    def test_disabled_returns_unchanged_chain(self, tmp_path):
        filters = ["[v_sub_1]copy[v_out]"]
        out = apply_disclosure_overlay(
            filters, DisclosureSettings(enabled=False), 80, tmp_path
        )
        assert out is filters or out == filters
        assert out[-1] == "[v_sub_1]copy[v_out]"

    def test_terminal_not_producing_v_out_is_logged_and_skipped(self, tmp_path):
        # Only a terminal that doesn't produce [v_out] at all is unrecoverable.
        # Anything else can be re-pointed, so leaving the chain alone is
        # reserved for shapes we genuinely can't reason about.
        filters = ["[v0]something_else[v_other]"]
        out = apply_disclosure_overlay(filters, DisclosureSettings(), 80, tmp_path)
        assert out == filters

    def test_content_aware_ass_terminal_gets_the_overlay(self, tmp_path):
        # The ffmpeg content-aware subtitle path ends with an ass= filter that
        # produces [v_out] directly, leaving no copy no-op to rewrite. The
        # disclosure is a required on-frame surface, so it must still land.
        filters = [
            "[0:v]scale=1080:1920[v0]",
            "[v0]ass='/tmp/subtitles_content_aware.ass'[v_out]",
        ]
        out = apply_disclosure_overlay(filters, DisclosureSettings(), 80, tmp_path)

        assert "drawtext=" in out[-1]
        assert out[-1].endswith("[v_out]")
        # The ass filter is preserved, re-pointed at the intermediate label.
        assert "ass='/tmp/subtitles_content_aware.ass'" in out[1]
        assert out[1].endswith("[v_pre_overlay]")
        # And the drawtext consumes that label, so the graph stays connected.
        assert out[-1].startswith("[v_pre_overlay]drawtext=")

    def test_normalization_is_idempotent_on_copy_terminal(self, tmp_path):
        # A chain that already ends in the no-op must not grow an extra entry.
        filters = ["[v0]scale=2:2[v1]", "[v1]copy[v_out]"]
        out = apply_disclosure_overlay(filters, DisclosureSettings(), 80, tmp_path)
        assert len(out) == len(filters)

    def test_empty_filter_list_returns_unchanged(self, tmp_path):
        out = apply_disclosure_overlay([], DisclosureSettings(), 80, tmp_path)
        assert out == []

    def test_input_stream_is_carried_through(self, tmp_path):
        filters = ["[v_subtitle_3]copy[v_out]"]
        out = apply_disclosure_overlay(filters, DisclosureSettings(), 80, tmp_path)
        # The disclosure drawtext takes the same input stream that the copy did.
        assert out[-1].startswith("[v_subtitle_3]drawtext=")

    def test_custom_disclosure_text_is_burned(self, tmp_path):
        filters = ["[v_sub_1]copy[v_out]"]
        apply_disclosure_overlay(
            filters, DisclosureSettings(text="#publi"), 80, tmp_path
        )
        assert _disclosure_text(tmp_path) == "#publi"

    def test_skipped_overlay_writes_no_file(self, tmp_path):
        # A disabled or unmatched chain returns before touching the disk.
        apply_disclosure_overlay(
            ["[v_sub_1]copy[v_out]"], DisclosureSettings(enabled=False), 80, tmp_path
        )
        assert not (tmp_path / "disclosure_text.txt").exists()
