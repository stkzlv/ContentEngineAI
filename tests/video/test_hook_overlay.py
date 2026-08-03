"""Tests for the Phase 1.2c hook overlay (closes #102, #160)."""

from __future__ import annotations

import pytest

from src.video.assembler.overlay_builder import (
    _escape_drawtext_textfile,
    _estimate_hook_text_width,
    _fit_hook_lines,
    apply_hook_overlay,
    build_hook_drawtext,
    extract_hook_line,
    resolve_hook_line,
)
from src.video.config.visual_models import HookOverlaySettings


class TestExtractHookLine:
    def test_first_sentence_period(self) -> None:
        text = "This $15 hub replaced my $200 one. The second line."
        assert extract_hook_line(text, 12) == "This $15 hub replaced my $200 one"

    def test_first_sentence_exclamation(self) -> None:
        text = "Best earbuds under $50! Trust me on this."
        assert extract_hook_line(text, 12) == "Best earbuds under $50"

    def test_first_sentence_question(self) -> None:
        text = "USB-C or Lightning? Pick one."
        assert extract_hook_line(text, 12) == "USB-C or Lightning"

    def test_truncates_to_max_words(self) -> None:
        text = "One two three four five six seven eight nine ten eleven."
        out = extract_hook_line(text, 5)
        assert out == "One two three four five..."

    def test_empty_returns_empty(self) -> None:
        assert extract_hook_line("", 7) == ""
        assert extract_hook_line("   \n  ", 7) == ""

    def test_no_terminator_returns_whole_text(self) -> None:
        text = "Short line with no period"
        assert extract_hook_line(text, 10) == "Short line with no period"


class TestResolveHookLine:
    """The authored headline wins over first-sentence extraction (#160, 1.9)."""

    def test_headline_wins_verbatim(self) -> None:
        # A headline is used as-is, not re-extracted, even if it looks like prose.
        out = resolve_hook_line(
            "This $15 hub replaced my $200 one",
            "Some totally different spoken first sentence. And more.",
            7,
        )
        assert out == "This $15 hub replaced my $200 one"

    def test_falls_back_to_extraction_without_headline(self) -> None:
        out = resolve_hook_line(None, "First sentence here. Second one.", 7)
        assert out == "First sentence here"

    def test_empty_headline_falls_back(self) -> None:
        out = resolve_hook_line("", "First sentence here. Second.", 7)
        assert out == "First sentence here"

    def test_neither_returns_empty(self) -> None:
        assert resolve_hook_line(None, None, 7) == ""
        assert resolve_hook_line("", "", 7) == ""


class TestBuildHookDrawtext:
    def test_filter_shape(self, tmp_path) -> None:
        settings = HookOverlaySettings(enabled=True, duration_sec=1.5)
        out = build_hook_drawtext(
            settings,
            "Best earbuds",  # short: stays a single line at 1080px
            subtitle_font_size_pixels=72,
            frame_width=1080,
            temp_dir=tmp_path,
            input_stream="[v_sub]",
            output_stream="[v_hook]",
        )
        assert out.startswith("[v_sub]drawtext=")
        assert out.endswith("[v_hook]")
        # Text lives in a textfile, not inline (inline apostrophes corrupt the
        # multi-filter chain), so the line content is on disk, not in `out`.
        assert "textfile=" in out
        assert (tmp_path / "hook_line_0.txt").read_text() == "Best earbuds"
        # single line, no intermediate stream
        assert out.count("drawtext=") == 1
        # 72 * 1.35 = 97.2 -> 97
        assert "fontsize=97" in out
        # Centre-horizontal
        assert "x=(w-text_w)/2" in out
        # time-gated
        assert "enable=between(t\\,0\\,1.500)" in out

    def test_disabled_background_skips_box(self, tmp_path) -> None:
        settings = HookOverlaySettings(enabled=True, background_enabled=False)
        out = build_hook_drawtext(
            settings,
            "hook line",
            subtitle_font_size_pixels=60,
            frame_width=1080,
            temp_dir=tmp_path,
            input_stream="[in]",
            output_stream="[out]",
        )
        assert "box=1" not in out

    def test_enabled_background_includes_box(self, tmp_path) -> None:
        settings = HookOverlaySettings(enabled=True, background_enabled=True)
        out = build_hook_drawtext(
            settings,
            "hook line",
            subtitle_font_size_pixels=60,
            frame_width=1080,
            temp_dir=tmp_path,
            input_stream="[in]",
            output_stream="[out]",
        )
        assert "box=1:boxcolor=black@0.5" in out

    def test_fontsize_floor_at_8(self, tmp_path) -> None:
        """Even with tiny narration size + low size_factor, never go below 8px."""
        settings = HookOverlaySettings(enabled=True, size_factor=1.0)
        out = build_hook_drawtext(
            settings,
            "x",
            subtitle_font_size_pixels=4,  # ridiculous low; clamp expected
            frame_width=1080,
            temp_dir=tmp_path,
            input_stream="[in]",
            output_stream="[out]",
        )
        assert "fontsize=8" in out


class TestTextfileEscaping:
    r"""textfile= drops the quoting layer but NOT drawtext's text expansion.

    Verified against real renders: an unescaped ``%`` makes FFmpeg log
    ``Stray %``, exit 0, and draw nothing at all for that line; an unescaped
    ``\`` is silently swallowed. Both must be escaped in the file contents.
    """

    def test_percent_is_escaped(self) -> None:
        assert _escape_drawtext_textfile("Save 40% on this hub") == (
            r"Save 40\% on this hub"
        )

    def test_backslash_is_escaped(self) -> None:
        assert _escape_drawtext_textfile("a \\ b") == "a \\\\ b"

    def test_backslash_escaped_before_percent(self) -> None:
        # Order matters: doubling the backslash first keeps the marker added
        # for % from being re-escaped into a literal backslash.
        assert _escape_drawtext_textfile("100%\\") == "100\\%\\\\"

    def test_quotes_and_colons_pass_through(self) -> None:
        # These are exactly what textfile= exists to avoid escaping.
        assert _escape_drawtext_textfile("you're set: really") == "you're set: really"

    def test_written_line_is_escaped(self, tmp_path) -> None:
        settings = HookOverlaySettings(enabled=True)
        build_hook_drawtext(
            settings,
            "Save 40% here",
            subtitle_font_size_pixels=72,
            frame_width=1080,
            temp_dir=tmp_path,
            input_stream="[in]",
            output_stream="[out]",
        )
        assert (tmp_path / "hook_line_0.txt").read_text() == r"Save 40\% here"


class TestHookFit:
    """Long hooks wrap to <=2 lines and shrink to fit the frame (#160)."""

    def test_long_hook_wraps_to_two_lines(self, tmp_path) -> None:
        settings = HookOverlaySettings(enabled=True)
        out = build_hook_drawtext(
            settings,
            "This cheap gadget quietly replaced my whole expensive desk setup",
            subtitle_font_size_pixels=72,
            frame_width=1080,
            temp_dir=tmp_path,
            input_stream="[v_sub]",
            output_stream="[v_hook]",
        )
        # Two stacked drawtexts chained through an intermediate stream.
        assert out.count("drawtext=") == 2
        assert "[v_hkl1]" in out
        assert out.startswith("[v_sub]drawtext=")
        assert out.endswith("[v_hook]")
        # Each line is written to its own textfile.
        assert (tmp_path / "hook_line_0.txt").exists()
        assert (tmp_path / "hook_line_1.txt").exists()
        # The second line carries a vertical offset (base + line height).
        assert "+" in out.split(";")[1]

    def test_fit_lines_stay_within_frame(self) -> None:
        settings = HookOverlaySettings(enabled=True)
        frame_width = 1080
        max_px = int(frame_width * settings.max_width_fraction)
        lines, font_size = _fit_hook_lines(
            "This cheap gadget quietly replaced my whole desk setup",
            97,
            frame_width,
            settings,
        )
        assert 1 <= len(lines) <= settings.max_lines
        assert all(
            _estimate_hook_text_width(line, font_size) <= max_px for line in lines
        )

    def test_fit_shrinks_font_when_unwrappable(self) -> None:
        settings = HookOverlaySettings(enabled=True)
        # A single very wide token can't be wrapped, so the font must shrink.
        lines, font_size = _fit_hook_lines("W" * 60, 97, 1080, settings)
        assert font_size < 97
        assert len(lines) == 1

    def test_short_hook_unchanged(self) -> None:
        settings = HookOverlaySettings(enabled=True)
        lines, font_size = _fit_hook_lines("Best earbuds", 97, 1080, settings)
        assert lines == ["Best earbuds"]
        assert font_size == 97

    def test_floor_truncation_is_marked_and_logged(self, caplog) -> None:
        """Dropping overflow at the font floor must be visible, not silent."""
        settings = HookOverlaySettings(enabled=True, max_lines=1)
        # A frame too narrow to fit two words even at 8px forces the drop.
        with caplog.at_level("WARNING"):
            lines, font_size = _fit_hook_lines(
                "alpha bravo charlie delta echo", 97, 40, settings
            )
        assert len(lines) == 1
        assert lines[0].endswith("...")
        assert font_size == 8
        assert "Hook overlay truncated" in caplog.text

    def test_no_ellipsis_when_everything_fits(self) -> None:
        settings = HookOverlaySettings(enabled=True)
        lines, _ = _fit_hook_lines("Best cheap hub wins", 97, 1080, settings)
        assert not any(line.endswith("...") for line in lines)


class TestApplyHookOverlay:
    def test_disabled_returns_unchanged(self, tmp_path) -> None:
        settings = HookOverlaySettings(enabled=False)
        filters = ["a;b", "stream_xcopy[v_out]"]
        assert (
            apply_hook_overlay(filters, settings, "hook", 60, 1080, tmp_path) is filters
        )

    def test_empty_text_returns_unchanged(self, tmp_path) -> None:
        settings = HookOverlaySettings(enabled=True)
        filters = ["a", "stream_xcopy[v_out]"]
        assert apply_hook_overlay(filters, settings, "", 60, 1080, tmp_path) is filters

    def test_empty_filters_returns_unchanged(self, tmp_path) -> None:
        settings = HookOverlaySettings(enabled=True)
        assert apply_hook_overlay([], settings, "hook", 60, 1080, tmp_path) == []

    def test_unexpected_terminal_returns_unchanged(self, tmp_path, caplog) -> None:
        settings = HookOverlaySettings(enabled=True)
        filters = ["a", "stream_xnocopy[v_other]"]
        out = apply_hook_overlay(filters, settings, "hook", 60, 1080, tmp_path)
        assert out is filters
        assert any("unexpected shape" in r.message for r in caplog.records)

    def test_rewrite_preserves_terminal_copy(self, tmp_path) -> None:
        """Hook layer must keep copy[v_out] alive for the disclosure layer."""
        settings = HookOverlaySettings(enabled=True)
        filters = ["scaled;padded[v_sub_3];", "[v_sub_3]copy[v_out]"]
        out = apply_hook_overlay(filters, settings, "Hook line", 60, 1080, tmp_path)
        # Original list extended by one; terminal still copy[v_out]
        assert len(out) == 3
        assert out[-1] == "[v_hook]copy[v_out]"
        # Hook drawtext sits in position -2
        assert out[-2].startswith("[v_sub_3]drawtext=")
        assert out[-2].endswith("[v_hook]")
        # Time-gated to duration_sec
        assert "enable=between(t\\,0\\,1.500)" in out[-2]


class TestHookPlusDisclosureStack:
    """Hook overlay must compose cleanly with the disclosure overlay rewrite.

    Order in core.py is: subtitle builder emits `<input>copy[v_out]`, hook
    rewrite produces `<input>drawtext_hook[v_hook];[v_hook]copy[v_out]`,
    disclosure rewrite then consumes the new terminal `copy[v_out]` and
    emits `[v_hook]drawtext_disclosure[v_out]`. Final shape is 3 entries:
    the original prefix, the hook drawtext, and the disclosure drawtext.
    """

    def test_hook_then_disclosure_produces_three_entries(self, tmp_path) -> None:
        from src.video.assembler.overlay_builder import (
            apply_disclosure_overlay,
            apply_hook_overlay,
        )
        from src.video.config.visual_models import DisclosureSettings

        hook = HookOverlaySettings(enabled=True)
        disclosure = DisclosureSettings(enabled=True)
        filters = ["scaled;padded[v_sub];", "[v_sub]copy[v_out]"]

        hooked = apply_hook_overlay(filters, hook, "first sentence", 60, 1080, tmp_path)
        final = apply_disclosure_overlay(hooked, disclosure, 60, tmp_path)

        # Three filter entries: prefix, hook drawtext, disclosure drawtext.
        assert len(final) == 3
        # Hook lives in slot 1, consuming the original [v_sub] and emitting [v_hook].
        assert final[1].startswith("[v_sub]drawtext=")
        assert final[1].endswith("[v_hook]")
        # Disclosure lives in slot 2, consuming [v_hook] and emitting [v_out].
        assert final[2].startswith("[v_hook]drawtext=")
        assert final[2].endswith("[v_out]")

    def test_disclosure_only_when_hook_disabled(self, tmp_path) -> None:
        from src.video.assembler.overlay_builder import (
            apply_disclosure_overlay,
            apply_hook_overlay,
        )
        from src.video.config.visual_models import DisclosureSettings

        hook = HookOverlaySettings(enabled=False)
        disclosure = DisclosureSettings(enabled=True)
        filters = ["prefix", "[v_sub]copy[v_out]"]

        hooked = apply_hook_overlay(filters, hook, "first sentence", 60, 1080, tmp_path)
        final = apply_disclosure_overlay(hooked, disclosure, 60, tmp_path)

        # Hook was disabled, so the filter list grew by zero. Disclosure
        # still rewrites the terminal copy[v_out] into a drawtext.
        assert len(final) == 2
        assert final[-1].startswith("[v_sub]drawtext=")
        assert final[-1].endswith("[v_out]")


class TestApostropheHandling:
    r"""The hook must survive an apostrophe in the first sentence.

    Inline ``text='...'`` corrupts silently inside the assembler's multi-filter
    ``-filter_complex`` chain: the ``'\''`` exit/reenter escape works on a lone
    ``-vf`` drawtext but not when another filter follows, where FFmpeg swallows
    the drawtext's own trailing args as literal text. A hook like "you're ..."
    then rendered as tiny garbage. The fix routes hook text through
    ``textfile=``, which needs no text escaping.
    """

    def test_disclosure_escaper_still_uses_exit_reenter(self) -> None:
        # The disclosure overlay still inlines its text (config-controlled,
        # e.g. "#ad", no apostrophe), so its escaper is unchanged.
        from src.video.assembler.overlay_builder import _escape_drawtext_text

        assert _escape_drawtext_text("you're") == r"you'\''re"

    def test_apostrophe_hook_uses_textfile(self, tmp_path) -> None:
        settings = HookOverlaySettings(enabled=True)
        out = build_hook_drawtext(
            settings,
            "you're trying",
            subtitle_font_size_pixels=72,
            frame_width=1080,
            temp_dir=tmp_path,
            input_stream="[v_sub]",
            output_stream="[v_hook]",
        )
        # No inline text at all; text is on disk with the literal apostrophe.
        assert "textfile=" in out
        assert "text='" not in out
        assert (tmp_path / "hook_line_0.txt").read_text() == "you're trying"


class TestVideoSettingsField:
    def test_default_has_disabled_hook_overlay(self) -> None:
        from src.video.config.visual_models import VideoSettings

        vs = VideoSettings(resolution=(1080, 1920), frame_rate=30)
        assert vs.hook_overlay.enabled is False
        assert vs.hook_overlay.duration_sec == 1.5

    def test_size_factor_bounds(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HookOverlaySettings(size_factor=0.5)
        with pytest.raises(ValidationError):
            HookOverlaySettings(size_factor=3.0)

    def test_duration_sec_bounds(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HookOverlaySettings(duration_sec=0.1)
        with pytest.raises(ValidationError):
            HookOverlaySettings(duration_sec=4.0)
