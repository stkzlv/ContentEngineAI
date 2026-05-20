"""Tests for the Phase 1.2c hook overlay (closes #102)."""

from __future__ import annotations

import pytest

from src.video.assembler.overlay_builder import (
    apply_hook_overlay,
    build_hook_drawtext,
    extract_hook_line,
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


class TestBuildHookDrawtext:
    def test_filter_shape(self) -> None:
        settings = HookOverlaySettings(enabled=True, duration_sec=1.5)
        out = build_hook_drawtext(
            settings,
            "Best earbuds under fifty",
            subtitle_font_size_pixels=72,
            input_stream="[v_sub]",
            output_stream="[v_hook]",
        )
        assert out.startswith("[v_sub]drawtext=")
        assert out.endswith("[v_hook]")
        assert "Best earbuds under fifty" in out
        # 72 * 1.35 = 97.2 -> 97
        assert "fontsize=97" in out
        # Centre-horizontal
        assert "x=(w-text_w)/2" in out
        # time-gated
        assert "enable='between(t,0,1.500)'" in out

    def test_disabled_background_skips_box(self) -> None:
        settings = HookOverlaySettings(enabled=True, background_enabled=False)
        out = build_hook_drawtext(
            settings,
            "hook line",
            subtitle_font_size_pixels=60,
            input_stream="[in]",
            output_stream="[out]",
        )
        assert "box=1" not in out

    def test_enabled_background_includes_box(self) -> None:
        settings = HookOverlaySettings(enabled=True, background_enabled=True)
        out = build_hook_drawtext(
            settings,
            "hook line",
            subtitle_font_size_pixels=60,
            input_stream="[in]",
            output_stream="[out]",
        )
        assert "box=1:boxcolor=black@0.5" in out

    def test_fontsize_floor_at_8(self) -> None:
        """Even with tiny narration size + low size_factor, never go below 8px."""
        settings = HookOverlaySettings(enabled=True, size_factor=1.0)
        out = build_hook_drawtext(
            settings,
            "x",
            subtitle_font_size_pixels=4,  # ridiculous low; clamp expected
            input_stream="[in]",
            output_stream="[out]",
        )
        assert "fontsize=8" in out


class TestApplyHookOverlay:
    def test_disabled_returns_unchanged(self) -> None:
        settings = HookOverlaySettings(enabled=False)
        filters = ["a;b", "stream_xcopy[v_out]"]
        assert apply_hook_overlay(filters, settings, "hook", 60) is filters

    def test_empty_text_returns_unchanged(self) -> None:
        settings = HookOverlaySettings(enabled=True)
        filters = ["a", "stream_xcopy[v_out]"]
        assert apply_hook_overlay(filters, settings, "", 60) is filters

    def test_empty_filters_returns_unchanged(self) -> None:
        settings = HookOverlaySettings(enabled=True)
        assert apply_hook_overlay([], settings, "hook", 60) == []

    def test_unexpected_terminal_returns_unchanged(self, caplog) -> None:
        settings = HookOverlaySettings(enabled=True)
        filters = ["a", "stream_xnocopy[v_other]"]
        out = apply_hook_overlay(filters, settings, "hook", 60)
        assert out is filters
        assert any("unexpected shape" in r.message for r in caplog.records)

    def test_rewrite_preserves_terminal_copy(self) -> None:
        """Hook layer must keep copy[v_out] alive for the disclosure layer."""
        settings = HookOverlaySettings(enabled=True)
        filters = ["scaled;padded[v_sub_3];", "[v_sub_3]copy[v_out]"]
        out = apply_hook_overlay(filters, settings, "Hook line", 60)
        # Original list extended by one; terminal still copy[v_out]
        assert len(out) == 3
        assert out[-1] == "[v_hook]copy[v_out]"
        # Hook drawtext sits in position -2
        assert out[-2].startswith("[v_sub_3]drawtext=")
        assert out[-2].endswith("[v_hook]")
        # Time-gated to duration_sec
        assert "enable='between(t,0,1.500)'" in out[-2]


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
