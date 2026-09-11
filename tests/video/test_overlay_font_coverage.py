"""Overlay text in a non-Latin script renders or refuses loudly (#392).

Drawtext takes one face and has no fallback chain, so a script the default
face does not carry rendered as empty notdef boxes at full width, silently,
with ffmpeg exiting 0. The resolver asks fontconfig for a face covering
every glyph; config validation refuses operator text nothing installed can
draw.
"""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video.assembler.font_resolver import (
    OverlayFontError,
    fontfile_for_text,
)

CJK_SAMPLE = "个人资料链接查看更多优惠信息"

# A codepoint from an unassigned plane: no font anywhere covers it, which
# makes the loud-failure branch deterministic on every box.
UNCOVERABLE = "\U0003fffd"


def _fc_available() -> bool:
    return shutil.which("fc-list") is not None


def _cjk_covered() -> bool:
    if not _fc_available():
        return False
    out = subprocess.run(
        ["fc-list", ":charset=4e2a", "file"], capture_output=True, text=True
    )
    return bool(out.stdout.strip())


class TestTheResolver:
    def test_ascii_needs_no_fontfile(self):
        assert fontfile_for_text("#ad", strict=True) is None
        assert fontfile_for_text("https://lnk.bio/x", strict=True) is None

    @pytest.mark.skipif(not _cjk_covered(), reason="no CJK font installed")
    def test_cjk_resolves_to_a_covering_face(self):
        path = fontfile_for_text(CJK_SAMPLE, strict=True)
        assert path is not None and path.exists()

    @pytest.mark.skipif(not _fc_available(), reason="fontconfig not installed")
    def test_uncoverable_text_refuses_loudly_in_strict_mode(self):
        with pytest.raises(OverlayFontError) as excinfo:
            fontfile_for_text(f"deal {UNCOVERABLE}", strict=True)
        # The message reprs the glyphs, so an unprintable one appears as
        # its escape -- which is the readable form for exactly this case.
        assert repr(UNCOVERABLE).strip("'") in str(
            excinfo.value
        ), "the missing glyph is named"

    @pytest.mark.skipif(not _fc_available(), reason="fontconfig not installed")
    def test_uncoverable_text_degrades_at_render_time(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            assert fontfile_for_text(f"deal {UNCOVERABLE}", strict=False) is None
        assert any("No installed scalable font" in r.message for r in caplog.records)


@pytest.mark.skipif(not _cjk_covered(), reason="no CJK font installed")
class TestTheBuildersCarryTheFace:
    def test_the_disclosure_filter_names_a_fontfile(self, tmp_path):
        from src.video.assembler.overlay_builder import build_disclosure_drawtext
        from src.video.config.visual_models import DisclosureSettings

        settings = DisclosureSettings(enabled=True, text=CJK_SAMPLE)
        filt = build_disclosure_drawtext(settings, 40, tmp_path, "[0:v]", "[v_out]")
        assert "fontfile=" in filt

    def test_an_ascii_disclosure_stays_on_the_default_face(self, tmp_path):
        from src.video.assembler.overlay_builder import build_disclosure_drawtext
        from src.video.config.visual_models import DisclosureSettings

        settings = DisclosureSettings(enabled=True, text="#ad")
        filt = build_disclosure_drawtext(settings, 40, tmp_path, "[0:v]", "[v_out]")
        assert "fontfile=" not in filt

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
    def test_the_filter_renders(self, tmp_path):
        """The frame is the evidence, not the filter string: the filter
        with the resolved face must at least be accepted by ffmpeg; the
        glyph content was verified from a rendered frame when this shipped.
        """
        from src.video.assembler.overlay_builder import build_disclosure_drawtext
        from src.video.config.visual_models import DisclosureSettings

        settings = DisclosureSettings(enabled=True, text=CJK_SAMPLE)
        filt = build_disclosure_drawtext(settings, 40, tmp_path, "[0:v]", "[v_out]")
        out = tmp_path / "frame.png"
        r = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=gray:size=1080x400:d=1",
                "-filter_complex",
                filt,
                "-map",
                "[v_out]",
                "-frames:v",
                "1",
                "-y",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr[-300:]
        assert out.exists() and out.stat().st_size > 0


class TestConfigValidationRefusesTheUndrawable:
    def _config(
        self,
        disclosure_text: str,
        *,
        upper: SimpleNamespace | None = None,
    ) -> SimpleNamespace:
        merged = SimpleNamespace(
            video_settings=SimpleNamespace(
                upper_line=upper
                or SimpleNamespace(enabled=False, source="custom", custom_text="")
            )
        )
        return SimpleNamespace(
            video_settings=SimpleNamespace(
                disclosure_overlay=SimpleNamespace(enabled=True, text=disclosure_text),
            ),
            video_profiles={"p1": object()},
            get_profile_merged_settings=lambda name, overrides: merged,
        )

    @pytest.mark.skipif(not _fc_available(), reason="fontconfig not installed")
    def test_an_undrawable_disclosure_is_a_config_error(self):
        from src.video.config_validator import VideoConfigValidator

        errors = VideoConfigValidator()._validate_overlay_glyph_coverage(
            self._config(f"pub {UNCOVERABLE}")
        )
        assert errors, "an undrawable operator string must refuse at load"
        assert "disclosure_overlay.text" in errors[0]

    def test_a_drawable_disclosure_passes(self):
        from src.video.config_validator import VideoConfigValidator

        errors = VideoConfigValidator()._validate_overlay_glyph_coverage(
            self._config("#publi")
        )
        assert errors == []

    @pytest.mark.skipif(not _fc_available(), reason="fontconfig not installed")
    def test_the_merged_profile_shape_is_what_counts(self):
        """A profile can enable an upper line the base leaves off, so the
        validator reads the MERGED settings -- the flat base said disabled
        and the old check silently passed the undrawable text (#392 review).
        """
        from src.video.config_validator import VideoConfigValidator

        upper = SimpleNamespace(
            enabled=True, source="custom", custom_text=f"deal {UNCOVERABLE}"
        )
        errors = VideoConfigValidator()._validate_overlay_glyph_coverage(
            self._config("#ad", upper=upper)
        )
        assert errors and "p1" in errors[0]

    @pytest.mark.skipif(not _fc_available(), reason="fontconfig not installed")
    def test_custom_text_is_ignored_when_the_source_is_not_custom(self):
        """`custom_text` never renders under another source, so undrawable
        text there must not refuse a config whose drawn text is fine.
        """
        from src.video.config_validator import VideoConfigValidator

        upper = SimpleNamespace(
            enabled=True,
            source="affiliate_link",
            custom_text=f"deal {UNCOVERABLE}",
        )
        errors = VideoConfigValidator()._validate_overlay_glyph_coverage(
            self._config("#ad", upper=upper)
        )
        assert errors == []


class TestTheReviewCases:
    """The two shapes pass 1 demonstrated failing (#392 review)."""

    @pytest.mark.skipif(not _fc_available(), reason="fontconfig not installed")
    def test_a_bitmap_only_face_is_never_chosen(self):
        """Noto Color Emoji cannot be sized by drawtext -- choosing it
        aborts the whole render. Bitmap coverage counts as no coverage, so
        this either resolves a scalable face or degrades to the default.
        """
        # Pure emoji: with any ASCII beside it the joint query already
        # fails on this box, so only the emoji-alone shape can reach a
        # bitmap face and prove the scalable filter.
        result = fontfile_for_text("🔥🔥", strict=False)
        if result is not None:
            out = subprocess.run(
                ["fc-list", ":charset=1f525:outline=true:color=false", "file"],
                capture_output=True,
                text=True,
            ).stdout
            assert str(result) in out, "a non-scalable face was chosen"

    @pytest.mark.skipif(not _fc_available(), reason="fontconfig not installed")
    def test_a_mixed_script_face_covers_the_latin_half_too(self):
        """The whole string draws with one face, so a face chosen for its
        Devanagari alone turned the Latin half into boxes -- the query
        carries every codepoint now.
        """
        result = fontfile_for_text("ऑफ़र deals link in bio", strict=False)
        if result is not None:
            out = subprocess.run(
                ["fc-list", ":charset=61 911:outline=true:color=false", "file"],
                capture_output=True,
                text=True,
            ).stdout
            assert str(result) in out, "the chosen face does not cover Latin"
