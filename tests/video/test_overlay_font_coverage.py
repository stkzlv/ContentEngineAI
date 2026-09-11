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
        assert any("No installed font" in r.message for r in caplog.records)


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
    def _config(self, disclosure_text: str) -> SimpleNamespace:
        return SimpleNamespace(
            video_settings=SimpleNamespace(
                disclosure_overlay=SimpleNamespace(enabled=True, text=disclosure_text),
                upper_line=SimpleNamespace(enabled=False, custom_text=""),
            ),
            video_profiles={},
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
