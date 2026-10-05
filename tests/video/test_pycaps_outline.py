"""A black outline round each caption word (REQ-VID-158), held off.

pycaps draws template CSS pixels at 2 x height / 1280 frame pixels, and a
stroke painted under the fill shows half its width outside the glyph, so a
6-pixel outline on a 1920-pixel frame is a 4-pixel CSS stroke.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.video.pycaps_engine import renderer
from tests.video.test_pycaps_sentence_case import _build


def _added_css(builder) -> list[str]:
    return [a[0] for n, a in builder.calls if n == "add_css_content"]


@pytest.mark.req("REQ-VID-158")
def test_the_outline_width_is_in_frame_pixels() -> None:
    css = renderer.outline_css(6, 1920)

    assert "-webkit-text-stroke: 4.000px #000" in css
    assert "paint-order: stroke fill" in css and "text-shadow: none" in css
    # Half the frame height, half the scale: the same CSS width doubles.
    assert "-webkit-text-stroke: 8.000px" in renderer.outline_css(6, 960)


@pytest.mark.req("REQ-VID-158")
def test_the_build_appends_the_outline_only_when_set(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(renderer, "_frame_height", lambda video: 1920)

    on = _build(monkeypatch, tmp_path, outline_px=6)
    off = _build(monkeypatch, tmp_path, outline_px=0)

    assert any("-webkit-text-stroke: 4.000px" in css for css in _added_css(on))
    assert not any("text-stroke" in css for css in _added_css(off))


def test_the_outline_lands_after_sentence_case(monkeypatch, tmp_path) -> None:
    """Both rules set `.word`; the outline's `text-shadow: none` must win."""
    monkeypatch.setattr(renderer, "_frame_height", lambda video: 1920)

    css = _added_css(
        _build(monkeypatch, tmp_path, outline_px=6, force_sentence_case=True)
    )

    assert "text-transform" in css[0] and "text-stroke" in css[1]


def test_pictex_keeps_the_template_edge(monkeypatch, tmp_path, caplog) -> None:
    builder = _build(monkeypatch, tmp_path, outline_px=6, renderer="pictex")

    assert not any("text-stroke" in css for css in _added_css(builder))
    assert "needs the css renderer" in caplog.text


def test_an_unreadable_video_falls_back_to_the_bundled_height(tmp_path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    assert renderer._frame_height(tmp_path / "missing.mp4") == 1920
    assert renderer._frame_height(broken) == 1920


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_the_probe_reads_a_real_height(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i"]
        + ["color=c=black:s=360x640:d=0.2", str(clip)],
        check=True,
    )

    assert renderer._frame_height(clip) == 640


def test_the_scale_matches_the_installed_pycaps() -> None:
    pytest.importorskip("pycaps", reason="optional group not installed")
    from pycaps.renderer.css_subtitle_renderer import CssSubtitleRenderer

    expected = 2.0 * 1920 / CssSubtitleRenderer.REFERENCE_VIDEO_HEIGHT
    assert renderer._css_scale(1920) == pytest.approx(expected)
