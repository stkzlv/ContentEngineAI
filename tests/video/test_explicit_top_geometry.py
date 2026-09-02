"""The geometry reported for an explicit top is where the content lands.

`apply_aspect_ratio_mode` returns a `VisualGeometry` that the FFmpeg engine's
content-aware captions are placed from. With `video_top_percent` set, the filter
puts the content's top row at `int(target_height * percent)`: both `pad`'s y
and `overlay`'s y are that literal. The geometry used to add a centring term
inside the band, so for a 1920x1080 source in a 1080x1920 frame at 0.25 the
frame had the content at row 480 and the geometry said 1136.

The centred branch was never wrong, and every bundled profile centres, which
is why nothing showed. A fork setting `video_vertical_align: top` got captions
positioned against a band the content was not in (#343).

The rendered frame decides, not the arithmetic. A unit assertion that the
geometry equals `int(target_height * percent)` would also pass if the filter
were changed to centre and the geometry left alone, which is the other way to
make the two agree, and is not what the config documents.
"""

from __future__ import annotations

import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from src.video.assembler.visual_builder import VisualFilterBuilder

TARGET_W, TARGET_H = 1080, 1920
SOURCE_W, SOURCE_H = 1920, 1080
TOP = 0.25


def _apply(mode: str, *, top: float | None, band: int = TARGET_H):
    builder = VisualFilterBuilder.__new__(VisualFilterBuilder)
    builder.config = MagicMock()
    builder.config.aspect_ratio = {"smart_scale_tolerance": 0.10}
    return builder.apply_aspect_ratio_mode(
        "[0:v]",
        mode,
        TARGET_W,
        TARGET_H,
        SOURCE_W,
        SOURCE_H,
        output_label="[v0_scaled]",
        target_content_height=band,
        video_top_percent=top,
    )


@pytest.mark.unit
class TestTheArithmetic:
    @pytest.mark.parametrize("mode", ["letterbox", "blur-fill"])
    @pytest.mark.parametrize("band", [TARGET_H, 960])
    def test_the_top_is_the_percent_of_the_frame(self, mode, band):
        """Whatever the band height, the filter's y is the same literal."""
        _, _, geometry = _apply(mode, top=TOP, band=band)

        assert geometry is not None
        assert geometry.rendered_y == int(TARGET_H * TOP)

    @pytest.mark.parametrize("mode", ["letterbox", "blur-fill"])
    def test_centring_is_unchanged(self, mode):
        _, _, geometry = _apply(mode, top=None)

        assert geometry is not None
        assert geometry.rendered_y == (TARGET_H - geometry.rendered_h) // 2


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestTheFrameItActuallyProduces:
    """Issue #343's acceptance bar: the content band's first row on a frame."""

    @staticmethod
    def _render(tmp_path, mode, *, top, band):
        source = tmp_path / "source.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=white:size={SOURCE_W}x{SOURCE_H}:duration=1:rate=5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            check=True,
            capture_output=True,
        )
        filter_string, label, geometry = _apply(mode, top=top, band=band)
        frame = tmp_path / f"{mode}-{top}-{band}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-filter_complex",
                f"{filter_string}{label}",
                "-map",
                label,
                "-frames:v",
                "1",
                str(frame),
            ],
            check=True,
            capture_output=True,
        )
        return frame, geometry

    @staticmethod
    def _content_rows(frame) -> tuple[int, int]:
        """First and last row that is the source's white.

        Letterbox surrounds the content with black; blur-fill with a blurred,
        darkened copy of it, which for a white source is a flat grey. Neither
        reaches the source's own white, so one predicate serves both modes.
        """
        from PIL import Image

        image = Image.open(frame).convert("RGB")
        width, height = image.size
        x = width // 2
        rows = [y for y in range(height) if sum(image.getpixel((x, y))) >= 740]
        assert rows, "no content rows found"
        return rows[0], rows[-1]

    @pytest.mark.parametrize("mode", ["letterbox", "blur-fill"])
    @pytest.mark.parametrize("band", [TARGET_H, 960])
    def test_the_content_band_starts_at_rendered_y(self, tmp_path, mode, band):
        frame, geometry = self._render(tmp_path, mode, top=TOP, band=band)

        assert geometry is not None
        first, last = self._content_rows(frame)
        assert first == geometry.rendered_y
        assert last == pytest.approx(geometry.rendered_y + geometry.rendered_h, abs=2)

    @pytest.mark.parametrize("mode", ["letterbox", "blur-fill"])
    def test_the_centred_band_still_starts_at_rendered_y(self, tmp_path, mode):
        """The branch that was right stays right."""
        frame, geometry = self._render(tmp_path, mode, top=None, band=TARGET_H)

        assert geometry is not None
        first, _ = self._content_rows(frame)
        assert first == pytest.approx(geometry.rendered_y, abs=1)
