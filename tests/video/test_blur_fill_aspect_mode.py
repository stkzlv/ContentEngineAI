"""A landscape source fills the frame instead of sitting in black bars.

A scraped product video is typically 1920x1080 and the output frame is
1080x1920. Scaled to the frame width that is a 608px content band, so
`letterbox` left 68% of the frame black -- measured at 68.3% on a real render,
and reproduced here through the real filtergraph rather than asserted from the
arithmetic.

`smart-scale` could not avoid it. The aspect difference for 16:9 into 9:16 is
2.16 against a tolerance of 0.10, so the tolerance would have to exceed 2.16
for any landscape clip to reach `crop-to-fit`; it resolved to `letterbox`
unconditionally for the one source type the scraper actually returns.

The geometry assertions are the load-bearing ones. `blur-fill` changes only
what surrounds the content band, so a difference in reported geometry would
move every caption and the disclosure overlay with it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from src.video.assembler.visual_builder import VisualFilterBuilder

TARGET_W, TARGET_H = 1080, 1920
SOURCE_W, SOURCE_H = 1920, 1080


def _builder() -> VisualFilterBuilder:
    builder = VisualFilterBuilder.__new__(VisualFilterBuilder)
    builder.config = MagicMock()
    builder.config.aspect_ratio = {"smart_scale_tolerance": 0.10}
    return builder


def _apply(mode: str, **kwargs):
    return _builder().apply_aspect_ratio_mode(
        "[0:v]",
        mode,
        TARGET_W,
        TARGET_H,
        SOURCE_W,
        SOURCE_H,
        output_label="[v0_scaled]",
        target_content_height=TARGET_H,
        **kwargs,
    )


@pytest.mark.unit
class TestSmartScaleReachesIt:
    def test_a_landscape_source_no_longer_resolves_to_letterbox(self):
        """The branch that made every product clip 68% black."""
        filter_string, _, _ = _apply("smart-scale", video_top_percent=None)

        assert "gblur" in filter_string
        assert ":black" not in filter_string

    def test_a_near_square_source_still_crops(self):
        """`smart-scale`'s near branch is unchanged."""
        filter_string, _, _ = _builder().apply_aspect_ratio_mode(
            "[0:v]", "smart-scale", TARGET_W, TARGET_H, 1000, 1900
        )

        assert "crop=" in filter_string
        assert "gblur" not in filter_string

    def test_letterbox_is_still_reachable_by_name(self):
        """Naming it is how a profile opts back into black bars."""
        filter_string, _, _ = _apply("letterbox", video_top_percent=None)

        assert ":black" in filter_string
        assert "gblur" not in filter_string


@pytest.mark.unit
class TestTheGeometryIsUnchanged:
    """Captions are placed from this, so it must not move."""

    @pytest.mark.parametrize("top", [None, 0.25])
    def test_blur_fill_reports_the_letterbox_geometry(self, top):
        _, _, letterbox = _apply("letterbox", video_top_percent=top)
        _, _, blur_fill = _apply("blur-fill", video_top_percent=top)

        assert letterbox is not None
        assert blur_fill == letterbox

    def test_the_content_band_is_full_width_and_not_full_height(self):
        """Guards the numbers the geometry equality is asserted against.

        Two modes agreeing on a geometry that was wrong for both would pass
        the test above.
        """
        _, _, geometry = _apply("blur-fill", video_top_percent=None)

        assert geometry is not None
        assert geometry.rendered_w == TARGET_W
        assert geometry.rendered_h == pytest.approx(
            TARGET_W * SOURCE_H / SOURCE_W, abs=1
        )
        assert geometry.rendered_y > 0


@pytest.mark.unit
class TestTheChainIsWellFormed:
    def test_the_labels_are_namespaced_by_the_output_label(self):
        """Two segments share one filtergraph, so a fixed label would clash."""
        first, _, _ = _builder().apply_aspect_ratio_mode(
            "[0:v]",
            "blur-fill",
            TARGET_W,
            TARGET_H,
            SOURCE_W,
            SOURCE_H,
            output_label="[v0_scaled]",
        )
        second, _, _ = _builder().apply_aspect_ratio_mode(
            "[1:v]",
            "blur-fill",
            TARGET_W,
            TARGET_H,
            SOURCE_W,
            SOURCE_H,
            output_label="[v1_scaled]",
        )

        assert "[v0_scaled_bg]" in first
        assert "[v1_scaled_bg]" in second
        assert not set(first.split(";")) & set(second.split(";"))

    def test_a_generated_output_label_is_a_valid_label(self):
        """No caller omits it today, but the blur-fill chain breaks if one does.

        The default used to be `f"{input_label}_scaled"`, giving
        `[0:v]_scaled`: brackets in the middle and a `:` FFmpeg reads as an
        argument separator. Every mode returns that label and the caller
        appends it, so every mode was rejected with "Trailing garbage after a
        filter" -- letterbox included, measured at exit 234. blur-fill
        additionally builds four internal labels from it.
        """
        builder = _builder()
        filter_string, label, _ = builder.apply_aspect_ratio_mode(
            "[0:v]", "blur-fill", TARGET_W, TARGET_H, SOURCE_W, SOURCE_H
        )

        assert label == "[0_v_scaled]"
        # The input label is the one place a `:` legitimately appears, and it
        # appears once. Every label the function derives must be word-only.
        labels = re.findall(r"\[([^\[\]]*)\]", filter_string)
        assert labels.count("0:v") == 1
        derived = [name for name in labels if name != "0:v"]
        assert derived
        assert all(re.fullmatch(r"\w+", name) for name in derived), derived

    def test_it_does_not_emit_its_own_output_label(self):
        """The caller appends it; emitting one here duplicates the label."""
        filter_string, label, _ = _apply("blur-fill", video_top_percent=None)

        assert label == "[v0_scaled]"
        assert not filter_string.endswith(label)

    def test_the_configured_sigma_stays_in_full_frame_terms(self):
        """The background is blurred at 1/6 scale as a speed optimisation.

        The config value must not be reinterpreted by that: doubling it in
        YAML has to double the apparent blur, whatever the internal factor.
        """
        weak, _, _ = _apply("blur-fill", video_top_percent=None, blur_sigma=6.0)
        strong, _, _ = _apply("blur-fill", video_top_percent=None, blur_sigma=12.0)

        assert "gblur=sigma=1.0000" in weak
        assert "gblur=sigma=2.0000" in strong


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestTheFrameItActuallyProduces:
    """The unit tests above assert on a string FFmpeg might still reject."""

    @staticmethod
    def _render(tmp_path, mode):
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
                f"testsrc2=size={SOURCE_W}x{SOURCE_H}:duration=1:rate=5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            check=True,
            capture_output=True,
        )
        filter_string, label, _ = _apply(mode, video_top_percent=None)
        frame = tmp_path / f"{mode}.png"
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
        return frame

    @staticmethod
    def _black_row_fraction(frame):
        from PIL import Image

        image = Image.open(frame).convert("RGB")
        width, height = image.size
        black = 0
        for y in range(height):
            samples = [image.getpixel((x, y)) for x in range(0, width, 60)]
            if all(isinstance(p, tuple) and sum(p) < 24 for p in samples):
                black += 1
        return black / height

    def test_letterbox_wastes_most_of_the_frame(self, tmp_path):
        """The defect, reproduced rather than quoted."""
        fraction = self._black_row_fraction(self._render(tmp_path, "letterbox"))

        assert fraction > 0.6

    def test_a_generated_label_still_renders(self, tmp_path):
        """The chain built without an explicit `output_label` must parse."""
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
                f"testsrc2=size={SOURCE_W}x{SOURCE_H}:duration=1:rate=5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            check=True,
            capture_output=True,
        )
        filter_string, label, _ = _builder().apply_aspect_ratio_mode(
            "[0:v]", "blur-fill", TARGET_W, TARGET_H, SOURCE_W, SOURCE_H
        )
        frame = tmp_path / "defaulted.png"
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

        assert frame.exists()

    def test_blur_fill_wastes_none_of_it(self, tmp_path):
        fraction = self._black_row_fraction(self._render(tmp_path, "blur-fill"))

        assert fraction == 0.0
