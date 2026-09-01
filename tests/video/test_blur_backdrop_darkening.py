"""The blurred backdrop is darkened, because captions sit on it.

`blur-fill` replaced the letterbox bars with a blurred copy of the source, so
the caption band stopped being solid black and started varying with the shot.
Measured on a real `product_video_primary` render (issue #344), the band ran
102-165 of 255 across three frames, and white caption text over the light end
is 2.5:1 against the 4.5:1 WCAG AA floor `docs/subtitle-best-practices.md`
requires.

`colorlevels` scales rather than subtracts, which is why it is not
`eq=brightness`. Measured on synthetic frames: a 39/255 backdrop goes to 0
under `eq=brightness=-0.25`, losing the surround entirely, and to 23 under
`colorlevels` at the same visual weight on a bright frame.

The darkening applies to the backdrop only. The content band is composited on
top afterwards, so the product photo or video is untouched.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from src.video.assembler.visual_builder import _build_image_placement

TARGET_W, TARGET_H = 1080, 1920


def _placement(**kw):
    args = {
        "index": 0,
        "vf_scale": "scale=1080:810",
        "width": TARGET_W,
        "height": TARGET_H,
        "target_y": 555,
        "pad_color": "black",
        "pix_fmt": "yuv420p",
        "background_fill": "blur",
        "blur_sigma": 20.0,
        "blur_darken": 0.6,
        "out_label": "[v_out]",
    }
    args.update(kw)
    return _build_image_placement(**args)


class TestTheFilterCarriesTheDarkening:
    def test_the_backdrop_is_darkened(self):
        chain = _placement()

        assert "colorlevels=romax=0.6:gomax=0.6:bomax=0.6" in chain

    def test_the_factor_is_the_one_passed_in(self):
        """Not a constant baked into the chain.

        A hardcoded 0.6 satisfies the presence check above, so the config
        field would reach the assembler and do nothing -- the silent-drop
        class `CLAUDE.md` records for profile-level fields.
        """
        chain = _placement(blur_darken=0.35)

        assert "romax=0.35:gomax=0.35:bomax=0.35" in chain
        assert "0.6" not in chain.split("colorlevels=")[1].split(",")[0]

    def test_it_darkens_the_backdrop_and_not_the_content(self):
        """Order matters: the overlay of the sharp copy comes after.

        Moving `colorlevels` past the overlay would dim the product photo
        along with its surround, which is a worse render than the one being
        fixed.
        """
        chain = _placement()
        darken_at = chain.index("colorlevels=")
        overlay_at = chain.index("overlay=")

        assert darken_at < overlay_at
        # And it sits on the backdrop branch, the one terminating at [bgb_0],
        # rather than on the [fg_0] branch that carries the sharp copy.
        backdrop_branch = next(
            seg for seg in chain.split(";") if seg.endswith("[bgb_0]")
        )
        sharp_branch = next(seg for seg in chain.split(";") if seg.endswith("[fgs_0]"))
        assert "colorlevels=" in backdrop_branch
        assert "colorlevels=" not in sharp_branch

    def test_a_solid_pad_is_unaffected(self):
        """`color` fill has no backdrop to darken."""
        chain = _placement(background_fill="color")

        assert "colorlevels" not in chain


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestItRendersAndActuallyDarkens:
    """Through real FFmpeg, not by reading the string.

    The v0.90.0 blur-fill work shipped a filtergraph that FFmpeg refused to
    parse, and a substring assertion said nothing about it.
    """

    @staticmethod
    def _band_mean(path, tmp_path):
        """Mean luma of a band that is backdrop, not content.

        The sharp copy occupies rows 555..1365 at this geometry, so a band
        starting at 1250 straddles it and averages in the undarkened product
        image. Captions land below the content, which is what this reads.
        """
        out = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                "crop=1080:200:0:1450,format=gray",
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
            check=True,
        ).stdout
        return sum(out) / len(out)

    @staticmethod
    def _render(chain, src, dest):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(src),
                "-filter_complex",
                chain,
                "-map",
                "[v_out]",
                "-frames:v",
                "1",
                "-y",
                str(dest),
            ],
            check=True,
            capture_output=True,
        )

    def _bright_source(self, tmp_path):
        src = tmp_path / "src.png"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=0xC8C8B4:s=1200x800:d=1",
                "-frames:v",
                "1",
                "-y",
                str(src),
            ],
            check=True,
            capture_output=True,
        )
        return src

    def test_the_graph_parses(self, tmp_path):
        """An unparseable chain exits 234 and writes nothing."""
        src = self._bright_source(tmp_path)
        dest = tmp_path / "out.png"

        self._render(_placement(), src, dest)

        assert dest.exists() and dest.stat().st_size > 0

    def test_the_caption_band_gets_darker(self, tmp_path):
        """The measurement the issue asks for, on a light source.

        Asserting only that the filter is present would pass against a
        `colorlevels` that scaled nothing.
        """
        src = self._bright_source(tmp_path)
        with_darken = tmp_path / "after.png"
        without = tmp_path / "before.png"

        self._render(_placement(), src, with_darken)
        self._render(_placement(blur_darken=1.0), src, without)

        after = self._band_mean(with_darken, tmp_path)
        before = self._band_mean(without, tmp_path)

        assert after < before, f"backdrop not darkened: {before:.1f} -> {after:.1f}"
        # 0.6 of a ~195 backdrop lands near 117, which is 4.6:1 against
        # white text. The bound is loose enough to survive a codec rounding
        # difference and tight enough that a no-op filter fails it.
        assert after < 130
