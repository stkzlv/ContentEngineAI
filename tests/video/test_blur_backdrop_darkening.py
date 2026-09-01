"""The blurred backdrop is darkened, because captions sit on it.

`blur-fill` replaced the letterbox bars with a blurred copy of the source, so
the caption band stopped being solid black and started varying with the shot.
Measured on a real `product_video_primary` render (issue #344), the band ran
102-165 of 255 across three frames. The base caption style is white fill with
a black stroke and stays legible over anything; what a bright backdrop costs
is the margin, with the fill at 2.5:1 against the light end.

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
from unittest.mock import MagicMock

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


def _video_chain(**kw):
    """The video half of blur-fill, which `_build_image_placement` does not cover."""
    from unittest.mock import MagicMock

    from src.video.assembler.visual_builder import VisualFilterBuilder

    builder = VisualFilterBuilder.__new__(VisualFilterBuilder)
    builder.config = MagicMock()
    builder.config.aspect_ratio = {"smart_scale_tolerance": 0.10}
    args = {"blur_sigma": 20.0, "blur_darken": 0.6, "video_top_percent": None}
    args.update(kw)
    filter_string, _, _ = builder.apply_aspect_ratio_mode(
        "[0:v]",
        "blur-fill",
        TARGET_W,
        TARGET_H,
        1920,
        1080,
        output_label="[v0_scaled]",
        target_content_height=TARGET_H,
        **args,
    )
    return filter_string


class TestTheVideoChainIsDarkenedToo:
    """The other half of blur-fill, and the half nothing guarded.

    Deleting the darkening from the video chain left the whole suite green,
    because every test above drives `_build_image_placement`. A scraped
    product clip goes through this path, not that one.
    """

    def test_the_backdrop_is_darkened(self):
        assert "colorlevels=romax=0.6:gomax=0.6:bomax=0.6" in _video_chain()

    def test_the_factor_is_the_one_passed_in(self):
        chain = _video_chain(blur_darken=0.35)

        assert "romax=0.35:gomax=0.35:bomax=0.35" in chain

    @pytest.mark.asyncio
    async def test_it_reads_its_own_field_and_not_the_image_one(self, tmp_path):
        """Drives `build_visual_chain`, where the two fields are read.

        Passing `blur_darken=` straight to `apply_aspect_ratio_mode` tests
        the builder and skips the call site, so swapping the two fields
        there stayed green. This renders a landscape clip through the real
        chain with the two set apart, so only the video one may appear.
        """
        from unittest.mock import AsyncMock

        from src.video.assembler.visual_builder import VisualFilterBuilder
        from src.video.config import config as video_config

        source = tmp_path / "clip.mp4"
        source.write_bytes(b"")

        inspector = MagicMock()
        inspector.is_video.return_value = True
        inspector.get_media_dimensions = AsyncMock(return_value=(1920, 1080))
        inspector.get_video_dimensions = AsyncMock(return_value=(1920, 1080))
        inspector.get_media_duration = AsyncMock(return_value=20.0)

        strategy = MagicMock()
        strategy.assemble = AsyncMock(return_value=([(source, 20.0, True)], "stub"))
        strategy_factory = MagicMock()
        strategy_factory.get_strategy.return_value = strategy

        settings = video_config.get_profile_merged_settings("product_video_primary")
        settings.video_settings.video_aspect_mode = "blur-fill"
        settings.video_settings.video_background_blur_darken = 0.31
        settings.video_settings.image_background_blur_darken = 0.22

        builder = VisualFilterBuilder(
            media_inspector=inspector,
            config=video_config,
            strategy_factory=strategy_factory,
            profile_settings=settings,
        )
        parts, *_ = await builder.build_visual_chain(
            visual_inputs=[source],
            total_video_duration=20.0,
            is_relative_mode=False,
            video_settings_dict=settings.video_settings.model_dump(),
        )
        chain = "\n".join(parts)

        assert "romax=0.31" in chain, "the video chain did not read its own field"
        assert "0.22" not in chain, "the video chain read the image field"

    def test_the_darkening_precedes_the_upscale(self):
        """`colorlevels` is RGB-only.

        Left after `scale` it runs the whole backdrop at frame size in RGB
        and converts back before the overlay. Measured on one interleaved
        run of a 5s clip, that placement cost about 2.4x the filter time of
        this one.
        """
        chain = _video_chain()
        darken_at = chain.index("colorlevels=")
        upscale_at = chain.index(f"scale={TARGET_W}:{TARGET_H}")

        assert darken_at < upscale_at
        assert "format=yuv420p" in chain[darken_at:upscale_at]

    def test_one_point_zero_emits_no_filter(self):
        """The documented opt-out must cost nothing.

        A `colorlevels` at 1.0 is a visual no-op but still forces the RGB
        round trip, so an operator opting out paid the whole price.
        """
        assert "colorlevels" not in _video_chain(blur_darken=1.0)


class TestTheImageChainAlsoSkipsTheNoOp:
    def test_one_point_zero_emits_no_filter(self):
        assert "colorlevels" not in _placement(blur_darken=1.0)

    def test_the_darkening_precedes_the_scale(self):
        """Source resolution, not frame resolution, for the same RGB reason."""
        chain = _placement()
        darken_at = chain.index("colorlevels=")
        scale_at = chain.index(f"scale={TARGET_W}:{TARGET_H}")

        assert darken_at < scale_at
