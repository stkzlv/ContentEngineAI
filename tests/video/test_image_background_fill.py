"""A product photo does not cover a 9:16 frame; something has to fill the rest.

Measured on a real render (`B0FC5S16YM`, `slideshow_images4`), the frame was
42-52% pure black across four samples: `image_width_percent: 1.0` scales the
image to the frame width, and a landscape source then occupies only its
natural height. Nothing filled the remainder -- there was no option to.

`blur` fills it with a scaled, blurred copy of the same image. Verified
against real FFmpeg: a 1500x1163 product photo through the generated
filtergraph produces a 1080x1920 frame with 0% black rows.
"""

from __future__ import annotations

import pytest

from src.video.assembler.visual_builder import _build_image_placement


def _placement(**kw):
    args = {
        "index": 0,
        "vf_scale": "scale=1080:810",
        "width": 1080,
        "height": 1920,
        "target_y": 555,
        "pad_color": "black",
        "pix_fmt": "yuv420p",
        "background_fill": "blur",
        "blur_sigma": 20.0,
        "out_label": "[v_temp_0]",
    }
    args.update(kw)
    return _build_image_placement(**args)


class TestTheFilterShape:
    def test_blur_covers_the_frame_rather_than_fitting_inside_it(self):
        """`increase` plus a crop, not `decrease`.

        Fitting the backdrop would letterbox the backdrop itself, which is the
        defect one layer down.
        """
        chain = _placement()

        assert "force_original_aspect_ratio=increase" in chain
        assert "crop=1080:1920" in chain

    def test_blur_composites_the_sharp_image_on_top(self):
        chain = _placement()

        assert "gblur=sigma=20.0" in chain
        assert "overlay=(W-w)/2:555" in chain

    def test_the_sharp_copy_is_scaled_by_the_callers_expression(self):
        """The backdrop must not replace the existing sizing logic."""
        chain = _placement(vf_scale="scale=980:735")

        assert "[fg_0]scale=980:735" in chain

    def test_color_is_byte_identical_to_what_shipped(self):
        """The default must not change any existing render."""
        chain = _placement(background_fill="color")

        assert chain == (
            "[0:v]scale=1080:810,setsar=1,"
            "pad=1080:1920:(ow-iw)/2:555:color=black,"
            "format=yuv420p[v_temp_0]"
        )

    def test_labels_are_scoped_to_the_index(self):
        """Two images in one graph would collide on a shared label name."""
        first = _placement(index=0)
        second = _placement(index=1)

        assert "[bg_0]" in first and "[bg_0]" not in second
        assert "[bg_1]" in second

    def test_an_unknown_fill_falls_back_to_the_solid_pad(self):
        """A value the model rejects should still not produce a broken graph."""
        chain = _placement(background_fill="something-else")

        assert "pad=1080:1920" in chain
        assert "gblur" not in chain


class TestTheProfileWiring:
    """Three conditions, per CLAUDE.md, or the override is swallowed silently.

    Declared on `VideoSettings`, declared on `VideoProfile`, and listed in the
    `_collect_overrides` map. Missing the third is the quiet one: no warning,
    no test failure, the profile value simply never arrives.
    """

    @pytest.fixture(scope="class")
    def config(self):
        from src.video.config import load_video_config  # noqa: F401
        from src.video.config_adapter import load_video_config_modular

        return load_video_config_modular()

    def test_the_bundled_profiles_fill_the_frame(self, config):
        for name in config.video_profiles:
            if name == "base":
                continue
            merged = config.get_profile_merged_settings(name)
            assert merged.video_settings.image_background_fill == "blur", name

    def test_a_profile_override_reaches_the_merged_settings(self, config):
        """The `_collect_overrides` condition, exercised rather than assumed."""
        from copy import deepcopy

        patched = deepcopy(config)
        profile = patched.video_profiles["slideshow_images4"]
        profile.image_background_fill = "color"

        merged = patched.get_profile_merged_settings("slideshow_images4")

        assert merged.video_settings.image_background_fill == "color"
