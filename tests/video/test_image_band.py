"""A product image stays out of the caption block (#368).

The assembler reserved a strip at the bottom of the frame for captions and
then centred the image in the whole frame. The captions are not at the
bottom: the shipped pycaps offset puts the block at 63-75% of the height and
the FFmpeg engine clamps its caption to the 65% safe-zone floor, so a tall
product image centred in the frame ran to 87% and the captions sat on it. A
centred 16:9 video ends near 66%, clear of a single-line caption and a few
rows into a two-line block, which is why the video profiles showed it far
less.

The band the image is centred in now ends above the caption block, for
whichever engine will burn it. The frame decides, not the arithmetic: the
last test renders the emitted filter and reads the image's last row.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.video.assembler.visual_band import (
    VisualBand,
    caption_band_top,
    visual_band,
)
from src.video.config import config

FRAME_H = 1920
FRAME_W = 1080
FONT = {"base_font_height_percent": 0.05, "reserved_space_font_multiplier": 1.3}


@pytest.mark.unit
class TestWhereTheCaptionBlockBegins:
    def test_pycaps_bottom_anchor_with_the_shipped_offset(self) -> None:
        settings = {
            "pycaps": {
                "vertical_align": "bottom",
                "vertical_align_offset": -0.20,
                "caption_block_height": 0.14,
            }
        }
        # Block bottom at 0.95 - 0.20 = 0.75, block 0.14 tall: top at 0.61.
        assert caption_band_top(FRAME_H, settings, "pycaps", **FONT) == int(
            FRAME_H * 0.61
        )

    def test_pycaps_top_anchor_is_the_offset(self) -> None:
        settings = {"pycaps": {"vertical_align": "top", "vertical_align_offset": 0.7}}
        assert caption_band_top(FRAME_H, settings, "pycaps", **FONT) == int(
            FRAME_H * 0.7
        )

    def test_pycaps_centre_anchor_follows_the_library_formula(self) -> None:
        settings = {
            "pycaps": {
                "vertical_align": "center",
                "vertical_align_offset": -0.1,
                "caption_block_height": 0.2,
            }
        }
        # (container - element) * (offset + 0.5) = 0.8 * 0.4 of the frame.
        assert caption_band_top(FRAME_H, settings, "pycaps", **FONT) == int(
            FRAME_H * 0.8 * 0.4
        )

    def test_no_offset_assumes_the_documented_block(self) -> None:
        """With no offset the template places the block; the doc's 52% band."""
        settings = {
            "pycaps": {"vertical_align_offset": None, "caption_block_height": 0.14}
        }
        assert caption_band_top(FRAME_H, settings, "pycaps", **FONT) == int(
            FRAME_H * (0.52 - 0.07)
        )

    def test_ffmpeg_ends_at_the_safe_zone_floor(self) -> None:
        """One styled line above `max_y`, the estimate the old reserve used."""
        settings = {"font_size_scale": 1.0}
        expected = int(FRAME_H * 0.651 - FRAME_H * 0.05 * 1.3)
        assert (
            caption_band_top(FRAME_H, settings, "ffmpeg", safe_zone_max_y=0.651, **FONT)
            == expected
        )

    def test_ffmpeg_is_the_fallback_for_an_unresolved_engine(self) -> None:
        settings = {"font_size_scale": 1.0}
        assert caption_band_top(FRAME_H, settings, None, **FONT) == caption_band_top(
            FRAME_H, settings, "ffmpeg", **FONT
        )


@pytest.mark.unit
class TestTheBand:
    def test_starts_at_the_header_when_centred_else_the_offset(self) -> None:
        band = visual_band(FRAME_H, caption_top=1171, top_offset=864, centred=True)
        assert band.top == int(FRAME_H * 0.141)
        band = visual_band(FRAME_H, caption_top=1171, top_offset=288, centred=False)
        assert band.top == 288

    def test_ends_a_gap_above_the_captions(self) -> None:
        band = visual_band(FRAME_H, caption_top=1171, top_offset=288, centred=False)
        assert band.bottom == 1171 - int(FRAME_H * 0.02)

    def test_a_block_reaching_the_header_gives_an_empty_band_not_a_negative_one(
        self,
    ) -> None:
        band = visual_band(FRAME_H, caption_top=100, top_offset=288, centred=False)
        assert band.height == 0

    def test_centring_is_inside_the_band(self) -> None:
        band = VisualBand(top=288, bottom=1133)
        assert band.centred_y(845) == 288
        assert band.centred_y(445) == 288 + 200
        assert band.centred_y(2000) == 288


async def _chain_for(
    tmp_path: Path,
    image: Path,
    engine: str,
    orig: tuple[int, int],
    *,
    profile: str = "slideshow_images1",
    overrides: dict | None = None,
    timeline: list | None = None,
):
    from src.video.assembler.visual_builder import VisualFilterBuilder

    inspector = MagicMock()
    inspector.is_video.side_effect = lambda path: path.suffix == ".mp4"
    inspector.get_media_dimensions = AsyncMock(return_value=orig)
    inspector.get_video_dimensions = AsyncMock(return_value=(1920, 1080))
    inspector.get_media_duration = AsyncMock(return_value=3.0)
    strategy = MagicMock()
    strategy.assemble = AsyncMock(
        return_value=(timeline or [(image, 3.0, False)], "stubbed")
    )
    factory = MagicMock()
    factory.get_strategy.return_value = strategy
    settings = config.get_profile_merged_settings(profile, overrides)
    builder = VisualFilterBuilder(
        media_inspector=inspector,
        config=config,
        strategy_factory=factory,
        profile_settings=settings,
        subtitle_engine=engine,
    )
    return await builder.build_visual_chain(
        visual_inputs=[item[0] for item in (timeline or [(image, 3.0, False)])],
        total_video_duration=3.0,
        is_relative_mode=True,
        video_settings_dict=settings.video_settings.model_dump(),
    )


def _caption_top(engine: str) -> int:
    settings = config.get_profile_merged_settings("slideshow_images1", None)
    vs = config.video_settings
    return caption_band_top(
        FRAME_H,
        settings.subtitle_settings.model_dump(),
        engine,
        base_font_height_percent=vs.base_font_height_percent,
        reserved_space_font_multiplier=vs.reserved_space_font_multiplier,
        safe_zone_max_y=settings.subtitle_settings.safe_zone.max_y,
    )


@pytest.mark.unit
class TestTheShippedImageProfile:
    """`slideshow_images1` under the bundled config, both engines."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("engine", ["pycaps", "ffmpeg"])
    @pytest.mark.parametrize("orig", [(1080, 1350), (1000, 1000), (1600, 900)])
    async def test_the_geometry_ends_above_the_captions(
        self, tmp_path, engine, orig
    ) -> None:
        image = tmp_path / "product.png"
        image.write_bytes(b"")
        *_, geometries, _ = await _chain_for(tmp_path, image, engine, orig)
        geometry = geometries[0]
        assert geometry.rendered_y >= int(FRAME_H * 0.141)
        assert geometry.rendered_y + geometry.rendered_h <= _caption_top(engine)

    @pytest.mark.asyncio
    async def test_a_tall_image_no_longer_reaches_the_lower_frame(self, tmp_path):
        """The defect in numbers: 87% of the frame, captions from 61%."""
        image = tmp_path / "product.png"
        image.write_bytes(b"")
        *_, geometries, _ = await _chain_for(tmp_path, image, "pycaps", (1080, 1350))
        bottom = geometries[0].rendered_y + geometries[0].rendered_h
        assert bottom < FRAME_H * 0.65


@pytest.mark.unit
class TestTheOtherProfilesAreNotSqueezed:
    @pytest.mark.asyncio
    async def test_a_fill_image_on_a_video_profile_starts_at_the_header(
        self, tmp_path
    ) -> None:
        """`product_video_sequential` sets `image_top_position_percent: 0.45`
        under centring, where the field has never applied. The first cut of
        this fix took it as the band's top and rendered the profile's fill
        images as 215x269 thumbnails at row 864 (review finding). A centred
        image starts at the header zone whatever that field says.
        """
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        image = tmp_path / "fill.png"
        image.write_bytes(b"")
        *_, geometries, _ = await _chain_for(
            tmp_path,
            image,
            "pycaps",
            (1080, 1350),
            profile="product_video_sequential",
            timeline=[(clip, 2.0, True), (image, 1.0, False)],
        )
        fill = geometries[1]
        assert fill.rendered_y == int(FRAME_H * 0.141)
        assert fill.rendered_h >= 800
        assert fill.rendered_y + fill.rendered_h <= _caption_top("pycaps")

    @pytest.mark.asyncio
    async def test_a_top_aligned_image_starts_at_its_offset(self, tmp_path) -> None:
        image = tmp_path / "product.png"
        image.write_bytes(b"")
        *_, geometries, _ = await _chain_for(
            tmp_path,
            image,
            "pycaps",
            (1080, 1350),
            overrides={
                "video_settings.image_vertical_align": "top",
                "video_settings.image_top_position_percent": 0.20,
            },
        )
        top = geometries[0]
        assert top.rendered_y == int(FRAME_H * 0.20)
        assert top.rendered_y + top.rendered_h <= _caption_top("pycaps") - int(
            FRAME_H * 0.02
        )

    @pytest.mark.asyncio
    async def test_a_caption_block_above_the_top_offset_falls_back_to_the_frame(
        self, tmp_path
    ) -> None:
        """No band at all: a top-anchored block at 5% under an image top of
        10%. The old whole-frame placement applies rather than a zero height,
        on the top-aligned path as well as the centred one (review finding).
        """
        image = tmp_path / "product.png"
        image.write_bytes(b"")
        *_, geometries, _ = await _chain_for(
            tmp_path,
            image,
            "pycaps",
            (1080, 1350),
            overrides={
                "video_settings.image_vertical_align": "top",
                "video_settings.image_top_position_percent": 0.10,
                "subtitle_settings.pycaps.vertical_align": "top",
                "subtitle_settings.pycaps.vertical_align_offset": 0.05,
            },
        )
        top = geometries[0]
        assert top.rendered_y == int(FRAME_H * 0.10)
        assert top.rendered_h == 1350  # natural height, not a squeezed band


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestTheFrameItActuallyProduces:
    """#368's acceptance bar: the image's last row on a rendered frame."""

    @pytest.mark.asyncio
    async def test_the_last_red_row_is_above_the_caption_block(self, tmp_path):
        from PIL import Image

        image = tmp_path / "product.png"
        Image.new("RGB", (1080, 1350), (255, 0, 0)).save(image)
        filter_parts, input_cmd, _, _, geometries, _ = await _chain_for(
            tmp_path, image, "pycaps", (1080, 1350)
        )
        frame = tmp_path / "frame.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *input_cmd]
            + ["-filter_complex", filter_parts[0], "-map", "[v_proc_0]"]
            + ["-frames:v", "1", str(frame)],
            check=True,
            timeout=120,
        )
        with Image.open(frame) as img:
            rgb = img.convert("RGB")
            column = [rgb.getpixel((FRAME_W // 2, y)) for y in range(rgb.height)]
        red_rows = [
            y
            for y, pixel in enumerate(column)
            if isinstance(pixel, tuple) and pixel[0] > 200
        ]
        assert red_rows, "the image is not in the frame"
        assert red_rows[0] == geometries[0].rendered_y
        assert red_rows[-1] <= _caption_top("pycaps") - int(FRAME_H * 0.02)
        assert red_rows[-1] < FRAME_H * 0.65
