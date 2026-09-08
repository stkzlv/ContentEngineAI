"""The rows a product image may occupy once the captions have taken theirs.

Both subtitle engines place their captions from configuration the assembler
never read. The image branch reserved a strip at the bottom of the frame and
then centred the image in the *whole* frame, so a tall product image ran to
87% of the height while the shipped pycaps offset put the caption block at
63-75% (#368). A centred 16:9 video ends near 66%: clear of a single-line
caption and a few rows into a two-line block, which is why the video
profiles showed it far less.

`caption_band_top` answers where the caption block begins, per engine, and
`visual_band` turns that into the band the image is fitted and centred in.
On the shipped config the block's bottom edge is at 75% of the frame and,
at the configured block height, its top at 63%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.video.config.constants import SAFE_ZONE_MAX_Y, SAFE_ZONE_MIN_Y

# pycaps anchors its bottom-aligned block at 95% of the frame plus the offset
# (`LayoutUtils.get_vertical_alignment_position`).
_PYCAPS_BOTTOM_BASE = 0.95
# Where docs/subtitle-best-practices.md puts the block when the template's
# own layout wins: centred around 52% of the frame.
_DOC_BLOCK_CENTRE = 0.52
# Whitespace kept between the image's last row and the caption block.
_GAP_FRACTION = 0.02


@dataclass(frozen=True)
class VisualBand:
    """Rows the content may occupy: `top` inclusive, `bottom` exclusive."""

    top: int
    bottom: int

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def centred_y(self, content_height: int) -> int:
        return self.top + max(0, (self.height - content_height) // 2)


def _fraction(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def caption_band_top(
    frame_height: int,
    subtitle_settings: dict[str, Any],
    engine: str | None,
    *,
    base_font_height_percent: float,
    reserved_space_font_multiplier: float,
    safe_zone_max_y: float = SAFE_ZONE_MAX_Y,
) -> int:
    """First row of the caption block, in pixels from the top of the frame.

    pycaps: the block's extent comes from its anchor and offset and from
    `caption_block_height`, an estimate that has to be configured because the
    template's CSS decides the real one. With no explicit offset the template
    positions the block itself, and the best-practices band is assumed.

    FFmpeg: the caption's centre is clamped to the safe zone, so the block
    ends at `max_y` and is one styled line tall, the same estimate the old
    bottom reserve used.
    """
    pycaps = subtitle_settings.get("pycaps") or {}
    if engine == "pycaps":
        block = _fraction(pycaps.get("caption_block_height"), 0.12)
        align = pycaps.get("vertical_align") or "bottom"
        offset = pycaps.get("vertical_align_offset")
        if offset is None:
            top = _DOC_BLOCK_CENTRE - block / 2
        elif align == "bottom":
            top = _PYCAPS_BOTTOM_BASE + float(offset) - block
        elif align == "top":
            top = float(offset)
        else:
            top = (1.0 - block) * (float(offset) + 0.5)
        return int(frame_height * max(0.0, top))

    scale = _fraction(subtitle_settings.get("font_size_scale"), 1.0)
    line_px = frame_height * base_font_height_percent * scale
    line_px *= reserved_space_font_multiplier
    return int(frame_height * safe_zone_max_y - line_px)


def upper_line_bottom(
    frame_height: int,
    upper_line: Any,
    subtitle_font_size_pixels: int,
    drawable_text: str | None,
) -> int:
    """Last row the static upper line occupies, or 0 when none is drawn.

    The line is drawn at `vertical_position` and is one line of text tall.
    The height is the font size plus the box padding drawtext adds on both
    sides (`boxborderw=8`), which is what the image has to clear.

    `drawable_text` is the text the overlay will actually draw, and is
    required rather than defaulted: reserving on `enabled` alone pushed the
    image down for a line the assembler then declined to draw -- a topic with
    no affiliate link, or a URL too long to trim -- so the image came out
    smaller and lower than the config asked for, silently. A default would
    let a caller reintroduce that by omission. None or empty means no line
    and no rows.
    """
    if upper_line is None or not getattr(upper_line, "enabled", False):
        return 0
    if not drawable_text:
        return 0
    size_factor = _fraction(getattr(upper_line, "size_factor", 0.55), 0.55)
    position = _fraction(getattr(upper_line, "vertical_position", 0.16), 0.16)
    font_px = max(8, int(round(subtitle_font_size_pixels * size_factor)))
    box_padding = 16 if getattr(upper_line, "background_enabled", True) else 0
    return int(frame_height * position) + font_px + box_padding


def visual_band(
    frame_height: int,
    *,
    caption_top: int,
    top_offset: int,
    centred: bool,
    safe_zone_min_y: float = SAFE_ZONE_MIN_Y,
    upper_line_bottom_px: int = 0,
) -> VisualBand:
    """The band an image is fitted into.

    A centred image starts at the platform header zone; a top-aligned one at
    the configured top offset, which is what that field has always meant
    (`product_video_sequential` sets it to 0.45 under centring and expects no
    effect). Both end a small gap above the caption block. The band is empty
    when the caption block reaches its top; the caller then fits the image
    from the band's top to the bottom of the frame rather than into a zero
    height, keeping the header clear and accepting the block over it.

    The static upper line takes its rows from the top of the band, plus the
    same gap kept above the captions (#88). Without that the image is fitted
    under the header and the line is drawn on top of it, which is the defect
    the band exists to prevent at the other end.
    """
    top = int(frame_height * safe_zone_min_y) if centred else top_offset
    if upper_line_bottom_px:
        top = max(top, upper_line_bottom_px + int(frame_height * _GAP_FRACTION))
    bottom = caption_top - int(frame_height * _GAP_FRACTION)
    return VisualBand(top=top, bottom=max(top, bottom))
