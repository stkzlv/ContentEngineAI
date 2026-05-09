"""Persistent on-frame disclosure overlay (FTC `#ad`, Spain `#publi`, etc.).

Burns a fixed-corner text overlay into every produced video so the disclosure
is visible across the full clip duration regardless of which subtitle engine
ran. Mounts as the last filter in the video filter chain, replacing the
existing `copy[v_out]` no-op so the overlay output IS the assembler's final
video stream.
"""

from __future__ import annotations

import logging

from src.video.config.visual_models import DisclosureSettings

logger = logging.getLogger(__name__)


def _escape_drawtext_text(text: str) -> str:
    """Escape characters that break FFmpeg drawtext's `text=` arg.

    Special chars in drawtext's text argument: backslash, single quote, colon,
    percent. Use FFmpeg's escape sequences. The arg is wrapped in single
    quotes by the caller.
    """
    return (
        text.replace("\\", r"\\")
        .replace("'", r"\'")
        .replace(":", r"\:")
        .replace("%", r"\%")
    )


def _position_expressions(
    position: str,
    margin_x_percent: float,
    margin_y_percent: float,
) -> tuple[str, str]:
    """Return (x, y) FFmpeg drawtext position expressions for the chosen corner.

    `text_w` and `text_h` are FFmpeg drawtext intrinsics (rendered text size).
    Margins are fractions of frame width/height.
    """
    margin_x = f"w*{margin_x_percent}"
    margin_y = f"h*{margin_y_percent}"

    if position == "top-left":
        return margin_x, margin_y
    if position == "top-right":
        return f"w-text_w-{margin_x}", margin_y
    if position == "bottom-left":
        return margin_x, f"h-text_h-{margin_y}"
    if position == "bottom-right":
        return f"w-text_w-{margin_x}", f"h-text_h-{margin_y}"

    # Pydantic Literal validation should prevent this, but be defensive.
    logger.warning(
        "Unknown disclosure position %r; falling back to top-right", position
    )
    return f"w-text_w-{margin_x}", margin_y


def build_disclosure_drawtext(
    settings: DisclosureSettings,
    subtitle_font_size_pixels: int,
    input_stream: str,
    output_stream: str,
) -> str:
    """Build a single FFmpeg drawtext filter that renders the disclosure.

    Parameters
    ----------
    settings:
        Configured disclosure overlay settings.
    subtitle_font_size_pixels:
        Pixel size of the narration caption font, used as the reference for
        ``size_factor``. Set to a sensible default (e.g. ``frame_height * 0.05``)
        when caption size is unavailable.
    input_stream:
        FFmpeg stream label feeding into this filter, e.g. ``[v_subtitle_3]``.
    output_stream:
        FFmpeg stream label this filter produces, typically ``[v_out]``.

    Returns
    -------
    A filter string suitable for appending to the FFmpeg filter graph.

    """
    font_size = max(8, int(round(subtitle_font_size_pixels * settings.size_factor)))
    text = _escape_drawtext_text(settings.text)
    x_expr, y_expr = _position_expressions(
        settings.position,
        settings.margin_x_percent,
        settings.margin_y_percent,
    )

    parts = [
        f"{input_stream}drawtext=",
        f"text='{text}':",
        f"fontsize={font_size}:",
        f"fontcolor={settings.font_color}:",
        f"borderw={settings.outline_thickness}:",
        f"bordercolor={settings.outline_color}:",
    ]
    if settings.background_enabled:
        parts.append(f"box=1:boxcolor={settings.background_color}:boxborderw=8:")
    parts.append(f"x={x_expr}:y={y_expr}{output_stream}")

    return "".join(parts)


def apply_disclosure_overlay(
    video_filters: list[str],
    settings: DisclosureSettings,
    subtitle_font_size_pixels: int,
) -> list[str]:
    """Inject the disclosure overlay as the final filter before ``[v_out]``.

    The subtitle builder ends its chain with a no-op ``copy[v_out]`` filter to
    rename the last intermediate stream. This function rewrites that line into
    a drawtext filter that renders the disclosure on top of the subtitle output
    and produces ``[v_out]`` directly. The caller's filter chain stays the same
    length and end-label.

    When ``settings.enabled`` is False, the input list is returned unchanged.
    When the input does not end with ``copy[v_out]`` (unexpected shape), the
    chain is also returned unchanged and a warning is logged.
    """
    if not settings.enabled:
        return video_filters

    if not video_filters:
        logger.warning("Disclosure overlay skipped: empty video_filters list")
        return video_filters

    last = video_filters[-1]
    # The subtitle builder always emits "...copy[v_out]" as the final entry.
    # Find the input stream label and rewrite the line to drawtext+v_out.
    if "copy[v_out]" not in last:
        logger.warning(
            "Disclosure overlay skipped: last filter has unexpected shape: %r",
            last,
        )
        return video_filters

    input_stream = last.replace("copy[v_out]", "")
    rewritten = build_disclosure_drawtext(
        settings,
        subtitle_font_size_pixels,
        input_stream=input_stream,
        output_stream="[v_out]",
    )
    return [*video_filters[:-1], rewritten]
