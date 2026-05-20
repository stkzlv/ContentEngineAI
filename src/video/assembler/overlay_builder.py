"""Persistent on-frame overlays burned by the assembler.

Two overlays live here:

- `apply_disclosure_overlay`: FTC `#ad` / Spain `#publi` corner badge,
  visible for the full clip duration. Last filter in the chain; rewrites
  the subtitle builder's `copy[v_out]` no-op so the overlay output is the
  final video stream.

- `apply_hook_overlay` (Phase 1.2c, closes #102): centre-upper static text
  rendering the first sentence of the spoken script, visible for the first
  ``duration_sec`` seconds only. Inserted BEFORE the disclosure overlay so
  the disclosure stays on top in the z-order. Preserves the chain's
  terminal ``copy[v_out]`` so the disclosure rewrite still finds the
  expected shape.
"""

from __future__ import annotations

import logging

from src.video.config.visual_models import DisclosureSettings, HookOverlaySettings

logger = logging.getLogger(__name__)


def _escape_drawtext_text(text: str) -> str:
    r"""Escape characters that break FFmpeg drawtext's `text=` arg.

    Special chars in drawtext's text argument: backslash, single quote, colon,
    percent. The arg is wrapped in single quotes by the caller.

    Apostrophes use the close-quote / backslash-quote / open-quote pattern
    (``'\''``) rather than ``\'``. The ``\'`` form is documented and works
    on a standalone drawtext, but breaks when the drawtext sits inside a
    multi-filter filtergraph chain — FFmpeg's parser consumes characters
    past the intended quote boundary and absorbs the rest of the chain
    into the drawtext args, producing a confusing ``Option 'st' not found``
    style error from later filters. The exit/reenter pattern is the same
    shell-style trick used to embed apostrophes in any single-quoted
    string; reliable across filter contexts.
    """
    return (
        text.replace("\\", r"\\")
        .replace("'", r"'\''")
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


def _truncate_to_words(text: str, max_words: int) -> str:
    """Trim ``text`` to ``max_words`` words, appending an ellipsis when cut."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(",.;:!?") + "..."


def extract_hook_line(script: str, max_words: int) -> str:
    """Pull the first sentence of the script as the hook overlay text.

    Splits on the first sentence terminator (``.!?``), strips, and caps to
    ``max_words``. Returns an empty string when the script is empty or all
    whitespace; the caller skips the overlay in that case.
    """
    if not script or not script.strip():
        return ""

    head = script.strip()
    for terminator in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = head.find(terminator)
        if 0 < idx < len(head):
            head = head[: idx + 1].rstrip(".!? ")
            break
    return _truncate_to_words(head, max_words)


def build_hook_drawtext(
    settings: HookOverlaySettings,
    hook_text: str,
    subtitle_font_size_pixels: int,
    input_stream: str,
    output_stream: str,
) -> str:
    r"""Build a time-gated centre-upper FFmpeg drawtext for the hook overlay.

    The drawtext is `enable`-gated to the first ``duration_sec`` seconds so
    it disappears for the rest of the clip without changing the filter
    graph length. Position is centred horizontally (``(w-text_w)/2``) and
    placed ``margin_y_percent`` from the top.

    The ``enable`` expression uses backslash-escaped commas (``\,``) per
    FFmpeg's filtergraph escaping rules. Single-quoting the expression
    (``enable='between(t,0,X)'``) is the documented form but breaks in
    practice when the filter sits inside a comma-separated chain — FFmpeg's
    parser still splits at the inner commas. ``between(t\,0\,X)`` survives
    cleanly. Same trick applies to any FFmpeg filter expression with
    embedded commas (``if``, ``lt``, ``gte``, etc.).
    """
    font_size = max(8, int(round(subtitle_font_size_pixels * settings.size_factor)))
    text = _escape_drawtext_text(hook_text)
    x_expr = "(w-text_w)/2"
    y_expr = f"h*{settings.margin_y_percent}"

    parts = [
        f"{input_stream}drawtext=",
        f"text='{text}':",
        f"fontsize={font_size}:",
        f"fontcolor={settings.font_color}:",
        f"borderw={settings.outline_thickness}:",
        f"bordercolor={settings.outline_color}:",
    ]
    if settings.background_enabled:
        parts.append(f"box=1:boxcolor={settings.background_color}:boxborderw=12:")
    parts.append(f"x={x_expr}:y={y_expr}:")
    parts.append(f"enable=between(t\\,0\\,{settings.duration_sec:.3f}){output_stream}")

    return "".join(parts)


def apply_hook_overlay(
    video_filters: list[str],
    settings: HookOverlaySettings,
    hook_text: str,
    subtitle_font_size_pixels: int,
) -> list[str]:
    """Insert the hook overlay before the disclosure overlay's rewrite slot.

    Rewrites the trailing ``copy[v_out]`` into a time-gated hook drawtext
    that produces an intermediate stream, then appends a new
    ``copy[v_out]`` so :func:`apply_disclosure_overlay` still finds the
    terminal shape it expects. Order in core.py is: hook → disclosure →
    [v_out], giving the disclosure top z-order.

    No-ops (returns input unchanged) when:

    - ``settings.enabled`` is False.
    - ``hook_text`` is empty after extraction (script was empty or all
      whitespace).
    - The terminal filter does not match ``copy[v_out]`` (logged at WARN;
      keeps the disclosure pipeline from breaking on unexpected shapes).
    """
    if not settings.enabled:
        return video_filters
    if not hook_text:
        logger.debug("Hook overlay skipped: empty hook text.")
        return video_filters
    if not video_filters:
        logger.warning("Hook overlay skipped: empty video_filters list.")
        return video_filters

    last = video_filters[-1]
    if "copy[v_out]" not in last:
        logger.warning(
            "Hook overlay skipped: last filter has unexpected shape: %r", last
        )
        return video_filters

    input_stream = last.replace("copy[v_out]", "")
    hook_filter = build_hook_drawtext(
        settings,
        hook_text,
        subtitle_font_size_pixels,
        input_stream=input_stream,
        output_stream="[v_hook]",
    )
    # Preserve the terminal copy[v_out] so the disclosure rewrite still works.
    new_terminal = "[v_hook]copy[v_out]"
    return [*video_filters[:-1], hook_filter, new_terminal]
