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
from pathlib import Path

from src.video.config.visual_models import DisclosureSettings, HookOverlaySettings

logger = logging.getLogger(__name__)


def _escape_drawtext_text(text: str) -> str:
    r"""Escape characters that break FFmpeg drawtext's `text=` arg.

    Special chars in drawtext's text argument: backslash, single quote, colon,
    percent. The arg is wrapped in single quotes by the caller.

    WARNING: apostrophes are NOT reliably escapable here. The close-quote /
    backslash-quote / open-quote pattern (``'\''``) survives a standalone
    ``-vf`` drawtext but still corrupts inside a multi-filter ``-filter_complex``
    chain (another filter present): FFmpeg swallows the drawtext's own trailing
    args as text. This escaper is used only by the disclosure overlay, whose
    text is config-controlled and apostrophe-free (``#ad`` / ``#publi``). For
    ARBITRARY text in a filter chain (e.g. the hook overlay), pass it via
    ``textfile=`` instead, which needs no text escaping. See
    ``build_hook_drawtext``.
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


def resolve_hook_line(
    hook_headline: str | None, hook_text: str | None, max_words: int
) -> str:
    """Choose the hook overlay line: an authored headline wins over the script.

    When a distinct authored ``hook_headline`` is available it is used verbatim
    (it is already short and already the final line, so it is not re-extracted).
    This is what makes the hook read as a designed headline rather than a copy of
    the first spoken sentence the running captions already show. Otherwise the
    first sentence of the spoken ``hook_text`` is extracted and capped, the
    pre-headline behaviour. Returns an empty string when neither is available, so
    the overlay becomes a no-op.
    """
    if hook_headline:
        return hook_headline
    if hook_text:
        return extract_hook_line(hook_text, max_words)
    return ""


# Approximate per-character width factors for FFmpeg's built-in hook font (the
# hook drawtext sets no ``fontfile``). Kept as constants so the fit logic stays a
# pure function testable without a loaded config, mirroring the character classes
# in unified_subtitle_generator.estimate_text_width_pixels.
# Average glyph width as a fraction of font size for FFmpeg's default drawtext
# font (no fontfile). Measured against real renders: 0.5 underestimated width by
# ~13%, so long hooks that the estimator judged as fitting rendered edge-to-edge
# (#160). 0.58 matches the observed rendered width so max_width_fraction holds.
_HOOK_WIDTH_TO_HEIGHT_RATIO = 0.58
_NARROW_CHARS = "iIjl|':;,.`!"
_WIDE_CHARS = "mwMWAGOQ@"


def _estimate_hook_text_width(text: str, font_size: int) -> int:
    """Estimate the rendered pixel width of a hook line at ``font_size``."""
    avg = font_size * _HOOK_WIDTH_TO_HEIGHT_RATIO
    total = 0.0
    for char in text:
        if char in _NARROW_CHARS:
            total += avg * 0.4
        elif char in _WIDE_CHARS:
            total += avg * 1.2
        elif char == " ":
            total += avg * 0.3
        else:
            total += avg
    return int(total)


def _wrap_hook_words(words: list[str], font_size: int, max_px: int) -> list[str]:
    """Greedy word-wrap so each line's estimated width stays within ``max_px``."""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and _estimate_hook_text_width(candidate, font_size) > max_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _fit_hook_lines(
    hook_text: str,
    start_font_size: int,
    frame_width: int,
    settings: HookOverlaySettings,
) -> tuple[list[str], int]:
    """Fit the hook to the frame width (#160).

    Wrap into at most ``settings.max_lines`` lines each within
    ``max_width_fraction`` of the frame width; when wrapping alone can't fit
    (too many words, or a single word wider than the frame) shrink the font and
    re-wrap, down to the 8px floor. Returns the wrapped lines and their font
    size. A hook that already fits on one line returns unchanged.
    """
    max_px = max(1, int(frame_width * settings.max_width_fraction))
    words = hook_text.split()
    font_size = start_font_size
    while True:
        lines = _wrap_hook_words(words, font_size, max_px) or [hook_text]
        fits = len(lines) <= settings.max_lines and all(
            _estimate_hook_text_width(line, font_size) <= max_px for line in lines
        )
        if fits or font_size <= 8:
            return lines[: settings.max_lines], font_size
        font_size = max(8, font_size - 6)


def build_hook_drawtext(
    settings: HookOverlaySettings,
    hook_text: str,
    subtitle_font_size_pixels: int,
    frame_width: int,
    temp_dir: Path,
    input_stream: str,
    output_stream: str,
) -> str:
    r"""Build a time-gated centre-upper FFmpeg hook overlay.

    The hook is wrapped to at most ``settings.max_lines`` lines sized to the
    frame width, and the font shrinks when wrapping alone can't fit, so a long
    hook never clips off-frame (#160). Each wrapped line is rendered as its own
    horizontally-centred ``drawtext`` (``(w-text_w)/2``), stacked vertically and
    chained with ``;`` (the filter separator core.py joins the list with); a
    single-line hook produces one drawtext.

    Each line's text is passed via ``textfile=`` (written to ``temp_dir``), not
    inline ``text=``. Inline text with a literal apostrophe silently corrupts:
    the ``'\''`` exit/reenter escape works on a standalone ``-vf`` drawtext but
    NOT inside the assembler's multi-filter ``-filter_complex`` chain, where the
    parser swallows the drawtext's own trailing args as text (so a hook like
    "you're ..." renders its args as tiny garbage). ``textfile=`` sidesteps all
    text escaping. The subtitle builder uses the same approach. Only the file
    path needs escaping (colons).

    The drawtext is `enable`-gated to the first ``duration_sec`` seconds so it
    disappears for the rest of the clip without changing the filter graph length.

    The ``enable`` expression uses backslash-escaped commas (``\,``) per FFmpeg's
    filtergraph escaping rules. Single-quoting the expression
    (``enable='between(t,0,X)'``) is the documented form but breaks in practice
    when the filter sits inside a comma-separated chain — FFmpeg's parser still
    splits at the inner commas. ``between(t\,0\,X)`` survives cleanly. Same trick
    applies to any FFmpeg filter expression with embedded commas.
    """
    start_font = max(8, int(round(subtitle_font_size_pixels * settings.size_factor)))
    lines, font_size = _fit_hook_lines(hook_text, start_font, frame_width, settings)
    line_height = int(font_size * 1.2)
    x_expr = "(w-text_w)/2"

    drawtexts: list[str] = []
    last_index = len(lines) - 1
    for i, line in enumerate(lines):
        stream_in = input_stream if i == 0 else f"[v_hkl{i}]"
        stream_out = output_stream if i == last_index else f"[v_hkl{i + 1}]"
        line_file = temp_dir / f"hook_line_{i}.txt"
        line_file.write_text(line, encoding="utf-8")
        text_path = line_file.as_posix().replace(":", r"\:")
        y_expr = (
            f"h*{settings.margin_y_percent}+{i * line_height}"
            if i
            else f"h*{settings.margin_y_percent}"
        )
        parts = [
            f"{stream_in}drawtext=",
            f"textfile='{text_path}':",
            f"fontsize={font_size}:",
            f"fontcolor={settings.font_color}:",
            f"borderw={settings.outline_thickness}:",
            f"bordercolor={settings.outline_color}:",
        ]
        if settings.background_enabled:
            parts.append(f"box=1:boxcolor={settings.background_color}:boxborderw=12:")
        parts.append(f"x={x_expr}:y={y_expr}:")
        parts.append(f"enable=between(t\\,0\\,{settings.duration_sec:.3f}){stream_out}")
        drawtexts.append("".join(parts))

    return ";".join(drawtexts)


def apply_hook_overlay(
    video_filters: list[str],
    settings: HookOverlaySettings,
    hook_text: str,
    subtitle_font_size_pixels: int,
    frame_width: int,
    temp_dir: Path,
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
        frame_width,
        temp_dir,
        input_stream=input_stream,
        output_stream="[v_hook]",
    )
    # Preserve the terminal copy[v_out] so the disclosure rewrite still works.
    new_terminal = "[v_hook]copy[v_out]"
    return [*video_filters[:-1], hook_filter, new_terminal]
