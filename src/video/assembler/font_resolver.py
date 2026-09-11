"""Resolve a font file that actually covers an overlay's text (#392).

FFmpeg's drawtext takes one face and has no fallback chain, so text in a
script the resolved default does not carry renders as empty notdef boxes at
full width, silently, with ffmpeg exiting 0. Observed with a CJK disclosure
override: fourteen empty rectangles where the operator's text should be.

Coverage is asked of fontconfig: ``fc-list ':charset=<codepoints>' file``
returns only faces that carry every requested glyph, so an empty answer
means nothing installed can draw the text -- the case that must fail loudly
rather than render boxes.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ASCII is carried by every face fontconfig would resolve for `Sans`, so
# only text beyond it needs a coverage check. This keeps the common case
# (`#ad`, `#publi`, URLs) free of a subprocess call per render.
_ASCII_CEILING = 0x7F


class OverlayFontError(ValueError):
    """No installed font covers every glyph of an overlay string."""


def _covering_files(codepoints: set[int]) -> list[Path]:
    charset = " ".join(f"{cp:x}" for cp in sorted(codepoints))
    proc = subprocess.run(
        ["fc-list", f":charset={charset}", "file"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        raise OSError(proc.stderr.strip() or "fc-list failed")
    files = []
    for line in proc.stdout.splitlines():
        path = line.split(":", 1)[0].strip()
        if path:
            files.append(Path(path))
    return files


def fontfile_for_text(text: str, *, strict: bool) -> Path | None:
    """A font file covering every glyph of ``text``, or None for the default.

    None means "let drawtext resolve its default face": returned for pure
    ASCII, and -- in non-strict mode -- whenever coverage cannot be
    established, because a render-time caller degrading to the old behavior
    beats losing the render. ``strict`` is for config validation, where an
    operator string nothing installed can draw must refuse with the glyphs
    named instead of shipping boxes.
    """
    codepoints = {ord(c) for c in text if not c.isspace() and ord(c) > _ASCII_CEILING}
    if not codepoints:
        return None

    try:
        files = _covering_files(codepoints)
    except (OSError, subprocess.TimeoutExpired) as e:
        # No fontconfig (or a broken one): coverage is unknowable here, so
        # strict validation passes rather than refusing every non-Latin
        # config on such systems.
        logger.debug("Font coverage check unavailable: %s", e)
        return None

    if not files:
        glyphs = "".join(sorted(chr(cp) for cp in codepoints))
        message = (
            f"No installed font covers the overlay text {text!r} "
            f"(uncovered glyphs: {glyphs!r}); drawtext would render empty "
            "boxes. Install a face carrying the script (e.g. Noto CJK) or "
            "change the configured text."
        )
        if strict:
            raise OverlayFontError(message)
        logger.warning("%s", message)
        return None

    # Prefer a sans face for visual consistency with the default; fall back
    # to whatever covers.
    for candidate in files:
        if "sans" in candidate.name.lower():
            return candidate
    return files[0]
