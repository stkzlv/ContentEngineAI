"""Read a product page's gallery and listing videos from its image block (REQ-SCR-086).

The page carries its gallery in an inline script as
`'colorImages': { 'initial': A.$.parseJSON('[...]') }`, one entry per image
with `hiRes` (sometimes null), `large` and `thumb`, and the listing's videos
as `'videos': A.$.parseJSON('[...]')` with `url`, `title` and
`durationSeconds`. The JSON sits inside a JavaScript string literal, so the
literal is decoded the way the browser would before the JSON is parsed.

Only `initial` is read: the page also carries other colour variants'
galleries in another script, and reading every `hiRes` on the page mixed
them in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

PARSE_CALL = "parseJSON('"

# The script that holds the landing colour's gallery, read in one call.
_IMAGE_BLOCK_JS = """
for (const s of document.querySelectorAll('script:not([src])')) {
    const t = s.textContent || '';
    if (t.includes("'colorImages'") && t.includes("'initial'")) return t;
}
return null;
"""

_ESCAPES = {
    "\\": "\\",
    "'": "'",
    '"': '"',
    "/": "/",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
}


@dataclass(frozen=True)
class ListingVideo:
    url: str
    title: str = ""
    duration_sec: float | None = None


@dataclass(frozen=True)
class ImageBlock:
    images: list[str] = field(default_factory=list)
    videos: list[ListingVideo] = field(default_factory=list)


def _js_string_at(text: str, start: int) -> str | None:
    """Decode the single-quoted JavaScript literal whose body starts at `start`."""
    out: list[str] = []
    i = start
    while i < len(text):
        char = text[i]
        if char == "'":
            return "".join(out)
        if char != "\\":
            out.append(char)
            i += 1
            continue
        nxt = text[i + 1 : i + 2]
        if nxt == "u" and i + 6 <= len(text):
            out.append(chr(int(text[i + 2 : i + 6], 16)))
            i += 6
        elif nxt == "x" and i + 4 <= len(text):
            out.append(chr(int(text[i + 2 : i + 4], 16)))
            i += 4
        else:
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
    return None


def _parsed_after(text: str, key: str) -> list[Any] | None:
    """The JSON array passed to the first `parseJSON('...')` after `key`."""
    at = text.find(key)
    if at < 0:
        return None
    call = text.find(PARSE_CALL, at)
    # The call has to belong to this key: a comma between them means the key
    # held something else and the call is a later property's.
    if call < 0 or "," in text[at + len(key) : call]:
        return None
    try:
        literal = _js_string_at(text, call + len(PARSE_CALL))
        value = json.loads(literal) if literal is not None else None
    except (ValueError, IndexError) as exc:
        logger.debug("Image block %s did not parse: %s", key, exc)
        return None
    return value if isinstance(value, list) else None


def parse_image_block(text: str) -> ImageBlock:
    """The gallery, `hiRes` falling back to `large`, and the listing videos."""
    images: list[str] = []
    for entry in _parsed_after(text, "'initial'") or []:
        if not isinstance(entry, dict):
            continue
        url = entry.get("hiRes") or entry.get("large")
        if isinstance(url, str) and url and url not in images:
            images.append(url)
    videos: list[ListingVideo] = []
    for entry in _parsed_after(text, "'videos'") or []:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url:
            continue
        duration = entry.get("durationSeconds")
        videos.append(
            ListingVideo(
                url=url,
                title=str(entry.get("title") or ""),
                duration_sec=float(duration)
                if isinstance(duration, int | float)
                else None,
            )
        )
    return ImageBlock(images=images, videos=videos)


def read_image_block(driver: Any) -> ImageBlock:
    """The page's image block, empty when the page has none or it can't be read."""
    try:
        text = driver.run_js(_IMAGE_BLOCK_JS)
    except Exception as exc:  # noqa: BLE001 - the element fallback takes over
        logger.debug("Could not read the image block: %s", exc)
        return ImageBlock()
    if not isinstance(text, str):
        return ImageBlock()
    return parse_image_block(text)
