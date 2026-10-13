"""Where a render's cover lives, and reading it back (design 0011).

The render writes the cover (`src.video.cover`); the publisher reads it. Both
sides name the files here, since the publisher may not import the video
package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

COVER_SUFFIX = ".cover.jpg"
RECORD_SUFFIX = ".cover.json"


def cover_paths(video: Path) -> tuple[Path, Path]:
    """The cover image and its record for `video`, beside it."""
    return (
        video.with_name(video.stem + COVER_SUFFIX),
        video.with_name(video.stem + RECORD_SUFFIX),
    )


@dataclass(frozen=True)
class CoverRecord:
    """A video's cover: the image and the frame's time in the video."""

    image: Path
    at_ms: int


def read_cover(video: Path) -> tuple[CoverRecord | None, str]:
    """The cover recorded for `video`, or None with the reason there is none."""
    image, record = cover_paths(video)
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "no cover record"
    except (OSError, ValueError) as exc:
        return None, f"cover record unreadable ({exc})"
    at_ms = data.get("at_ms") if isinstance(data, dict) else None
    # bool is an int subclass; a record saying `true` is not a time.
    if not isinstance(at_ms, int) or isinstance(at_ms, bool) or at_ms < 0:
        return None, "cover record has no frame time"
    try:
        size = image.stat().st_size if image.is_file() else 0
    except OSError:
        size = 0
    if size == 0:
        return None, "cover image missing"
    return CoverRecord(image=image, at_ms=at_ms), ""
