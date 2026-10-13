"""A render's cover image (REQ-VID-014, design 0011).

The cover is a frame of the finished video while the hook headline is on
screen: the opening visual with the headline, which sits at 28% of the frame
height, inside the centred 3:4 area Instagram's profile grid crops to. A
small record beside it keeps the frame's time, so a platform that picks its
cover by timestamp (TikTok) can point at the same frame.

Each video gets its own cover, named after it, because a product renders
once per profile and a platform may be sent any of those videos.

A cover that can't be written is logged and skipped: the video is the
render's product, the cover only presents it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.utils.async_io import async_run_ffmpeg

logger = logging.getLogger(__name__)

COVER_SUFFIX = ".cover.jpg"
RECORD_SUFFIX = ".cover.json"
COVER_TIMEOUT_SEC = 60.0


def cover_paths(video: Path) -> tuple[Path, Path]:
    """The cover image and its record for `video`, beside it."""
    return (
        video.with_name(video.stem + COVER_SUFFIX),
        video.with_name(video.stem + RECORD_SUFFIX),
    )


def cover_time_sec(hook_duration_sec: float) -> float:
    """Halfway through the hook, so the headline is fully on screen."""
    return round(max(hook_duration_sec, 0.0) / 2, 3)


async def write_cover(
    video: Path, at_sec: float, ffmpeg_path: str = "ffmpeg"
) -> Path | None:
    """Write the cover and its record beside `video`; None on failure."""
    out, record = cover_paths(video)
    # A previous render's cover must not outlive a failed write.
    out.unlink(missing_ok=True)
    record.unlink(missing_ok=True)
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{at_sec:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out),
    ]
    ok, _, stderr = await async_run_ffmpeg(cmd, timeout_sec=COVER_TIMEOUT_SEC)
    if not ok or not out.is_file() or out.stat().st_size == 0:
        logger.warning("Cover not written for %s: %s", video, stderr.strip()[:200])
        out.unlink(missing_ok=True)
        return None
    try:
        record.write_text(
            json.dumps(
                {
                    "file": out.name,
                    "video": video.name,
                    "at_ms": round(at_sec * 1000),
                }
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Cover record not written for %s: %s", video, exc)
    logger.info("Cover written: %s (at %.2fs)", out, at_sec)
    return out
