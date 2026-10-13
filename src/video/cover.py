"""A render's cover image (REQ-VID-014, design 0011).

The cover is a frame of the finished video while the hook headline is on
screen: the opening visual with the headline, which sits at 28% of the frame
height, inside the centred 3:4 area Instagram's profile grid crops to. A
small record beside it keeps the frame's time, so a platform that picks its
cover by timestamp (TikTok) can point at the same frame.

A cover that can't be written is logged and skipped: the video is the
render's product, the cover only presents it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

COVER_FILE = "cover.jpg"
COVER_RECORD = "cover.json"


def cover_time_sec(hook_duration_sec: float) -> float:
    """Halfway through the hook, so the headline is fully on screen."""
    return round(max(hook_duration_sec, 0.0) / 2, 3)


async def write_cover(
    video: Path, run_root: Path, at_sec: float, ffmpeg_path: str = "ffmpeg"
) -> Path | None:
    """Write `cover.jpg` and `cover.json` into `run_root`; None on failure."""
    out = run_root / COVER_FILE
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
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
    except OSError as exc:
        logger.warning("Cover not written for %s: %s", video, exc)
        return None
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        logger.warning(
            "Cover not written for %s: ffmpeg exited %s: %s",
            video,
            proc.returncode,
            stderr.decode(errors="replace").strip()[:200],
        )
        return None
    try:
        (run_root / COVER_RECORD).write_text(
            json.dumps({"file": COVER_FILE, "at_ms": round(at_sec * 1000)}),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Cover record not written for %s: %s", video, exc)
    logger.info("Cover written: %s (at %.2fs)", out, at_sec)
    return out
