"""Where the spoken words in a voiceover end (design 0002).

The voiceover file's duration includes whatever silence the TTS provider pads
after the last word, so `peak` and `loop` endings measure the speech instead:
the start of a silence that runs to the end of the file.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Quieter than this, for at least this long, counts as silence.
SILENCE_NOISE_DB = -45
SILENCE_MIN_SEC = 0.15
# A silence ending this close to the file's end runs to the end.
EOF_SLACK_SEC = 0.05

_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_speech_end(silencedetect_log: str, duration: float) -> float:
    """The speech end from `silencedetect` output for a file of `duration`.

    A silence that runs to the end of the file ends the speech where it
    starts. ffmpeg 8 closes such a silence with a `silence_end` at the file's
    end; older releases leave it open. A file that ends on sound ends at
    `duration`.
    """
    starts = [float(m) for m in _START.findall(silencedetect_log)]
    ends = [float(m) for m in _END.findall(silencedetect_log)]
    if not starts:
        return duration
    open_at_end = len(starts) > len(ends)
    if open_at_end or ends[-1] >= duration - EOF_SLACK_SEC:
        return min(max(starts[-1], 0.0), duration)
    return duration


async def speech_end_sec(audio_path: Path, ffmpeg_path: str, duration: float) -> float:
    """Seconds into `audio_path` where the last spoken word ends.

    Falls back to `duration`, the whole file, when the measurement fails: a
    render that keeps a little silence beats one that cuts a word.
    """
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_SEC}",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
    except OSError as exc:
        logger.warning("Could not measure speech end of %s: %s", audio_path, exc)
        return duration
    if proc.returncode != 0:
        logger.warning(
            "Could not measure speech end of %s: ffmpeg exited %s",
            audio_path,
            proc.returncode,
        )
        return duration
    return parse_speech_end(stderr.decode(errors="replace"), duration)
