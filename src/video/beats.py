"""Music beats and cuts snapped to them (design 0005).

Beats come from `librosa.beat.beat_track`, once per track, cached beside it as
`<track>.beats.json` keyed to the file's size and modification time. librosa
is optional: without it the detection warns once and returns nothing, and
the cuts stay where the timeline put them.

Snapping moves each crossfade midpoint to the nearest beat within a window.
A move is skipped when it would shorten a segment below the minimum, reach the
render's end, or lengthen a video clip (a still can hold longer; a clip may
have no frames to spare). Only the cuts move; the voiceover and the captions
keep their timing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_SUFFIX = ".beats.json"


def _cache_path(track: Path) -> Path:
    return track.with_name(track.name + _CACHE_SUFFIX)


def _stamp(track: Path) -> dict[str, int]:
    st = track.stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _read_cache(track: Path) -> list[float] | None:
    try:
        data = json.loads(_cache_path(track).read_text(encoding="utf-8"))
        beats = data.get("beats") if data.get("stamp") == _stamp(track) else None
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(beats, list) or not all(
        isinstance(b, int | float) for b in beats
    ):
        return None
    return [float(b) for b in beats]


def detect_beats(track: Path) -> list[float] | None:
    """Beat times in seconds, from the cache or librosa; None when unavailable."""
    cached = _read_cache(track)
    if cached is not None:
        return cached
    try:
        import librosa.beat  # type: ignore[import-untyped, import-not-found, unused-ignore]
        from librosa.util.exceptions import (  # type: ignore[import-untyped, import-not-found, unused-ignore]
            LibrosaError,
        )
    except ImportError:
        logger.warning("Beat snapping needs librosa: poetry install --with beats")
        return None
    try:
        samples, rate = librosa.load(str(track), sr=None, mono=True)
        _, frames = librosa.beat.beat_track(y=samples, sr=rate)
        beats = [float(t) for t in librosa.frames_to_time(frames, sr=rate)]
    except (OSError, ValueError, RuntimeError, AttributeError, LibrosaError) as exc:
        logger.warning("Beat detection failed for %s: %s", track.name, exc)
        return None
    try:
        _cache_path(track).write_text(
            json.dumps({"beats": beats, "stamp": _stamp(track)}), encoding="utf-8"
        )
    except OSError as exc:
        logger.debug("Could not cache beats for %s: %s", track.name, exc)
    return beats


def snap_durations(
    durations: list[float],
    is_video: list[bool],
    transition_sec: float,
    beats: list[float],
    window_sec: float,
    min_segment_sec: float,
    end_sec: float,
) -> list[float]:
    """Segment durations with each cut moved to the nearest beat in the window.

    A cut is the middle of the crossfade between two segments. Moving it by
    `delta` lengthens one neighbour and shortens the other by the same amount,
    so the timeline's total length does not change.
    """
    out = list(durations)
    offset = 0.0
    for i in range(1, len(out)):
        offset += out[i - 1] - transition_sec
        cut = offset + transition_sec / 2
        near = [b for b in beats if abs(b - cut) <= window_sec]
        if not near:
            continue
        delta = min(near, key=lambda b: abs(b - cut)) - cut
        longer = i - 1 if delta > 0 else i
        if is_video[longer]:
            continue
        before, after = out[i - 1] + delta, out[i] - delta
        if min(before, after) < min_segment_sec or cut + delta >= end_sec:
            continue
        out[i - 1], out[i] = before, after
        offset += delta
    return out
