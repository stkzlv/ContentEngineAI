"""Vocal event windows and the caption filter (REQ-CNT-183).

A performed take voices each delivery segment separately, so the time span of
every segment that carries a vocal event is known. After transcription, a word
inside such a span that matches what the tag was heard as in the probe
(`PERFORMED_TAG_TRANSCRIPTIONS`) is dropped from the caption words, but only
while the transcript holds more of that word than the script does: a word the
script says is never dropped.
"""

from __future__ import annotations

import json
import logging
import re
import wave
from collections import Counter
from pathlib import Path
from typing import Any

from src.video.config.audio_models import PERFORMED_TAG_TRANSCRIPTIONS
from src.video.delivery_plan import DeliverySegment

logger = logging.getLogger(__name__)

EVENTS_FILE = "vocal_events.json"
# Whisper's word times are coarse and the silence trim is estimated, so a
# window reaches this far past each end of its segment.
WINDOW_PAD_SEC = 0.3

_WORD = re.compile(r"[a-z0-9']+")


def _norm(word: str) -> str:
    return "".join(_WORD.findall(word.lower()))


def event_windows(
    segments: list[DeliverySegment], durations_sec: list[float]
) -> list[dict[str, Any]]:
    """One window per event: its segment's span in the joined audio, gap
    included, since an event after the last word can sound in the gap.
    """
    windows: list[dict[str, Any]] = []
    if len(durations_sec) != len(segments):
        return windows
    start = 0.0
    for segment, duration in zip(segments, durations_sec, strict=True):
        end = start + duration + segment.gap_after_ms / 1000
        for event in segment.events:
            windows.append({"tag": event.tag, "start": start, "end": end})
        start = end
    return windows


def leading_silence_sec(wav_path: Path, threshold_db: float) -> float:
    """Seconds before the first sample louder than `threshold_db` (dBFS)."""
    import numpy as np

    with wave.open(str(wav_path)) as w:
        width, channels, rate = w.getsampwidth(), w.getnchannels(), w.getframerate()
        data = w.readframes(w.getnframes())
    if width != 2 or not data:
        return 0.0
    samples = np.abs(np.frombuffer(data, dtype="<i2").astype(np.int32))
    loud = np.nonzero(samples > 32768 * 10 ** (threshold_db / 20))[0]
    return float(loud[0] // channels) / rate if loud.size else 0.0


def shift_windows(
    windows: list[dict[str, Any]], offset_sec: float
) -> list[dict[str, Any]]:
    """Move windows earlier by `offset_sec`, for audio trimmed at the start."""
    return [
        {**w, "start": max(0.0, w["start"] - offset_sec), "end": w["end"] - offset_sec}
        for w in windows
    ]


def write_windows(path: Path, windows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(windows, indent=2), encoding="utf-8")


def read_windows(path: Path | None) -> list[dict[str, Any]]:
    """The recorded windows, or none when the file is missing or unreadable."""
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            {"tag": str(w["tag"]), "start": float(w["start"]), "end": float(w["end"])}
            for w in data
        ]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning("Vocal events unreadable at %s: %s", path, exc)
        return []


def words_to_drop(
    words: list[tuple[str, float, float]],
    windows: list[dict[str, Any]],
    script: str,
) -> set[int]:
    """Indexes of transcript words heard from a vocal event, not spoken.

    `words` holds (text, start, end). A word is a candidate when it overlaps
    an event's padded window and is one of that tag's recorded transcriptions.
    Candidates are dropped only while the transcript holds more of the word
    than the script.
    """
    if not windows:
        return set()
    script_counts = Counter(_norm(w) for w in script.split())
    heard_counts = Counter(_norm(w[0]) for w in words)
    drop: set[int] = set()
    for i, (text, start, end) in enumerate(words):
        word = _norm(text)
        for window in windows:
            heard_as = PERFORMED_TAG_TRANSCRIPTIONS.get(window["tag"], frozenset())
            if word not in heard_as:
                continue
            if end < window["start"] - WINDOW_PAD_SEC:
                continue
            if start > window["end"] + WINDOW_PAD_SEC:
                continue
            if heard_counts[word] > script_counts[word]:
                drop.add(i)
                heard_counts[word] -= 1
            break
    return drop


def filter_timings(
    timings: list[dict[str, Any]], windows: list[dict[str, Any]], script: str | None
) -> list[dict[str, Any]]:
    """The flat timings (`word`, `start`, `end`) without event words."""
    if not windows or not script:
        return timings
    drop = words_to_drop(
        [(t["word"], float(t["start"]), float(t["end"])) for t in timings],
        windows,
        script,
    )
    return [t for i, t in enumerate(timings) if i not in drop]


def filter_result(
    result: dict[str, Any], windows: list[dict[str, Any]], script: str | None
) -> dict[str, Any]:
    """Drop event words from a Whisper result's segments, in place."""
    if not windows or not script or not isinstance(result.get("segments"), list):
        return result
    flat: list[tuple[int, int, dict[str, Any]]] = []
    for si, segment in enumerate(result["segments"]):
        for wi, word in enumerate(segment.get("words") or []):
            if all(k in word for k in ("word", "start", "end")):
                flat.append((si, wi, word))
    drop = words_to_drop(
        [(w["word"], float(w["start"]), float(w["end"])) for _, _, w in flat],
        windows,
        script,
    )
    if not drop:
        return result
    gone = {(flat[i][0], flat[i][1]) for i in drop}
    for si, segment in enumerate(result["segments"]):
        words = segment.get("words")
        if not words:
            continue
        kept = [w for wi, w in enumerate(words) if (si, wi) not in gone]
        if len(kept) != len(words):
            segment["words"] = kept
            segment["text"] = "".join(str(w.get("word", "")) for w in kept)
    if isinstance(result.get("text"), str):
        result["text"] = "".join(str(s.get("text", "")) for s in result["segments"])
    logger.info(
        "Caption filter dropped %d word(s) heard from vocal events: %s",
        len(drop),
        ", ".join(sorted(flat[i][2]["word"].strip() for i in drop)),
    )
    return result
