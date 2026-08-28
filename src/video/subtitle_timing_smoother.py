"""Post-processing for raw Whisper word timings.

Vanilla Whisper rounds word timestamps to whole seconds by default, producing
flicker and uneven segment durations in karaoke-style captions. This module
applies four best-practice smoothing rules (see docs/subtitle-best-practices.md
section 5) to the flat word timing list *before* it reaches either the FFmpeg
or pycaps subtitle engine.

The rules are:
1. Clamp minimum word duration (prevents imperceptible flashes)
2. Merge inter-word gaps shorter than a threshold into the preceding word
3. Hold the last word of each segment slightly past audio end
4. Lead audio so the word appears just before it's spoken
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Characters that can continue a word rather than start one. Whisper emits a
# thousands separator, a decimal point or a hyphen as its own token, so
# "80,000" arrives as "80" + ",000" and the renderers space-join it into
# "80 ,000". The decimal case is worse than cosmetic: "2.4GHz" becomes
# "2 4GHz", which reads as a different number.
_CONTINUATION_STARTS = (",", ".", "-", "'", "\u2019")


def _is_continuation(prev_word: str, word: str) -> bool:
    """Whether `word` continues `prev_word` rather than beginning a new one.

    Requires the separator to be followed by an alphanumeric and the previous
    token to end in one, so a token that merely *carries* punctuation is left
    alone. That distinction is the whole rule: in a script listing channels
    "1," "6," and "11." are three separate words with trailing punctuation,
    while ",000" is the tail of the number before it.
    """
    stripped = word.strip()
    if len(stripped) < 2 or not stripped.startswith(_CONTINUATION_STARTS):
        return False
    if not stripped[1].isalnum():
        return False
    return bool(prev_word.strip()) and prev_word.strip()[-1].isalnum()


def _join_continuations(
    words: list[dict[str, Any]], text_key: str, end_key: str
) -> list[dict[str, Any]]:
    """Fold continuation tokens into the word they continue.

    Runs before the timing rules: merging two words into one changes what the
    gap and duration rules are looking at, and a rule that has already padded
    a token about to disappear has padded nothing.
    """
    if not words:
        return words

    merged: list[dict[str, Any]] = [dict(words[0])]
    for current in words[1:]:
        previous = merged[-1]
        if _is_continuation(
            str(previous.get(text_key, "")), str(current.get(text_key, ""))
        ):
            previous[text_key] = (
                str(previous[text_key]).rstrip() + str(current[text_key]).strip()
            )
            # The joined word is spoken across both spans.
            previous[end_key] = current[end_key]
            continue
        merged.append(dict(current))
    return merged


def smooth_word_timings(
    timings: list[dict[str, Any]],
    *,
    min_word_sec: float = 0.12,
    gap_merge_sec: float = 0.08,
    hold_last_sec: float = 0.20,
    lead_sec: float = 0.04,
    segment_gap_threshold_sec: float = 0.40,
    hook_lead_sec: float = 0.0,
    hook_lead_word_count: int = 0,
) -> list[dict[str, Any]]:
    """Apply best-practice smoothing to raw STT word timings.

    Operates on the flat list format produced by ``_extract_word_timings``::

        [{"word": "Hello", "start_time": 0.0, "end_time": 0.6}, ...]

    Returns a new list; the input is not mutated.

    Args:
    ----
        timings: Word timing dicts with ``word``, ``start_time``, ``end_time``.
        min_word_sec: Rule 1 — minimum word duration. Words shorter than this
            are extended to this length. Default 120 ms.
        gap_merge_sec: Rule 2 — gaps between consecutive words shorter than
            this are absorbed into the preceding word. Default 80 ms.
        hold_last_sec: Rule 3 — extra hold time added to the last word of
            each detected segment. Default 200 ms.
        lead_sec: Rule 4 — how far ahead of the audio each word should
            appear. Default 40 ms.
        segment_gap_threshold_sec: Silence gap longer than this is treated as
            a segment boundary (used by rule 3). Default 400 ms matches
            ``word_timestamp_pause_threshold`` in subtitles.yaml.
        hook_lead_sec: Phase 1.2f hook-line lead. Extra shift applied to the
            first ``hook_lead_word_count`` words on top of ``lead_sec``.
            Default 0 (off). Research band is 0.10-0.30s.
        hook_lead_word_count: Number of leading words to apply
            ``hook_lead_sec`` to. Default 0 (off). Pair with the burned-in
            hook overlay so silent viewers parse the opening line before
            any audio cue.

    """
    if not timings:
        return timings

    # Rule 0: fold continuation tokens before anything measures the list.
    out = _join_continuations([dict(t) for t in timings], "word", "end_time")

    # Rule 4: lead — shift start earlier so the word appears just before
    # it's spoken. Don't touch end_time; that still marks the audio offset.
    hook_lead_n = max(0, hook_lead_word_count) if hook_lead_sec > 0 else 0
    for idx, t in enumerate(out):
        extra = hook_lead_sec if idx < hook_lead_n else 0.0
        t["start_time"] = max(0.0, t["start_time"] - lead_sec - extra)

    # Rule 2: merge short inter-word gaps into the preceding word.
    # Walk forward so each adjustment only affects the immediate pair.
    for i in range(1, len(out)):
        gap = out[i]["start_time"] - out[i - 1]["end_time"]
        if 0 <= gap < gap_merge_sec:
            out[i - 1]["end_time"] = out[i]["start_time"]

    # Rule 1: clamp minimum word duration. Extend end_time if needed.
    for t in out:
        dur = t["end_time"] - t["start_time"]
        if dur < min_word_sec:
            t["end_time"] = t["start_time"] + min_word_sec

    # Rule 3: hold the last word of each segment past audio end.
    # Detect segment boundaries via gaps exceeding the threshold.
    for i in range(len(out)):
        is_last_overall = i == len(out) - 1
        is_segment_end = False
        if not is_last_overall:
            gap_to_next = out[i + 1]["start_time"] - out[i]["end_time"]
            is_segment_end = gap_to_next >= segment_gap_threshold_sec
        if is_last_overall or is_segment_end:
            out[i]["end_time"] += hold_last_sec

    count = len(out)
    logger.debug(
        "Smoothed %d word timings (lead=%.0fms, min=%.0fms, "
        "gap=%.0fms, hold=%.0fms)",
        count,
        lead_sec * 1000,
        min_word_sec * 1000,
        gap_merge_sec * 1000,
        hold_last_sec * 1000,
    )
    return out


def smooth_whisper_result_dict(
    result_w: dict[str, Any],
    *,
    min_word_sec: float = 0.12,
    gap_merge_sec: float = 0.08,
    hold_last_sec: float = 0.20,
    lead_sec: float = 0.04,
    segment_gap_threshold_sec: float = 0.40,
    hook_lead_sec: float = 0.0,
    hook_lead_word_count: int = 0,
) -> dict[str, Any]:
    """Apply the same smoothing rules to the raw Whisper result dict.

    Pycaps consumes the ``whisper_json`` format directly (segments → words →
    start/end), so we need to smooth *that* dict in addition to the flat list.
    Returns a deep-enough copy; the original dict is not mutated.

    The word-level keys in Whisper's raw dict are ``"start"`` and ``"end"``
    (not ``"start_time"``/``"end_time"`` as in the extracted flat list).
    """
    import copy

    if not result_w or "segments" not in result_w:
        return result_w

    out = copy.deepcopy(result_w)
    # Phase 1.2f hook lead applies to the first N words across the entire
    # transcript, not the first N words of each segment.
    hook_lead_remaining = max(0, hook_lead_word_count) if hook_lead_sec > 0 else 0

    for segment in out["segments"]:
        words = segment.get("words")
        if not words:
            continue

        # Rule 0: fold continuation tokens before anything measures the list.
        # This is the path pycaps consumes, so it is the one that decides what
        # the burned caption reads.
        words = _join_continuations(words, "word", "end")
        segment["words"] = words

        # Rule 4: lead
        for w in words:
            if "start" in w:
                extra = hook_lead_sec if hook_lead_remaining > 0 else 0.0
                w["start"] = max(0.0, w["start"] - lead_sec - extra)
                if hook_lead_remaining > 0:
                    hook_lead_remaining -= 1

        # Rule 2: merge short gaps
        for i in range(1, len(words)):
            if "start" in words[i] and "end" in words[i - 1]:
                gap = words[i]["start"] - words[i - 1]["end"]
                if 0 <= gap < gap_merge_sec:
                    words[i - 1]["end"] = words[i]["start"]

        # Rule 1: clamp minimum duration
        for w in words:
            if "start" in w and "end" in w:
                dur = w["end"] - w["start"]
                if dur < min_word_sec:
                    w["end"] = w["start"] + min_word_sec

        # Rule 3: hold last word of this segment
        if words and "end" in words[-1]:
            words[-1]["end"] += hold_last_sec

        # Update segment-level start/end to match smoothed word boundaries
        first_start = next((w["start"] for w in words if "start" in w), None)
        last_end = next((w["end"] for w in reversed(words) if "end" in w), None)
        if first_start is not None:
            segment["start"] = first_start
        if last_end is not None:
            segment["end"] = last_end

    return out
