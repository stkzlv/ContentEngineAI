"""Tests for subtitle timing post-processing (smoother).

Covers the four best-practice rules applied to raw Whisper word timings:
1. Minimum word duration clamp
2. Short inter-word gap merge
3. Last-word-of-segment hold
4. Audio lead (pre-display offset)

Also covers the parallel path that smooths the raw Whisper result dict
for pycaps consumption.
"""

from __future__ import annotations

import copy

import pytest

from src.video.subtitle_timing_smoother import (
    smooth_whisper_result_dict,
    smooth_word_timings,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start_time": start, "end_time": end}


def _make_sentence() -> list[dict]:
    """A short sentence with realistic Whisper timings."""
    return [
        _word("This", 0.50, 0.72),
        _word("is", 0.74, 0.80),
        _word("a", 0.82, 0.86),
        _word("test", 0.88, 1.20),
        _word("sentence", 1.22, 1.80),
    ]


def _make_two_segments() -> list[dict]:
    """Two segments separated by a gap > 0.4s."""
    return [
        _word("First", 0.50, 0.90),
        _word("part", 0.92, 1.30),
        # gap of 0.6s — segment boundary
        _word("Second", 1.90, 2.30),
        _word("part", 2.32, 2.70),
    ]


def _make_whisper_dict(words_per_segment: list[list[dict]]) -> dict:
    """Build a minimal Whisper result dict from word lists.

    Word dicts here use Whisper's raw keys ("start"/"end"), not the
    extracted flat-list keys ("start_time"/"end_time").
    """
    segments = []
    for i, words in enumerate(words_per_segment):
        seg_words = [
            {"word": w["word"], "start": w["start"], "end": w["end"]} for w in words
        ]
        segments.append(
            {
                "id": i,
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": " ".join(w["word"] for w in words),
                "words": seg_words,
            }
        )
    return {"language": "en", "text": "test", "segments": segments}


# ---------------------------------------------------------------------------
# smooth_word_timings
# ---------------------------------------------------------------------------


class TestSmoothWordTimings:
    """Tests for the flat-list smoother."""

    def test_empty_input(self):
        assert smooth_word_timings([]) == []

    def test_single_word(self):
        timings = [_word("Hello", 1.0, 1.5)]
        result = smooth_word_timings(timings)
        assert len(result) == 1
        # Rule 4: lead shifts start earlier
        assert result[0]["start_time"] == pytest.approx(0.96)
        # Rule 3: hold adds to end (single word = last of segment)
        assert result[0]["end_time"] == pytest.approx(1.70)

    def test_does_not_mutate_input(self):
        timings = [_word("Hello", 1.0, 1.5)]
        original = copy.deepcopy(timings)
        smooth_word_timings(timings)
        assert timings == original

    def test_lead_shifts_start_earlier(self):
        timings = [_word("Word", 0.50, 1.00)]
        result = smooth_word_timings(timings, lead_sec=0.10)
        assert result[0]["start_time"] == pytest.approx(0.40)

    def test_lead_clamps_to_zero(self):
        timings = [_word("Word", 0.02, 0.50)]
        result = smooth_word_timings(timings, lead_sec=0.10)
        assert result[0]["start_time"] == 0.0

    def test_min_word_duration_clamp(self):
        # Word with 0.04s duration, shorter than default 0.12s
        timings = [_word("a", 1.00, 1.04)]
        result = smooth_word_timings(
            timings, min_word_sec=0.12, lead_sec=0.0, hold_last_sec=0.0
        )
        assert result[0]["end_time"] == pytest.approx(1.12)

    def test_gap_merge(self):
        # Two words with a 0.05s gap (under default 0.08s threshold)
        timings = [
            _word("Hello", 1.00, 1.30),
            _word("world", 1.35, 1.80),
        ]
        result = smooth_word_timings(
            timings, gap_merge_sec=0.08, lead_sec=0.0, hold_last_sec=0.0
        )
        # Gap of 0.05s should be merged: first word's end → second word's start
        # After lead (0.0), word1 end should extend to word2 start
        assert result[0]["end_time"] == pytest.approx(result[1]["start_time"])

    def test_gap_not_merged_when_large(self):
        # Two words with a 0.20s gap (above 0.08s threshold)
        timings = [
            _word("Hello", 1.00, 1.30),
            _word("world", 1.50, 1.80),
        ]
        result = smooth_word_timings(
            timings, gap_merge_sec=0.08, lead_sec=0.0, hold_last_sec=0.0
        )
        # Gap is too large — don't merge
        assert result[0]["end_time"] == pytest.approx(1.30)

    def test_hold_last_word_of_segment(self):
        timings = _make_two_segments()
        result = smooth_word_timings(
            timings,
            hold_last_sec=0.20,
            lead_sec=0.0,
            segment_gap_threshold_sec=0.40,
        )
        # "part" at index 1 is last word of first segment
        assert result[1]["end_time"] == pytest.approx(1.30 + 0.20)
        # "part" at index 3 is last word overall
        assert result[3]["end_time"] == pytest.approx(2.70 + 0.20)

    def test_hold_not_applied_to_mid_segment_words(self):
        timings = _make_two_segments()
        result = smooth_word_timings(
            timings,
            hold_last_sec=0.20,
            lead_sec=0.0,
            gap_merge_sec=0.0,  # disable gap merge to isolate hold rule
            segment_gap_threshold_sec=0.40,
        )
        # "First" at index 0 is NOT the last word of a segment — no hold added
        assert result[0]["end_time"] == pytest.approx(0.90)

    def test_all_rules_combined(self):
        """Smoke test with all rules active on a realistic sentence."""
        timings = _make_sentence()
        result = smooth_word_timings(timings)
        assert len(result) == len(timings)
        # All start times should be non-negative
        for t in result:
            assert t["start_time"] >= 0.0
        # All words should have at least min_word_sec duration
        for t in result:
            dur = t["end_time"] - t["start_time"]
            assert dur >= 0.12 - 1e-9  # float tolerance

    def test_custom_params(self):
        timings = [_word("test", 2.00, 2.50)]
        result = smooth_word_timings(
            timings,
            min_word_sec=0.20,
            gap_merge_sec=0.10,
            hold_last_sec=0.30,
            lead_sec=0.05,
        )
        # lead: 2.00 - 0.05 = 1.95
        assert result[0]["start_time"] == pytest.approx(1.95)
        # hold: 2.50 + 0.30 = 2.80
        assert result[0]["end_time"] == pytest.approx(2.80)


# ---------------------------------------------------------------------------
# smooth_whisper_result_dict
# ---------------------------------------------------------------------------


class TestSmoothWhisperResultDict:
    """Tests for the raw Whisper dict smoother (pycaps path)."""

    def test_empty_or_missing_segments(self):
        assert smooth_whisper_result_dict({}) == {}
        assert smooth_whisper_result_dict({"text": "hi"}) == {"text": "hi"}

    def test_does_not_mutate_input(self):
        raw_words = [
            {"word": "Hello", "start": 0.50, "end": 0.90},
            {"word": "world", "start": 0.92, "end": 1.30},
        ]
        original = _make_whisper_dict([raw_words])
        frozen = copy.deepcopy(original)
        smooth_whisper_result_dict(original)
        assert original == frozen

    def test_lead_applied(self):
        raw_words = [
            {"word": "Hello", "start": 0.50, "end": 0.90},
        ]
        result = smooth_whisper_result_dict(
            _make_whisper_dict([raw_words]), lead_sec=0.10
        )
        words = result["segments"][0]["words"]
        assert words[0]["start"] == pytest.approx(0.40)

    def test_segment_boundaries_updated(self):
        raw_words = [
            {"word": "Hello", "start": 0.50, "end": 0.90},
            {"word": "world", "start": 0.92, "end": 1.30},
        ]
        result = smooth_whisper_result_dict(
            _make_whisper_dict([raw_words]),
            lead_sec=0.04,
            hold_last_sec=0.20,
        )
        seg = result["segments"][0]
        # Segment start should match first word's smoothed start
        assert seg["start"] == pytest.approx(seg["words"][0]["start"])
        # Segment end should match last word's smoothed end (with hold)
        assert seg["end"] == pytest.approx(seg["words"][-1]["end"])

    def test_multiple_segments(self):
        seg1_words = [
            {"word": "First", "start": 0.50, "end": 0.90},
        ]
        seg2_words = [
            {"word": "Second", "start": 2.00, "end": 2.50},
        ]
        result = smooth_whisper_result_dict(
            _make_whisper_dict([seg1_words, seg2_words]),
            hold_last_sec=0.20,
            lead_sec=0.0,
        )
        # Each segment's last word gets the hold
        assert result["segments"][0]["words"][0]["end"] == pytest.approx(1.10)
        assert result["segments"][1]["words"][0]["end"] == pytest.approx(2.70)

    def test_min_word_duration_in_dict(self):
        raw_words = [
            {"word": "a", "start": 1.00, "end": 1.02},  # 20ms — too short
        ]
        result = smooth_whisper_result_dict(
            _make_whisper_dict([raw_words]),
            min_word_sec=0.12,
            lead_sec=0.0,
            hold_last_sec=0.0,
        )
        w = result["segments"][0]["words"][0]
        assert w["end"] == pytest.approx(1.12)
