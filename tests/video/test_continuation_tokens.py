"""Whisper emits a word's tail as its own token, and the renderers space-join.

Three instances measured on real renders:

    ' 80'  + ',000'   -> burned as `80 ,000`
    ' 2'   + '.4GHz'  -> burned as `2 4GHz`
    ' go'  + '-to.'   -> burned as `go  -to.`

The decimal case is the one that changes meaning rather than spacing: the
point is dropped entirely, so a viewer reads a different number.

The rule cannot key on digits, or on punctuation generally. The same wifi
script that produced `2 4GHz` also lists channels as `1,` `6,` `11.` -- three
genuinely separate words carrying trailing punctuation. What distinguishes
them is where the separator sits.
"""

from __future__ import annotations

import pytest

from src.video.subtitle_timing_smoother import (
    smooth_whisper_result_dict,
    smooth_word_timings,
)


def _flat(*pairs):
    return [{"word": w, "start_time": s, "end_time": e} for w, s, e in pairs]


class TestTheFlatListPath:
    """Consumed by the FFmpeg subtitle engine."""

    @pytest.mark.parametrize(
        "tokens,expected",
        [
            ([(" 80", 12.5, 13.0), (",000", 12.9, 13.6)], [" 80,000"]),
            ([(" 2", 6.5, 6.7), (".4GHz", 6.7, 7.8)], [" 2.4GHz"]),
            ([(" go", 20.0, 20.3), ("-to.", 20.3, 20.9)], [" go-to."]),
            ([(" it", 1.0, 1.2), ("'s", 1.2, 1.4)], [" it's"]),
        ],
    )
    def test_a_continuation_is_folded_into_the_word_before_it(self, tokens, expected):
        assert [w["word"] for w in smooth_word_timings(_flat(*tokens))] == expected

    def test_trailing_punctuation_does_not_merge_words(self):
        """The channel list from the same script that produced `2 4GHz`."""
        tokens = _flat((" 1,", 17.4, 17.8), (" 6,", 18.0, 18.4), (" 11.", 18.8, 19.4))

        assert [w["word"] for w in smooth_word_timings(tokens)] == [
            " 1,",
            " 6,",
            " 11.",
        ]

    def test_a_sentence_boundary_does_not_merge(self):
        """A separator followed by a space continues nothing."""
        tokens = _flat((" channel", 1.0, 1.4), (". Then", 1.5, 2.0))

        assert len(smooth_word_timings(tokens)) == 2

    def test_the_joined_word_spans_both_tokens(self):
        """It is spoken across both, so the caption has to hold that long.

        Asserted as a lower bound: the segment-hold rule adds its 200 ms on
        top afterwards, and pinning the exact value would make this a test of
        that rule instead.
        """
        joined = smooth_word_timings(_flat((" 80", 12.5, 13.0), (",000", 12.9, 13.6)))

        assert len(joined) == 1
        assert joined[0]["end_time"] >= 13.6


class TestThePycapsPath:
    """The nested dict is what the burned captions are rendered from.

    Fixing only the flat list would leave the defect visible in exactly the
    renders where it was measured, since the bundled config uses pycaps.
    """

    def test_a_continuation_is_folded(self):
        result = smooth_whisper_result_dict(
            {
                "segments": [
                    {
                        "words": [
                            {"word": " 2", "start": 6.5, "end": 6.7},
                            {"word": ".4GHz", "start": 6.7, "end": 7.8},
                        ]
                    }
                ]
            }
        )

        assert [w["word"] for w in result["segments"][0]["words"]] == [" 2.4GHz"]

    def test_the_input_is_not_mutated(self):
        """Its docstring promises this, and the join edits words in place."""
        original = {
            "segments": [
                {
                    "words": [
                        {"word": " 80", "start": 12.5, "end": 13.0},
                        {"word": ",000", "start": 12.9, "end": 13.6},
                    ]
                }
            ]
        }

        smooth_whisper_result_dict(original)

        assert [w["word"] for w in original["segments"][0]["words"]] == [" 80", ",000"]
