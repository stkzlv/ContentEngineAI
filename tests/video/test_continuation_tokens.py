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
    join_continuations_in_result,
    join_continuations_in_timings,
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
        assert [
            w["word"] for w in join_continuations_in_timings(_flat(*tokens))
        ] == expected

    def test_trailing_punctuation_does_not_merge_words(self):
        """The channel list from the same script that produced `2 4GHz`."""
        tokens = _flat((" 1,", 17.4, 17.8), (" 6,", 18.0, 18.4), (" 11.", 18.8, 19.4))

        assert [w["word"] for w in join_continuations_in_timings(tokens)] == [
            " 1,",
            " 6,",
            " 11.",
        ]

    def test_a_sentence_boundary_does_not_merge(self):
        """A separator followed by a space continues nothing."""
        tokens = _flat((" channel", 1.0, 1.4), (". Then", 1.5, 2.0))

        assert len(join_continuations_in_timings(tokens)) == 2

    def test_the_joined_word_spans_both_tokens(self):
        """It is spoken across both, so the caption has to hold that long.

        The join runs outside the smoothing rules now, so nothing pads it
        afterwards and the span is exactly the two tokens.
        """
        joined = join_continuations_in_timings(
            _flat((" 80", 12.5, 13.0), (",000", 12.9, 13.6))
        )

        assert len(joined) == 1
        assert joined[0]["end_time"] == pytest.approx(13.6)


class TestThePycapsPath:
    """The nested dict is what the burned captions are rendered from.

    Fixing only the flat list would leave the defect visible in exactly the
    renders where it was measured, since the bundled config uses pycaps.
    """

    def test_a_continuation_is_folded(self):
        result = join_continuations_in_result(
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

        join_continuations_in_result(original)

        assert [w["word"] for w in original["segments"][0]["words"]] == [" 80", ",000"]


class TestTheClausesThatWereUnpinned:
    """Two halves of the rule had no test and could be deleted silently."""

    def test_a_word_starting_with_an_apostrophe_is_not_a_continuation(self):
        """Whisper marks a new word with a leading space; `'em` has one.

        Stripping before the check threw that signal away and burned
        `get 'em` as `get'em`.
        """
        joined = join_continuations_in_timings(
            _flat(
                (" get", 1.0, 1.3),
                (" 'em", 1.3, 1.6),
                (" the", 1.7, 1.9),
                (" '90s", 1.9, 2.3),
            )
        )

        assert [w["word"] for w in joined] == [" get", " 'em", " the", " '90s"]

    def test_the_previous_word_must_end_alphanumeric(self):
        """Otherwise a separator chains onto punctuation and reads as noise."""
        joined = join_continuations_in_timings(
            _flat((" wow!", 1.0, 1.3), (",000", 1.3, 1.6))
        )

        assert len(joined) == 2

    def test_a_unicode_apostrophe_continues_a_word(self):
        """Whisper emits U+2019 where the source used a typographic quote."""
        joined = join_continuations_in_timings(
            _flat((" don", 1.0, 1.3), ("’t", 1.3, 1.6))
        )

        assert [w["word"] for w in joined] == [" don’t"]

    def test_a_word_missing_its_end_does_not_raise(self):
        """Every neighbouring rule guards; this one indexed directly."""
        result = join_continuations_in_result(
            {
                "segments": [
                    {
                        "words": [
                            {"word": " 80", "start": 1.0, "end": 1.3},
                            {"word": ",000", "start": 1.3},
                        ]
                    }
                ]
            }
        )

        assert [w["word"] for w in result["segments"][0]["words"]] == [" 80,000"]


class TestItIsNotGatedBehindTimingSmoothing:
    """The four smoothing rules are cosmetic and an operator may turn them off.

    A caption reading `2 4GHz` for `2.4GHz` is not a timing preference, so the
    join runs before that gate.
    """

    def test_the_stt_path_joins_outside_the_gate(self):
        import ast
        from pathlib import Path

        tree = ast.parse(Path("src/video/stt_functions.py").read_text())
        gate = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and "enabled" in ast.dump(n.test)
        )
        joins_inside = [
            c
            for c in ast.walk(gate)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id.startswith("join_continuations")
        ]

        assert not joins_inside, (
            "the rejoin sits inside the timing-smoothing gate; disabling a "
            "cosmetic timing knob would restore `2 4GHz`"
        )


class TestTheTemplateDoesNotEatTheDecimalPoint:
    """Joining the word is only half the fix on the default engine.

    `word-focus`, one of the two templates in the bundled pool, ships
    `RemovePunctuationMarksEffect(['.'])`, which is `text.replace('.', '')` --
    every period, not just a trailing one. Handing it the joined `2.4GHz`
    burns `24GHz`, and `$1,299.99` burns `$1,29999`. That is the same harm the
    join exists to remove, arriving by another route.
    """

    def test_the_effect_is_dropped_from_a_template_that_ships_it(self):
        from pycaps.template import TemplateFactory, TemplateLoader

        from src.video.pycaps_engine.renderer import _drop_punctuation_stripping

        builder = TemplateLoader(TemplateFactory().create("word-focus")).load(False)
        before = [type(e).__name__ for e in builder._caps_pipeline._text_effects]

        _drop_punctuation_stripping(builder)

        after = [type(e).__name__ for e in builder._caps_pipeline._text_effects]
        assert "RemovePunctuationMarksEffect" in before, (
            "word-focus no longer ships the effect; if upstream removed it, "
            "this guard and its rationale can go"
        )
        assert "RemovePunctuationMarksEffect" not in after

    def test_a_template_without_it_is_untouched(self):
        from pycaps.template import TemplateFactory, TemplateLoader

        from src.video.pycaps_engine.renderer import _drop_punctuation_stripping

        builder = TemplateLoader(TemplateFactory().create("explosive")).load(False)
        before = list(builder._caps_pipeline._text_effects)

        _drop_punctuation_stripping(builder)

        assert builder._caps_pipeline._text_effects == before

    def test_a_renamed_pycaps_attribute_degrades_rather_than_raises(self):
        from src.video.pycaps_engine.renderer import _drop_punctuation_stripping

        class Builder:
            pass

        _drop_punctuation_stripping(Builder())
