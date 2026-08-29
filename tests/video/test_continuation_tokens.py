"""Whisper emits a word's tail as its own token, and the renderers space-join.

Three instances measured on real renders:

    ' 80'  + ',000'   -> burned as `80 ,000`
    ' 2'   + '.4GHz'  -> burned as `2 4GHz`
    ' go'  + '-to.'   -> burned as `go  -to.`

The decimal case changes meaning rather than spacing: the point is dropped, so
a viewer reads a different number.

The rule cannot key on digits, or on punctuation generally. The same wifi
script that produced `2 4GHz` lists channels as `1,` `6,` `11.` -- three
genuinely separate words carrying trailing punctuation. What distinguishes
them is where the separator sits, and whether Whisper marked a new word with
a leading space.
"""

from __future__ import annotations

import pytest

from src.video.stt_functions import _extract_word_timings
from src.video.subtitle_timing_smoother import join_continuations_in_result


def _result(*words: str) -> dict:
    return {
        "segments": [
            {
                "words": [
                    {"word": w, "start": i * 0.3, "end": i * 0.3 + 0.25}
                    for i, w in enumerate(words)
                ]
            }
        ]
    }


def _pycaps_words(*words: str) -> list[str]:
    """What the burned captions are rendered from."""
    return [
        w["word"]
        for w in join_continuations_in_result(_result(*words))["segments"][0]["words"]
    ]


def _ffmpeg_words(*words: str) -> list[str]:
    """The FFmpeg engine's list, built the way production builds it.

    The join runs on the result dict and the flat list is extracted from it
    afterwards. Extracting first and joining second protected pycaps only:
    `_extract_word_timings` strips each word, so the leading space -- the one
    signal separating a continuation from a word beginning with an apostrophe
    -- was already gone. FFmpeg is what a default install renders with, since
    the pycaps group is optional.
    """
    return [
        w["word"]
        for w in _extract_word_timings(join_continuations_in_result(_result(*words)))
    ]


class TestBothEnginesGetTheJoin:
    @pytest.mark.parametrize(
        "tokens,expected",
        [
            ((" 80", ",000"), "80,000"),
            ((" 2", ".4GHz"), "2.4GHz"),
            ((" go", "-to."), "go-to."),
            ((" it", "'s"), "it's"),
            ((" don", "’t"), "don’t"),
        ],
    )
    def test_a_continuation_is_folded(self, tokens, expected):
        assert _ffmpeg_words(*tokens) == [expected]
        assert _pycaps_words(*tokens) == [f" {expected}"]

    @pytest.mark.parametrize(
        "tokens",
        [
            (" 1,", " 6,", " 11."),
            (" wow!", ",000"),
            (" channel", ". Then"),
            (" get", " 'em"),
            (" the", " '90s"),
        ],
    )
    def test_these_are_separate_words(self, tokens):
        """Trailing punctuation, a non-alphanumeric follower, and Whisper's
        own new-word marker -- a leading space -- each keep words apart.
        """
        assert len(_ffmpeg_words(*tokens)) == len(tokens)
        assert len(_pycaps_words(*tokens)) == len(tokens)

    def test_the_joined_word_spans_both_tokens(self):
        joined = join_continuations_in_result(_result(" 80", ",000"))
        word = joined["segments"][0]["words"][0]

        assert word["start"] == pytest.approx(0.0)
        assert word["end"] == pytest.approx(0.55)

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

    def test_the_input_is_not_mutated(self):
        original = _result(" 80", ",000")

        join_continuations_in_result(original)

        assert len(original["segments"][0]["words"]) == 2


class TestTheWiringIsWhereItHasToBe:
    """Both wirings were deletable with the whole suite green."""

    @staticmethod
    def _tree(path: str):
        import ast
        from pathlib import Path

        return ast.parse(Path(path).read_text())

    def test_the_join_runs_before_the_flat_list_is_extracted(self):
        """Joining afterwards fixes pycaps and breaks FFmpeg."""
        import ast

        tree = self._tree("src/video/stt_functions.py")
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            and n.name == "generate_subtitles_with_whisper"
        )
        join_line = next(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "join_continuations_in_result"
        )
        extract_line = next(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_extract_word_timings"
        )

        assert join_line < extract_line

    def test_the_join_is_not_inside_the_timing_gate(self):
        """Those four rules are cosmetic and may be switched off."""
        import ast

        tree = self._tree("src/video/stt_functions.py")
        gate = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and "enabled" in ast.dump(n.test)
        )

        assert not [
            c
            for c in ast.walk(gate)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "join_continuations_in_result"
        ]

    def test_the_renderer_drops_the_punctuation_effect(self):
        """Deleting the call left the suite green."""
        import ast

        tree = self._tree("src/video/pycaps_engine/renderer.py")

        assert [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_drop_punctuation_stripping"
        ], "the pycaps pipeline no longer drops the effect that deletes periods"


class TestTheTemplateDoesNotEatTheDecimalPoint:
    """`word-focus` ships `RemovePunctuationMarksEffect(['.'])`.

    Implemented as `text.replace('.', '')` -- every period, not just a
    trailing one. Handing it the joined `2.4GHz` burns `24GHz`, and
    `$1,299.99` burns `$1,29999`: the same harm the join exists to remove,
    arriving by another route and reading as a confident wrong figure.
    """

    def test_the_effect_is_dropped_from_a_template_that_ships_it(self):
        pytest.importorskip(
            "pycaps",
            reason="optional group not installed (poetry install --with pycaps)",
        )
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
        pytest.importorskip(
            "pycaps",
            reason="optional group not installed (poetry install --with pycaps)",
        )
        from pycaps.template import TemplateFactory, TemplateLoader

        from src.video.pycaps_engine.renderer import _drop_punctuation_stripping

        builder = TemplateLoader(TemplateFactory().create("explosive")).load(False)
        before = list(builder._caps_pipeline._text_effects)

        _drop_punctuation_stripping(builder)

        assert builder._caps_pipeline._text_effects == before


def test_a_renamed_pycaps_attribute_degrades_rather_than_raises():
    """Needs no pycaps, so it stays outside the skipped class."""
    from src.video.pycaps_engine.renderer import _drop_punctuation_stripping

    class Builder:
        pass

    _drop_punctuation_stripping(Builder())
