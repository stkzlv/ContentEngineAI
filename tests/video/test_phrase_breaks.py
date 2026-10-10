"""Caption segments end at phrase boundaries (REQ-VID-157)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.video.config.subtitle_models import PycapsSettings
from src.video.pycaps_engine.phrase_breaks import (
    break_score,
    segment_end,
    split_words,
)

REPO = Path(__file__).resolve().parents[2]


def _segments(text: str, max_chars: int = 25, min_chars: int = 10) -> list[str]:
    words = text.split()
    ends = split_words(words, max_chars, min_chars)
    return [" ".join(words[a:b]) for a, b in zip([0, *ends], ends, strict=False)]


def _greedy(words: list[str], max_chars: int, min_chars: int) -> list[int]:
    """The `limit_by_chars` rule as pycaps applies it, for comparison."""
    ends, start = [], 0
    while start < len(words):
        end, chars = start, 0
        while end < len(words) and chars + len(words[end]) <= max_chars:
            chars += len(words[end])
            end += 1
        end = max(end, start + 1)
        if sum(len(w) for w in words[end:]) < min_chars:
            end = len(words)
        ends.append(end)
        start = end
    return ends


@pytest.mark.unit
@pytest.mark.req("REQ-VID-157")
class TestWhereASegmentEnds:
    def test_a_clause_starts_on_its_conjunction(self) -> None:
        assert _segments("Your router is slow because it needs a reboot.") == [
            "Your router is slow",
            "because it needs a reboot.",
        ]

    def test_never_after_an_article(self) -> None:
        segments = _segments(
            "You do this by unplugging the power for 30 seconds, then plugging it back in."
        )

        assert not any(s.split()[-1] in {"the", "a", "your"} for s in segments)
        assert "the power for 30 seconds," in segments

    @pytest.mark.parametrize(
        ("text", "together"),
        [
            (
                "Simply plug it in and press the button to start the cleaning cycle",
                "plug it in",
            ),
            (
                "Next you need to turn on the power switch located at the back",
                "turn on",
            ),
            ("Then log out of every account you share with others", "log out"),
        ],
    )
    def test_a_phrasal_verb_keeps_its_particle(self, text: str, together: str) -> None:
        for limits in ((25, 10), (20, 15)):
            segments = _segments(text, *limits)
            assert any(together in s for s in segments), (limits, segments)

    @pytest.mark.parametrize("word", ["under", "through", "has", "been"])
    def test_more_binding_words(self, word: str) -> None:
        assert break_score([word, "the"], 1) < 0

    def test_a_comma_is_the_best_break(self) -> None:
        assert _segments("To fix this, first close unnecessary browser tabs.")[0] == (
            "To fix this,"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "Open Settings on your iPhone 15 Pro and tap Battery now.",
            "This charger tops up a Galaxy S24 Ultra in under an hour.",
            "Press Windows 11 Start and then search for Snipping Tool app.",
        ],
    )
    def test_a_name_stays_on_one_screen(self, text: str) -> None:
        segments = _segments(text)
        for name in (
            "iPhone 15 Pro",
            "Galaxy S24 Ultra",
            "Windows 11",
            "Snipping Tool",
        ):
            if name in text:
                assert any(name in s for s in segments), segments

    def test_a_capital_after_a_full_stop_is_not_a_name(self) -> None:
        """'Open' starts a sentence, so 'Open | Settings' may be split."""
        assert break_score(["done.", "Open", "Settings"], 2) >= 0
        assert break_score(["the", "Snipping", "Tool"], 2) < 0

    def test_a_break_too_early_for_the_minimum_is_skipped(self) -> None:
        words = "Hi, restart your router today and wait please".split()

        assert segment_end(words, 0, 25, 10) > 1

    def test_the_limits_hold(self) -> None:
        text = (
            "Before this smartwatch, I used to pull my phone out for everything, "
            "and now I can see texts and even take calls right from my wrist."
        )
        words = text.split()
        ends = split_words(words, 25, 10)
        for a, b in zip([0, *ends], ends, strict=False):
            assert sum(len(w) for w in words[a:b]) <= 25 or b == len(words)
        assert ends[-1] == len(words)

    def test_with_no_good_break_it_splits_as_pycaps_does(self) -> None:
        words = "aaaa bbbb cccc dddd eeee ffff gggg hhhh".split()

        assert split_words(words, 12, 4) == _greedy(words, 12, 4)

    def test_a_short_remainder_joins_the_segment_as_in_pycaps(self) -> None:
        words = "Unplug the router and wait a bit".split()

        assert segment_end(words, 0, 25, 15) == len(words)

    def test_a_word_longer_than_the_limit_is_its_own_segment(self) -> None:
        assert segment_end(["Supercalifragilistic", "is", "long"], 0, 10, 2) == 1


class _Builder:
    def __init__(self, splitters: list) -> None:
        self._caps_pipeline = types.SimpleNamespace(
            _layout_options=MagicMock(),
            _text_effects=[],
            _semantic_tagger=types.SimpleNamespace(_ai_rules={}),
            _sound_effects=[],
            _animators=[],
            _segment_splitters=splitters,
        )

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            return self

        return _record


class LimitByCharsSplitter:
    """Named like pycaps' splitter, carrying its limits."""

    def __init__(self, max_limit: int, min_limit: int, avoid: int = 0) -> None:
        self._max_limit = max_limit
        self._min_limit = min_limit
        self._avoid_finishing_segment_with_word_shorter_than = avoid


class SplitIntoSentencesSplitter:
    pass


def _fake_pycaps(monkeypatch: pytest.MonkeyPatch, builder: _Builder) -> None:
    template = types.ModuleType("pycaps.template")
    template.TemplateFactory = lambda: types.SimpleNamespace(create=lambda n: n)

    class _Loader:
        def __init__(self, template):
            pass

        def with_input_video(self, _):
            return self

        def load(self, _):
            return builder

    template.TemplateLoader = _Loader
    transcriber = types.ModuleType("pycaps.transcriber")
    transcriber.TranscriptFormat = types.SimpleNamespace(WHISPER_JSON="whisper")
    renderer = types.ModuleType("pycaps.renderer")
    renderer.PictexSubtitleRenderer = lambda: "pictex-renderer"
    root = types.ModuleType("pycaps")
    root.template, root.transcriber, root.renderer = template, transcriber, renderer
    for name, mod in {
        "pycaps": root,
        "pycaps.template": template,
        "pycaps.transcriber": transcriber,
        "pycaps.renderer": renderer,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


def _build(monkeypatch, tmp_path: Path, splitters: list, **settings) -> list:
    from src.video.pycaps_engine.renderer import PycapsRenderer

    builder = _Builder(splitters)
    _fake_pycaps(monkeypatch, builder)
    monkeypatch.setattr(
        "src.video.pycaps_engine.renderer.merge_layout_with_template",
        lambda *a, **k: MagicMock(),
    )
    PycapsRenderer._build_pipeline(
        PycapsRenderer.__new__(PycapsRenderer),
        input_video=tmp_path / "in.mp4",
        transcript_path=tmp_path / "t.json",
        output_video=tmp_path / "out.mp4",
        template_name="word-focus",
        visual_bounds=None,
        settings=PycapsSettings(outline_px=0, **settings),
    )
    result: list = builder._caps_pipeline._segment_splitters
    return result


@pytest.mark.unit
@pytest.mark.req("REQ-VID-157")
class TestTheSwitch:
    def test_on_swaps_the_char_splitter_and_keeps_its_limits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        sentences = SplitIntoSentencesSplitter()
        splitters = _build(
            monkeypatch,
            tmp_path,
            [sentences, LimitByCharsSplitter(20, 15)],
            phrase_breaks=True,
        )

        assert splitters[0] is sentences
        assert type(splitters[1]).__name__ == "_PhraseBoundarySplitter"
        assert (splitters[1]._max_limit, splitters[1]._min_limit) == (20, 15)

    def test_off_keeps_the_template_splitter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        original = LimitByCharsSplitter(20, 15)

        assert _build(monkeypatch, tmp_path, [original]) == [original]

    def test_a_splitter_with_the_short_word_rule_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        original = LimitByCharsSplitter(20, 15, avoid=3)

        assert _build(monkeypatch, tmp_path, [original], phrase_breaks=True) == [
            original
        ]

    def test_the_shipped_config_turns_it_on(self) -> None:
        import yaml

        raw = yaml.safe_load((REPO / "config" / "subtitles.yaml").read_text())

        assert PycapsSettings(**raw["subtitle_settings"]["pycaps"]).phrase_breaks


@pytest.mark.unit
@pytest.mark.req("REQ-VID-157")
class TestAgainstTheRealLibrary:
    @pytest.fixture
    def pycaps(self):
        return pytest.importorskip(
            "pycaps",
            reason="optional group not installed (poetry install --with pycaps)",
        )

    @staticmethod
    def _document(text: str):
        from pycaps.common import Document, Line, Segment, TimeFragment, Word

        words = text.split()
        document = Document()
        segment = Segment(time=TimeFragment(start=0.0, end=float(len(words))))
        line = Line(time=segment.time)
        line.words.set_all(
            [
                Word(text=w, time=TimeFragment(start=float(i), end=i + 0.9))
                for i, w in enumerate(words)
            ]
        )
        segment.lines.add(line)
        document.segments.add(segment)
        return document

    @staticmethod
    def _texts(document) -> list[str]:
        return [" ".join(w.text for w in s.lines[0].words) for s in document.segments]

    def test_both_pool_templates_get_the_swap(self, pycaps):
        from pycaps.template import TemplateFactory, TemplateLoader

        from src.video.pycaps_engine.renderer import _use_phrase_breaks

        for name in ("explosive", "word-focus"):
            builder = TemplateLoader(TemplateFactory().create(name)).load(False)
            _use_phrase_breaks(builder)
            kinds = [
                type(s).__name__ for s in builder._caps_pipeline._segment_splitters
            ]
            assert "LimitByCharsSplitter" not in kinds, name
            assert "_PhraseBoundarySplitter" in kinds, name

    def test_the_real_split_moves_the_break(self, pycaps):
        from pycaps.transcriber import LimitByCharsSplitter as Real

        from src.video.pycaps_engine.renderer import _PhraseBoundarySplitter

        text = "Your router is slow because it needs a reboot."
        stock, ours = self._document(text), self._document(text)
        Real(max_limit=25, min_limit=10).split(stock)
        _PhraseBoundarySplitter(25, 10).split(ours)

        assert self._texts(stock) == [
            "Your router is slow because it",
            "needs a reboot.",
        ]
        assert self._texts(ours) == [
            "Your router is slow",
            "because it needs a reboot.",
        ]
        times = [(s.time.start, s.time.end) for s in ours.segments]
        assert times == [(0.0, 3.9), (4.0, 8.9)]

    def test_with_no_better_break_it_matches_the_library(self, pycaps):
        from pycaps.transcriber import LimitByCharsSplitter as Real

        from src.video.pycaps_engine.renderer import _PhraseBoundarySplitter

        text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh"
        stock, ours = self._document(text), self._document(text)
        Real(max_limit=12, min_limit=4).split(stock)
        _PhraseBoundarySplitter(12, 4).split(ours)

        assert self._texts(ours) == self._texts(stock)
