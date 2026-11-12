"""Lines inside a pycaps caption break at phrase boundaries (REQ-VID-157)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.video.config.subtitle_models import PycapsSettings
from src.video.phrase_breaks import line_ends

pytestmark = pytest.mark.req("REQ-VID-157")

REPO = Path(__file__).resolve().parents[2]
SPACE = 10


def _widths(words: list[str]) -> list[float]:
    return [10.0 * len(w) for w in words]


def _lines(text: str, max_width: float) -> list[str]:
    words = text.split()
    ends = line_ends(words, _widths(words), SPACE, max_width)
    return [" ".join(words[a:b]) for a, b in zip([0, *ends], ends, strict=False)]


def _pycaps_greedy(widths: list[float], space: float, max_width: float) -> list[int]:
    """The pycaps `LineSplitter` loop, as end indices, for comparison."""
    ends: list[int] = []
    count, total = 0, 0.0
    for index, width in enumerate(widths):
        if total + width + space <= max_width:
            count += 1
            total += space + width
        else:
            if count:
                ends.append(index)
            count, total = 1, width
    ends.append(len(widths))
    return ends


class TestWhereALineBreaks:
    def test_a_line_does_not_end_on_an_article(self) -> None:
        # Greedy: "Tap the" | "button".
        assert _lines("Tap the button", 100) == ["Tap", "the button"]

    def test_a_name_stays_on_one_line(self) -> None:
        # Greedy: "Get iPhone" | "15 now".
        assert _lines("Get iPhone 15 now", 130) == ["Get", "iPhone 15 now"]

    def test_a_break_never_adds_a_line(self) -> None:
        # Moving "the" down would need a third line, so the greedy break stays.
        words = "Open the Settings app".split()
        greedy = _pycaps_greedy(_widths(words), SPACE, 100)
        assert line_ends(words, _widths(words), SPACE, 100) == greedy

    def test_with_no_better_break_it_matches_pycaps(self) -> None:
        words = "Battery holds charge seven hours".split()
        for max_width in (90, 120, 160, 250):
            assert line_ends(words, _widths(words), SPACE, max_width) == _pycaps_greedy(
                _widths(words), SPACE, max_width
            )

    def test_a_word_wider_than_the_line_sits_alone(self) -> None:
        assert _lines("Supercalifragilistic is long", 100) == [
            "Supercalifragilistic",
            "is long",
        ]

    def test_one_line_is_left_alone(self) -> None:
        assert _lines("Tap the button", 500) == ["Tap the button"]

    def test_lines_never_outnumber_or_overflow_pycaps(self) -> None:
        rng = random.Random(7)  # noqa: S311 - a fixed seed for test cases
        vocab = "the a to of is your iPhone 15 battery charger cable turns on".split()
        for _ in range(500):
            words = [rng.choice(vocab) for _ in range(rng.randint(1, 7))]
            widths = [rng.uniform(20, 140) for _ in words]
            max_width = rng.uniform(120, 400)
            ours = line_ends(words, widths, SPACE, max_width)
            greedy = _pycaps_greedy(widths, SPACE, max_width)
            assert len(ours) <= len(greedy), (words, widths, max_width)
            assert ours[-1] == len(words)
            start = 0
            for end in ours:
                line = widths[start:end]
                assert (
                    end - start == 1
                    or sum(line) + SPACE * (len(line) - 1) <= max_width + SPACE
                ), (words, widths, max_width)
                start = end


class TestTheSwitch:
    def test_off_by_default_in_the_model(self) -> None:
        assert PycapsSettings().phrase_line_breaks is False

    def test_the_shipped_config_turns_it_on(self) -> None:
        import yaml

        raw = yaml.safe_load((REPO / "config" / "subtitles.yaml").read_text())
        assert PycapsSettings(**raw["subtitle_settings"]["pycaps"]).phrase_line_breaks


class TestAgainstTheRealLibrary:
    @pytest.fixture
    def pycaps(self):
        return pytest.importorskip("pycaps")

    @staticmethod
    def _segment(texts: list[str]):
        from pycaps.common import ElementLayout, Line, Segment, Size, TimeFragment, Word

        segment = Segment(time=TimeFragment(start=0.0, end=float(len(texts))))
        line = Line(time=segment.time)
        line.words.set_all(
            [
                Word(
                    text=t,
                    time=TimeFragment(start=float(i), end=i + 0.9),
                    max_layout=ElementLayout(size=Size(width=10 * len(t), height=40)),
                )
                for i, t in enumerate(texts)
            ]
        )
        segment.lines.add(line)
        return segment

    @staticmethod
    def _lines_of(segment) -> list[str]:
        return [" ".join(w.text for w in line.words) for line in segment.lines]

    def test_the_swapped_splitter_moves_the_break(self, pycaps) -> None:
        from pycaps.layout import LineSplitter, SubtitleLayoutOptions

        from src.video.pycaps_engine.renderer import _phrase_line_splitter

        # 100 px of a 125 px frame at ratio 0.8, with no word spacing set,
        # so the spacing is 0 and "Tap the" is 60 px of 100.
        options = SubtitleLayoutOptions(max_width_ratio=0.8)
        stock, ours = (
            self._segment(["Tap", "the", "button"]),
            self._segment(["Tap", "the", "button"]),
        )
        LineSplitter(options)._split_segment_into_lines(stock, 125)
        _phrase_line_splitter(options)._split_segment_into_lines(ours, 125)
        assert self._lines_of(stock) == ["Tap the", "button"]
        assert self._lines_of(ours) == ["Tap", "the button"]

    def test_the_minimum_line_count_still_splits_a_short_caption(self, pycaps) -> None:
        # Templates such as line-focus ask for two lines at least.
        from pycaps.layout import LineSplitter, SubtitleLayoutOptions

        from src.video.pycaps_engine.renderer import _phrase_line_splitter

        options = SubtitleLayoutOptions(max_width_ratio=0.8, min_number_of_lines=2)
        stock, ours = (
            self._segment(["Tap", "the", "button"]),
            self._segment(["Tap", "the", "button"]),
        )
        LineSplitter(options)._split_segment_into_lines(stock, 500)
        _phrase_line_splitter(options)._split_segment_into_lines(ours, 500)
        assert len(self._lines_of(ours)) == 2
        assert self._lines_of(ours) == self._lines_of(stock)

    def test_the_switch_replaces_the_splitter_after_prepare(self, pycaps) -> None:
        from pycaps.template import TemplateFactory, TemplateLoader

        from src.video.pycaps_engine.renderer import _use_phrase_line_breaks

        builder = TemplateLoader(TemplateFactory().create("explosive")).load(False)
        pipeline = builder._caps_pipeline
        calls: list[str] = []
        pipeline.prepare = lambda: calls.append("prepared")
        _use_phrase_line_breaks(builder)
        pipeline.prepare()
        assert calls == ["prepared"]
        assert type(pipeline._line_splitter).__name__ == "_PhraseBoundaryLineSplitter"
