"""`timing_smoothing` has to reach the smoother (#397).

`config/subtitles.yaml` carries a `subtitle_settings.timing_smoothing` block
with six documented keys and none of them arrived. `_build_subtitle_base`
enumerates keys explicitly, so a block it does not name never reaches the
merged settings however completely the YAML defines it: the model's
`default_factory=dict` wins and the smoother ran on its own function
defaults.

`lead_sec: 0.04` happens to equal the code default, which is why the rendered
output matched what the config asked for -- by coincidence rather than by
wiring. `hook_lead_sec` and `hook_lead_word_count` do not match theirs, so
that feature had never run.
"""

import ast
from pathlib import Path

import pytest

from src.video.config import config
from src.video.subtitle_timing_smoother import smooth_word_timings

# The kwargs `stt_functions` filters the block down to before calling the
# smoother. `enabled` is read separately and is not a kwarg, so splatting the
# whole block would raise.
SMOOTHER_KWARGS = (
    "min_word_sec",
    "gap_merge_sec",
    "hold_last_sec",
    "lead_sec",
    "hook_lead_sec",
    "hook_lead_word_count",
)


class TestTheBlockReachesTheMergedSettings:
    def test_the_global_block_survives_the_profile_merge(self):
        merged = config.get_profile_merged_settings("slideshow_stock", {})
        block = merged.subtitle_settings.timing_smoothing

        assert (
            block
        ), "timing_smoothing arrived empty; the smoother would run on defaults"
        for key in SMOOTHER_KWARGS:
            assert key in block, f"{key} was dropped between the YAML and the merge"

    def test_the_shipped_values_arrive_intact(self):
        """Not just present: the configured numbers, not the defaults."""
        block = config.get_profile_merged_settings(
            "slideshow_stock", {}
        ).subtitle_settings.timing_smoothing

        assert block["lead_sec"] == pytest.approx(0.04)
        assert block["hook_lead_sec"] == pytest.approx(0.20)
        assert block["hook_lead_word_count"] == 3

    def test_the_block_is_named_in_the_base_builder(self):
        """The builder enumerates keys, so an unnamed block cannot arrive."""
        source = Path("src/video/config/core_models.py").read_text()
        assert '"timing_smoothing": ss.get("timing_smoothing"' in source


class TestTheCallSiteCanUseIt:
    def test_the_filtered_kwargs_are_accepted_by_the_smoother(self):
        """`enabled` is in the block and is not a kwarg.

        Splatting the whole block raises `TypeError`, so the filtering in
        `stt_functions` is load-bearing, not tidiness.
        """
        block = config.get_profile_merged_settings(
            "slideshow_stock", {}
        ).subtitle_settings.timing_smoothing
        assert "enabled" in block

        kwargs = {k: block[k] for k in SMOOTHER_KWARGS if k in block}
        timings = [
            {"word": "a", "start_time": 0.0, "end_time": 0.35},
            {"word": "b", "start_time": 0.35, "end_time": 0.70},
        ]
        smooth_word_timings(timings, **kwargs)  # must not raise

        with pytest.raises(TypeError):
            smooth_word_timings(timings, **block)

    def test_the_smoother_gets_the_block_not_a_hardcoded_set(self):
        """The call site must read the config, not restate the defaults."""
        tree = ast.parse(Path("src/video/subtitle_utils.py").read_text())

        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "timing_smoothing"
        ]
        assert reads, (
            "subtitle_utils must read timing_smoothing from the settings it "
            "was handed; without it the smoother runs on function defaults"
        )


class TestWhatTurningItOnDoes:
    """The hook lead has never run, so record what it now does.

    #397 warns that repairing the plumbing activates `hook_lead_sec: 0.20`
    on the first three words, and that a line break inside those words
    produces a 240ms overlap on the flat list where every overlap had been
    40ms. That is true and measured here, so the number is not a surprise to
    the next reader.
    """

    @staticmethod
    def _contiguous(n: int = 10, step: float = 0.35):
        """Whisper emits contiguous windows: each word starts where the last ended."""
        return [
            {
                "word": f"w{i}",
                "start_time": round(i * step, 3),
                "end_time": round((i + 1) * step, 3),
            }
            for i in range(n)
        ]

    def _kwargs(self):
        block = config.get_profile_merged_settings(
            "slideshow_stock", {}
        ).subtitle_settings.timing_smoothing
        return {k: block[k] for k in SMOOTHER_KWARGS if k in block}

    def test_the_first_three_words_get_the_extra_lead(self):
        raw = self._contiguous()
        out = smooth_word_timings(raw, **self._kwargs())

        # Word 1's shift is truncated at zero, so measure words 2 and 3.
        for i in (1, 2):
            shift = raw[i]["start_time"] - out[i]["start_time"]
            assert shift == pytest.approx(
                0.24
            ), f"word {i} shifted {shift}s; expected lead 0.04 + hook 0.20"

    def test_the_fourth_word_gets_the_base_lead_only(self):
        raw = self._contiguous()
        out = smooth_word_timings(raw, **self._kwargs())

        shift = raw[3]["start_time"] - out[3]["start_time"]
        assert shift == pytest.approx(0.04), "the hook lead must cover 3 words only"

    def test_the_overlap_it_creates_is_the_documented_240ms(self):
        """Recorded, not accepted silently.

        The FFmpeg path clamps a segment's end to its successor's start, so
        ASS dialogue lines still do not overlap. The flat list and the dict
        the pycaps engine reads carry it.
        """
        raw = self._contiguous()
        out = smooth_word_timings(raw, **self._kwargs())

        overlaps = [
            a["end_time"] - b["start_time"]
            for a, b in zip(out, out[1:], strict=False)
            if a["end_time"] > b["start_time"] + 1e-9
        ]
        assert overlaps
        assert max(overlaps) == pytest.approx(0.24)
