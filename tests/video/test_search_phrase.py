"""Where the search phrase appears in each render (REQ-CNT-055)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video.render_choices import choices_from_context, report
from src.video.search_phrase import (
    captions,
    contains,
    coverage_lines,
    placement,
    search_phrase,
)


def test_contains_needs_every_significant_word() -> None:
    assert contains("Level up with this Smart Watch!", "smart watch")
    assert contains("watch, but smart", "smart watch")
    assert contains("Before this smartwatch, I", "smart watch")
    # A spoken answer drops the question words of a topic title, but a
    # product keyword keeps every word.
    title = "Why your laptop fan runs when idle"
    assert contains("Your laptop fan runs when idle because", title, question=True)
    assert not contains("Your laptop fan runs when idle because", title)
    assert not contains("This bottle opener is great", "can opener")
    assert not contains("A smart ring", "smart watch")
    assert not contains(None, "smart watch")
    assert not contains("anything", "a")


def test_key_words_keep_the_order_and_drop_repeats() -> None:
    from src.video.search_phrase import key_words

    assert key_words("Why wifi drops and wifi lags", question=True) == [
        "wifi",
        "drops",
        "lags",
    ]


def test_the_phrase_is_the_keyword_or_the_topic_title() -> None:
    assert search_phrase(SimpleNamespace(keyword=" smart watch ")) == "smart watch"
    topic = SimpleNamespace(topic="t", title="Why wifi drops", keyword="router")
    assert search_phrase(topic) == "Why wifi drops"
    assert search_phrase(SimpleNamespace(keyword="")) is None
    assert search_phrase(None) is None


def test_captions_are_read_the_way_the_publisher_reads_them(tmp_path) -> None:
    (tmp_path / "metadata_tiktok.json").write_text(json.dumps({"description": "tt"}))
    (tmp_path / "metadata_instagram.json").write_text("{broken")

    assert captions(tmp_path) == {"tiktok": "tt"}
    # A unified file wins over a platform's own, as at publish time.
    (tmp_path / "metadata.json").write_text(json.dumps({"description": "unified"}))
    assert captions(tmp_path) == dict.fromkeys(
        ("youtube", "tiktok", "instagram"), "unified"
    )
    assert captions(tmp_path / "missing") == {}


@pytest.mark.req("REQ-CNT-055")
def test_placement_checks_each_place() -> None:
    late = "x" * 60 + " smart watch"

    result = placement(
        "smart watch",
        "This smart watch tracks sleep. It also rings.",
        "Smart watch, no phone",
        {"youtube": "Smart watch that rings", "tiktok": late},
    )

    assert result == {
        "phrase": "smart watch",
        "spoken": True,
        "headline": True,
        "captions": {"youtube": True, "tiktok": False},
    }
    assert placement(None, "x", "y", {}) is None


@pytest.mark.req("REQ-CNT-055")
def test_the_row_and_the_report_carry_it(tmp_path: Path) -> None:
    root = tmp_path / "B0X"
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps({"description": "This smart watch rings."})
    )
    ctx = SimpleNamespace(
        state={"hook_headline": "A smart watch for runners"},
        config=SimpleNamespace(
            video_settings=SimpleNamespace(
                first_frame_pre_motion=False, video_transition_duration=0.3
            )
        ),
        product=SimpleNamespace(keyword="smart watch"),
        profile_name="p",
        profile=SimpleNamespace(
            video_assembly_mode="sequential",
            first_frame_pre_motion=None,
            video_transition_duration=0.5,
        ),
        run_paths={"music_info_file": None, "run_root": root},
        script="It rings. This smart watch also tracks sleep.",
    )

    row = choices_from_context(ctx)

    assert row["search_phrase"]["spoken"] is False
    assert row["search_phrase"]["headline"] is True
    assert row["search_phrase"]["captions"]["youtube"] is True
    lines = report([row], 0.6, 0.5)
    assert "  first spoken sentence 0/1, hook headline 1/1, " in lines[-1]
    assert lines[-1].endswith("every caption opening 1/1, all three 0/1")


def test_no_phrases_means_no_report_lines() -> None:
    assert coverage_lines([{"profile": "p"}]) == []


@pytest.mark.req("REQ-CNT-055")
def test_a_topic_row_reads_its_title_as_a_question(tmp_path: Path) -> None:
    root = tmp_path / "topic-x"
    root.mkdir()
    ctx = SimpleNamespace(
        state={"hook_headline": "Laptop fan runs when idle"},
        config=SimpleNamespace(
            video_settings=SimpleNamespace(
                first_frame_pre_motion=False, video_transition_duration=0.3
            )
        ),
        product=SimpleNamespace(
            topic="t", title="Why your laptop fan runs when idle", keyword=""
        ),
        profile_name="p",
        profile=SimpleNamespace(
            video_assembly_mode="sequential",
            first_frame_pre_motion=None,
            video_transition_duration=0.5,
        ),
        run_paths={"music_info_file": None, "run_root": root},
        script="Your laptop fan runs when idle because of updates.",
    )

    placed = choices_from_context(ctx)["search_phrase"]

    assert placed["spoken"] is True and placed["headline"] is True


@pytest.mark.req("REQ-CNT-055")
def test_a_series_prefix_and_opening_line_are_skipped() -> None:
    prefix = "A long series prefix that fills nearly all of the sixty: "
    script = "Myth check: they die in the cold. This desk lamp lasts a day."

    result = placement(
        "desk lamp",
        script,
        None,
        {
            "youtube": prefix + "desk lamp batteries",
            "tiktok": "x" * 60 + " desk lamp",
        },
        series_prefix=prefix,
        opening="Myth check: they die in the cold.",
    )

    assert result is not None
    assert result["spoken"] is True
    assert result["captions"] == {"youtube": True, "tiktok": False}
    # Without the series values, both fall past the measured places.
    plain = placement("desk lamp", script, None, {"youtube": prefix + "desk lamp"})
    assert plain is not None and plain["spoken"] is False
    assert plain["captions"]["youtube"] is False


@pytest.mark.req("REQ-CNT-055")
def test_an_opening_the_script_doesnt_start_with_is_not_skipped() -> None:
    result = placement(
        "desk lamp",
        "This desk lamp rings. It also tracks sleep.",
        None,
        {},
        opening="Myth check: it rings.",
    )
    assert result is not None and result["spoken"] is True


@pytest.mark.req("REQ-CNT-055")
def test_a_series_row_skips_its_prefix_and_opening(tmp_path: Path) -> None:
    from src.video.config.series_models import SeriesConfig

    root = tmp_path / "topic-y"
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps({"description": PREFIX + "desk lamp batteries"})
    )
    ctx = SimpleNamespace(
        state={},
        config=SimpleNamespace(
            series={
                "myth": SeriesConfig(
                    title_prefix=PREFIX, opening_line="Myth check: {claim}."
                )
            },
            video_settings=SimpleNamespace(
                first_frame_pre_motion=False, video_transition_duration=0.3
            ),
        ),
        product=SimpleNamespace(
            keyword="desk lamp", series="myth", claim="they die in the cold"
        ),
        profile_name="p",
        profile=SimpleNamespace(
            video_assembly_mode="sequential",
            first_frame_pre_motion=None,
            video_transition_duration=0.5,
        ),
        run_paths={"music_info_file": None, "run_root": root},
        script="Myth check: they die in the cold. This desk lamp lasts a day.",
    )

    placed = choices_from_context(ctx)["search_phrase"]

    assert placed["spoken"] is True
    assert placed["captions"]["youtube"] is True


PREFIX = "A long series prefix that fills nearly all of the sixty: "
