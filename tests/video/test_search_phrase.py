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
    # A spoken answer drops the question words of a topic title.
    assert contains(
        "Your laptop fan runs when idle because", "Why your laptop fan runs when idle"
    )
    assert not contains("A smart ring", "smart watch")
    assert not contains(None, "smart watch")
    assert not contains("anything", "a")


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
