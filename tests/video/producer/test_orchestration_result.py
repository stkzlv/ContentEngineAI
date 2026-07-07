"""Tests for the producer result sentinel parsing."""

from pathlib import Path

from src.video.producer.orchestration import failed_step_from_result


def test_failed_step_parsed_from_sentinel():
    assert failed_step_from_result("FAILED:generate_subtitles") == "generate_subtitles"


def test_failed_step_empty_step_reads_unknown():
    assert failed_step_from_result("FAILED:") == "unknown"


def test_non_failure_results_return_none():
    # success paths (str or Path), the skip sentinel, and a caller-level None
    # (set by the caller's own timeout/exception handling) are not failures
    assert failed_step_from_result("outputs/B0ABC123/video.mp4") is None
    assert failed_step_from_result(Path("outputs/B0ABC123/video.mp4")) is None
    assert failed_step_from_result("SKIPPED") is None
    assert failed_step_from_result(None) is None
