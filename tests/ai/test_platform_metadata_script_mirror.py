"""Tests for the video-script mirror plumbing in platform metadata.

Covers `_read_video_script` (helper that reads the script.txt artifact and
threads it into the per-platform LLM prompts so caption templates can mirror
the script's closing engagement-bait line).
"""

from pathlib import Path

import pytest

from src.ai.platform_metadata import _read_video_script


class TestReadVideoScript:
    def test_returns_none_when_intermediate_paths_is_none(self):
        assert _read_video_script(None) is None

    def test_returns_none_when_intermediate_paths_is_empty(self):
        assert _read_video_script({}) is None

    def test_returns_none_when_script_key_missing(self, tmp_path: Path):
        paths = {"description": tmp_path / "description.txt"}
        assert _read_video_script(paths) is None

    def test_returns_none_when_script_file_does_not_exist(self, tmp_path: Path):
        paths = {"script": tmp_path / "missing-script.txt"}
        assert _read_video_script(paths) is None

    def test_returns_script_text_when_file_exists(self, tmp_path: Path):
        script_path = tmp_path / "script.txt"
        script = (
            "Best 65W chargers under $50 for tech-savvy young adults. "
            "This is a great charger. "
            "Most people only need two ports, but three is usually better. "
            "Link in bio if you want one."
        )
        script_path.write_text(script, encoding="utf-8")
        assert _read_video_script({"script": script_path}) == script

    def test_strips_surrounding_whitespace(self, tmp_path: Path):
        script_path = tmp_path / "script.txt"
        script_path.write_text("\n\n  hello  \n\n", encoding="utf-8")
        assert _read_video_script({"script": script_path}) == "hello"

    def test_empty_script_file_returns_none(self, tmp_path: Path):
        script_path = tmp_path / "script.txt"
        script_path.write_text("   \n\n  ", encoding="utf-8")
        assert _read_video_script({"script": script_path}) is None

    def test_unreadable_file_returns_none(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """File present but unreadable: helper returns None and logs WARN."""
        script_path = tmp_path / "script.txt"
        script_path.write_text("ok", encoding="utf-8")

        def fake_read_text(*args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "read_text", fake_read_text)

        import logging

        with caplog.at_level(logging.WARNING, logger="src.ai.platform_metadata"):
            result = _read_video_script({"script": script_path})

        assert result is None
        assert any("Failed to read script" in r.getMessage() for r in caplog.records)
