"""Test suite for result types."""

from pathlib import Path

import pytest

from src.video.result_types import (
    AudioResult,
    SubtitleResult,
    VideoResult,
)


class TestSubtitleResult:
    """Test cases for SubtitleResult."""

    def test_subtitle_result_success(self):
        """Test successful subtitle result creation."""
        result = SubtitleResult(
            success=True,
            path=Path("/test/subtitles.srt"),
            format="srt",
            segments_created=10,
            generation_method="timing_based",
            timing_source="whisper",
        )

        assert result.success is True
        assert result.path == Path("/test/subtitles.srt")
        assert result.format == "srt"
        assert result.segments_created == 10
        assert result.generation_method == "timing_based"
        assert result.timing_source == "whisper"
        assert len(result.errors) == 0

    def test_subtitle_result_failure(self):
        """Test failed subtitle result creation."""
        result = SubtitleResult(
            success=False,
            path=None,
            format="srt",
            errors=["Failed to generate subtitles", "No audio found"],
        )

        assert result.success is False
        assert result.path is None
        assert result.format == "srt"
        assert len(result.errors) == 2
        assert "Failed to generate subtitles" in result.errors

    def test_subtitle_result_add_error(self):
        """Test adding errors to subtitle result."""
        result = SubtitleResult(success=True, path=None, format="srt")

        result.add_error("Test error message")

        assert len(result.errors) == 1
        assert result.errors[0] == "Test error message"

    def test_subtitle_result_add_multiple_errors(self):
        """Test adding multiple errors to subtitle result."""
        result = SubtitleResult(success=True, path=None, format="srt")

        result.add_error("Error 1")
        result.add_error("Error 2")

        assert len(result.errors) == 2
        assert "Error 1" in result.errors
        assert "Error 2" in result.errors


class TestAudioResult:
    """Test cases for AudioResult."""

    def test_audio_result_success(self):
        """Test successful audio result creation."""
        result = AudioResult(
            success=True,
            path=Path("/test/audio.wav"),
            duration_sec=45.5,
        )

        assert result.success is True
        assert result.path == Path("/test/audio.wav")
        assert result.duration_sec == 45.5

    def test_audio_result_failure(self):
        """Test failed audio result creation."""
        result = AudioResult(
            success=False,
            path=None,
            errors=["TTS provider failed"],
        )

        assert result.success is False
        assert result.path is None
        assert "TTS provider failed" in result.errors


class TestVideoResult:
    """Test cases for VideoResult."""

    def test_video_result_success(self):
        """Test successful video result creation."""
        result = VideoResult(
            success=True,
            path=Path("/test/video.mp4"),
            duration_sec=60.0,
            resolution=(1920, 1080),
        )

        assert result.success is True
        assert result.path == Path("/test/video.mp4")
        assert result.duration_sec == 60.0
        assert result.resolution == (1920, 1080)

    def test_video_result_add_error(self):
        """Test adding error to video result."""
        result = VideoResult(success=False, path=None)

        result.add_error("FFmpeg failed")

        assert "FFmpeg failed" in result.errors
