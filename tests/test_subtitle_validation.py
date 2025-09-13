"""Test suite for subtitle validation utilities.

This module tests the consolidated subtitle validation functions that were
extracted from duplicated code in subtitle_generator.py and subtitle_utils.py.
"""

import tempfile
from pathlib import Path

from src.video.subtitle_validation import (
    validate_ass_file,
    validate_srt_file,
    validate_subtitle_segments,
)


class TestValidateSrtFile:
    """Test cases for validate_srt_file function."""

    def test_valid_srt_file(self):
        """Test validation of a properly formatted SRT file."""
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
This is a test subtitle
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()
            srt_path = Path(f.name)

        try:
            assert validate_srt_file(srt_path) is True
            assert validate_srt_file(srt_path, debug_mode=True) is True
        finally:
            srt_path.unlink()

    def test_empty_srt_file(self):
        """Test validation of an empty SRT file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write("")
            f.flush()
            srt_path = Path(f.name)

        try:
            assert validate_srt_file(srt_path) is False
        finally:
            srt_path.unlink()

    def test_nonexistent_srt_file(self):
        """Test validation of a nonexistent SRT file."""
        srt_path = Path("/nonexistent/path/test.srt")
        assert validate_srt_file(srt_path) is False

    def test_invalid_timing_srt(self):
        """Test validation of SRT with invalid timing (start >= end)."""
        srt_content = """1
00:00:03,000 --> 00:00:01,000
Invalid timing
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()
            srt_path = Path(f.name)

        try:
            assert validate_srt_file(srt_path) is False
        finally:
            srt_path.unlink()

    def test_empty_text_segments_only(self):
        """Test SRT with only empty text segments."""
        srt_content = """1
00:00:01,000 --> 00:00:03,000


2
00:00:04,000 --> 00:00:06,000

"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()
            srt_path = Path(f.name)

        try:
            assert validate_srt_file(srt_path) is False
        finally:
            srt_path.unlink()

    def test_mixed_valid_empty_segments(self):
        """Test SRT with mix of valid and empty segments."""
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Valid text

2
00:00:04,000 --> 00:00:06,000


3
00:00:07,000 --> 00:00:09,000
Another valid segment
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()
            srt_path = Path(f.name)

        try:
            assert validate_srt_file(srt_path) is True
        finally:
            srt_path.unlink()


class TestValidateSubtitleSegments:
    """Test cases for validate_subtitle_segments function."""

    def test_valid_segments(self):
        """Test validation of properly formatted subtitle segments."""
        segments = [
            {"text": "Hello world", "start": 1.0, "end": 3.0},
            {"text": "This is a test", "start": 4.0, "end": 6.0},
        ]
        assert validate_subtitle_segments(segments) is True
        assert validate_subtitle_segments(segments, debug_mode=True) is True

    def test_empty_segments_list(self):
        """Test validation of empty segments list."""
        assert validate_subtitle_segments([]) is False

    def test_missing_required_fields(self):
        """Test segments with missing required fields."""
        segments = [
            {"text": "Hello", "start": 1.0},  # Missing 'end'
            {"start": 2.0, "end": 4.0},  # Missing 'text'
        ]
        assert validate_subtitle_segments(segments) is False

    def test_invalid_timing_values(self):
        """Test segments with invalid timing values."""
        segments = [
            {"text": "Test", "start": -1.0, "end": 2.0},  # Negative start
        ]
        assert validate_subtitle_segments(segments) is False

        segments = [
            {"text": "Test", "start": 3.0, "end": 1.0},  # Start >= end
        ]
        assert validate_subtitle_segments(segments) is False

    def test_empty_text_segments(self):
        """Test segments with empty or whitespace-only text."""
        segments = [
            {"text": "", "start": 1.0, "end": 2.0},
            {"text": "   ", "start": 3.0, "end": 4.0},
            {"text": "Valid text", "start": 5.0, "end": 6.0},
        ]
        assert validate_subtitle_segments(segments) is True

    def test_non_numeric_timing(self):
        """Test segments with non-numeric timing values."""
        segments = [
            {"text": "Test", "start": "1.0", "end": 2.0},  # String start time
        ]
        assert validate_subtitle_segments(segments) is False

    def test_all_invalid_segments(self):
        """Test when all segments are invalid."""
        segments = [
            {"text": "", "start": 1.0, "end": 2.0},  # Empty text
            {"text": "  ", "start": 3.0, "end": 4.0},  # Whitespace only
        ]
        assert validate_subtitle_segments(segments) is False


class TestValidateAssFile:
    """Test cases for validate_ass_file function."""

    def test_valid_ass_file(self):
        """Test validation of a properly formatted ASS file."""
        ass_content = """[Script Info]
Title: Test
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour
Style: Default,Arial,16,&H00ffffff,&H00000000

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello world
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,This is a test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False) as f:
            f.write(ass_content)
            f.flush()
            ass_path = Path(f.name)

        try:
            assert validate_ass_file(ass_path) is True
            assert validate_ass_file(ass_path, debug_mode=True) is True
        finally:
            ass_path.unlink()

    def test_nonexistent_ass_file(self):
        """Test validation of a nonexistent ASS file."""
        ass_path = Path("/nonexistent/path/test.ass")
        assert validate_ass_file(ass_path) is False

    def test_empty_ass_file(self):
        """Test validation of an empty ASS file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False) as f:
            f.write("")
            f.flush()
            ass_path = Path(f.name)

        try:
            assert validate_ass_file(ass_path) is False
        finally:
            ass_path.unlink()

    def test_missing_required_sections(self):
        """Test ASS file with missing required sections."""
        ass_content = """[Script Info]
Title: Test

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,16
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False) as f:
            f.write(ass_content)
            f.flush()
            ass_path = Path(f.name)

        try:
            # Missing [Events] section
            assert validate_ass_file(ass_path) is False
        finally:
            ass_path.unlink()

    def test_no_dialogue_lines(self):
        """Test ASS file with no dialogue lines."""
        ass_content = """[Script Info]
Title: Test

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,16

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False) as f:
            f.write(ass_content)
            f.flush()
            ass_path = Path(f.name)

        try:
            assert validate_ass_file(ass_path) is False
        finally:
            ass_path.unlink()


# Integration tests
class TestIntegration:
    """Integration tests for subtitle validation functions."""

    def test_srt_validation_consistency(self):
        """Test that SRT validation is consistent across different inputs."""
        # Create a valid SRT file
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Consistent validation test

2
00:00:04,000 --> 00:00:06,000
Second segment
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()
            srt_path = Path(f.name)

        try:
            # Should be valid both in normal and debug mode
            assert validate_srt_file(srt_path, debug_mode=False) is True
            assert validate_srt_file(srt_path, debug_mode=True) is True
        finally:
            srt_path.unlink()

    def test_validation_functions_independence(self):
        """Test that validation functions don't interfere with each other."""
        # Test that calling one validation function doesn't affect others
        segments = [{"text": "Test", "start": 1.0, "end": 2.0}]

        # Call segment validation
        result1 = validate_subtitle_segments(segments)

        # Call file validation with nonexistent file
        result2 = validate_srt_file(Path("/nonexistent.srt"))

        # Call segment validation again - should have same result
        result3 = validate_subtitle_segments(segments)

        assert result1 is True
        assert result2 is False
        assert result3 is True
        assert result1 == result3
