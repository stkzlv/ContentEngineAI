"""Unit tests for the subtitle utilities module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.video.subtitle_utils import (
    adjust_subtitle_timing,
    convert_timestamps_to_seconds,
    get_subtitle_info,
    slice_subtitles,
)
from src.video.subtitle_validation import validate_srt_file


class TestSubtitleUtils:
    """Test the subtitle utilities functions."""

    @pytest.fixture
    def sample_srt_content(self) -> str:
        """Sample SRT content for testing."""
        return """1
00:00:01,000 --> 00:00:03,500
Hello world, this is a test.

2
00:00:04,000 --> 00:00:07,200
This is the second subtitle line.

3
00:00:08,000 --> 00:00:10,500
Final subtitle for testing.
"""

    @pytest.fixture
    def sample_srt_file(self, temp_dir: Path, sample_srt_content: str) -> Path:
        """Create a sample SRT file for testing."""
        srt_path = temp_dir / "sample.srt"
        srt_path.write_text(sample_srt_content, encoding="utf-8")
        return srt_path

    @pytest.fixture
    def malformed_srt_content(self) -> str:
        """Malformed SRT content for testing validation."""
        return """1
00:00:01,000 --> 00:00:00,500
Invalid timing - end before start

2
00:00:04,000 --> 00:00:07,200

3
00:00:08,000 --> 00:00:10,500
Missing text in subtitle 2
"""

    @pytest.fixture
    def malformed_srt_file(self, temp_dir: Path, malformed_srt_content: str) -> Path:
        """Create a malformed SRT file for testing."""
        srt_path = temp_dir / "malformed.srt"
        srt_path.write_text(malformed_srt_content, encoding="utf-8")
        return srt_path

    def test_validate_srt_file_valid(self, sample_srt_file: Path):
        """Test validation of a valid SRT file."""
        result = validate_srt_file(sample_srt_file, debug_mode=True)
        assert result is True

    def test_validate_srt_file_nonexistent(self, temp_dir: Path):
        """Test validation of non-existent SRT file."""
        nonexistent_path = temp_dir / "nonexistent.srt"
        result = validate_srt_file(nonexistent_path)
        assert result is False

    def test_validate_srt_file_empty(self, temp_dir: Path):
        """Test validation of empty SRT file."""
        empty_path = temp_dir / "empty.srt"
        empty_path.write_text("", encoding="utf-8")
        result = validate_srt_file(empty_path)
        assert result is False

    def test_validate_srt_file_malformed(self, malformed_srt_file: Path):
        """Test validation of malformed SRT file."""
        result = validate_srt_file(malformed_srt_file)
        assert result is False

    def test_get_subtitle_info_valid(self, sample_srt_file: Path):
        """Test getting info from a valid SRT file."""
        result = get_subtitle_info(sample_srt_file)

        assert result is not None
        assert result["file_path"] == str(sample_srt_file)
        assert result["segment_count"] == 3
        assert result["is_valid"] is True
        assert result["first_segment_start"] == 1000  # 1 second in ms
        assert result["last_segment_end"] == 10500  # 10.5 seconds in ms
        assert result["duration_ms"] == 10500

    def test_get_subtitle_info_nonexistent(self, temp_dir: Path):
        """Test getting info from non-existent file."""
        nonexistent_path = temp_dir / "nonexistent.srt"
        result = get_subtitle_info(nonexistent_path)
        assert result is None

    def test_get_subtitle_info_empty(self, temp_dir: Path):
        """Test getting info from empty SRT file."""
        empty_path = temp_dir / "empty.srt"
        empty_path.write_text("", encoding="utf-8")
        result = get_subtitle_info(empty_path)

        assert result is not None
        assert result["segment_count"] == 0
        assert result["duration_ms"] == 0
        assert result["is_valid"] is False

    def test_adjust_subtitle_timing_delay(self, sample_srt_file: Path, temp_dir: Path):
        """Test delaying subtitle timing by 2 seconds."""
        output_path = temp_dir / "delayed.srt"

        result = adjust_subtitle_timing(sample_srt_file, 2000, output_path)

        assert result is not None
        assert result.exists()

        # Verify timing adjustment
        original_info = get_subtitle_info(sample_srt_file)
        adjusted_info = get_subtitle_info(result)

        assert original_info is not None
        assert adjusted_info is not None
        assert (
            adjusted_info["first_segment_start"]
            == original_info["first_segment_start"] + 2000
        )
        assert (
            adjusted_info["last_segment_end"]
            == original_info["last_segment_end"] + 2000
        )

    def test_adjust_subtitle_timing_advance(
        self, sample_srt_file: Path, temp_dir: Path
    ):
        """Test advancing subtitle timing by 1 second."""
        output_path = temp_dir / "advanced.srt"

        result = adjust_subtitle_timing(sample_srt_file, -1000, output_path)

        assert result is not None
        assert result.exists()

        # Verify timing adjustment
        original_info = get_subtitle_info(sample_srt_file)
        adjusted_info = get_subtitle_info(result)

        assert original_info is not None
        assert adjusted_info is not None
        assert (
            adjusted_info["first_segment_start"]
            == original_info["first_segment_start"] - 1000
        )
        assert (
            adjusted_info["last_segment_end"]
            == original_info["last_segment_end"] - 1000
        )

    def test_adjust_subtitle_timing_inplace(self, sample_srt_file: Path):
        """Test adjusting subtitle timing in-place (no output path specified)."""
        original_info = get_subtitle_info(sample_srt_file)

        result = adjust_subtitle_timing(sample_srt_file, 500)  # No output path

        assert result is not None
        assert result == sample_srt_file  # Should be same path

        # Verify timing was adjusted
        adjusted_info = get_subtitle_info(sample_srt_file)
        assert original_info is not None
        assert adjusted_info is not None
        assert (
            adjusted_info["first_segment_start"]
            == original_info["first_segment_start"] + 500
        )

    def test_adjust_subtitle_timing_nonexistent(self, temp_dir: Path):
        """Test adjusting timing of non-existent file."""
        nonexistent_path = temp_dir / "nonexistent.srt"
        output_path = temp_dir / "output.srt"

        result = adjust_subtitle_timing(nonexistent_path, 1000, output_path)
        assert result is None

    def test_slice_subtitles_middle_range(self, sample_srt_file: Path, temp_dir: Path):
        """Test slicing subtitles from middle time range (4s-8s)."""
        output_path = temp_dir / "sliced.srt"

        result = slice_subtitles(sample_srt_file, 4000, 8000, output_path)

        assert result is not None
        assert result.exists()

        # Verify sliced content
        sliced_info = get_subtitle_info(result)
        assert sliced_info is not None
        assert sliced_info["segment_count"] == 1  # Should capture subtitle 2

        # Check that timing starts from 0
        srt_content = result.read_text()
        assert "00:00:00,000" in srt_content  # Should start from 0
        assert "This is the second subtitle line" in srt_content

    def test_slice_subtitles_overlapping_range(
        self, sample_srt_file: Path, temp_dir: Path
    ):
        """Test slicing subtitles with overlapping range (2s-9s)."""
        output_path = temp_dir / "overlapping.srt"

        result = slice_subtitles(sample_srt_file, 2000, 9000, output_path)

        assert result is not None
        assert result.exists()

        # Verify segments captured (should get subtitles 1, 2, and 3 which
        # overlap this range)
        sliced_info = get_subtitle_info(result)
        assert sliced_info is not None
        assert sliced_info["segment_count"] >= 1  # At least one segment should overlap

    def test_slice_subtitles_no_overlap(self, sample_srt_file: Path, temp_dir: Path):
        """Test slicing subtitles with no overlapping range."""
        output_path = temp_dir / "no_overlap.srt"

        # Try to slice between subtitles where there's no content
        result = slice_subtitles(sample_srt_file, 11000, 12000, output_path)

        # Should return None as no subtitles found in range
        assert result is None

    def test_slice_subtitles_nonexistent(self, temp_dir: Path):
        """Test slicing non-existent file."""
        nonexistent_path = temp_dir / "nonexistent.srt"
        output_path = temp_dir / "output.srt"

        result = slice_subtitles(nonexistent_path, 1000, 5000, output_path)
        assert result is None

    def test_convert_timestamps_to_seconds(self, sample_srt_file: Path, temp_dir: Path):
        """Test converting SRT timestamps to seconds format."""
        output_path = temp_dir / "seconds.txt"

        result = convert_timestamps_to_seconds(sample_srt_file, output_path)

        assert result is not None
        assert result.exists()

        # Verify seconds format
        content = result.read_text()
        assert "1.000s --> 3.500s" in content
        assert "4.000s --> 7.200s" in content
        assert "8.000s --> 10.500s" in content
        assert "Hello world, this is a test." in content

    def test_convert_timestamps_nonexistent(self, temp_dir: Path):
        """Test converting timestamps of non-existent file."""
        nonexistent_path = temp_dir / "nonexistent.srt"
        output_path = temp_dir / "output.txt"

        result = convert_timestamps_to_seconds(nonexistent_path, output_path)
        assert result is None

    @patch("src.video.subtitle_validation.pysrt")
    def test_validate_srt_file_pysrt_exception(self, mock_pysrt, sample_srt_file: Path):
        """Test validation when pysrt raises exception."""
        mock_pysrt.open.side_effect = Exception("pysrt error")

        result = validate_srt_file(sample_srt_file)
        assert result is False

    @patch("src.video.subtitle_utils.pysrt")
    def test_adjust_subtitle_timing_pysrt_exception(
        self, mock_pysrt, sample_srt_file: Path, temp_dir: Path
    ):
        """Test timing adjustment when pysrt raises exception."""
        mock_pysrt.open.side_effect = Exception("pysrt error")
        output_path = temp_dir / "output.srt"

        result = adjust_subtitle_timing(sample_srt_file, 1000, output_path)
        assert result is None

    @patch("src.video.subtitle_utils.pysrt")
    def test_slice_subtitles_pysrt_exception(
        self, mock_pysrt, sample_srt_file: Path, temp_dir: Path
    ):
        """Test slicing when pysrt raises exception."""
        mock_pysrt.open.side_effect = Exception("pysrt error")
        output_path = temp_dir / "output.srt"

        result = slice_subtitles(sample_srt_file, 1000, 5000, output_path)
        assert result is None

    @patch("src.video.subtitle_utils.pysrt")
    def test_get_subtitle_info_pysrt_exception(self, mock_pysrt, sample_srt_file: Path):
        """Test info extraction when pysrt raises exception."""
        mock_pysrt.open.side_effect = Exception("pysrt error")

        result = get_subtitle_info(sample_srt_file)
        assert result is None

    def test_edge_case_single_subtitle(self, temp_dir: Path):
        """Test functions with single subtitle file."""
        single_content = """1
00:00:02,000 --> 00:00:05,000
Single subtitle line.
"""
        srt_path = temp_dir / "single.srt"
        srt_path.write_text(single_content, encoding="utf-8")

        # Test validation
        assert validate_srt_file(srt_path) is True

        # Test info
        info = get_subtitle_info(srt_path)
        assert info is not None
        assert info["segment_count"] == 1
        assert info["duration_ms"] == 5000

        # Test slicing
        output_path = temp_dir / "sliced_single.srt"
        result = slice_subtitles(srt_path, 1000, 6000, output_path)
        assert result is not None

        # Test timing adjustment
        output_path2 = temp_dir / "adjusted_single.srt"
        result2 = adjust_subtitle_timing(srt_path, -1000, output_path2)
        assert result2 is not None

    def test_unicode_content_handling(self, temp_dir: Path):
        """Test handling of Unicode characters in subtitles."""
        unicode_content = """1
00:00:01,000 --> 00:00:03,000
Hello 世界! Testing emojis and special chars ñoël.

2
00:00:04,000 --> 00:00:07,000
Тестирование кириллицы и العربية text.
"""
        srt_path = temp_dir / "unicode.srt"
        srt_path.write_text(unicode_content, encoding="utf-8")

        # Test all functions handle Unicode correctly
        assert validate_srt_file(srt_path) is True

        info = get_subtitle_info(srt_path)
        assert info is not None
        assert info["segment_count"] == 2

        # Test slicing preserves Unicode - slice to get both subtitles
        output_path = temp_dir / "unicode_sliced.srt"
        result = slice_subtitles(
            srt_path, 500, 8000, output_path
        )  # Extended range to capture both
        assert result is not None

        content = result.read_text(encoding="utf-8")
        assert "世界" in content
        assert "ñoël" in content
        # The second subtitle should now be included in the slice
        assert "кириллицы" in content or "Тестирование" in content

    def test_large_file_handling(self, temp_dir: Path):
        """Test handling of files with many subtitle segments."""
        # Generate a large SRT file with 100 segments
        large_content_lines = []
        for i in range(1, 101):
            start_ms = (i - 1) * 2000  # 2 seconds apart
            end_ms = start_ms + 1500  # 1.5 second duration

            start_time = (
                f"{start_ms//3600000:02d}:{(start_ms//60000)%60:02d}:"
                f"{(start_ms//1000)%60:02d},{start_ms%1000:03d}"
            )
            end_time = (
                f"{end_ms//3600000:02d}:{(end_ms//60000)%60:02d}:"
                f"{(end_ms//1000)%60:02d},{end_ms%1000:03d}"
            )

            large_content_lines.extend(
                [
                    str(i),
                    f"{start_time} --> {end_time}",
                    f"This is subtitle number {i} for testing.",
                    "",
                ]
            )

        large_content = "\n".join(large_content_lines)
        srt_path = temp_dir / "large.srt"
        srt_path.write_text(large_content, encoding="utf-8")

        # Test functions work with large files
        assert validate_srt_file(srt_path) is True

        info = get_subtitle_info(srt_path)
        assert info is not None
        assert info["segment_count"] == 100

        # Test slicing from large file
        output_path = temp_dir / "large_sliced.srt"
        result = slice_subtitles(srt_path, 50000, 100000, output_path)  # 50s-100s range
        assert result is not None

        sliced_info = get_subtitle_info(result)
        assert sliced_info is not None
        assert sliced_info["segment_count"] > 0
