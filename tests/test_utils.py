"""Unit tests for utility functions."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.utils import (
    cleanup_temp_dirs,
    ensure_dirs_exist,
    format_timestamp,
    get_filename_from_url,
    sanitize_filename,
    take_screenshot,
)
from src.utils.script_sanitizer import sanitize_script


class TestEnsureDirsExist:
    """Test directory creation utilities."""

    @pytest.mark.unit
    def test_ensure_dirs_exist_file_path(self, temp_dir: Path):
        """Test creating directories for a file path."""
        file_path = temp_dir / "subdir" / "nested" / "file.txt"
        ensure_dirs_exist(file_path)

        assert file_path.parent.exists()
        assert file_path.parent.is_dir()

    @pytest.mark.unit
    def test_ensure_dirs_exist_directory_path(self, temp_dir: Path):
        """Test creating a directory path."""
        dir_path = temp_dir / "newdir" / "nested"
        ensure_dirs_exist(dir_path)

        assert dir_path.exists()
        assert dir_path.is_dir()

    def test_ensure_dirs_exist_existing_directory(self, temp_dir: Path):
        """Test handling existing directories."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        ensure_dirs_exist(existing_dir)
        assert existing_dir.exists()

    def test_ensure_dirs_exist_permission_error(self, temp_dir: Path):
        """Test handling permission errors gracefully."""
        # This test verifies that permission errors are logged but don't crash
        with (
            patch("src.utils.logger.error") as mock_logger,
            patch.object(
                Path, "mkdir", side_effect=PermissionError("Permission denied")
            ),
        ):
            file_path = temp_dir / "test" / "file.txt"
            ensure_dirs_exist(file_path)

            mock_logger.assert_called_once()


class TestSanitizeFilename:
    """Test filename sanitization."""

    @pytest.mark.unit
    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        result = sanitize_filename("test file.txt")
        assert result == "test_file.txt"

    @pytest.mark.unit
    def test_sanitize_filename_special_chars(self):
        """Test sanitization of special characters."""
        result = sanitize_filename("file@#$%^&*()+.txt")
        assert result == "file@#$%^&_()+.txt"  # Only invalid chars are replaced with _

    @pytest.mark.unit
    def test_sanitize_filename_multiple_dots(self):
        """Test handling of multiple dots."""
        result = sanitize_filename("file..name..txt")
        assert result == "file..name..txt"  # Dots are preserved

    @pytest.mark.unit
    def test_sanitize_filename_empty(self):
        """Test handling empty filename."""
        result = sanitize_filename("")
        assert result == "file"  # Current fallback name

    @pytest.mark.unit
    def test_sanitize_filename_whitespace_only(self):
        """Test handling whitespace-only filename."""
        result = sanitize_filename("   ")
        assert result == "file"  # Current fallback name


class TestGetFilenameFromUrl:
    """Test URL filename extraction."""

    def test_get_filename_from_url_basic(self):
        """Test basic URL filename extraction."""
        url = "https://example.com/image.jpg"
        result = get_filename_from_url(url, "test_id")
        assert "test_id" in result
        assert result.endswith(".jpg")

    def test_get_filename_from_url_with_query_params(self):
        """Test URL with query parameters."""
        url = "https://example.com/image.jpg?size=large&format=jpg"
        result = get_filename_from_url(url, "test_id")
        assert "test_id" in result
        assert result.endswith(".jpg")

    def test_get_filename_from_url_with_fragment(self):
        """Test URL with fragment."""
        url = "https://example.com/image.jpg#section"
        result = get_filename_from_url(url, "test_id")
        assert "test_id" in result
        assert result.endswith(".jpg")

    def test_get_filename_from_url_no_extension(self):
        """Test URL without file extension."""
        url = "https://example.com/image"
        result = get_filename_from_url(url, "test_id")
        assert "test_id" in result
        assert "image" in result

    def test_get_filename_from_url_complex_path(self):
        """Test URL with complex path."""
        url = "https://example.com/path/to/deep/nested/image.jpg"
        result = get_filename_from_url(url, "test_id")
        assert "test_id" in result
        assert result.endswith(".jpg")

    def test_get_filename_from_url_with_content_type(self):
        """Test filename extraction with content type hint."""
        url = "https://example.com/data"
        result = get_filename_from_url(url, "test_id", "image/jpeg")
        assert "test_id" in result
        assert result.endswith(".jpg")

    def test_get_filename_from_url_generic_name(self):
        """Test filename extraction with generic URL name."""
        url = "https://example.com/index"
        result = get_filename_from_url(url, "test_id", "video/mp4")
        assert "test_id" in result
        assert result.endswith(".mp4")


class TestFormatTimestamp:
    """Test timestamp formatting."""

    def test_format_timestamp_seconds(self):
        """Test formatting seconds."""
        result = format_timestamp(61.5)
        assert result == "00:01:01,500"

    def test_format_timestamp_minutes(self):
        """Test formatting minutes."""
        result = format_timestamp(125.75)
        assert result == "00:02:05,750"

    def test_format_timestamp_hours(self):
        """Test formatting hours."""
        result = format_timestamp(3661.25)
        assert result == "01:01:01,250"

    def test_format_timestamp_zero(self):
        """Test formatting zero seconds."""
        result = format_timestamp(0.0)
        assert result == "00:00:00,000"

    def test_format_timestamp_fractional(self):
        """Test formatting fractional seconds."""
        result = format_timestamp(0.123)
        assert result == "00:00:00,123"


class TestCleanupTempDirs:
    """Test temporary directory cleanup."""

    def test_cleanup_temp_dirs_empty_list(self):
        """Test cleanup with empty directory list."""
        cleanup_temp_dirs()
        # Should not raise any exceptions

    def test_cleanup_temp_dirs_nonexistent(self, temp_dir: Path):
        """Test cleanup of non-existent directories."""
        nonexistent_dir = temp_dir / "nonexistent"
        cleanup_temp_dirs(nonexistent_dir)
        # Should not raise any exceptions

    def test_cleanup_temp_dirs_existing(self, temp_dir: Path):
        """Test cleanup of existing directories."""
        test_dir = temp_dir / "test_cleanup"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")

        cleanup_temp_dirs(test_dir)
        assert not test_dir.exists()

    def test_cleanup_temp_dirs_with_files(self, temp_dir: Path):
        """Test cleanup of directories with files."""
        test_dir = temp_dir / "test_with_files"
        test_dir.mkdir()

        # Create some test files
        (test_dir / "file1.txt").write_text("test1")
        (test_dir / "file2.txt").write_text("test2")
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "nested.txt").write_text("nested")

        cleanup_temp_dirs(test_dir)
        assert not test_dir.exists()


class TestDownloadFile:
    """Test file download functionality."""

    @pytest.mark.skip(reason="Complex async HTTP mocking needs integration setup")
    @pytest.mark.asyncio
    async def test_download_file_success(self, temp_dir: Path):
        """Test successful file download."""
        pass

    @pytest.mark.skip(reason="Complex async HTTP mocking needs integration setup")
    @pytest.mark.asyncio
    async def test_download_file_http_error(self, temp_dir: Path):
        """Test download with HTTP error."""
        pass

    @pytest.mark.skip(reason="Complex async HTTP mocking needs integration setup")
    @pytest.mark.asyncio
    async def test_download_file_network_error(self, temp_dir: Path):
        """Test download with network error."""
        pass


class TestTakeScreenshot:
    """Test screenshot functionality."""

    @pytest.mark.asyncio
    async def test_take_screenshot(self, temp_dir: Path):
        """Test taking a screenshot."""
        # Mock page object
        mock_page = AsyncMock()

        await take_screenshot(mock_page, temp_dir, "test_screenshot")

        # Verify screenshot was called
        mock_page.screenshot.assert_called_once()
        call_args = mock_page.screenshot.call_args
        assert "path" in call_args[1]
        # Convert Path to string for endswith check
        assert str(call_args[1]["path"]).endswith("test_screenshot.png")


class TestScriptSanitizer:
    """Test script sanitization functionality."""

    def test_sanitize_script_basic(self):
        """Test basic script sanitization."""
        script = "This is a **bold** test script with `code` and [links](url)."
        result = sanitize_script(script)

        # Should remove markdown formatting
        assert "**bold**" not in result
        assert "`code`" in result  # Backticks are not handled by this function
        assert "code" in result  # Content remains
        assert "This is a bold test script with" in result
        # Note: The function doesn't remove square brackets, so [links] remains

    def test_sanitize_script_empty(self):
        """Test sanitization of empty script."""
        result = sanitize_script("")
        assert result == ""

    def test_sanitize_script_whitespace(self):
        """Test sanitization of whitespace-only script."""
        result = sanitize_script("   \n\t   ")
        assert result == ""

    def test_sanitize_script_special_chars(self):
        """Test sanitization of special characters."""
        script = "Script with special chars: @#$%^&()"
        result = sanitize_script(script)
        # Should preserve most special characters (but * gets removed)
        assert "@#$%^&()" in result
        assert "*" not in result

    def test_sanitize_script_multiple_spaces(self):
        """Test normalization of multiple spaces."""
        script = "Multiple    spaces   should   be   normalized"
        result = sanitize_script(script)
        assert "  " not in result  # No double spaces
        assert "Multiple spaces should be normalized" in result

    def test_sanitize_script_newlines(self):
        """Test handling of newlines."""
        script = "Line 1\nLine 2\n\nLine 3"
        result = sanitize_script(script)
        # Should normalize newlines but preserve line structure
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
