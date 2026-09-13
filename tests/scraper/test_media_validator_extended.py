"""Tests for media validation functionality.

This module provides comprehensive test coverage for image and video validation,
including format checking, dimension validation, and quality control.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
from PIL import Image

from src.scraper.amazon.media_validator import (
    MediaValidationResult,
    extract_video_metadata,
    generate_validation_report,
    validate_media_batch,
    validate_url_accessibility,
    verify_image_file,
    verify_video_file,
)


class TestMediaValidationResult:
    """Test cases for MediaValidationResult class."""

    def test_media_validation_result_creation(self):
        """Test creating MediaValidationResult."""
        file_path = Path("/test/image.jpg")
        validation_data = {"width": 1920, "height": 1080}
        issues = ["Test issue"]

        result = MediaValidationResult(file_path, True, validation_data, issues)

        assert result.file_path == file_path
        assert result.is_valid is True
        assert result.validation_data == validation_data
        assert result.issues == issues

    def test_media_validation_result_to_dict(self):
        """Test converting MediaValidationResult to dictionary."""
        file_path = Path("/test/image.jpg")
        validation_data = {"width": 1920, "height": 1080}
        issues = ["Test issue"]

        result = MediaValidationResult(file_path, False, validation_data, issues)
        result_dict = result.to_dict()

        expected = {
            "file_path": str(file_path),
            "is_valid": False,
            "validation_data": validation_data,
            "issues": issues,
            "metadata": {},
        }

        assert result_dict == expected


class TestImageValidation:
    """Test cases for image validation functionality."""

    @pytest.fixture
    def temp_image_file(self):
        """Create a temporary valid image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create a small valid JPEG image
            img = Image.new("RGB", (100, 100), color="red")
            img.save(f.name, "JPEG")
            yield Path(f.name)
            Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def temp_large_image_file(self):
        """Create a temporary large image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create a large valid JPEG image (2000x1500)
            img = Image.new("RGB", (2000, 1500), color="blue")
            img.save(f.name, "JPEG")
            yield Path(f.name)
            Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "global_settings": {
                "image_config": {
                    "min_high_res_dimension": 1500,
                    "min_high_res_file_size": 10000,
                }
            }
        }

    def test_verify_image_file_valid_large_image(
        self, temp_large_image_file, mock_config
    ):
        """Test verification of valid large image."""
        with patch("src.scraper.amazon.media_validator.CONFIG", mock_config):
            result = verify_image_file(temp_large_image_file)

        assert result.is_valid is True
        assert result.validation_data["width"] == 2000
        assert result.validation_data["height"] == 1500
        assert result.validation_data["max_dimension"] == 2000
        assert result.validation_data["format"] == "JPEG"
        assert len(result.issues) == 0

    def test_verify_image_file_small_image_fails(self, temp_image_file, mock_config):
        """Test verification fails for small image."""
        with patch("src.scraper.amazon.media_validator.CONFIG", mock_config):
            result = verify_image_file(temp_image_file)

        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any(
            "do not meet minimum requirement" in issue for issue in result.issues
        )

    def test_verify_image_file_nonexistent_file(self, mock_config):
        """Test verification of non-existent file."""
        nonexistent_file = Path("/nonexistent/image.jpg")

        with patch("src.scraper.amazon.media_validator.CONFIG", mock_config):
            result = verify_image_file(nonexistent_file)

        assert result.is_valid is False
        assert any("File does not exist" in issue for issue in result.issues)

    def test_verify_image_file_empty_file(self, mock_config):
        """Test verification of empty file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create empty file
            empty_file = Path(f.name)

        try:
            with patch("src.scraper.amazon.media_validator.CONFIG", mock_config):
                result = verify_image_file(empty_file)

            assert result.is_valid is False
            assert any(
                "File size 0 bytes is below minimum" in issue for issue in result.issues
            )
        finally:
            empty_file.unlink(missing_ok=True)

    def test_verify_image_file_custom_parameters(self, temp_large_image_file):
        """Test verification with custom parameters."""
        result = verify_image_file(
            temp_large_image_file, min_dimension=1000, min_file_size=5000
        )

        assert result.is_valid is True
        assert result.validation_data["expected_min_dimension"] == 1000
        assert result.validation_data["expected_min_file_size"] == 5000


class TestVideoMetadataExtraction:
    """Test cases for video metadata extraction functionality."""

    @patch("subprocess.run")
    def test_extract_video_metadata_success(self, mock_subprocess):
        """Test successful metadata extraction with all fields."""
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "10.5",
                    "bit_rate": "2000000",
                },
                {"codec_type": "audio", "codec_name": "aac", "bit_rate": "128000"},
            ],
            "format": {
                "duration": "10.5",
                "size": "2621440",
                "bit_rate": "2000000",
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            },
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is not None
            assert metadata["duration"] == 10.5
            assert metadata["width"] == 1920
            assert metadata["height"] == 1080
            assert metadata["codec"] == "h264"
            assert metadata["format"] == "mov,mp4,m4a,3gp,3g2,mj2"
            assert metadata["bitrate"] == 2000000
            assert metadata["has_audio"] is True
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_no_audio(self, mock_subprocess):
        """Test metadata extraction for video without audio."""
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h265",
                    "width": 3840,
                    "height": 2160,
                    "duration": "30.0",
                    "bit_rate": "8000000",
                }
            ],
            "format": {
                "duration": "30.0",
                "size": "30000000",
                "bit_rate": "8000000",
                "format_name": "mp4",
            },
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is not None
            assert metadata["has_audio"] is False
            assert metadata["codec"] == "h265"
            assert metadata["width"] == 3840
            assert metadata["height"] == 2160
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_duration_from_format(self, mock_subprocess):
        """Test metadata extraction when duration is only in format section."""
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "vp9",
                    "width": 1280,
                    "height": 720,
                }
            ],
            "format": {
                "duration": "15.75",
                "size": "5242880",
                "format_name": "webm",
            },
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is not None
            assert metadata["duration"] == 15.75
            assert metadata["bitrate"] is None
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_bitrate_from_format(self, mock_subprocess):
        """Test metadata extraction when bitrate is only in format section."""
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "av1",
                    "width": 1920,
                    "height": 1080,
                    "duration": "20.0",
                }
            ],
            "format": {
                "duration": "20.0",
                "size": "10485760",
                "bit_rate": "4194304",
                "format_name": "mp4",
            },
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is not None
            assert metadata["bitrate"] == 4194304
        finally:
            video_file.unlink(missing_ok=True)

    def test_extract_video_metadata_nonexistent_file(self):
        """Test metadata extraction for non-existent file."""
        nonexistent_file = Path("/nonexistent/video.mp4")
        metadata = extract_video_metadata(nonexistent_file)

        assert metadata is None

    @patch("subprocess.run")
    def test_extract_video_metadata_ffprobe_failure(self, mock_subprocess):
        """Test metadata extraction when FFprobe fails."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "FFprobe error: Invalid file"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"corrupted video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is None
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_invalid_json(self, mock_subprocess):
        """Test metadata extraction when FFprobe returns invalid JSON."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "not valid json {["

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is None
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_no_video_streams(self, mock_subprocess):
        """Test metadata extraction when no video streams found."""
        ffprobe_output = {
            "streams": [{"codec_type": "audio", "codec_name": "aac"}],
            "format": {"duration": "10.0", "format_name": "mp4"},
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake audio data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is None
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_incomplete_data(self, mock_subprocess):
        """Test metadata extraction with incomplete video stream data."""
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    # Missing height and duration
                }
            ],
            "format": {"format_name": "mp4"},
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is None
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_timeout(self, mock_subprocess):
        """Test metadata extraction handles timeout gracefully."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("ffprobe", 30)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is None
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_subprocess_error(self, mock_subprocess):
        """Test metadata extraction handles subprocess errors."""
        mock_subprocess.side_effect = subprocess.SubprocessError("Process failed")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is None
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_unexpected_exception(self, mock_subprocess):
        """Test metadata extraction handles unexpected exceptions."""
        mock_subprocess.side_effect = RuntimeError("Unexpected error")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is None
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_invalid_duration_type(self, mock_subprocess):
        """Test metadata extraction handles invalid duration type."""
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "invalid",  # Invalid duration
                }
            ],
            "format": {
                "duration": "not_a_number",  # Invalid duration
                "format_name": "mp4",
            },
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is None
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_invalid_bitrate_type(self, mock_subprocess):
        """Test metadata extraction handles invalid bitrate type."""
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "10.0",
                    "bit_rate": "not_a_number",
                }
            ],
            "format": {
                "duration": "10.0",
                "bit_rate": "invalid",
                "format_name": "mp4",
            },
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is not None
            assert metadata["bitrate"] is None
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_video_metadata_multiple_video_streams(self, mock_subprocess):
        """Test metadata extraction uses first video stream when multiple exist."""
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "10.0",
                },
                {
                    "codec_type": "video",
                    "codec_name": "h265",
                    "width": 3840,
                    "height": 2160,
                    "duration": "10.0",
                },
            ],
            "format": {"duration": "10.0", "format_name": "mp4"},
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            metadata = extract_video_metadata(video_file)

            assert metadata is not None
            assert metadata["codec"] == "h264"
            assert metadata["width"] == 1920
            assert metadata["height"] == 1080
        finally:
            video_file.unlink(missing_ok=True)


class TestVideoValidation:
    """Test cases for video validation functionality."""

    def test_verify_video_file_nonexistent_file(self):
        """Test verification of non-existent video file."""
        nonexistent_file = Path("/nonexistent/video.mp4")

        result = verify_video_file(nonexistent_file)

        assert result.is_valid is False
        assert any("File does not exist" in issue for issue in result.issues)

    def test_verify_video_file_empty_file(self):
        """Test verification of empty video file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            empty_file = Path(f.name)

        try:
            result = verify_video_file(empty_file)

            assert result.is_valid is False
            assert any("File is empty" in issue for issue in result.issues)
        finally:
            empty_file.unlink(missing_ok=True)

    def test_verify_video_file_invalid_extension(self):
        """Test verification of file with invalid extension."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a video")
            invalid_file = Path(f.name)

        try:
            result = verify_video_file(invalid_file)

            assert result.is_valid is False
            assert any("FFprobe failed" in issue for issue in result.issues)
        finally:
            invalid_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_verify_video_file_ffprobe_success(self, mock_subprocess):
        """Test video verification with successful FFprobe."""
        # Mock FFprobe output
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "10.5",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "10.5", "size": "1048576", "bit_rate": "1000000"},
        }

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(ffprobe_output)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            with patch(
                "builtins.open", mock_open(read_data=b"\x00\x00\x00\x18ftypmp41")
            ):
                result = verify_video_file(video_file)

            assert result.is_valid is True
            assert result.validation_data["width"] == 1920
            assert result.validation_data["height"] == 1080
            assert result.validation_data["duration"] == 10.5
            assert result.validation_data["video_codec"] == "h264"
            assert result.validation_data["audio_codec"] == "aac"
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_verify_video_file_ffprobe_failure(self, mock_subprocess):
        """Test video verification with FFprobe failure."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "FFprobe error"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            video_file = Path(f.name)

        try:
            result = verify_video_file(video_file)

            assert result.is_valid is False
            assert any("FFprobe failed with error" in issue for issue in result.issues)
        finally:
            video_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_verify_video_file_html_content(self, mock_subprocess):
        """Test video verification detects HTML content."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Invalid data"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"<html><head><title>Not a video</title></head></html>")
            html_file = Path(f.name)

        try:
            result = verify_video_file(html_file)

            assert result.is_valid is False
            assert any("FFprobe failed" in issue for issue in result.issues)
        finally:
            html_file.unlink(missing_ok=True)


class TestBatchValidation:
    """Test cases for batch media validation."""

    @pytest.fixture
    def temp_files(self):
        """Create temporary test files."""
        files = []

        # Valid large image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (2000, 1500), color="green")
            img.save(f.name, "JPEG")
            files.append(Path(f.name))

        # Small image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(f.name, "JPEG")
            files.append(Path(f.name))

        # Empty file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            files.append(Path(f.name))

        yield files

        # Cleanup
        for file_path in files:
            file_path.unlink(missing_ok=True)

    def test_validate_media_batch_auto_detection(self, temp_files):
        """Test batch validation with auto-detection."""
        results = validate_media_batch(temp_files, media_type="auto")

        assert len(results) == 3
        assert all(isinstance(r, MediaValidationResult) for r in results)

        # First file should be valid (large image)
        assert results[0].is_valid is True
        assert results[0].validation_data["file_type"] == "image"

        # Second file should be invalid (small image)
        assert results[1].is_valid is False
        assert results[1].validation_data["file_type"] == "image"

        # Third file should be invalid (empty video)
        assert results[2].is_valid is False
        assert results[2].validation_data["file_type"] == "video"

    def test_validate_media_batch_forced_image_type(self, temp_files):
        """Test batch validation with forced image type."""
        results = validate_media_batch(temp_files, media_type="image")

        assert len(results) == 3
        assert all(r.validation_data["file_type"] == "image" for r in results)


class TestValidationReporting:
    """Test cases for validation reporting functionality."""

    def test_generate_validation_report_empty_results(self):
        """Test generating report with empty results."""
        report = generate_validation_report([])

        assert report["summary"]["total_files"] == 0
        assert report["summary"]["valid_files"] == 0
        assert report["summary"]["invalid_files"] == 0
        assert report["summary"]["success_rate"] == 0

    def test_generate_validation_report_mixed_results(self):
        """Test generating report with mixed results."""
        # Create mock results
        valid_result = MediaValidationResult(
            Path("/test/valid.jpg"),
            True,
            {
                "file_type": "image",
                "width": 1920,
                "height": 1080,
                "actual_file_size": 50000,
            },
            [],
        )

        invalid_result = MediaValidationResult(
            Path("/test/invalid.mp4"),
            False,
            {"file_type": "video"},
            ["File does not exist", "Invalid format"],
        )

        results = [valid_result, invalid_result]
        report = generate_validation_report(results)

        assert report["summary"]["total_files"] == 2
        assert report["summary"]["valid_files"] == 1
        assert report["summary"]["invalid_files"] == 1
        assert report["summary"]["success_rate"] == 50.0
        assert report["summary"]["image_files"] == 1
        assert report["summary"]["video_files"] == 1

        # Check common issues
        assert "File does not exist" in report["common_issues"]
        assert "Invalid format" in report["common_issues"]

    def test_generate_validation_report_save_to_file(self):
        """Test saving validation report to file."""
        valid_result = MediaValidationResult(
            Path("/test/valid.jpg"), True, {"file_type": "image"}, []
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.json"
            generate_validation_report([valid_result], output_path)

            assert output_path.exists()
            with open(output_path) as f:
                saved_report = json.load(f)

            assert saved_report["summary"]["total_files"] == 1
            assert saved_report["summary"]["valid_files"] == 1


class TestURLValidation:
    """Test cases for URL accessibility validation."""

    @patch("requests.head")
    def test_validate_url_accessibility_success(self, mock_head):
        """Test successful URL validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "image/jpeg",
            "content-length": "50000",
        }
        mock_response.url = "https://example.com/image.jpg"
        mock_head.return_value = mock_response

        is_accessible, metadata = validate_url_accessibility(
            "https://example.com/image.jpg"
        )

        assert is_accessible is True
        assert metadata["status_code"] == "200"
        assert metadata["content_type"] == "image/jpeg"
        assert metadata["content_length"] == "50000"
        assert metadata["redirected"] == "False"

    @patch("requests.head")
    def test_validate_url_accessibility_failure(self, mock_head):
        """Test failed URL validation."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.url = "https://example.com/notfound.jpg"
        mock_head.return_value = mock_response

        is_accessible, metadata = validate_url_accessibility(
            "https://example.com/notfound.jpg"
        )

        assert is_accessible is False
        assert metadata["status_code"] == "404"

    @patch("requests.head")
    def test_validate_url_accessibility_html_warning(self, mock_head):
        """Test URL validation detects HTML content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com/page.html"
        mock_head.return_value = mock_response

        is_accessible, metadata = validate_url_accessibility(
            "https://example.com/page.html"
        )

        assert is_accessible is True
        assert "warning" in metadata
        assert "HTML content" in metadata["warning"]

    @patch("requests.head")
    def test_validate_url_accessibility_timeout(self, mock_head):
        """Test URL validation with timeout."""
        import requests

        mock_head.side_effect = requests.exceptions.Timeout()

        is_accessible, metadata = validate_url_accessibility(
            "https://example.com/slow.jpg"
        )

        assert is_accessible is False
        assert metadata["error"] == "Request timeout"


class TestIntegrationScenarios:
    """Integration test scenarios for media validation."""

    @pytest.fixture
    def mock_config_full(self):
        """Full mock configuration for integration tests."""
        return {
            "global_settings": {
                "image_config": {
                    "min_high_res_dimension": 1500,
                    "min_high_res_file_size": 10000,
                }
            },
            "scrapers": {
                "amazon": {
                    "http_headers": {
                        "media_download": {
                            "User-Agent": "Mozilla/5.0 Test Browser",
                            "Accept": "*/*",
                        }
                    }
                }
            },
        }

    def test_complete_validation_workflow(self, mock_config_full):
        """Test complete validation workflow from URL to final report."""
        # Create test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create valid image
            valid_img_path = temp_path / "valid.jpg"
            img = Image.new("RGB", (2000, 1500), color="blue")
            img.save(valid_img_path, "JPEG")

            # Create invalid image (too small)
            invalid_img_path = temp_path / "invalid.jpg"
            small_img = Image.new("RGB", (100, 100), color="red")
            small_img.save(invalid_img_path, "JPEG")

            with patch("src.scraper.amazon.media_validator.CONFIG", mock_config_full):
                # Test individual validations
                valid_result = verify_image_file(valid_img_path)
                invalid_result = verify_image_file(invalid_img_path)

                assert valid_result.is_valid is True
                assert invalid_result.is_valid is False

                # Test batch validation
                batch_results = validate_media_batch([valid_img_path, invalid_img_path])
                assert len(batch_results) == 2
                assert batch_results[0].is_valid is True
                assert batch_results[1].is_valid is False

                # Test report generation
                report_path = temp_path / "validation_report.json"
                report = generate_validation_report(batch_results, report_path)

                assert report["summary"]["total_files"] == 2
                assert report["summary"]["valid_files"] == 1
                assert report["summary"]["invalid_files"] == 1
                assert report["summary"]["success_rate"] == 50.0
                assert report_path.exists()

    def test_error_handling_with_corrupted_files(self):
        """Test validation handles corrupted files gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create file with invalid image data
            corrupted_file = temp_path / "corrupted.jpg"
            corrupted_file.write_bytes(b"this is not an image file")

            result = verify_image_file(corrupted_file)

            assert result.is_valid is False
            assert len(result.issues) > 0
            assert any(
                "Failed to open/process image" in issue for issue in result.issues
            )
