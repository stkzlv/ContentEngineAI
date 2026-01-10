"""Unit tests for MediaValidator utility."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.scraper.amazon.media_validator import (
    extract_video_metadata,
    verify_image_file,
    verify_video_file,
    validate_media_batch,
    generate_validation_report,
    MediaValidationResult,
)

pytestmark = pytest.mark.unit

@pytest.fixture
def mock_image_path(tmp_path):
    """Create a mock image file."""
    path = tmp_path / "test_image.jpg"
    path.write_bytes(b"mock image content")
    return path

@pytest.fixture
def mock_video_path(tmp_path):
    """Create a mock video file."""
    path = tmp_path / "test_video.mp4"
    # Use valid MP4 signature
    path.write_bytes(b"\x00\x00\x00\x18ftypmp4")
    return path

class TestImageValidation:
    """Tests for verify_image_file."""

    @patch("src.scraper.amazon.media_validator.Image.open")
    def test_verify_image_success(self, mock_open, mock_image_path):
        """Test successful image validation."""
        mock_img = MagicMock()
        mock_img.size = (2000, 2000)
        mock_img.format = "JPEG"
        mock_img.mode = "RGB"
        mock_open.return_value.__enter__.return_value = mock_img
        
        result = verify_image_file(mock_image_path, min_dimension=1000, min_file_size=10)
        
        assert result.is_valid is True
        assert result.validation_data["width"] == 2000
        assert result.validation_data["format"] == "JPEG"
        assert len(result.issues) == 0

    @patch("src.scraper.amazon.media_validator.Image.open")
    def test_verify_image_too_small(self, mock_open, mock_image_path):
        """Test image failing dimension check."""
        mock_img = MagicMock()
        mock_img.size = (500, 500)
        mock_img.format = "JPEG"
        mock_open.return_value.__enter__.return_value = mock_img
        
        result = verify_image_file(mock_image_path, min_dimension=1000)
        
        assert result.is_valid is False
        assert any("dimensions" in issue for issue in result.issues)

    def test_verify_image_nonexistent(self, tmp_path):
        """Test validation of nonexistent file."""
        result = verify_image_file(tmp_path / "nonexistent.jpg")
        assert result.is_valid is False
        assert "File does not exist" in result.issues[0]

class TestVideoValidation:
    """Tests for verify_video_file and metadata extraction."""

    @patch("src.scraper.amazon.media_validator.subprocess.run")
    def test_extract_video_metadata_success(self, mock_run, mock_video_path):
        """Test successful metadata extraction via FFprobe."""
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "10.5",
                    "bit_rate": "5000000"
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac"
                }
            ],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "10.5",
                "size": "6562500"
            }
        }
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(ffprobe_output))
        
        metadata = extract_video_metadata(mock_video_path)
        
        assert metadata is not None
        assert metadata["duration"] == 10.5
        assert metadata["width"] == 1920
        assert metadata["codec"] == "h264"
        assert metadata["has_audio"] is True

    @patch("src.scraper.amazon.media_validator.subprocess.run")
    def test_verify_video_success(self, mock_run, mock_video_path):
        """Test successful video validation."""
        ffprobe_output = {
            "streams": [{"codec_type": "video", "width": 1280, "height": 720, "duration": "5.0", "codec_name": "h264"}],
            "format": {"duration": "5.0", "format_name": "mp4"}
        }
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(ffprobe_output))
        
        result = verify_video_file(mock_video_path, min_duration=2.0, min_dimension=480)
        
        assert result.is_valid is True
        assert result.metadata["width"] == 1280
        assert len(result.issues) == 0

    @patch("src.scraper.amazon.media_validator.subprocess.run")
    def test_verify_video_invalid_format(self, mock_run, mock_video_path):
        """Test video failing format/signature check."""
        # Use a clearly HTML start to trigger the specific HTML warning
        mock_video_path.write_bytes(b"<html><head>")
        
        # Simulate FFprobe raising an exception (e.g. timeout or subprocess error)
        # This causes the code to fall through to the signature check
        mock_run.side_effect = subprocess.SubprocessError("FFprobe failed")
        
        result = verify_video_file(mock_video_path)
        
        assert result.is_valid is False
        # Should contain the FFprobe error AND the HTML content warning
        assert any("FFprobe subprocess error" in issue for issue in result.issues)
        assert any("HTML content" in issue for issue in result.issues)

class TestBatchAndReport:
    """Tests for batch validation and reporting."""

    @patch("src.scraper.amazon.media_validator.verify_image_file")
    @patch("src.scraper.amazon.media_validator.verify_video_file")
    def test_validate_media_batch(self, mock_video, mock_image, tmp_path):
        """Test batch validation of mixed media."""
        img_path = tmp_path / "img.jpg"
        vid_path = tmp_path / "vid.mp4"
        
        mock_image.return_value = MediaValidationResult(img_path, True, {"file_type": "image"})
        mock_video.return_value = MediaValidationResult(vid_path, True, {"file_type": "video"})
        
        results = validate_media_batch([img_path, vid_path])
        
        assert len(results) == 2
        assert mock_image.called
        assert mock_video.called

    def test_generate_validation_report(self):
        """Test report generation from results."""
        results = [
            MediaValidationResult(
                Path("i1.jpg"), 
                True, 
                {
                    "file_type": "image", 
                    "width": 2000, 
                    "height": 2000,  # Added height
                    "actual_file_size": 20000
                }
            ),
            MediaValidationResult(Path("v1.mp4"), False, {"file_type": "video"}, issues=["Corrupt"]),
        ]
        
        report = generate_validation_report(results)
        
        assert report["summary"]["total_files"] == 2
        assert report["summary"]["valid_files"] == 1
        assert report["summary"]["invalid_files"] == 1
        assert "Corrupt" in report["common_issues"]
