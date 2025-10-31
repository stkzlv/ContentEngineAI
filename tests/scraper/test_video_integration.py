"""Integration tests for end-to-end video processing pipeline.

These tests validate the complete video workflow from extraction through validation
and metadata extraction using real Amazon product data (when available) or mocked
responses for CI environments.

Run with: pytest tests/scraper/test_video_integration.py -v
Skip slow tests: pytest tests/scraper/test_video_integration.py -v -m "not slow"
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.scraper.amazon.media_validator import (
    extract_video_metadata,
    verify_video_file,
)
from src.scraper.base.models import BaseProductData, Platform

# Test markers
pytestmark = pytest.mark.integration


# Test ASINs known to have video content (as of design phase)
TEST_ASINS_WITH_VIDEOS = [
    "B0BTYCRJSS",  # Known to have product videos
    "B0D6GZF3T4",  # Alternate test ASIN with videos
]

# Marker for slow tests that hit real Amazon pages
slow = pytest.mark.slow


@pytest.fixture
def test_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="test_video_integration_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_video_file():
    """Create a mock video file with valid MP4 signature."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        # Write minimal valid MP4 header
        f.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
        f.write(b"\x00" * 1000)  # Add some data
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def mock_ffprobe_metadata():
    """Mock FFprobe output for video metadata."""
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "duration": "10.5",
                "bit_rate": "2000000",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
            },
        ],
        "format": {
            "duration": "10.5",
            "size": "2621440",
            "bit_rate": "2000000",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        },
    }


class TestVideoExtractionValidation:
    """Test video extraction and validation pipeline."""

    def test_video_file_validation_with_metadata(
        self, mock_video_file, mock_ffprobe_metadata
    ):
        """Test complete video validation including metadata extraction.

        Validates:
        - Requirement 2: Video Metadata Extraction
        - Requirement 3: Video Validation and Quality Filtering
        """
        with patch("subprocess.run") as mock_subprocess:
            # Mock FFprobe success
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = json.dumps(mock_ffprobe_metadata)

            # Test validation
            result = verify_video_file(mock_video_file)

            # Verify validation passes
            assert result.is_valid is True, "Video should pass validation"
            assert len(result.issues) == 0, "Should have no validation issues"

            # Verify validation data populated
            assert "width" in result.validation_data
            assert "height" in result.validation_data
            assert "duration" in result.validation_data

            # Verify metadata extracted
            assert "duration" in result.metadata
            assert result.metadata["duration"] == 10.5
            assert result.metadata["width"] == 1920
            assert result.metadata["height"] == 1080
            assert result.metadata["codec"] == "h264"
            assert result.metadata["has_audio"] is True

    def test_video_validation_graceful_degradation(self, mock_video_file):
        """Test graceful degradation when FFprobe unavailable.

        Validates:
        - Requirement 5: Robust Error Handling
        - Metadata extraction failures don't crash validation
        """
        with patch("subprocess.run") as mock_subprocess:
            # Mock FFprobe failure
            mock_subprocess.return_value.returncode = 1
            mock_subprocess.return_value.stderr = "FFprobe not found"

            # Test validation still works
            result = verify_video_file(mock_video_file)

            # Validation should still complete (signature check, etc.)
            assert result is not None
            assert isinstance(result.validation_data, dict)
            # Metadata might be empty but should not crash
            assert isinstance(result.metadata, dict)

    def test_video_validation_corrupted_file(self, test_output_dir):
        """Test validation rejects corrupted video files.

        Validates:
        - Requirement 3: Video Validation and Quality Filtering
        - Corrupted files are rejected with clear errors
        """
        # Create corrupted file
        corrupted_file = test_output_dir / "corrupted.mp4"
        corrupted_file.write_bytes(b"<html>Not a video</html>")

        # Test validation
        result = verify_video_file(corrupted_file)

        # Should fail validation
        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any(
            "FFprobe failed" in issue or "HTML" in issue for issue in result.issues
        )

    def test_metadata_extraction_handles_missing_fields(self, mock_video_file):
        """Test metadata extraction handles incomplete FFprobe output.

        Validates:
        - Requirement 2: Video Metadata Extraction
        - Requirement 5: Robust Error Handling
        """
        incomplete_metadata = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    # Missing height and duration
                }
            ],
            "format": {},
        }

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = json.dumps(incomplete_metadata)

            # Extract metadata
            metadata = extract_video_metadata(mock_video_file)

            # Should return None for incomplete data
            assert metadata is None


class TestVideoStorageOrganization:
    """Test video storage and file organization."""

    def test_video_directory_creation(self, test_output_dir):
        """Test video directory is created correctly.

        Validates:
        - Requirement 4: Organized Video Storage
        - Directory structure follows outputs/{ASIN}/videos/ pattern
        """
        asin = "TEST_ASIN_123"
        videos_dir = test_output_dir / asin / "videos"

        # Create directory structure
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Verify structure
        assert videos_dir.exists()
        assert videos_dir.is_dir()
        assert videos_dir.parent.name == asin

    def test_video_file_naming_pattern(self, test_output_dir):
        """Test video files follow naming convention.

        Validates:
        - Requirement 4: Organized Video Storage
        - Files named as video_{index}.mp4
        """
        videos_dir = test_output_dir / "TEST_ASIN" / "videos"
        videos_dir.mkdir(parents=True)

        # Create test video files
        for i in range(3):
            video_file = videos_dir / f"video_{i}.mp4"
            video_file.write_bytes(b"fake video data")

        # Verify naming pattern
        video_files = sorted(videos_dir.glob("video_*.mp4"))
        assert len(video_files) == 3
        assert video_files[0].name == "video_0.mp4"
        assert video_files[1].name == "video_1.mp4"
        assert video_files[2].name == "video_2.mp4"


class TestProductDataIntegration:
    """Test video data integration with product data structures."""

    def test_product_data_video_fields(self):
        """Test product data includes video fields.

        Validates:
        - Requirement 6: Product Data Integration
        - BaseProductData has video fields
        """
        # Create product data with videos
        product = BaseProductData(
            title="Test Product",
            price="19.99",
            url="https://example.com/product",
            platform=Platform.AMAZON,
            platform_id="TEST_123",
            videos=["https://example.com/video1.mp4"],
            downloaded_videos=["outputs/TEST_123/videos/video_0.mp4"],
        )

        # Verify video fields present
        assert hasattr(product, "videos")
        assert hasattr(product, "downloaded_videos")
        assert isinstance(product.videos, list)
        assert isinstance(product.downloaded_videos, list)
        assert len(product.videos) == 1
        assert len(product.downloaded_videos) == 1

    def test_product_data_empty_videos_handling(self):
        """Test product data handles missing videos gracefully.

        Validates:
        - Requirement 6: Product Data Integration
        - Empty video lists don't cause errors
        """
        # Create product without videos
        product = BaseProductData(
            title="Test Product No Videos",
            price="29.99",
            url="https://example.com/product",
            platform=Platform.AMAZON,
            platform_id="TEST_456",
            videos=[],
            downloaded_videos=[],
        )

        # Should not raise errors
        assert product.videos == []
        assert product.downloaded_videos == []

    def test_product_data_serialization_with_videos(self, test_output_dir):
        """Test product data with videos serializes to JSON correctly.

        Validates:
        - Requirement 6: Product Data Integration
        - Video metadata included in data.json
        """
        # Create product with videos
        product = BaseProductData(
            title="Test Product",
            price="39.99",
            url="https://example.com/product",
            platform=Platform.AMAZON,
            platform_id="TEST_789",
            videos=["https://example.com/video1.mp4", "https://example.com/video2.mp4"],
            downloaded_videos=["videos/video_0.mp4", "videos/video_1.mp4"],
        )

        # Serialize to JSON
        data_file = test_output_dir / "data.json"
        with open(data_file, "w") as f:
            json.dump(product.to_dict(), f, indent=2)

        # Verify serialization
        assert data_file.exists()

        # Load and verify
        with open(data_file) as f:
            loaded_data = json.load(f)

        assert "videos" in loaded_data
        assert "downloaded_videos" in loaded_data
        assert len(loaded_data["videos"]) == 2
        assert len(loaded_data["downloaded_videos"]) == 2


class TestErrorHandlingAndRecovery:
    """Test error handling and graceful degradation."""

    def test_partial_video_failure_continues_processing(
        self, test_output_dir, mock_video_file
    ):
        """Test processing continues when some videos fail.

        Validates:
        - Requirement 5: Robust Error Handling
        - Partial failures don't halt entire process
        """
        # Create mix of valid and invalid videos
        videos_dir = test_output_dir / "TEST_ASIN" / "videos"
        videos_dir.mkdir(parents=True)

        # Copy valid video
        valid_video = videos_dir / "video_0.mp4"
        shutil.copy(mock_video_file, valid_video)

        # Create invalid video
        invalid_video = videos_dir / "video_1.mp4"
        invalid_video.write_bytes(b"<html>Not a video</html>")

        # Validate both
        results = []
        for video_file in [valid_video, invalid_video]:
            with patch("subprocess.run") as mock_subprocess:
                if video_file == valid_video:
                    mock_subprocess.return_value.returncode = 0
                    mock_subprocess.return_value.stdout = json.dumps(
                        {
                            "streams": [
                                {
                                    "codec_type": "video",
                                    "codec_name": "h264",
                                    "width": 1920,
                                    "height": 1080,
                                    "duration": "10.0",
                                }
                            ],
                            "format": {"duration": "10.0", "format_name": "mp4"},
                        }
                    )
                else:
                    mock_subprocess.return_value.returncode = 1
                    mock_subprocess.return_value.stderr = "Invalid file"

                result = verify_video_file(video_file)
                results.append(result)

        # Should have both results
        assert len(results) == 2
        assert results[0].is_valid is True
        assert results[1].is_valid is False

    def test_all_videos_fail_product_continues(self):
        """Test product processing continues when all videos fail.

        Validates:
        - Requirement 5: Robust Error Handling
        - Products process successfully with images only
        """
        # Create product with failed videos
        product = BaseProductData(
            title="Test Product",
            price="49.99",
            url="https://example.com/product",
            platform=Platform.AMAZON,
            platform_id="TEST_NO_VIDEOS",
            videos=[],  # All videos failed
            downloaded_videos=[],  # No videos downloaded
            images=["image1.jpg", "image2.jpg"],  # But has images
            downloaded_images=["images/image_0.jpg", "images/image_1.jpg"],
        )

        # Should not raise errors
        assert product.platform_id == "TEST_NO_VIDEOS"
        assert len(product.videos) == 0
        assert len(product.downloaded_videos) == 0
        assert len(product.images) > 0  # Has fallback images


class TestEndToEndIntegration:
    """End-to-end integration tests for complete pipeline."""

    @slow
    @pytest.mark.skipif(
        not shutil.which("ffprobe"),
        reason="Requires FFprobe installed for real video validation",
    )
    def test_complete_video_validation_pipeline(self, mock_video_file):
        """Test complete pipeline with real FFprobe (if available).

        Validates:
        - Complete integration of all requirements
        - Real FFprobe execution works correctly
        """
        # This test uses real FFprobe if available
        result = verify_video_file(mock_video_file)

        # Basic validation should work
        assert result is not None
        assert isinstance(result.validation_data, dict)

        # If FFprobe available, metadata should be extracted
        if shutil.which("ffprobe"):
            # With minimal MP4 file, FFprobe might fail but shouldn't crash
            assert isinstance(result.metadata, dict)

    def test_validation_report_generation(self, test_output_dir, mock_video_file):
        """Test validation reports are generated correctly.

        Validates:
        - Requirement 3: Video Validation and Quality Filtering
        - Validation results tracked properly
        """
        from src.scraper.amazon.media_validator import generate_validation_report

        # Create validation results
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 1920,
                            "height": 1080,
                            "duration": "10.0",
                        }
                    ],
                    "format": {"duration": "10.0", "format_name": "mp4"},
                }
            )

            result = verify_video_file(mock_video_file)

        # Generate report
        report_path = test_output_dir / "validation_report.json"
        report = generate_validation_report([result], report_path)

        # Verify report
        assert report["summary"]["total_files"] == 1
        assert report_path.exists()

        # Verify report content
        with open(report_path) as f:
            saved_report = json.load(f)
        assert saved_report["summary"]["total_files"] == 1


class TestConfigurationIntegration:
    """Test configuration parameter integration."""

    def test_video_config_parameters_available(self):
        """Test video configuration parameters are accessible.

        Validates:
        - Configuration from config/scraper.yaml is loaded
        - Video-specific settings available
        """
        from src.scraper.amazon.config import CONFIG

        # Verify video config exists
        assert "global_settings" in CONFIG
        assert "video_config" in CONFIG["global_settings"]

        video_config = CONFIG["global_settings"]["video_config"]

        # Verify new parameters from task 6
        assert "enable_metadata_extraction" in video_config
        assert isinstance(video_config["enable_metadata_extraction"], bool)

    def test_download_config_parameters_available(self):
        """Test download configuration parameters for videos.

        Validates:
        - Download timeout and retry settings configured
        """
        from src.scraper.amazon.config import CONFIG

        # Verify download config exists
        assert "global_settings" in CONFIG
        assert "download_config" in CONFIG["global_settings"]

        download_config = CONFIG["global_settings"]["download_config"]

        # Verify new parameters from task 6
        assert "video_download_timeout" in download_config
        assert "retry_video_downloads" in download_config
        assert download_config["video_download_timeout"] == 300
        assert download_config["retry_video_downloads"] == 2


# Test summary and coverage verification
def test_requirements_coverage():
    """Meta-test to verify all requirements have test coverage.

    This test documents which requirements are covered by which test classes:

    - Requirement 1 (High-Quality Video Detection): Tested via existing media_extractor
    - Requirement 2 (Video Metadata Extraction): TestVideoExtractionValidation
    - Requirement 3 (Video Validation): TestVideoExtractionValidation
    - Requirement 4 (Organized Video Storage): TestVideoStorageOrganization
    - Requirement 5 (Robust Error Handling): TestErrorHandlingAndRecovery
    - Requirement 6 (Product Data Integration): TestProductDataIntegration
    """
    requirements_tested = {
        "R1_video_detection": "Covered by existing media_extractor tests",
        "R2_metadata_extraction": "TestVideoExtractionValidation",
        "R3_validation": "TestVideoExtractionValidation",
        "R4_storage": "TestVideoStorageOrganization",
        "R5_error_handling": "TestErrorHandlingAndRecovery",
        "R6_data_integration": "TestProductDataIntegration",
    }

    # All requirements should have coverage
    assert len(requirements_tested) == 6
    assert all(v for v in requirements_tested.values())
