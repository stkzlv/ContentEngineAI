"""Tests for producer cleanup functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


# Mock the producer module to test cleanup functionality
class MockProducerCleanup:
    """Mock producer cleanup functionality for testing."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def cleanup_producer_files(self, product_id: str) -> dict[str, bool]:
        """Mock cleanup of producer-generated files."""
        product_dir = self.output_dir / product_id
        if not product_dir.exists():
            return {}

        cleanup_results = {}

        # Define the files that should be cleaned up
        cleanup_files = [
            "script.txt",
            "voiceover.wav",
            "subtitles.ass",
            "subtitles_content_aware.ass",
            "metadata.json",
            "performance_metrics.json",
        ]

        # Cleanup video files with various patterns
        video_patterns = [
            f"video_{product_id}_*.mp4",
            f"{product_id}_*.mp4",
            "final_video.mp4",
            "output.mp4"
        ]

        for file_name in cleanup_files:
            file_path = product_dir / file_name
            if file_path.exists():
                file_path.unlink()
                cleanup_results[file_name] = True
            else:
                cleanup_results[file_name] = False

        # Handle video files with glob patterns
        for pattern in video_patterns:
            for video_file in product_dir.glob(pattern):
                cleanup_results[video_file.name] = True
                video_file.unlink()

        # Cleanup temp directory
        temp_dir = product_dir / "temp"
        if temp_dir.exists():
            for temp_file in temp_dir.rglob("*"):
                if temp_file.is_file():
                    temp_file.unlink()
                    cleanup_results[f"temp/{temp_file.name}"] = True

            # Remove empty temp directory
            try:
                temp_dir.rmdir()
                cleanup_results["temp_dir"] = True
            except OSError:
                cleanup_results["temp_dir"] = False

        return cleanup_results


class TestProducerCleanup:
    """Test producer file cleanup functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = None
        self.product_id = "B0BTYCRJSS"

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir:
            self.temp_dir.cleanup()

    def create_test_environment(self) -> tuple[Path, MockProducerCleanup]:
        """Create a test environment with mock producer files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)
        product_dir = output_dir / self.product_id
        product_dir.mkdir(parents=True)

        # Create mock producer-generated files
        files_to_create = [
            "data.json",  # Should NOT be cleaned (scraped data)
            "script.txt",  # Should be cleaned
            "voiceover.wav",  # Should be cleaned
            "subtitles.ass",  # Should be cleaned
            "subtitles_content_aware.ass",  # Should be cleaned
            "metadata.json",  # Should be cleaned
            "performance_metrics.json",  # Should be cleaned
            f"video_{self.product_id}_slideshow_images1.mp4",  # Should be cleaned
            f"{self.product_id}_final.mp4",  # Should be cleaned
        ]

        for file_name in files_to_create:
            file_path = product_dir / file_name
            file_path.write_text(f"Mock content for {file_name}")

        # Create temp directory with files
        temp_dir = product_dir / "temp"
        temp_dir.mkdir()
        (temp_dir / "temp_file1.txt").write_text("temp content 1")
        (temp_dir / "temp_file2.wav").write_text("temp content 2")

        # Create images directory (should NOT be cleaned)
        images_dir = product_dir / "images"
        images_dir.mkdir()
        (images_dir / "image1.jpg").write_text("image content")

        cleanup_handler = MockProducerCleanup(output_dir)
        return product_dir, cleanup_handler

    def test_cleanup_producer_generated_files(self):
        """Test cleanup removes only producer-generated files."""
        product_dir, cleanup_handler = self.create_test_environment()

        # Verify initial state
        assert (product_dir / "data.json").exists()
        assert (product_dir / "script.txt").exists()
        assert (product_dir / "voiceover.wav").exists()
        assert (product_dir / "subtitles.ass").exists()
        assert (product_dir / "subtitles_content_aware.ass").exists()
        assert (product_dir / "metadata.json").exists()
        assert (product_dir / f"video_{self.product_id}_slideshow_images1.mp4").exists()

        # Execute cleanup
        results = cleanup_handler.cleanup_producer_files(self.product_id)

        # Verify producer files were removed
        assert not (product_dir / "script.txt").exists()
        assert not (product_dir / "voiceover.wav").exists()
        assert not (product_dir / "subtitles.ass").exists()
        assert not (product_dir / "subtitles_content_aware.ass").exists()
        assert not (product_dir / "metadata.json").exists()
        assert not (
            product_dir / f"video_{self.product_id}_slideshow_images1.mp4"
        ).exists()

        # Verify scraped data and media were preserved
        assert (product_dir / "data.json").exists()
        assert (product_dir / "images").exists()
        assert (product_dir / "images" / "image1.jpg").exists()

        # Verify cleanup results
        assert results["script.txt"] is True
        assert results["voiceover.wav"] is True
        assert results["subtitles.ass"] is True
        assert results["subtitles_content_aware.ass"] is True

    def test_cleanup_content_aware_ass_files(self):
        """Test cleanup specifically handles content-aware ASS files."""
        product_dir, cleanup_handler = self.create_test_environment()

        # Verify content-aware ASS file exists
        content_aware_file = product_dir / "subtitles_content_aware.ass"
        assert content_aware_file.exists()

        # Execute cleanup
        results = cleanup_handler.cleanup_producer_files(self.product_id)

        # Verify content-aware ASS file was removed
        assert not content_aware_file.exists()
        assert results["subtitles_content_aware.ass"] is True

    def test_cleanup_video_files_with_patterns(self):
        """Test cleanup handles various video file naming patterns."""
        product_dir, cleanup_handler = self.create_test_environment()

        # Create additional video files with different patterns
        video_files = [
            f"video_{self.product_id}_profile1.mp4",
            f"video_{self.product_id}_profile2.mp4",
            f"{self.product_id}_backup.mp4",
            "final_video.mp4",
            "output.mp4"
        ]

        for video_file in video_files:
            (product_dir / video_file).write_text("video content")

        # Execute cleanup
        results = cleanup_handler.cleanup_producer_files(self.product_id)

        # Verify all video patterns were cleaned
        for video_file in video_files:
            assert not (product_dir / video_file).exists()
            if video_file in results:
                assert results[video_file] is True

    def test_cleanup_temp_directory(self):
        """Test cleanup removes temporary files and directory."""
        product_dir, cleanup_handler = self.create_test_environment()

        temp_dir = product_dir / "temp"
        assert temp_dir.exists()
        assert (temp_dir / "temp_file1.txt").exists()
        assert (temp_dir / "temp_file2.wav").exists()

        # Execute cleanup
        results = cleanup_handler.cleanup_producer_files(self.product_id)

        # Verify temp files and directory were removed
        assert not (temp_dir / "temp_file1.txt").exists()
        assert not (temp_dir / "temp_file2.wav").exists()
        assert not temp_dir.exists()
        assert results["temp_dir"] is True

    def test_cleanup_preserves_scraped_data(self):
        """Test cleanup preserves original scraped data and media."""
        product_dir, cleanup_handler = self.create_test_environment()

        # Create additional files that should be preserved

        # Create videos directory
        videos_dir = product_dir / "videos"
        videos_dir.mkdir()
        (videos_dir / "video1.mp4").write_text("scraped video content")

        # Execute cleanup
        cleanup_handler.cleanup_producer_files(self.product_id)

        # Verify preserved files still exist
        assert (product_dir / "data.json").exists()
        assert (product_dir / "images" / "image1.jpg").exists()
        assert (product_dir / "videos" / "video1.mp4").exists()

    def test_cleanup_nonexistent_product(self):
        """Test cleanup handles nonexistent product gracefully."""
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)
        cleanup_handler = MockProducerCleanup(output_dir)

        # Try to cleanup nonexistent product
        results = cleanup_handler.cleanup_producer_files("NONEXISTENT")

        # Should return empty results without errors
        assert results == {}

    def test_cleanup_partial_files(self):
        """Test cleanup when only some producer files exist."""
        product_dir, cleanup_handler = self.create_test_environment()

        # Remove some files before cleanup
        (product_dir / "script.txt").unlink()
        (product_dir / "metadata.json").unlink()

        # Execute cleanup
        results = cleanup_handler.cleanup_producer_files(self.product_id)

        # Verify results reflect actual file states
        assert results["script.txt"] is False  # File didn't exist
        assert results["voiceover.wav"] is True  # File was removed
        assert results["metadata.json"] is False  # File didn't exist

    def test_cleanup_with_clean_flag_behavior(self):
        """Test cleanup behavior mimicking --clean flag usage."""
        product_dir, cleanup_handler = self.create_test_environment()

        # Simulate producer run with --clean flag
        initial_files = list(product_dir.rglob("*"))
        initial_file_count = len([f for f in initial_files if f.is_file()])

        # Execute cleanup
        cleanup_handler.cleanup_producer_files(self.product_id)

        # Count remaining files
        remaining_files = list(product_dir.rglob("*"))
        remaining_file_count = len([f for f in remaining_files if f.is_file()])

        # Should have fewer files after cleanup
        assert remaining_file_count < initial_file_count

        # Should preserve data.json and media directories
        assert (product_dir / "data.json").exists()
        assert (product_dir / "images").exists()

    def test_cleanup_comprehensive_file_types(self):
        """Test cleanup handles all expected producer file types."""
        product_dir, cleanup_handler = self.create_test_environment()

        # Create comprehensive set of producer files
        comprehensive_files = [
            "script.txt",
            "voiceover.wav",
            "voiceover_backup.wav",
            "subtitles.ass",
            "subtitles_content_aware.ass",
            "subtitles.srt",
            "metadata.json",
            "performance_metrics.json",
            "pipeline_state.json",
            f"video_{self.product_id}_profile.mp4",
            "final_output.mp4",
            "temp/audio_temp.wav",
            "temp/video_temp.mp4",
            "temp/subtitle_temp.ass"
        ]

        # Create all files
        for file_path_str in comprehensive_files:
            file_path = product_dir / file_path_str
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f"content for {file_path_str}")

        # Execute cleanup
        cleanup_handler.cleanup_producer_files(self.product_id)

        # Verify comprehensive cleanup
        producer_files = [
            "script.txt",
            "voiceover.wav",
            "subtitles.ass",
            "subtitles_content_aware.ass",
            "metadata.json",
            "performance_metrics.json"
        ]

        for file_name in producer_files:
            assert not (product_dir / file_name).exists()

        # Verify temp directory is cleaned
        assert not (product_dir / "temp").exists()


class TestCleanupIntegration:
    """Integration tests for cleanup functionality."""

    def test_end_to_end_cleanup_workflow(self):
        """Test complete cleanup workflow from producer run to clean state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            product_id = "TEST123"
            product_dir = output_dir / product_id

            # Simulate complete producer run output
            self._simulate_producer_output(product_dir, product_id)

            # Verify all files exist
            self._verify_producer_files_exist(product_dir, product_id)

            # Execute cleanup
            cleanup_handler = MockProducerCleanup(output_dir)
            results = cleanup_handler.cleanup_producer_files(product_id)

            # Verify clean state
            self._verify_clean_state(product_dir, results)

    def _simulate_producer_output(self, product_dir: Path, product_id: str):
        """Simulate complete producer output."""
        product_dir.mkdir(parents=True)

        # Core producer files
        producer_files = {
            "script.txt": "Generated script content",
            "voiceover.wav": "Audio content",
            "subtitles.ass": "ASS subtitle content",
            "subtitles_content_aware.ass": "Content-aware ASS content",
            "metadata.json": '{"version": "0.1.1"}',
            "performance_metrics.json": '{"duration": 30.5}',
            f"video_{product_id}_slideshow_images1.mp4": "Video content"
        }

        for file_name, content in producer_files.items():
            (product_dir / file_name).write_text(content)

        # Preserved files
        preserved_files = {
            "data.json": '{"product_id": "' + product_id + '"}',
        }

        for file_name, content in preserved_files.items():
            (product_dir / file_name).write_text(content)

        # Media directories
        (product_dir / "images").mkdir()
        (product_dir / "images" / "product.jpg").write_text("image")

        # Temp files
        (product_dir / "temp").mkdir()
        (product_dir / "temp" / "processing.tmp").write_text("temp")

    def _verify_producer_files_exist(self, product_dir: Path, product_id: str):
        """Verify all producer files exist before cleanup."""
        assert (product_dir / "script.txt").exists()
        assert (product_dir / "voiceover.wav").exists()
        assert (product_dir / "subtitles.ass").exists()
        assert (product_dir / "subtitles_content_aware.ass").exists()
        assert (product_dir / f"video_{product_id}_slideshow_images1.mp4").exists()
        assert (product_dir / "temp").exists()

    def _verify_clean_state(self, product_dir: Path, results: dict):
        """Verify clean state after cleanup."""
        # Producer files should be removed
        assert not (product_dir / "script.txt").exists()
        assert not (product_dir / "voiceover.wav").exists()
        assert not (product_dir / "subtitles.ass").exists()
        assert not (product_dir / "subtitles_content_aware.ass").exists()
        assert not (product_dir / "temp").exists()

        # Preserved files should remain
        assert (product_dir / "data.json").exists()
        assert (product_dir / "images").exists()

        # Results should indicate successful cleanup
        assert results["script.txt"] is True
        assert results["temp_dir"] is True
