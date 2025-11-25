"""Tests for producer cleanup functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.video.config import VideoConfig
from src.video.producer.state import _clean_producer_files


class TestProducerCleanup:
    """Test producer file cleanup functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = None
        self.product_id = "B0BTYCRJSS"
        self.profile_name = "slideshow_images1"

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir:
            self.temp_dir.cleanup()

    def create_test_environment(self) -> tuple[Path, VideoConfig, dict]:
        """Create a test environment with mock producer files and config."""
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)
        product_dir = output_dir / self.product_id
        product_dir.mkdir(parents=True)

        # Create mock config
        config = MagicMock()

        # Mock output_structure
        files = MagicMock()
        files.script = "script.txt"
        files.description = "description.txt"
        files.voiceover = "voiceover.wav"
        files.subtitles = "subtitles.srt"
        files.final_video = "video_{product_id}_{profile}.mp4"
        files.attribution = "ATTRIBUTIONS.txt"

        temp_files = MagicMock()
        temp_files.metadata = "metadata.json"
        temp_files.ffmpeg_log = "ffmpeg_command.log"
        temp_files.performance = "performance.json"

        subdirs = MagicMock()
        subdirs.temp = "temp"

        config.output_structure.product_files = files
        config.output_structure.product_temp_files = temp_files
        config.output_structure.product_subdirs = subdirs

        # Mock path_config
        config.path_config.temp_dir = "temp"
        config.path_config.music_dir = "music"
        config.path_config.gathered_visuals = "gathered_visuals.json"
        config.path_config.cleanup.debug_file_patterns = ["debug_*.log"]

        # Create mock producer-generated files
        files_to_create = [
            "data.json",  # Should NOT be cleaned (scraped data)
            "script.txt",  # Should be cleaned
            "voiceover.wav",  # Should be cleaned
            "subtitles.ass",  # Should be cleaned
            "subtitles_content_aware.ass",  # Should be cleaned
            "subtitle_upper.ass", # Should be cleaned
            "metadata.json",  # Should be cleaned (in temp)
            "performance.json",  # Should be cleaned (in temp)
            f"video_{self.product_id}_{self.profile_name}.mp4",  # Should be cleaned
            f"video_{self.profile_name}.mp4", # Should be cleaned (old pattern)
            "ATTRIBUTIONS.txt", # Should be cleaned
            "gathered_visuals.json", # Should be cleaned
        ]

        # Create temp dir
        (product_dir / "temp").mkdir(exist_ok=True)

        # Create music dir
        (product_dir / "music").mkdir(exist_ok=True)
        (product_dir / "music" / "song.mp3").write_text("music")

        for file_name in files_to_create:
            if file_name in ["metadata.json", "performance.json"]:
                file_path = product_dir / "temp" / file_name
            else:
                file_path = product_dir / file_name

            file_path.write_text(f"Mock content for {file_name}")

        # Create debug file
        (product_dir / "debug_test.log").write_text("debug log")

        # Create images directory (should NOT be cleaned)
        images_dir = product_dir / "images"
        images_dir.mkdir(exist_ok=True)
        (images_dir / "image1.jpg").write_text("image content")

        run_paths = {"run_root": product_dir}

        return product_dir, config, run_paths

    def test_cleanup_producer_generated_files(self):
        """Test cleanup removes only producer-generated files."""
        product_dir, config, run_paths = self.create_test_environment()

        # Verify initial state
        assert (product_dir / "data.json").exists()
        assert (product_dir / "script.txt").exists()
        assert (product_dir / "temp" / "metadata.json").exists()
        assert (product_dir / "music" / "song.mp3").exists()

        # Execute cleanup
        _clean_producer_files(run_paths, config, self.product_id, self.profile_name)

        # Verify producer files were removed
        assert not (product_dir / "script.txt").exists()
        assert not (product_dir / "voiceover.wav").exists()
        assert not (product_dir / "subtitles.ass").exists()
        assert not (product_dir / "temp").exists() # Temp dir should be removed
        assert not (product_dir / "music").exists() # Music dir should be removed
        assert not (product_dir / "gathered_visuals.json").exists()
        assert not (product_dir / "ATTRIBUTIONS.txt").exists()
        assert not (product_dir / "debug_test.log").exists()

        # Verify scraped data and media were preserved
        assert (product_dir / "data.json").exists()
        assert (product_dir / "images").exists()
        assert (product_dir / "images" / "image1.jpg").exists()

    def test_cleanup_video_files_with_patterns(self):
        """Test cleanup handles various video file naming patterns."""
        product_dir, config, run_paths = self.create_test_environment()

        # Create additional video files with different patterns
        video_files = [
            f"video_{self.product_id}_other.mp4",
            f"video_{self.product_id}_test.mp4",
        ]

        for video_file in video_files:
            (product_dir / video_file).write_text("video content")

        # Execute cleanup
        _clean_producer_files(run_paths, config, self.product_id, self.profile_name)

        # Verify all video patterns were cleaned
        for video_file in video_files:
            assert not (product_dir / video_file).exists()

    def test_cleanup_preserves_scraped_data(self):
        """Test cleanup preserves original scraped data and media."""
        product_dir, config, run_paths = self.create_test_environment()

        # Create videos directory (scraped videos)
        videos_dir = product_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        (videos_dir / "video1.mp4").write_text("scraped video content")

        # Execute cleanup
        _clean_producer_files(run_paths, config, self.product_id, self.profile_name)

        # Verify preserved files still exist
        assert (product_dir / "data.json").exists()
        assert (product_dir / "images" / "image1.jpg").exists()
        assert (product_dir / "videos" / "video1.mp4").exists()
