"""Tests for outputs directory cleanup functionality.

Tests the cleanup logic that removes unexpected files and directories
while preserving expected structure and important files.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.video.video_config import (
    CleanupSettings,
)


class TestCleanupSettings:
    """Test cleanup configuration and validation."""

    @pytest.mark.unit
    def test_cleanup_settings_defaults(self):
        """Test that cleanup settings have sensible defaults."""
        settings = CleanupSettings()

        assert settings.enabled is True
        assert settings.dry_run is False
        assert settings.max_age_days == 7
        assert "*.md" in settings.preserve_patterns
        assert "*.tmp" in settings.force_cleanup_patterns
        assert settings.cleanup_empty_dirs is True
        assert settings.create_report is True

    @pytest.mark.unit
    def test_cleanup_settings_customization(self):
        """Test customizing cleanup settings."""
        settings = CleanupSettings(
            enabled=False,
            dry_run=True,
            max_age_days=14,
            preserve_patterns=["*.custom"],
            force_cleanup_patterns=["*.remove"],
            cleanup_empty_dirs=False,
            create_report=False,
            report_file="custom_report.json",
        )

        assert settings.enabled is False
        assert settings.dry_run is True
        assert settings.max_age_days == 14
        assert settings.preserve_patterns == ["*.custom"]
        assert settings.force_cleanup_patterns == ["*.remove"]
        assert settings.cleanup_empty_dirs is False
        assert settings.create_report is False
        assert settings.report_file == "custom_report.json"


class TestCleanupLogic:
    """Test cleanup logic with temporary directory structures."""

    @pytest.fixture
    def temp_outputs_dir(self):
        """Create a temporary outputs directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def test_config(self, temp_outputs_dir):
        """Create a test config with temporary directory."""
        from pathlib import Path

        from src.video.video_config import CleanupSettings, load_video_config

        # Load the real config file and override directories for testing
        config_path = Path(__file__).parent.parent / "config" / "video_producer.yaml"
        config = load_video_config(config_path)

        # Override output directory and cleanup settings for testing
        config.global_output_directory = str(temp_outputs_dir)
        config.cleanup_settings = CleanupSettings(
            enabled=True,
            dry_run=False,
            max_age_days=1,  # Short age for testing
            preserve_patterns=["*.md", "*.txt", "cache/**"],
            force_cleanup_patterns=["*.tmp", "*.temp", ".DS_Store"],
            cleanup_empty_dirs=True,
            create_report=True,
            report_file="cleanup_report.json",
        )

        # Manually set derived paths for testing
        config.global_output_root_path = temp_outputs_dir
        config.video_production_base_runs_path = temp_outputs_dir / "videos"
        config.general_video_producer_log_dir_path = (
            temp_outputs_dir / "logs" / "producer"
        )
        config.scraper_data_base_path = temp_outputs_dir / "data"

        return config

    @pytest.fixture
    def sample_file_structure(self, temp_outputs_dir):
        """Create a sample file structure for testing."""
        files_created = []

        # Expected structure
        (temp_outputs_dir / "videos/B123/profile1").mkdir(parents=True)
        (temp_outputs_dir / "videos/B123/profile1/B123_profile1.mp4").write_text(
            "video"
        )
        files_created.append("videos/B123/profile1/B123_profile1.mp4")

        # Product-centric structure (current)
        (temp_outputs_dir / "B0BTYCRJSS/output").mkdir(parents=True)
        (temp_outputs_dir / "B0BTYCRJSS/output/data.json").write_text("{}")
        files_created.append("B0BTYCRJSS/output/data.json")

        (temp_outputs_dir / "logs/producer").mkdir(parents=True)
        (temp_outputs_dir / "logs/producer/video_producer.log").write_text("logs")
        files_created.append("logs/producer/video_producer.log")

        # Files that should be preserved
        (temp_outputs_dir / "README.md").write_text("# README")
        files_created.append("README.md")

        (temp_outputs_dir / "notes.txt").write_text("notes")
        files_created.append("notes.txt")

        (temp_outputs_dir / "cache/models").mkdir(parents=True)
        (temp_outputs_dir / "cache/models/model.bin").write_text("model")
        files_created.append("cache/models/model.bin")

        # Files that should be force cleaned up
        (temp_outputs_dir / "temp_file.tmp").write_text("temp")
        files_created.append("temp_file.tmp")

        (temp_outputs_dir / "backup.temp").write_text("backup")
        files_created.append("backup.temp")

        (temp_outputs_dir / ".DS_Store").write_text("")
        files_created.append(".DS_Store")

        # Unexpected files/directories
        (temp_outputs_dir / "unexpected_dir").mkdir()
        (temp_outputs_dir / "unexpected_dir/file.txt").write_text("unexpected")
        files_created.append("unexpected_dir/file.txt")

        (temp_outputs_dir / "random_file.xyz").write_text("random")
        files_created.append("random_file.xyz")

        # Empty directories
        (temp_outputs_dir / "empty_dir").mkdir()
        (temp_outputs_dir / "videos/B123/profile1/temp/empty").mkdir(parents=True)

        return files_created

    @pytest.mark.integration
    def test_cleanup_dry_run(self, test_config, sample_file_structure):
        """Test cleanup in dry-run mode doesn't actually remove files."""
        # Run cleanup in dry-run mode
        result = test_config.cleanup_outputs_directory(dry_run=True)

        assert result["dry_run"] is True
        assert len(result["actions"]) > 0
        assert result["statistics"]["files_removed"] == 0
        assert result["statistics"]["directories_removed"] == 0

        # Verify no files were actually removed
        assert (test_config.global_output_root_path / "temp_file.tmp").exists()
        assert (test_config.global_output_root_path / ".DS_Store").exists()
        assert (test_config.global_output_root_path / "unexpected_dir").exists()

        # Check action types
        action_types = [action["action"] for action in result["actions"]]
        assert "would_remove_file" in action_types
        assert "would_remove_directory" in action_types

    @pytest.mark.integration
    def test_cleanup_actual_removal(self, test_config, sample_file_structure):
        """Test actual cleanup removes the correct files."""
        # Run actual cleanup
        result = test_config.cleanup_outputs_directory(dry_run=False)

        assert result["dry_run"] is False
        assert len(result["actions"]) > 0
        assert result["statistics"]["files_removed"] > 0

        # Verify force cleanup files were removed
        assert not (test_config.global_output_root_path / "temp_file.tmp").exists()
        assert not (test_config.global_output_root_path / ".DS_Store").exists()
        assert not (test_config.global_output_root_path / "backup.temp").exists()

        # Verify unexpected files/dirs were removed
        assert not (test_config.global_output_root_path / "unexpected_dir").exists()
        assert not (test_config.global_output_root_path / "random_file.xyz").exists()

        # Verify preserved files remain
        assert (test_config.global_output_root_path / "README.md").exists()
        assert (test_config.global_output_root_path / "notes.txt").exists()
        assert (test_config.global_output_root_path / "cache/models/model.bin").exists()

        # Verify expected structure remains
        video_path = (
            test_config.global_output_root_path
            / "videos/B123/profile1/B123_profile1.mp4"
        )
        data_path = (
            test_config.global_output_root_path
            / "data/amazon/run1/products/product.json"
        )
        assert video_path.exists()
        assert data_path.exists()
        assert (
            test_config.global_output_root_path / "logs/producer/video_producer.log"
        ).exists()

    @pytest.mark.integration
    def test_preserve_patterns(self, test_config, temp_outputs_dir):
        """Test that preserve patterns work correctly."""
        # Create files matching preserve patterns
        (temp_outputs_dir / "important.md").write_text("markdown")
        (temp_outputs_dir / "config.txt").write_text("config")
        (temp_outputs_dir / "cache/data").mkdir(parents=True)
        (temp_outputs_dir / "cache/data/cached.bin").write_text("cached")

        test_config.cleanup_outputs_directory(dry_run=False)

        # All preserve pattern files should remain
        assert (temp_outputs_dir / "important.md").exists()
        assert (temp_outputs_dir / "config.txt").exists()
        assert (temp_outputs_dir / "cache/data/cached.bin").exists()

    @pytest.mark.integration
    def test_force_cleanup_patterns(self, test_config, temp_outputs_dir):
        """Test that force cleanup patterns override preserve patterns."""
        # Create files that match both preserve and force cleanup patterns
        # txt normally preserved, ~ forces cleanup
        (temp_outputs_dir / "document.txt~").write_text("backup text file")
        # tmp forces cleanup
        (temp_outputs_dir / "important.tmp").write_text("temp file")

        test_config.cleanup_outputs_directory(dry_run=False)

        # Force cleanup should override preserve patterns
        assert not (temp_outputs_dir / "document.txt~").exists()
        assert not (temp_outputs_dir / "important.tmp").exists()

    @pytest.mark.integration
    def test_age_based_cleanup(self, test_config, temp_outputs_dir):
        """Test age-based cleanup respects file modification times."""
        # Create old file (older than max_age_days)
        old_file = temp_outputs_dir / "old_file.xyz"
        old_file.write_text("old content")

        # Set file modification time to be older than threshold
        old_time = (
            time.time() - (test_config.cleanup_settings.max_age_days + 1) * 24 * 3600
        )
        os.utime(old_file, (old_time, old_time))

        # Create new file (newer than max_age_days)
        new_file = temp_outputs_dir / "new_file.xyz"
        new_file.write_text("new content")

        test_config.cleanup_outputs_directory(dry_run=False)

        # Old file should be removed, new file should remain
        assert not old_file.exists()
        assert new_file.exists()

    @pytest.mark.integration
    def test_empty_directory_cleanup(self, test_config, temp_outputs_dir):
        """Test cleanup of empty directories."""
        # Create nested empty directories
        (temp_outputs_dir / "empty1").mkdir()
        (temp_outputs_dir / "nested/empty2").mkdir(parents=True)
        (temp_outputs_dir / "with_file/subdir").mkdir(parents=True)
        (temp_outputs_dir / "with_file/subdir/file.txt").write_text("content")

        test_config.cleanup_outputs_directory(dry_run=False)

        # Empty directories should be removed
        assert not (temp_outputs_dir / "empty1").exists()
        assert not (temp_outputs_dir / "nested/empty2").exists()
        # Parent should also be empty now
        assert not (temp_outputs_dir / "nested").exists()

        # Directory with files should remain
        assert (temp_outputs_dir / "with_file/subdir").exists()
        assert (temp_outputs_dir / "with_file/subdir/file.txt").exists()

    @pytest.mark.integration
    def test_cleanup_report_generation(self, test_config, sample_file_structure):
        """Test that cleanup reports are generated correctly."""
        result = test_config.cleanup_outputs_directory(dry_run=False)

        # Check report structure
        assert "timestamp" in result
        assert "dry_run" in result
        assert "config" in result
        assert "actions" in result
        assert "statistics" in result

        # Check statistics
        stats = result["statistics"]
        assert "files_removed" in stats
        assert "directories_removed" in stats
        assert "bytes_freed" in stats
        assert "errors" in stats

        # Check actions have required fields
        for action in result["actions"]:
            assert "action" in action
            assert "path" in action
            assert "timestamp" in action

        # Check that report file was created (if enabled)
        if test_config.cleanup_settings.create_report:
            report_path = (
                test_config.global_output_root_path
                / test_config.cleanup_settings.report_file
            )
            assert report_path.exists()

            # Verify report content
            with report_path.open() as f:
                report_data = json.load(f)
            assert report_data["statistics"] == result["statistics"]

    @pytest.mark.integration
    def test_cleanup_disabled(self, test_config, sample_file_structure):
        """Test that cleanup can be disabled via configuration."""
        test_config.cleanup_settings.enabled = False

        result = test_config.cleanup_outputs_directory()

        assert result["status"] == "disabled"
        assert len(result["actions"]) == 0

        # Verify no files were removed
        assert (test_config.global_output_root_path / "temp_file.tmp").exists()
        assert (test_config.global_output_root_path / ".DS_Store").exists()

    @pytest.mark.integration
    def test_override_dry_run(self, test_config, sample_file_structure):
        """Test that dry_run parameter can override config setting."""
        # Set config to actual cleanup
        test_config.cleanup_settings.dry_run = False

        # Override with dry_run=True
        result = test_config.cleanup_outputs_directory(dry_run=True)

        assert result["dry_run"] is True
        assert result["statistics"]["files_removed"] == 0

        # Files should still exist
        assert (test_config.global_output_root_path / "temp_file.tmp").exists()

    @pytest.mark.unit
    def test_path_expectation_logic(self, test_config, temp_outputs_dir):
        """Test the logic for determining expected vs unexpected paths."""
        expected_paths = test_config.get_expected_paths()

        # Test expected paths
        expected_video_path = temp_outputs_dir / "videos/product1/profile1/video.mp4"
        expected_data_path = temp_outputs_dir / "data/amazon/run1/products/data.json"
        expected_log_path = temp_outputs_dir / "logs/producer/app.log"

        assert test_config._is_path_expected(expected_video_path, expected_paths)
        assert test_config._is_path_expected(expected_data_path, expected_paths)
        assert test_config._is_path_expected(expected_log_path, expected_paths)

        # Test unexpected paths
        unexpected_path = temp_outputs_dir / "random/unexpected/file.txt"
        assert not test_config._is_path_expected(unexpected_path, expected_paths)

    @pytest.mark.unit
    def test_pattern_matching(self, test_config, temp_outputs_dir):
        """Test preserve and force cleanup pattern matching."""
        # Test preserve patterns
        assert test_config._should_preserve(temp_outputs_dir / "README.md")
        assert test_config._should_preserve(temp_outputs_dir / "config.txt")
        assert test_config._should_preserve(temp_outputs_dir / "cache/models/data.bin")
        assert not test_config._should_preserve(temp_outputs_dir / "random.xyz")

        # Test force cleanup patterns
        assert test_config._should_force_cleanup(temp_outputs_dir / "temp.tmp")
        assert test_config._should_force_cleanup(temp_outputs_dir / "backup.temp")
        assert test_config._should_force_cleanup(temp_outputs_dir / ".DS_Store")
        assert not test_config._should_force_cleanup(temp_outputs_dir / "normal.txt")

    @pytest.mark.integration
    def test_error_handling(self, test_config, temp_outputs_dir):
        """Test error handling during cleanup operations."""
        # Create a file we can't delete (simulate permission error)
        problem_file = temp_outputs_dir / "protected.tmp"
        problem_file.write_text("content")

        # Mock the unlink method to raise an exception
        with patch.object(Path, "unlink", side_effect=PermissionError("Access denied")):
            result = test_config.cleanup_outputs_directory(dry_run=False)

        # Should record error but continue
        assert result["statistics"]["errors"] > 0
        error_actions = [a for a in result["actions"] if a["action"] == "error"]
        assert len(error_actions) > 0
        assert "Access denied" in error_actions[0]["error"]

    @pytest.mark.integration
    def test_nonexistent_outputs_directory(self, test_config):
        """Test cleanup when outputs directory doesn't exist."""
        # Point to non-existent directory
        test_config.global_output_root_path = Path("/nonexistent/path")

        result = test_config.cleanup_outputs_directory()

        assert len(result["actions"]) == 0
        assert result["statistics"]["files_removed"] == 0


class TestCleanupUtilityScript:
    """Test the cleanup utility script functionality."""

    @pytest.mark.integration
    def test_cleanup_script_dry_run(self, temp_dir):
        """Test the cleanup utility script in dry-run mode."""
        # This would require testing the actual script
        # For now, we test the core logic through VideoConfig
        pass

    @pytest.mark.integration
    def test_cleanup_script_force_mode(self, temp_dir):
        """Test the cleanup utility script with force flag."""
        # This would test overriding disabled cleanup
        pass


if __name__ == "__main__":
    pytest.main([__file__])
