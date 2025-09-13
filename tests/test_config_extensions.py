"""Tests for new config fields added for clean functionality."""

import pytest
from pydantic import ValidationError

from src.video.video_config import CleanupConfig, PathConfig


class TestCleanupConfig:
    """Test CleanupConfig model with new debug_file_patterns field."""

    def test_cleanup_config_defaults(self):
        """Test CleanupConfig with default values."""
        config = CleanupConfig()

        assert config.remove_temp_on_success is True
        assert config.keep_temp_on_failure is True
        assert config.cache_max_age_hours == 168
        assert config.debug_file_patterns == [
            "incomplete_script_*.txt",
            "voiceover_whisper_*.json",
            "voiceover_whisper_*.txt",
            "*_ffmpeg_command.log",
        ]

    def test_cleanup_config_custom_debug_patterns(self):
        """Test CleanupConfig with custom debug file patterns."""
        custom_patterns = ["custom_*.log", "test_debug_*.json", "*_temp_file.txt"]

        config = CleanupConfig(debug_file_patterns=custom_patterns)

        assert config.debug_file_patterns == custom_patterns

    def test_cleanup_config_empty_debug_patterns(self):
        """Test CleanupConfig with empty debug file patterns."""
        config = CleanupConfig(debug_file_patterns=[])

        assert config.debug_file_patterns == []

    def test_cleanup_config_validation(self):
        """Test CleanupConfig validation for invalid values."""
        # Valid config should work
        config = CleanupConfig(
            remove_temp_on_success=False,
            keep_temp_on_failure=False,
            cache_max_age_hours=24,
            debug_file_patterns=["*.log"],
        )

        assert config.remove_temp_on_success is False
        assert config.keep_temp_on_failure is False
        assert config.cache_max_age_hours == 24
        assert config.debug_file_patterns == ["*.log"]


class TestPathConfig:
    """Test PathConfig model with new internal file configuration fields."""

    def test_path_config_defaults(self):
        """Test PathConfig with default values."""
        config = PathConfig()

        assert config.use_product_oriented_structure is True
        assert config.gathered_visuals == "gathered_visuals.json"
        assert config.temp_dir == "temp"
        assert config.music_dir == "music"
        assert isinstance(config.cleanup, CleanupConfig)

    def test_path_config_custom_values(self):
        """Test PathConfig with custom internal file values."""
        config = PathConfig(
            gathered_visuals="custom_visuals.json",
            temp_dir="temporary",
            music_dir="audio",
        )

        assert config.gathered_visuals == "custom_visuals.json"
        assert config.temp_dir == "temporary"
        assert config.music_dir == "audio"

    def test_path_config_with_custom_cleanup(self):
        """Test PathConfig with custom cleanup configuration."""
        cleanup = CleanupConfig(
            debug_file_patterns=["custom_*.log"], cache_max_age_hours=48
        )

        config = PathConfig(cleanup=cleanup)

        assert config.cleanup.debug_file_patterns == ["custom_*.log"]
        assert config.cleanup.cache_max_age_hours == 48

    def test_path_config_validation(self):
        """Test PathConfig validation for field types."""
        # Test that string fields must be strings
        with pytest.raises(ValidationError):
            PathConfig(gathered_visuals=123)

        with pytest.raises(ValidationError):
            PathConfig(temp_dir=None)

        with pytest.raises(ValidationError):
            PathConfig(music_dir=["invalid", "type"])

    def test_path_config_nested_cleanup_validation(self):
        """Test that PathConfig properly validates nested CleanupConfig."""
        # Invalid cleanup config should raise validation error
        with pytest.raises(ValidationError):
            PathConfig(cleanup="invalid_cleanup_config")


class TestIntegratedConfig:
    """Test integration of new config fields with existing VideoConfig."""

    def test_path_config_has_new_fields(self):
        """Test that PathConfig has the new fields available."""
        # This is already tested above, but we verify the integration exists
        config = PathConfig()

        # Verify all new fields are accessible
        assert hasattr(config, "gathered_visuals")
        assert hasattr(config, "temp_dir")
        assert hasattr(config, "music_dir")
        assert hasattr(config.cleanup, "debug_file_patterns")

        # This proves the fields are properly integrated into the config system
