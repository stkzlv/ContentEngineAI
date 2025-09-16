"""Unit tests for VideoConfigValidator.

This module tests the configuration validation functionality to ensure
runtime errors are caught early during startup.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.video.config_validator import (
    VideoConfigValidator,
    validate_config_and_exit_on_error,
)
from src.video.video_config import load_video_config


class TestVideoConfigValidator:
    """Test cases for VideoConfigValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = VideoConfigValidator()
        # Load a valid configuration as starting point
        config_path = Path(__file__).parent.parent / "config" / "video_producer.yaml"
        self.valid_config = load_video_config(config_path)

    def test_valid_config_passes_validation(self):
        """Test that valid configuration passes all validation checks."""
        errors = self.validator.validate_config(self.valid_config)

        # Should be no errors for valid config
        assert errors == []

    def test_ffmpeg_validation_missing_executable(self):
        """Test FFmpeg validation when executable is missing."""
        # Set FFmpeg path to nonexistent location
        self.valid_config.ffmpeg_settings.executable_path = "/nonexistent/ffmpeg"

        errors = self.validator.validate_config(self.valid_config)

        assert len(errors) >= 1
        assert any("FFmpeg not found" in error for error in errors)

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_ffmpeg_validation_execution_failure(self, mock_run, mock_which):
        """Test FFmpeg validation when executable exists but fails."""
        # Mock FFmpeg found but execution fails
        mock_which.return_value = "/usr/bin/ffmpeg"
        mock_run.return_value = MagicMock(returncode=1, stderr="FFmpeg error")

        errors = self.validator.validate_config(self.valid_config)

        assert any("FFmpeg execution failed" in error for error in errors)

    def test_directory_validation_nonexistent_output_dir(self):
        """Test directory validation for nonexistent output directory."""
        # Set output directory to path that cannot be created
        self.valid_config.global_output_directory = "/root/cannot_create"

        errors = self.validator.validate_config(self.valid_config)

        # Should find permission/creation error
        assert any("not writable" in error for error in errors)

    def test_font_validation_missing_directory(self):
        """Test font validation when directory is missing."""
        # Set font directory to nonexistent path
        self.valid_config.subtitle_settings.font_directory = "/nonexistent/fonts"

        errors = self.validator.validate_config(self.valid_config)

        assert any("Font directory not found" in error for error in errors)

    def test_font_validation_no_font_files(self):
        """Test font validation when directory exists but has no fonts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty directory
            font_dir = Path(temp_dir) / "fonts"
            font_dir.mkdir()

            self.valid_config.subtitle_settings.font_directory = str(font_dir)

            errors = self.validator.validate_config(self.valid_config)

            assert any("No font files found" in error for error in errors)

    def test_font_validation_with_valid_fonts(self):
        """Test font validation passes with valid font files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory with font files
            font_dir = Path(temp_dir) / "fonts"
            font_dir.mkdir()
            (font_dir / "arial.ttf").write_text("fake font content")

            self.valid_config.subtitle_settings.font_directory = str(font_dir)

            errors = self.validator.validate_config(self.valid_config)

            # Should not have font-related errors
            assert not any("font" in error.lower() for error in errors)

    def test_resolution_validation_invalid_format(self):
        """Test resolution validation with invalid format."""
        # Test invalid resolution format
        self.valid_config.video_settings.resolution = (0, 0)

        errors = self.validator.validate_config(self.valid_config)

        assert any("Invalid resolution dimensions" in error for error in errors)

    def test_resolution_validation_landscape_warning(self):
        """Test resolution validation logs warning for landscape format."""
        with patch("src.video.config_validator.logger") as mock_logger:
            # Set landscape resolution
            self.valid_config.video_settings.resolution = (1920, 1080)

            self.validator.validate_config(self.valid_config)

            # Should log warning about landscape format
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Landscape resolution detected" in warning_call

    def test_audio_settings_validation_extreme_volumes(self):
        """Test audio validation catches extreme volume levels."""
        # Set extreme volume levels
        self.valid_config.audio_settings.voiceover_volume_db = 20  # Too high
        self.valid_config.audio_settings.music_volume_db = 5  # Should be negative

        errors = self.validator.validate_config(self.valid_config)

        assert any("Voiceover volume too high" in error for error in errors)
        assert any("Music volume too high" in error for error in errors)

    def test_audio_settings_validation_invalid_fade_durations(self):
        """Test audio validation catches invalid fade durations."""
        # Set invalid fade durations
        self.valid_config.audio_settings.music_fade_in_duration = -1.0  # Negative
        self.valid_config.audio_settings.music_fade_out_duration = 15.0  # Too long

        errors = self.validator.validate_config(self.valid_config)

        assert any("Invalid fade-in duration" in error for error in errors)
        assert any("Invalid fade-out duration" in error for error in errors)

    def test_background_music_validation_missing_files(self):
        """Test validation of background music file paths."""
        # Set nonexistent music files
        from pathlib import Path

        self.valid_config.audio_settings.background_music_paths = [
            Path("/nonexistent/music1.mp3"),
            Path("/nonexistent/music2.mp3"),
        ]

        errors = self.validator.validate_config(self.valid_config)

        music_errors = [e for e in errors if "Background music file not found" in e]
        assert len(music_errors) == 2

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variable_validation_google_stt(self):
        """Test environment variable validation for Google Cloud STT."""
        # Enable Google STT without credentials
        if hasattr(self.valid_config, "google_stt_settings"):
            self.valid_config.google_stt_settings.enabled = True

        self.validator.validate_config(self.valid_config)

        # Should warn about missing Google credentials if enabled
        # Note: This test depends on the actual config structure

    def test_runtime_dependencies_validation(self):
        """Test validation of Python package dependencies."""
        errors = self.validator.validate_runtime_dependencies()

        # Should pass for current environment
        assert errors == []

    @patch("importlib.import_module")
    def test_runtime_dependencies_missing_packages(self, mock_import):
        """Test dependency validation when packages are missing."""
        # Mock missing package
        mock_import.side_effect = ImportError("No module named 'missing_package'")

        with patch("src.video.config_validator.__import__", side_effect=ImportError):
            errors = self.validator.validate_runtime_dependencies()

            # Should detect missing packages
            assert len(errors) > 0
            assert any("Required package not installed" in error for error in errors)

    def test_whisper_model_directory_creation(self):
        """Test validation creates missing model directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set model directory to path that needs creation
            model_dir = Path(temp_dir) / "models" / "whisper"
            self.valid_config.whisper_settings.model_download_root = str(model_dir)

            errors = self.validator.validate_config(self.valid_config)

            # Directory should be created, no errors
            assert model_dir.exists()
            model_errors = [e for e in errors if "model directory" in e.lower()]
            assert len(model_errors) == 0


class TestValidateConfigAndExitOnError:
    """Test cases for the convenience function."""

    def test_valid_config_no_exit(self):
        """Test that valid configuration does not cause exit."""
        config_path = Path(__file__).parent.parent / "config" / "video_producer.yaml"
        valid_config = load_video_config(config_path)

        # Should not raise SystemExit
        try:
            validate_config_and_exit_on_error(valid_config)
        except SystemExit:
            pytest.fail("Valid configuration should not cause exit")

    def test_invalid_config_causes_exit(self):
        """Test that invalid configuration causes SystemExit."""
        config_path = Path(__file__).parent.parent / "config" / "video_producer.yaml"
        invalid_config = load_video_config(config_path)

        # Make configuration invalid
        invalid_config.ffmpeg_settings.executable_path = "/nonexistent/ffmpeg"

        with pytest.raises(SystemExit) as exc_info:
            validate_config_and_exit_on_error(invalid_config)

        assert exc_info.value.code == 1


# Integration tests
class TestConfigValidationIntegration:
    """Integration tests for configuration validation."""

    def test_producer_config_validation_integration(self):
        """Test that validation integrates properly with producer configuration."""
        from src.video.video_config import load_video_config

        # Load actual configuration used by producer
        config_path = Path(__file__).parent.parent / "config" / "video_producer.yaml"
        config = load_video_config(config_path)

        validator = VideoConfigValidator()
        config_errors = validator.validate_config(config)
        dep_errors = validator.validate_runtime_dependencies()

        # Current production config should be valid
        all_errors = config_errors + dep_errors
        if all_errors:
            pytest.fail(f"Production configuration has validation errors: {all_errors}")

    def test_validation_covers_critical_components(self):
        """Test that validation covers all critical system components."""
        validator = VideoConfigValidator()

        # Check that validator has methods for all critical validations
        assert hasattr(validator, "_validate_ffmpeg")
        assert hasattr(validator, "_validate_directories")
        assert hasattr(validator, "_validate_fonts")
        assert hasattr(validator, "_validate_resolution")
        assert hasattr(validator, "_validate_audio_settings")
        assert hasattr(validator, "_validate_environment_variables")
        assert hasattr(validator, "_validate_file_paths")

        # Check that runtime dependency validation exists
        assert hasattr(validator, "validate_runtime_dependencies")
