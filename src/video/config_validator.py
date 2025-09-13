"""Configuration validation for ContentEngineAI video module.

This module provides runtime validation of VideoConfig to catch configuration
errors early and prevent runtime failures. Integrates with existing Pydantic
models in video_config.py.
"""

import logging
import os
import shutil
from pathlib import Path

from src.video.subtitle_positioning import (
    PositionAnchor,
    StylePreset,
    convert_legacy_config,
)
from src.video.video_config import VideoConfig

logger = logging.getLogger(__name__)


class VideoConfigValidator:
    """Validates VideoConfig for runtime compatibility and dependency availability."""

    def validate_config(self, config: VideoConfig) -> list[str]:
        """Validate configuration and return list of validation errors.

        Args:
        ----
            config: VideoConfig instance to validate

        Returns:
        -------
            List of validation error messages (empty if valid)

        """
        errors = []

        # FFmpeg validation (critical for video assembly)
        errors.extend(self._validate_ffmpeg(config))

        # Directory validation
        errors.extend(self._validate_directories(config))

        # Font validation (if subtitle generation enabled)
        errors.extend(self._validate_fonts(config))

        # Resolution validation for 9:16 vertical format
        errors.extend(self._validate_resolution(config))

        # Audio settings validation
        errors.extend(self._validate_audio_settings(config))

        # Unified subtitle configuration validation
        errors.extend(self._validate_unified_subtitle_config(config))

        # Environment variables validation
        errors.extend(self._validate_environment_variables(config))

        # File path validation
        errors.extend(self._validate_file_paths(config))

        return errors

    def _validate_ffmpeg(self, config: VideoConfig) -> list[str]:
        """Validate FFmpeg availability and functionality."""
        errors = []

        ffmpeg_path = (
            getattr(config.ffmpeg_settings, "executable_path", "ffmpeg") or "ffmpeg"
        )

        if not shutil.which(ffmpeg_path):
            errors.append(f"FFmpeg not found in PATH: {ffmpeg_path}")
            return errors  # No point checking further if FFmpeg not found

        # Test FFmpeg basic functionality
        try:
            import subprocess

            result = subprocess.run(
                [ffmpeg_path, "-version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                errors.append(f"FFmpeg execution failed: {result.stderr}")
        except Exception as e:
            errors.append(f"FFmpeg validation failed: {e}")

        return errors

    def _validate_directories(self, config: VideoConfig) -> list[str]:
        """Validate directory paths and permissions."""
        errors = []

        # Global output directory validation
        output_dir = Path(config.global_output_directory)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = output_dir / ".config_validation_test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            errors.append(f"Output directory not writable: {output_dir} ({e})")

        # Whisper model directory validation
        if hasattr(config, "whisper_settings") and config.whisper_settings:
            model_dir = getattr(config.whisper_settings, "model_download_root", None)
            if model_dir:
                model_path = Path(model_dir)
                if not model_path.exists():
                    try:
                        model_path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        errors.append(
                            f"Cannot create model directory: {model_path} ({e})"
                        )

        return errors

    def _validate_fonts(self, config: VideoConfig) -> list[str]:
        """Validate font directories and availability for subtitle rendering."""
        errors = []

        if not getattr(config.subtitle_settings, "enabled", True):
            return errors  # Skip font validation if subtitles disabled

        font_dir = getattr(config.subtitle_settings, "font_directory", None)
        if font_dir:
            font_path = Path(font_dir)
            if not font_path.is_dir():
                errors.append(f"Font directory not found: {font_dir}")
            else:
                # Check for common font files
                font_extensions = [".ttf", ".otf", ".woff", ".woff2"]
                font_files = [
                    f
                    for f in font_path.iterdir()
                    if f.suffix.lower() in font_extensions
                ]
                if not font_files:
                    errors.append(
                        f"No font files found in directory: {font_dir} "
                        f"(looking for {font_extensions})"
                    )

        return errors

    def _validate_resolution(self, config: VideoConfig) -> list[str]:
        """Validate video resolution settings."""
        errors = []

        resolution = getattr(config.video_settings, "resolution", (1080, 1920))
        if not isinstance(resolution, list | tuple) or len(resolution) != 2:
            errors.append(
                f"Invalid resolution format: {resolution} (expected [width, height])"
            )
            return errors

        width, height = resolution
        if not isinstance(width, int) or not isinstance(height, int):
            errors.append(f"Resolution values must be integers: {width}x{height}")
        elif width <= 0 or height <= 0:
            errors.append(f"Invalid resolution dimensions: {width}x{height}")
        elif width > height:
            # Warn about landscape format (expected 9:16 vertical)
            logger.warning(
                f"Landscape resolution detected: {width}x{height}. "
                "ContentEngineAI optimized for vertical 9:16 format."
            )

        return errors

    def _validate_audio_settings(self, config: VideoConfig) -> list[str]:
        """Validate audio configuration settings."""
        errors = []

        audio_settings = config.audio_settings

        # Volume level validation
        voiceover_db = getattr(audio_settings, "voiceover_volume_db", 0)
        music_db = getattr(audio_settings, "music_volume_db", -12)

        if voiceover_db > 10:
            errors.append(
                f"Voiceover volume too high: {voiceover_db}dB (max recommended: 10dB)"
            )
        if voiceover_db < -30:
            errors.append(
                f"Voiceover volume too low: {voiceover_db}dB (min recommended: -30dB)"
            )

        if music_db > 0:
            errors.append(
                f"Music volume too high: {music_db}dB (should be negative for background)"
            )
        if music_db < -50:
            errors.append(
                f"Music volume too low: {music_db}dB (min recommended: -50dB)"
            )

        # Fade duration validation
        fade_in = getattr(audio_settings, "music_fade_in_duration", 2.0)
        fade_out = getattr(audio_settings, "music_fade_out_duration", 3.0)

        if fade_in < 0 or fade_in > 10:
            errors.append(f"Invalid fade-in duration: {fade_in}s (range: 0-10s)")
        if fade_out < 0 or fade_out > 10:
            errors.append(f"Invalid fade-out duration: {fade_out}s (range: 0-10s)")

        return errors

    def _validate_environment_variables(self, config: VideoConfig) -> list[str]:
        """Validate required environment variables."""
        errors = []

        # Google Cloud STT validation
        if (
            hasattr(config, "google_stt_settings")
            and getattr(config.google_stt_settings, "enabled", False)
            and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        ):
            errors.append(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable required "
                "for Google Cloud STT (or disable google_stt_settings.enabled)"
            )

        # Freesound API validation
        freesound_key_var = getattr(
            config.audio_settings, "freesound_api_key_env_var", "FREESOUND_API_KEY"
        )
        if freesound_key_var and not os.getenv(freesound_key_var):
            logger.warning(
                f"Environment variable {freesound_key_var} not set. "
                "Freesound background music will be unavailable."
            )

        return errors

    def _validate_file_paths(self, config: VideoConfig) -> list[str]:
        """Validate file paths in configuration."""
        errors = []

        # Background music paths validation
        music_paths = getattr(config.audio_settings, "background_music_paths", [])
        for music_path in music_paths:
            if not Path(music_path).exists():
                errors.append(f"Background music file not found: {music_path}")

        # Profile-specific validation could be added here
        # (checking template files, effect files, etc.)

        return errors

    def validate_runtime_dependencies(self, config: VideoConfig) -> list[str]:
        """Validate runtime dependencies and system requirements.

        Args:
        ----
            config: VideoConfig instance to check settings against

        Returns:
        -------
            List of dependency validation errors

        """
        errors = []

        # Python package validation
        required_packages = [
            ("pysrt", "pysrt"),
            ("PIL", "Pillow"),
            ("yaml", "PyYAML"),
            ("pydantic", "pydantic"),
        ]

        # Add whisper only if enabled in config
        if (
            hasattr(config, "whisper_settings")
            and config.whisper_settings
            and config.whisper_settings.enabled
        ):
            required_packages.append(("whisper", "openai-whisper"))

        for package_name, pip_name in required_packages:
            try:
                __import__(package_name)
            except ImportError:
                errors.append(f"Required package not installed: {pip_name}")
            except Exception as e:
                # Handle compatibility issues for specific packages like whisper
                if package_name == "whisper":
                    logger.warning(f"Whisper package has compatibility issues: {e}")
                    # Don't treat whisper compatibility issues as fatal since we have fallback STT
                else:
                    errors.append(f"Required package not installed: {pip_name}")

        # System-level validation
        try:
            import psutil

            # Memory check (Whisper models can be large)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 2.0:
                logger.warning(
                    f"Low available memory: {available_memory_gb:.1f}GB. "
                    "Whisper models may cause OOM errors."
                )
        except ImportError:
            logger.warning("psutil not available for system resource validation")

        return errors

    def _validate_unified_subtitle_config(self, config: VideoConfig) -> list[str]:
        """Validate unified subtitle configuration and suggest optimizations.

        Args:
        ----
            config: VideoConfig instance to validate

        Returns:
        -------
            List of validation error/warning messages

        """
        errors = []

        if not hasattr(config, "subtitle_settings") or not config.subtitle_settings:
            return errors

        subtitle_settings = config.subtitle_settings

        try:
            # Test conversion from legacy to unified config
            unified_config = convert_legacy_config(subtitle_settings.__dict__)

            # Validate anchor value
            if hasattr(unified_config, "anchor"):
                try:
                    PositionAnchor(unified_config.anchor)
                except ValueError:
                    valid_anchors = [e.value for e in PositionAnchor]
                    errors.append(
                        f"Invalid subtitle anchor '{unified_config.anchor}'. "
                        f"Valid options: {valid_anchors}"
                    )

            # Validate style preset
            if hasattr(unified_config, "style_preset"):
                try:
                    StylePreset(unified_config.style_preset)
                except ValueError:
                    valid_presets = [e.value for e in StylePreset]
                    errors.append(
                        f"Invalid subtitle style preset '{unified_config.style_preset}'. "
                        f"Valid options: {valid_presets}"
                    )

            # Check configuration style and provide guidance
            has_unified_params = any(
                param in subtitle_settings.__dict__
                for param in ["anchor", "style_preset", "content_aware"]
            )
            has_legacy_params = hasattr(subtitle_settings, "positioning_mode")

            if has_legacy_params and not has_unified_params:
                logger.info(
                    "Legacy subtitle positioning detected - will be converted to unified config. "
                    "Consider updating config to use new unified parameters directly."
                )
            elif has_unified_params:
                logger.info("Using unified subtitle configuration ✓")
            else:
                logger.debug("Using default subtitle configuration")

            logger.debug("Unified subtitle configuration validation completed")

        except Exception as e:
            errors.append(
                f"Failed to validate unified subtitle configuration: {e}. "
                f"This may indicate incompatible legacy settings."
            )

        return errors


def validate_config_and_exit_on_error(config: VideoConfig) -> None:
    """Validate configuration and exit if errors found.

    This is a convenience function for use in main() functions
    to fail fast on configuration errors.

    Args:
    ----
        config: VideoConfig instance to validate

    Raises:
    ------
        SystemExit: If validation errors are found

    """
    validator = VideoConfigValidator()

    # Validate configuration
    config_errors = validator.validate_config(config)
    dependency_errors = validator.validate_runtime_dependencies(config)

    all_errors = config_errors + dependency_errors

    if all_errors:
        logger.critical("Configuration validation failed:")
        for error in all_errors:
            logger.critical(f"  ❌ {error}")
        logger.critical("Fix configuration errors before proceeding")
        logger.info("See CONFIGURATION.md for configuration guide")
        raise SystemExit(1)

    logger.info("✅ Configuration validation passed")
