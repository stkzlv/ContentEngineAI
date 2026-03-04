# src/config_manager.py
"""Unified configuration manager for both video and scraper systems.

This module provides a central entry point for configuration management
during the migration from monolithic to modular configuration structure.
It handles backward compatibility and provides a unified precedence system.
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.scraper.config_adapter import ScraperConfigAdapter
from src.utils.secrets import mask_secret
from src.video.config_adapter import ModularConfigAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecretDefinition:
    """Definition of a secret/credential required by the system."""

    name: str
    description: str
    required: bool
    setup_url: str
    alternative_names: tuple[str, ...] = ()

    def get_value(self) -> str | None:
        """Get the secret value from environment, checking alternative names."""
        value = os.environ.get(self.name)
        if value:
            return value
        for alt_name in self.alternative_names:
            value = os.environ.get(alt_name)
            if value:
                return value
        return None


# Define all secrets used by the system
SECRETS_REGISTRY: tuple[SecretDefinition, ...] = (
    # Required secrets
    SecretDefinition(
        name="GEMINI_API_KEY",
        description="Google Gemini API for LLM script generation",
        required=False,
        setup_url="https://aistudio.google.com/apikey",
    ),
    SecretDefinition(
        name="OPENROUTER_API_KEY",
        description="OpenRouter API for LLM script generation (fallback)",
        required=False,
        setup_url="https://openrouter.ai/",
    ),
    SecretDefinition(
        name="PEXELS_API_KEY",
        description="Pexels API for stock images/videos",
        required=True,
        setup_url="https://www.pexels.com/api/",
    ),
    SecretDefinition(
        name="FREESOUND_API_KEY",
        description="Freesound API for background music",
        required=True,
        setup_url="https://freesound.org/apiv2/apply/",
    ),
    # Optional secrets
    SecretDefinition(
        name="GOOGLE_APPLICATION_CREDENTIALS",
        description="Google Cloud credentials for TTS services",
        required=False,
        setup_url="https://console.cloud.google.com/",
    ),
    SecretDefinition(
        name="LATE_API_KEY",
        description="Late.dev API for social media publishing",
        required=False,
        setup_url="https://getlate.dev/",
        alternative_names=("PUBLISHER_API_KEY",),
    ),
    SecretDefinition(
        name="PICSEE_API_KEY",
        description="Picsee API for URL shortening",
        required=False,
        setup_url="https://picsee.io/",
    ),
)


@dataclass
class SecretsValidationResult:
    """Result of secrets validation."""

    valid: bool
    missing_required: list[SecretDefinition]
    missing_optional: list[SecretDefinition]
    present: list[SecretDefinition]

    def log_summary(self) -> None:
        """Log a summary of the validation result."""
        if self.present:
            logger.debug(
                f"Secrets configured: {', '.join(s.name for s in self.present)}"
            )
        if self.missing_optional:
            logger.info(
                f"Optional secrets not configured: "
                f"{', '.join(s.name for s in self.missing_optional)}"
            )
        if self.missing_required:
            for secret in self.missing_required:
                logger.error(
                    f"Missing required secret: {secret.name} - {secret.description}"
                )


class UnifiedConfigManager:
    """Unified configuration manager supporting both video and scraper systems."""

    def __init__(self, config_root: str = "config"):
        """Initialize the unified config manager."""
        self.config_root = Path(config_root)
        self.video_adapter = ModularConfigAdapter(config_root)
        self.scraper_adapter = ScraperConfigAdapter(config_root)

    def apply_precedence_rules(
        self, config: dict[str, Any], cli_overrides: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Apply unified precedence rules: CLI > ENV > YAML.

        Args:
        ----
            config: Base configuration from YAML files
            cli_overrides: Overrides from CLI arguments

        Returns:
        -------
            Configuration with precedence rules applied

        """
        # Start with YAML config (lowest precedence)
        final_config = config.copy()

        # Apply environment variable overrides (medium precedence)
        self._apply_env_overrides(final_config)

        # Apply CLI overrides (highest precedence)
        if cli_overrides:
            self._apply_cli_overrides(final_config, cli_overrides)

        return final_config

    def _apply_env_overrides(self, config: dict[str, Any]) -> None:
        """Apply environment variable overrides to config."""
        # Common environment variable patterns
        env_mappings = {
            # Debug mode override
            "DEBUG_MODE": ["debug_mode", "global_settings.debug_mode"],
            "CONTENT_ENGINE_DEBUG": ["debug_mode", "global_settings.debug_mode"],
            # Output directory override
            "CONTENT_ENGINE_OUTPUT": ["global_output_directory"],
            "OUTPUTS_DIR": [
                "global_output_directory",
                "scraper_output_config.base_directory",
            ],
            # Performance overrides
            "CONTENT_ENGINE_TIMEOUT": ["pipeline_timeout_sec"],
            "FFMPEG_THREADS": ["ffmpeg_settings.encoding.threads"],
            # Subtitle positioning
            "SUBTITLE_ANCHOR": ["subtitle_settings.anchor"],
            "SUBTITLE_MARGIN": ["subtitle_settings.margin"],
            "SUBTITLE_CONTENT_AWARE": ["subtitle_settings.content_aware"],
            # Subtitle styling
            "SUBTITLE_STYLE_PRESET": ["subtitle_settings.style_preset"],
            "SUBTITLE_FONT_SIZE_SCALE": ["subtitle_settings.font_size_scale"],
            "SUBTITLE_ALIGNMENT": ["subtitle_settings.horizontal_alignment"],
            "SUBTITLE_MAX_WIDTH_FRACTION": [
                "subtitle_settings.max_subtitle_width_fraction"
            ],
            # Subtitle randomization
            "SUBTITLE_RANDOMIZE_FONTS": ["subtitle_settings.randomize_fonts"],
            "SUBTITLE_RANDOMIZE_COLORS": ["subtitle_settings.randomize_colors"],
            "SUBTITLE_RANDOMIZE_EFFECTS": ["subtitle_settings.randomize_effects"],
            # Subtitle text formatting
            "SUBTITLE_MAX_LINE_LENGTH": ["subtitle_settings.max_line_length"],
            "SUBTITLE_MAX_WORDS_PER_LINE": ["subtitle_settings.max_words_per_line"],
            "SUBTITLE_MAX_DURATION": ["subtitle_settings.max_duration"],
            "SUBTITLE_MIN_DURATION": ["subtitle_settings.min_duration"],
            # Advanced subtitle styling
            "SUBTITLE_FONT": ["subtitle_settings.font_name"],
            "SUBTITLE_FONT_COLOR": ["subtitle_settings.font_color"],
            "SUBTITLE_OUTLINE_COLOR": ["subtitle_settings.outline_color"],
            "SUBTITLE_BACKGROUND_COLOR": ["subtitle_settings.background_color"],
        }

        for env_var, config_paths in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Apply the environment value to all specified config paths
                for path in config_paths:
                    self._set_nested_value(config, path, env_value)

    def _apply_cli_overrides(
        self, config: dict[str, Any], cli_overrides: dict[str, Any]
    ) -> None:
        """Apply CLI argument overrides to config."""
        # Common CLI override patterns (legacy short names)
        cli_mappings = {
            "debug": ["debug_mode", "global_settings.debug_mode"],
            "output_dir": [
                "global_output_directory",
                "scraper_output_config.base_directory",
            ],
            "timeout": ["pipeline_timeout_sec"],
            "headless": ["global_settings.browser_config.headless"],
            "clean": ["cleanup.remove_temp_on_success"],
        }

        for cli_key, config_paths in cli_mappings.items():
            if cli_key in cli_overrides:
                cli_value = cli_overrides[cli_key]
                for path in config_paths:
                    self._set_nested_value(config, path, cli_value)

        # Apply all other CLI overrides using dot notation
        # (e.g., "video_settings.image_top_position_percent")
        for cli_key, cli_value in cli_overrides.items():
            if cli_key not in cli_mappings:  # Skip already processed mappings
                self._set_nested_value(config, cli_key, cli_value)

    def _set_nested_value(self, config: dict[str, Any], path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        keys = path.split(".")
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        final_key = keys[-1]
        # Convert string values to appropriate types
        if isinstance(value, str):
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)

        current[final_key] = value

    def _get_video_fallback_config(
        self, cli_overrides: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Provide minimal fallback video configuration."""
        fallback_config = {
            "debug_mode": True,
            "global_output_directory": "outputs",
            "pipeline_timeout_sec": 900,
            "audio_settings": {
                "bitrate": "192k",
                "sample_rate": 48000,
                "channels": 2,
            },
            "video_settings": {
                "resolution": {"width": 1920, "height": 1080},
                "framerate": 30,
                "bitrate": "5M",
            },
            "subtitle_settings": {
                "enabled": True,
                "font_family": "Arial",
                "font_size": 48,
            },
            "ffmpeg_settings": {
                "encoding": {
                    "preset": "medium",
                    "crf": 23,
                    "threads": 0,
                }
            },
            "cleanup": {
                "remove_temp_on_success": True,
                "remove_temp_on_failure": False,
            },
        }
        return self.apply_precedence_rules(fallback_config, cli_overrides)

    def _get_scraper_fallback_config(
        self, cli_overrides: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Provide minimal fallback scraper configuration."""
        fallback_config = {
            "global_settings": {
                "debug_mode": True,
                "output_config": {
                    "base_directory": "outputs",
                    "file_patterns": {
                        "product_file": "{keyword}_products.json",
                        "image_file": "{asin}_image_{index}.{ext}",
                        "video_file": "{asin}_video_{index}.{ext}",
                    },
                },
                "retry_config": {
                    "default_max_retries": 3,
                    "base_delay": 1.0,
                    "max_delay": 60.0,
                    "backoff_factor": 2.0,
                    "use_jitter": True,
                    "jitter_factor": 0.5,
                },
                "browser_config": {
                    "max_products_per_search": 5,
                    "search_result_timeout": 10,
                },
            },
            "scrapers": {
                "amazon": {
                    "enabled": True,
                    "base_url": "https://www.amazon.com",
                    "max_products": 3,
                    "keywords": [],
                    "default_search_parameters": {
                        "min_price": None,
                        "max_price": None,
                        "min_rating": None,
                        "prime_only": False,
                        "free_shipping": False,
                        "brands": [],
                        "sort_order": "relevanceblender",
                        "category": None,
                    },
                }
            },
        }
        return self.apply_precedence_rules(fallback_config, cli_overrides)

    def get_video_config(self, cli_overrides: dict[str, Any] = None) -> dict[str, Any]:
        """Get video configuration with precedence rules applied."""
        try:
            base_config = self.video_adapter.get_merged_config_dict()
            return self.apply_precedence_rules(base_config, cli_overrides)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load video config, using fallback: {e}")
            return self._get_video_fallback_config(cli_overrides)

    def get_scraper_config(
        self, cli_overrides: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Get scraper configuration with precedence rules applied."""
        try:
            base_config = self.scraper_adapter.get_merged_config_dict()
            return self.apply_precedence_rules(base_config, cli_overrides)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load scraper config, using fallback: {e}")
            return self._get_scraper_fallback_config(cli_overrides)

    def validate_config_structure(self) -> dict[str, bool]:
        """Validate that all required configuration files exist."""
        validation_results = {}

        # Check consolidated config files
        consolidated_files = [
            "core.yaml",
            "video_production.yaml",
            "ai_services.yaml",
            "subtitles.yaml",
            "scraper.yaml",
            "performance.yaml",
        ]

        for file_path in consolidated_files:
            full_path = self.config_root / file_path
            validation_results[file_path] = full_path.exists()

        return validation_results

    def validate_required_secrets(
        self, exit_on_missing: bool = True
    ) -> SecretsValidationResult:
        """Validate that required secrets are configured.

        Args:
        ----
            exit_on_missing: If True, exit with error when required secrets are missing.
                Defaults to True for production use.

        Returns:
        -------
            SecretsValidationResult with validation details.

        """
        missing_required: list[SecretDefinition] = []
        missing_optional: list[SecretDefinition] = []
        present: list[SecretDefinition] = []

        for secret in SECRETS_REGISTRY:
            value = secret.get_value()
            if value:
                present.append(secret)
                logger.debug(f"Secret {secret.name}: {mask_secret(value)}")
            elif secret.required:
                missing_required.append(secret)
            else:
                missing_optional.append(secret)

        result = SecretsValidationResult(
            valid=len(missing_required) == 0,
            missing_required=missing_required,
            missing_optional=missing_optional,
            present=present,
        )

        # Log summary
        result.log_summary()

        # Exit if required secrets are missing
        if not result.valid and exit_on_missing:
            print("\n❌ Missing required environment variables:\n", file=sys.stderr)
            for secret in missing_required:
                alt_info = ""
                if secret.alternative_names:
                    alt_info = f" (or {', '.join(secret.alternative_names)})"
                print(
                    f"  • {secret.name}{alt_info}\n"
                    f"    Purpose: {secret.description}\n"
                    f"    Setup: {secret.setup_url}\n",
                    file=sys.stderr,
                )
            print(
                "See .env.example for configuration template.\n",
                file=sys.stderr,
            )
            sys.exit(1)

        return result


# Global instance for easy access
unified_config_manager = UnifiedConfigManager()


def get_unified_config_manager() -> UnifiedConfigManager:
    """Get the global unified configuration manager instance."""
    return unified_config_manager


def validate_modular_config() -> bool:
    """Validate that modular configuration is properly set up."""
    manager = get_unified_config_manager()
    validation_results = manager.validate_config_structure()

    all_valid = all(validation_results.values())
    if not all_valid:
        missing_files = [
            path for path, exists in validation_results.items() if not exists
        ]
        print(f"❌ Missing modular config files: {missing_files}")
        return False

    print("✅ All modular configuration files found")
    return True


if __name__ == "__main__":
    # Test the configuration manager
    if validate_modular_config():
        print("Modular configuration validation passed!")
    else:
        print("Modular configuration validation failed!")


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation from video config.

    Args:
    ----
        key: Dot-separated configuration key (e.g. 'video_settings.resolution.width')
        default: Default value if key is not found

    Returns:
    -------
        Configuration value or default

    """
    manager = get_unified_config_manager()
    # Prioritize video config as this is primarily used by video processing components
    current = manager.get_video_config()

    for k in key.split("."):
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default

    return current
