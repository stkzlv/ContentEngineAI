# src/config_manager.py
"""Unified configuration manager for both video and scraper systems.

This module provides a central entry point for configuration management
during the migration from monolithic to modular configuration structure.
It handles backward compatibility and provides a unified precedence system.
"""

import os
from pathlib import Path
from typing import Any

from src.scraper.config_adapter import ScraperConfigAdapter
from src.video.config_adapter import ModularConfigAdapter


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
            # API keys
            "OPENROUTER_API_KEY": ["llm_settings.api_key_env_var"],
            "OPENAI_API_KEY": ["tts_config.openai.api_key_env_var"],
            "ELEVENLABS_API_KEY": ["tts_config.elevenlabs.api_key_env_var"],
            "FREESOUND_API_KEY": ["stock_media_settings.freesound.api_key_env_var"],
            "GOOGLE_APPLICATION_CREDENTIALS": [
                "google_cloud_stt_settings.credentials_env_var"
            ],
            # Output directory override
            "CONTENT_ENGINE_OUTPUT": ["global_output_directory"],
            "OUTPUTS_DIR": [
                "global_output_directory",
                "scraper_output_config.base_directory",
            ],
            # Performance overrides
            "CONTENT_ENGINE_TIMEOUT": ["pipeline_timeout_sec"],
            "FFMPEG_THREADS": ["ffmpeg_settings.encoding.threads"],
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
        # Common CLI override patterns
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

    def get_video_config(self, cli_overrides: dict[str, Any] = None) -> dict[str, Any]:
        """Get video configuration with precedence rules applied."""
        base_config = self.video_adapter.get_merged_config_dict()
        return self.apply_precedence_rules(base_config, cli_overrides)

    def get_scraper_config(
        self, cli_overrides: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Get scraper configuration with precedence rules applied."""
        base_config = self.scraper_adapter.get_merged_config_dict()
        return self.apply_precedence_rules(base_config, cli_overrides)

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
