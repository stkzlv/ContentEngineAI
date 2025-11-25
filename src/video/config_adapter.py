# src/video/config_adapter.py
"""Backward compatibility adapter for video configuration system.

This module provides seamless backward compatibility during the migration
from monolithic video_producer.yaml to modular configuration structure.
All existing imports and usage patterns remain unchanged.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from src.video.config import VideoConfig, VideoProfile

logger = logging.getLogger(__name__)


class ModularConfigAdapter:
    """Adapter that merges modular configs to maintain backward compatibility."""

    def __init__(self, config_root: str = "config"):
        """Initialize the adapter with config root directory."""
        self.config_root = Path(config_root)
        self._merged_config: dict[str, Any] | None = None
        self._config_files_map = {
            "core": "core.yaml",
            "video_production": "video_production.yaml",
            "ai_services": "ai_services.yaml",
            "subtitles": "subtitles.yaml",
            "performance": "performance.yaml",
        }

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any]:
        """Load a YAML file and return its contents."""
        try:
            if file_path.exists():
                with open(file_path, encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    return content if isinstance(content, dict) else {}
            else:
                logger.warning(f"Config file not found: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            return {}

    def _merge_configs(self) -> dict[str, Any]:
        """Merge all modular config files into a single structure."""
        if self._merged_config is not None:
            return self._merged_config.copy()

        merged = {}

        # Load all consolidated configuration files
        config_files = [
            "core",
            "video_production",
            "ai_services",
            "subtitles",
            "performance",
        ]

        for config_name in config_files:
            config_data = self._load_yaml_file(
                self.config_root / self._config_files_map[config_name]
            )
            merged.update(config_data)

        # Legacy structure method removed - all structures now defined in YAML files
        # (output_structure, path_config, video_profiles, etc. in config/*.yaml)

        self._merged_config = merged
        return merged

    def get_merged_config_dict(self) -> dict[str, Any]:
        """Get the merged configuration as a dictionary."""
        return self._merge_configs()


def load_video_config_modular(
    config_path: str = None, cli_overrides: dict[str, Any] = None
) -> VideoConfig:
    """Load video configuration from modular structure.

    Args:
    ----
        config_path: Deprecated (kept for API compatibility, ignored)
        cli_overrides: CLI arguments to apply with precedence

    Returns:
    -------
        VideoConfig instance with all precedence rules applied

    """
    # Load from modular structure using unified config manager
    from src.config_manager import get_unified_config_manager

    manager = get_unified_config_manager()
    merged_config = manager.get_video_config(cli_overrides)

    logger.info("Loading video config from modular structure with precedence rules")

    # Create a temporary YAML string and parse it through existing VideoConfig
    # This ensures all Pydantic validation still works
    temp_yaml = yaml.dump(merged_config)
    temp_config = yaml.safe_load(temp_yaml)
    return VideoConfig(**temp_config)


# Alias for backward compatibility
load_video_config = load_video_config_modular
