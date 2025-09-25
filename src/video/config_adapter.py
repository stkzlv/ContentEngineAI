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

from src.video.video_config import VideoConfig, VideoProfile

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

        # Add legacy structure elements that existing code expects
        self._add_legacy_structure(merged)

        self._merged_config = merged
        return merged

    def _add_legacy_structure(self, config: dict[str, Any]) -> None:
        """Add legacy configuration structure elements."""
        # Ensure output_structure exists (expected by existing code)
        if "output_structure" not in config:
            config["output_structure"] = {
                "product_directory_pattern": "{product_id}",
                "product_files": {
                    "scraped_data": "data.json",
                    "script": "script.txt",
                    "description": "description.txt",
                    "voiceover": "voiceover.wav",
                    "subtitles": "subtitles.srt",
                    "final_video": "video_{product_id}_{profile}.mp4",
                    "metadata": "metadata.json",
                    "ffmpeg_log": "ffmpeg_command.log",
                    "performance": "performance.json",
                },
                "product_subdirs": {
                    "images": "images",
                    "videos": "videos",
                    "music": "music",
                    "temp": "temp",
                },
                "global_dirs": config.get("global_dirs", {}),
            }

        # Ensure path_config exists
        if "path_config" not in config:
            config["path_config"] = {
                "use_product_oriented_structure": True,
                "cleanup": config.get("cleanup", {}),
                "script": "script.txt",
                "attribution": "attributions.txt",
                "metadata": "{name}.json",
                "timestamped_log": "{component}_{timestamp}.log",
                "main_log": "{component}.log",
                "gathered_visuals": "gathered_visuals.json",
                "temp_dir": "temp",
                "music_dir": "music",
            }

        # Add other expected top-level sections
        if "media_settings" not in config:
            config["media_settings"] = {}

        if "api_settings" not in config:
            config["api_settings"] = {}

        if "text_processing" not in config:
            config["text_processing"] = {}

        if "audio_processing" not in config:
            config["audio_processing"] = {}

        # Add video profiles if not present
        if "video_profiles" not in config:
            config["video_profiles"] = {
                "slideshow_images1": {
                    "description": (
                        "Dynamically uses scraped product images to match voiceover "
                        "duration."
                    ),
                    "use_scraped_images": True,
                    "use_scraped_videos": False,
                    "use_stock_images": False,
                    "use_stock_videos": False,
                    "stock_image_count": 0,
                    "stock_video_count": 0,
                    "use_dynamic_image_count": True,
                    "image_width_percent": 0.85,
                    "image_top_position_percent": 0.07,
                    "preserve_aspect_ratio": True,
                    "subtitle_randomize_fonts": True,
                    "subtitle_randomize_colors": True,
                    "subtitle_randomize_effects": True,
                    "subtitle_settings": {
                        "anchor": "below_content",
                        "margin": 0.015,
                        "content_aware": True,
                        "style_preset": "random",
                        "font_size_scale": 1.1,
                        "max_line_length": 22,
                        "horizontal_alignment": "center",
                        "subtitle_format": "ass",
                    },
                }
            }

    def get_merged_config_dict(self) -> dict[str, Any]:
        """Get the merged configuration as a dictionary."""
        return self._merge_configs()


def load_video_config_modular(
    config_path: str = None, cli_overrides: dict[str, Any] = None
) -> VideoConfig:
    """Load video configuration using modular structure with backward compatibility.

    This function provides the same interface as the original load_video_config
    but sources data from modular config files when available, falling back
    to the monolithic file if modular configs are not found.

    Args:
    ----
        config_path: Path to config file (for backward compatibility)
        cli_overrides: CLI arguments to apply with precedence

    Returns:
    -------
        VideoConfig instance with all precedence rules applied

    """
    # Try to load from modular structure first
    try:
        # Use unified config manager for precedence handling
        from src.config_manager import get_unified_config_manager

        manager = get_unified_config_manager()
        merged_config = manager.get_video_config(cli_overrides)

        if merged_config:
            logger.info(
                "Loading video config from modular structure with precedence rules"
            )
            # Create a temporary YAML string and parse it through existing VideoConfig
            # This ensures all Pydantic validation still works
            temp_yaml = yaml.dump(merged_config)
            temp_config = yaml.safe_load(temp_yaml)
            return VideoConfig(**temp_config)

    except Exception as e:
        logger.warning(
            f"Failed to load modular config, falling back to monolithic: {e}"
        )

    # Fallback to original monolithic loading
    if config_path is None:
        config_path = "config/video_producer.yaml"

    logger.info(f"Loading video config from monolithic file: {config_path}")

    # Import original loading function
    from src.video.video_config import load_video_config as original_load_video_config

    return original_load_video_config(Path(config_path))


# Replace the original function with our backward-compatible version
def install_modular_config_adapter():
    """Install the modular config adapter to replace original loading function."""
    import src.video.video_config as video_config_module

    # Store original function for fallback
    if not hasattr(video_config_module, "_original_load_video_config"):
        video_config_module._original_load_video_config = (  # type: ignore
            video_config_module.load_video_config
        )

    # Create a wrapper that matches the original signature
    def load_video_config_wrapper(config_path: Path) -> VideoConfig:
        return load_video_config_modular(str(config_path))

    # Replace with our backward-compatible version
    video_config_module.load_video_config = load_video_config_wrapper

    logger.info("Modular config adapter installed successfully")
