# src/scraper/config_adapter.py
"""Backward compatibility adapter for scraper configuration system.

This module provides seamless backward compatibility during migration
from monolithic scrapers.yaml to modular configuration structure.
All existing CONFIG global usage patterns remain unchanged.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ScraperConfigAdapter:
    """Adapter that merges modular scraper configs for backward compatibility."""

    def __init__(self, config_root: str = "config"):
        """Initialize the adapter with config root directory."""
        self.config_root = Path(config_root)
        self._merged_config: dict[str, Any] | None = None

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any]:
        """Load a YAML file and return its contents."""
        try:
            if file_path.exists():
                with open(file_path, encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    return content if isinstance(content, dict) else {}
            else:
                logger.warning(f"Scraper config file not found: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading scraper config file {file_path}: {e}")
            return {}

    def _merge_scraper_configs(self) -> dict[str, Any]:
        """Load consolidated scraper config file."""
        if self._merged_config is not None:
            return self._merged_config.copy()

        # Load the consolidated scraper configuration
        merged = self._load_yaml_file(self.config_root / "scraper.yaml")

        if not merged:
            logger.warning(
                "Consolidated scraper config not found, creating minimal structure"
            )
            merged = {}

        # Ensure required structure exists for backward compatibility
        self._ensure_required_structure(merged)

        self._merged_config = merged
        return merged

    def _ensure_required_structure(self, config: dict[str, Any]) -> None:
        """Transform consolidated config to legacy structure for compatibility."""
        # If we have consolidated structure, transform it to legacy format
        if "global_settings" in config and "amazon" in config:
            # Already in consolidated format, transform to legacy
            legacy_config = {
                "global_settings": config.get("global_settings", {}),
                "scrapers": {"amazon": config.get("amazon", {})},
            }
            config.clear()
            config.update(legacy_config)

        # Ensure global_settings exists
        if "global_settings" not in config:
            config["global_settings"] = {}

        global_settings = config["global_settings"]

        # Set defaults for critical settings
        if "debug_mode" not in global_settings:
            global_settings["debug_mode"] = True

        if "output_config" not in global_settings:
            global_settings["output_config"] = {
                "base_directory": "outputs",
                "file_patterns": {
                    "product_file": "{keyword}_products.json",
                    "image_file": "{asin}_image_{index}.{ext}",
                    "video_file": "{asin}_video_{index}.{ext}",
                },
            }

        # Ensure scrapers section exists with Amazon config
        if "scrapers" not in config:
            config["scrapers"] = {}

        if "amazon" not in config["scrapers"]:
            config["scrapers"]["amazon"] = {
                "enabled": True,
                "platform": "amazon",
                "base_url": "https://www.amazon.com",
            }

    def get_merged_config_dict(self) -> dict[str, Any]:
        """Get the merged scraper configuration as a dictionary."""
        return self._merge_scraper_configs()


def load_scraper_config_modular(
    config_path: str = None, cli_overrides: dict[str, Any] = None
) -> dict[str, Any]:
    """Load scraper configuration using modular structure with backward compatibility.

    This function maintains the same interface as the original CONFIG loading
    but sources data from modular config files when available.

    Args:
    ----
        config_path: Path to config file (for backward compatibility)
        cli_overrides: CLI arguments to apply with precedence

    Returns:
    -------
        Configuration dictionary with all precedence rules applied

    """
    # Try to load from modular structure first
    try:
        # Use unified config manager for precedence handling
        from src.config_manager import get_unified_config_manager

        manager = get_unified_config_manager()
        merged_config = manager.get_scraper_config(cli_overrides)

        if merged_config:
            logger.info(
                "Loading scraper config from modular structure with precedence rules"
            )
            return merged_config

    except Exception as e:
        logger.warning(
            f"Failed to load modular scraper config, falling back to monolithic: {e}"
        )

    # Fallback to original monolithic loading
    if config_path is None:
        config_path = "config/scrapers.yaml"

    logger.info(f"Loading scraper config from monolithic file: {config_path}")

    try:
        # Load the monolithic file directly
        config_file = Path(config_path)
        if not config_file.exists():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_file = project_root / config_path

        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                result = yaml.safe_load(f)
                return result if isinstance(result, dict) else {}
        else:
            logger.error(
                f"Neither modular nor monolithic scraper config found: {config_path}"
            )
            return {}

    except Exception as e:
        logger.error(f"Error loading monolithic scraper config: {e}")
        return {}


def install_scraper_config_adapter():
    """Install the scraper config adapter to replace original CONFIG loading."""
    try:
        # Update the global CONFIG in amazon.config module
        import src.scraper.amazon.config as amazon_config_module

        # Load using our adapter with precedence rules
        new_config = load_scraper_config_modular()

        # Update the global CONFIG dictionary
        if hasattr(amazon_config_module, "CONFIG"):
            amazon_config_module.CONFIG.clear()
            amazon_config_module.CONFIG.update(new_config)
            logger.info("Scraper config adapter installed successfully")
        else:
            logger.warning("Amazon config module CONFIG not found")

    except Exception as e:
        logger.error(f"Failed to install scraper config adapter: {e}")


# Note: Auto-installation removed to prevent import-time side effects
# Adapters should be explicitly installed when needed
