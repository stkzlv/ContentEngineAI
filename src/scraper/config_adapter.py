# src/scraper/config_adapter.py
"""Backward compatibility adapter for scraper configuration system.

This module provides seamless backward compatibility during migration
from monolithic scrapers.yaml to modular configuration structure.
All existing CONFIG global usage patterns remain unchanged.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from src.scraper.config_models import ScraperConfig

logger = logging.getLogger(__name__)


class ScraperConfigAdapter:
    """Adapter that merges modular scraper configs for backward compatibility."""

    def __init__(self, config_root: str = "config"):
        """Initialize the adapter with config root directory."""
        self.config_root = Path(config_root)
        self._merged_config: dict[str, Any] | None = None

    def _get_minimal_config_fallback(self) -> dict[str, Any]:
        """Provide minimal configuration fallback when files can't be loaded."""
        return {
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
                        "prime_only": False,
                        "sort_order": "relevanceblender",
                    },
                }
            },
        }

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any]:
        """Load a YAML file and return its contents with enhanced fallback."""
        try:
            if file_path.exists():
                with open(file_path, encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict):
                        logger.debug(f"Successfully loaded config from {file_path}")
                        return content
                    else:
                        logger.warning(
                            f"Config file {file_path} contains invalid format, "
                            "using fallback"
                        )
                        return self._get_minimal_config_fallback()
            else:
                logger.warning(
                    f"Scraper config file not found: {file_path}, using fallback"
                )
                return self._get_minimal_config_fallback()
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}, using fallback")
            return self._get_minimal_config_fallback()
        except Exception as e:
            logger.error(
                f"Unexpected error loading scraper config file {file_path}: {e}, "
                "using fallback"
            )
            return self._get_minimal_config_fallback()

    def _merge_scraper_configs(self) -> dict[str, Any]:
        """Load consolidated scraper config file with enhanced fallback."""
        if self._merged_config is not None:
            return self._merged_config.copy()

        # Load the consolidated scraper configuration
        merged = self._load_yaml_file(self.config_root / "scraper.yaml")

        if not merged:
            logger.warning(
                "Consolidated scraper config not found, "
                "using minimal fallback structure"
            )
            merged = self._get_minimal_config_fallback()

        # Ensure required structure exists for backward compatibility
        try:
            self._ensure_required_structure(merged)
        except Exception as e:
            logger.error(f"Failed to ensure config structure: {e}, using fallback")
            merged = self._get_minimal_config_fallback()

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
    """Load scraper configuration from modular structure.

    Args:
    ----
        config_path: Deprecated (kept for API compatibility, ignored)
        cli_overrides: CLI arguments to apply with precedence

    Returns:
    -------
        Configuration dictionary with all precedence rules applied

    """
    # Load from modular structure using unified config manager
    from src.config_manager import get_unified_config_manager

    manager = get_unified_config_manager()
    merged_config = manager.get_scraper_config(cli_overrides)

    logger.info("Loading scraper config from modular structure with precedence rules")
    return merged_config


# Alias for backward compatibility
load_scraper_config = load_scraper_config_modular


def load_scraper_config_pydantic(
    config_path: str = "config/scraper.yaml", cli_overrides: dict[str, Any] = None
) -> "ScraperConfig":
    """Load scraper configuration as Pydantic models.

    Args:
    ----
        config_path: Path to configuration file (optional, for compatibility)
        cli_overrides: CLI arguments to apply with precedence

    Returns:
    -------
        Pydantic ScraperConfig instance with validated configuration

    """
    from src.scraper.config_models import ScraperConfig

    # Load dict config first
    config_dict = load_scraper_config_modular(config_path, cli_overrides)

    # Transform to Pydantic-compatible structure
    pydantic_dict = {
        "global_settings": config_dict.get("global_settings", {}),
        "amazon": config_dict.get("scrapers", {}).get("amazon", {}),
    }

    # Create and validate Pydantic model
    logger.info("Loading scraper config as Pydantic models")
    return ScraperConfig(**pydantic_dict)
