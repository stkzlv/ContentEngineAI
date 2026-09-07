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
        self._settings: ScraperConfig | None = None

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any]:
        """The file's mapping, or an empty one when there is no usable file.

        An empty mapping validates to the models' defaults, so a missing or
        malformed file is logged and read as the defaults rather than as a
        fallback dict of this module's own (#125).
        """
        if not file_path.exists():
            logger.warning(
                "Scraper config file not found: %s; using defaults", file_path
            )
            return {}
        try:
            with open(file_path, encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error("YAML parsing error in %s: %s; using defaults", file_path, e)
            return {}
        if not isinstance(content, dict):
            logger.warning("Config file %s is not a mapping; using defaults", file_path)
            return {}
        logger.debug("Successfully loaded config from %s", file_path)
        return content

    def _merge_scraper_configs(self) -> dict[str, Any]:
        """Load the consolidated scraper config file, validated."""
        if self._merged_config is not None:
            return self._merged_config.copy()

        merged = self._load_yaml_file(self.config_root / "scraper.yaml")
        # Validate through the typed models and hand out the defaults-filled
        # dict in the file's own shape. A misspelled key raises here; every
        # consumer that indexes the dict finds its key present (#125).
        from src.scraper.config_models import ScraperConfig

        self._settings = ScraperConfig.from_legacy_dict(merged)
        self._merged_config = self._settings.to_runtime_dict()
        return self._merged_config.copy()

    def get_settings(self) -> "ScraperConfig":
        """The validated, typed configuration behind `get_merged_config_dict`."""
        if self._settings is None:
            self._merge_scraper_configs()
        assert self._settings is not None
        return self._settings

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
