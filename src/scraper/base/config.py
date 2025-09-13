"""Multi-platform configuration management for the scraper architecture.

This module provides a centralized configuration system that supports multiple
e-commerce platforms while maintaining a consistent API. It handles platform-specific
settings, global defaults, and configuration validation.
"""

from pathlib import Path
from typing import Any

import yaml

from .models import Platform


class PlatformConfigManager:
    """Centralized configuration manager for multi-platform scraping.

    This class manages configuration for all supported platforms, providing
    a unified interface for accessing platform-specific settings while
    maintaining global defaults and validation.
    """

    def __init__(self, config_path: str = "config/scrapers.yaml"):
        """Initialize the configuration manager.

        Args:
        ----
            config_path: Path to the YAML configuration file

        """
        self.config_path = config_path
        self._config = self._load_config()
        self._validate_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        # Find config file relative to project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        config_file = project_root / self.config_path

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _validate_config(self) -> None:
        """Validate the loaded configuration structure."""
        required_sections = ["global_settings", "scrapers"]

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Required configuration section missing: {section}")

        # Validate global settings
        global_settings = self._config["global_settings"]
        required_global = ["output_config", "retry_config"]

        for setting in required_global:
            if setting not in global_settings:
                raise ValueError(f"Required global setting missing: {setting}")

    def get_global_settings(self) -> dict[str, Any]:
        """Get global configuration settings."""
        return self._config.get("global_settings", {})

    def get_platform_config(self, platform: Platform) -> dict[str, Any]:
        """Get configuration for a specific platform.

        Args:
        ----
            platform: The platform to get configuration for

        Returns:
        -------
            Platform-specific configuration dictionary

        Raises:
        ------
            ValueError: If platform is not configured

        """
        scrapers_config = self._config.get("scrapers", {})
        platform_config = scrapers_config.get(platform.value)

        if platform_config is None:
            raise ValueError(f"No configuration found for platform: {platform.value}")

        return platform_config

    def is_platform_enabled(self, platform: Platform) -> bool:
        """Check if a platform is enabled in the configuration.

        Args:
        ----
            platform: The platform to check

        Returns:
        -------
            True if platform is enabled, False otherwise

        """
        try:
            platform_config = self.get_platform_config(platform)
            return platform_config.get("enabled", False)
        except ValueError:
            return False

    def get_enabled_platforms(self) -> list[Platform]:
        """Get list of enabled platforms from configuration.

        Returns
        -------
            List of enabled Platform enums

        """
        enabled = []
        scrapers_config = self._config.get("scrapers", {})

        for platform_name, config in scrapers_config.items():
            if config.get("enabled", False):
                try:
                    platform = Platform(platform_name)
                    enabled.append(platform)
                except ValueError:
                    # Skip unknown platforms
                    continue

        return enabled

    def get_output_path(self, path_type: str, platform: Platform, **kwargs) -> Path:
        """Get platform-specific output path with variable substitution.

        Args:
        ----
            path_type: Type of path ('products', 'media', 'debug', etc.)
            platform: The platform this path is for
            **kwargs: Variables for path substitution (keyword, asin, etc.)

        Returns:
        -------
            Resolved Path object

        """
        global_settings = self.get_global_settings()
        output_config = global_settings.get("output_config", {})

        # Get base directory
        base_dir = Path(output_config.get("base_directory", "outputs"))

        # Get subdirectory structure
        subdirs = output_config.get("subdirectories", {})

        # Build path based on type
        if path_type == "platform":
            # Platform root: use temporary directory since legacy paths are deprecated
            platform_dir = subdirs.get("platform_dir", f"temp/{platform.value}")
            return base_dir / platform_dir

        elif path_type == "products":
            # Products directory: fallback to temp since new structure uses Botasaurus
            platform_path = self.get_output_path("platform", platform, **kwargs)
            products_dir = subdirs.get("products", "products")
            return platform_path / products_dir

        elif path_type == "media":
            # Media directory: fallback to temp since new structure uses
            # product-centric layout
            platform_path = self.get_output_path("platform", platform, **kwargs)
            media_dir = subdirs.get("media", "media")
            return platform_path / media_dir

        elif path_type == "debug":
            # Debug directory: fallback to temp
            platform_path = self.get_output_path("platform", platform, **kwargs)
            debug_dir = subdirs.get("debug", "debug")
            return platform_path / debug_dir

        else:
            raise ValueError(f"Unknown path type: {path_type}")

    def get_filename_pattern(self, file_type: str, **kwargs) -> str:
        """Get filename pattern with variable substitution.

        Args:
        ----
            file_type: Type of file ('product', 'image', 'video')
            **kwargs: Variables for filename substitution

        Returns:
        -------
            Filename with variables substituted

        """
        global_settings = self.get_global_settings()
        output_config = global_settings.get("output_config", {})
        file_patterns = output_config.get("file_patterns", {})

        if file_type == "product":
            pattern = file_patterns.get("product_file", "{keyword}_products.json")
        elif file_type == "image":
            pattern = file_patterns.get("image_file", "{asin}_image_{index}.{ext}")
        elif file_type == "video":
            pattern = file_patterns.get("video_file", "{asin}_video_{index}.{ext}")
        else:
            raise ValueError(f"Unknown file type: {file_type}")

        return pattern.format(**kwargs)

    def get_retry_config(self) -> dict[str, Any]:
        """Get retry configuration settings."""
        global_settings = self.get_global_settings()
        return global_settings.get("retry_config", {})

    def get_rate_limiting_config(self) -> dict[str, Any]:
        """Get rate limiting configuration settings."""
        global_settings = self.get_global_settings()
        return global_settings.get("rate_limiting", {})

    def get_platform_base_url(self, platform: Platform) -> str:
        """Get base URL for a platform.

        Args:
        ----
            platform: The platform to get URL for

        Returns:
        -------
            Base URL string

        """
        platform_config = self.get_platform_config(platform)
        return platform_config.get("base_url", "")

    def get_platform_max_products(self, platform: Platform) -> int:
        """Get maximum products setting for a platform.

        Args:
        ----
            platform: The platform to get setting for

        Returns:
        -------
            Maximum products per search

        """
        platform_config = self.get_platform_config(platform)
        return platform_config.get("max_products", 5)

    def get_platform_keywords(self, platform: Platform) -> list[str]:
        """Get default keywords for a platform.

        Args:
        ----
            platform: The platform to get keywords for

        Returns:
        -------
            List of default keywords

        """
        platform_config = self.get_platform_config(platform)
        return platform_config.get("keywords", [])

    def get_platform_search_defaults(self, platform: Platform) -> dict[str, Any]:
        """Get default search parameters for a platform.

        Args:
        ----
            platform: The platform to get defaults for

        Returns:
        -------
            Dictionary of default search parameters

        """
        platform_config = self.get_platform_config(platform)
        return platform_config.get("default_search_parameters", {})


# Global configuration manager instance
_config_manager: PlatformConfigManager | None = None


def get_config_manager(
    config_path: str = "config/scrapers.yaml",
) -> PlatformConfigManager:
    """Get the global configuration manager instance.

    Args:
    ----
        config_path: Path to configuration file (only used on first call)

    Returns:
    -------
        PlatformConfigManager instance

    """
    global _config_manager

    if _config_manager is None:
        _config_manager = PlatformConfigManager(config_path)

    return _config_manager


def get_output_path(path_type: str, platform: Platform, **kwargs) -> Path:
    """Convenience function to get output path.

    Args:
    ----
        path_type: Type of path ('products', 'media', 'debug', etc.)
        platform: The platform this path is for
        **kwargs: Variables for path substitution

    Returns:
    -------
        Resolved Path object

    """
    config_manager = get_config_manager()
    return config_manager.get_output_path(path_type, platform, **kwargs)


def get_filename_pattern(file_type: str, **kwargs) -> str:
    """Convenience function to get filename pattern.

    Args:
    ----
        file_type: Type of file ('product', 'image', 'video')
        **kwargs: Variables for filename substitution

    Returns:
    -------
        Filename with variables substituted

    """
    config_manager = get_config_manager()
    return config_manager.get_filename_pattern(file_type, **kwargs)
