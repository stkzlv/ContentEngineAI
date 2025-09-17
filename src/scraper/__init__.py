"""Multi-platform scraper package with factory and registry support.

This package provides a unified interface for scraping products from multiple
e-commerce platforms while maintaining platform-specific optimizations.

Architecture:
    - base/: Platform-agnostic foundation (models, config, utils)
    - amazon/: Amazon-specific implementation
    - [future]: ebay/, walmart/, etc.

Usage:
    # Using the factory
    scraper = ScraperFactory.create_scraper(Platform.AMAZON)
    products = scraper.scrape_products(["wireless earbuds"])

    # Direct platform import
    from src.scraper.amazon import BotasaurusAmazonScraper
    scraper = BotasaurusAmazonScraper()
    products = scraper.scrape_products(["B0BTYCRJSS"])
"""

import contextlib
import logging
from typing import Optional

from .base import (
    BaseProductData,
    BaseScraper,
    BaseSearchParameters,
    Platform,
    ScraperRegistry,
    get_config_manager,
)


class ScraperFactory:
    """Factory class for creating platform-specific scrapers.

    This factory provides a unified interface for creating scrapers while
    supporting automatic platform detection, configuration validation,
    and graceful error handling.
    """

    @classmethod
    def create_scraper(
        cls,
        platform: Platform,
        config_path: str | None = None,
        debug_mode: bool | None = None,
    ) -> BaseScraper:
        """Create a scraper instance for the specified platform.

        Args:
        ----
            platform: The platform to create a scraper for
            config_path: Optional path to configuration file
            debug_mode: Optional debug mode override

        Returns:
        -------
            Platform-specific scraper instance

        Raises:
        ------
            ValueError: If platform is not supported or not enabled
            ImportError: If platform module cannot be imported

        """
        # Check if platform is supported
        if not ScraperRegistry.is_platform_supported(platform):
            # Try to auto-import platform module
            cls._auto_import_platform(platform)

            # Check again after import attempt
            if not ScraperRegistry.is_platform_supported(platform):
                available = ScraperRegistry.get_available_platforms()
                raise ValueError(
                    f"Platform {platform.value} is not supported. "
                    f"Available platforms: {[p.value for p in available]}"
                )

        # Check if platform is enabled in configuration
        config_manager = get_config_manager(config_path or "config/scrapers.yaml")
        if not config_manager.is_platform_enabled(platform):
            raise ValueError(
                f"Platform {platform.value} is not enabled in configuration. "
                f"Set 'scrapers.{platform.value}.enabled: true' in config file."
            )

        # Get scraper class and create instance
        scraper_class = ScraperRegistry.get_scraper_class(platform)

        # Create instance with appropriate parameters
        if platform == Platform.AMAZON:
            # Amazon scraper takes config_path and debug_override
            return scraper_class(  # type: ignore[call-arg,misc]
                config_path=config_path or "config/scrapers.yaml",
                debug_override=debug_mode,
            )
        else:
            # Future platforms may have different constructors
            return scraper_class()  # type: ignore[misc]

    @classmethod
    def get_available_platforms(cls) -> list[Platform]:
        """Get list of available platforms.

        Attempts to auto-import all known platforms before returning the list.

        Returns
        -------
            List of supported Platform enums

        """
        # Auto-import known platforms
        for platform in Platform:
            with contextlib.suppress(Exception):
                cls._auto_import_platform(platform)

        return ScraperRegistry.get_available_platforms()

    @classmethod
    def get_enabled_platforms(cls, config_path: str | None = None) -> list[Platform]:
        """Get list of enabled platforms from configuration.

        Args:
        ----
            config_path: Optional path to configuration file

        Returns:
        -------
            List of enabled Platform enums

        """
        config_manager = get_config_manager(config_path or "config/scrapers.yaml")
        return config_manager.get_enabled_platforms()

    @classmethod
    def _auto_import_platform(cls, platform: Platform) -> None:
        """Attempt to auto-import a platform module.

        Args:
        ----
            platform: The platform to import

        Raises:
        ------
            ImportError: If platform module cannot be imported

        """
        try:
            if platform == Platform.AMAZON:
                from .amazon import BotasaurusAmazonScraper  # noqa: F401
            elif platform == Platform.EBAY or platform == Platform.WALMART:
                # Future implementation
                pass
            # Add other platforms as they are implemented

        except ImportError as e:
            logging.getLogger(__name__).debug(
                f"Could not import {platform.value} scraper: {e}"
            )
            raise ImportError(f"Platform {platform.value} module not available") from e


class MultiPlatformScraper:
    """Convenience class for scraping from multiple platforms simultaneously.

    This class provides a high-level interface for running scraping operations
    across multiple platforms with unified result handling.
    """

    def __init__(
        self,
        platforms: list[Platform] | None = None,
        config_path: str | None = None,
    ):
        """Initialize multi-platform scraper.

        Args:
        ----
            platforms: List of platforms to use (defaults to all enabled platforms)
            config_path: Optional path to configuration file

        """
        self.config_path = config_path

        if platforms is None:
            # Use all enabled platforms
            self.platforms = ScraperFactory.get_enabled_platforms(config_path)
        else:
            # Validate provided platforms
            enabled = ScraperFactory.get_enabled_platforms(config_path)
            self.platforms = [p for p in platforms if p in enabled]

        self._scrapers: dict[Platform, BaseScraper] = {}

    def get_scraper(self, platform: Platform) -> BaseScraper:
        """Get or create a scraper for the specified platform.

        Args:
        ----
            platform: The platform to get scraper for

        Returns:
        -------
            Platform-specific scraper instance

        """
        if platform not in self._scrapers:
            self._scrapers[platform] = ScraperFactory.create_scraper(
                platform, self.config_path
            )

        return self._scrapers[platform]

    def scrape_all_platforms(self, keywords: list[str]) -> dict[Platform, list]:
        """Scrape products from all configured platforms.

        Args:
        ----
            keywords: List of search terms or product IDs

        Returns:
        -------
            Dictionary mapping platforms to their scraped products

        """
        results = {}

        for platform in self.platforms:
            try:
                scraper = self.get_scraper(platform)
                products = scraper.scrape_products(keywords)
                results[platform] = products

                logging.getLogger(__name__).info(
                    f"Scraped {len(products)} products from {platform.value}"
                )

            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Failed to scrape {platform.value}: {e}"
                )
                results[platform] = []

        return results

    def cleanup(self):
        """Clean up all scrapers."""
        for scraper in self._scrapers.values():
            try:
                scraper.cleanup()
            except Exception as e:
                logging.getLogger(__name__).debug(f"Cleanup warning: {e}")

        self._scrapers.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()


# Key components already imported above

__all__ = [
    "ScraperFactory",
    "MultiPlatformScraper",
    "Platform",
    "BaseProductData",
    "BaseSearchParameters",
]
