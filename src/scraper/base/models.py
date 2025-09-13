"""Platform-agnostic base models for multi-platform scraper architecture.

This module defines the common data structures and interfaces that all platform-specific
scrapers should implement. It provides a unified API for different e-commerce platforms
while allowing platform-specific extensions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Platform(Enum):
    """Supported e-commerce platforms."""

    AMAZON = "amazon"
    EBAY = "ebay"
    WALMART = "walmart"
    SHOPIFY = "shopify"
    ETSY = "etsy"


class ProductStatus(Enum):
    """Product availability status."""

    AVAILABLE = "available"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


@dataclass
class BaseProductData:
    """Base product data structure shared across all platforms.

    This class defines the minimum required fields that every platform scraper
    should provide. Platform-specific scrapers can extend this class to add
    their own fields (e.g., ASIN for Amazon, Item ID for eBay).
    """

    # Core product identification
    title: str
    price: str
    url: str
    platform: Platform

    # Product details
    description: str = ""
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)

    # Search context
    keyword: str = ""
    search_position: int | None = None

    # Product metadata
    rating: str | None = None
    reviews_count: str | None = None
    brand: str | None = None
    category: str | None = None
    status: ProductStatus = ProductStatus.UNKNOWN

    # Media downloads
    downloaded_images: list[str] = field(default_factory=list)
    downloaded_videos: list[str] = field(default_factory=list)

    # Platform-specific ID (ASIN, Item ID, etc.)
    platform_id: str | None = None

    # Affiliate/monetization
    affiliate_link: str | None = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # All list fields are now initialized with default_factory
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "price": self.price,
            "url": self.url,
            "platform": self.platform.value,
            "description": self.description,
            "images": self.images,
            "videos": self.videos,
            "keyword": self.keyword,
            "search_position": self.search_position,
            "rating": self.rating,
            "reviews_count": self.reviews_count,
            "brand": self.brand,
            "category": self.category,
            "status": self.status.value,
            "downloaded_images": self.downloaded_images,
            "downloaded_videos": self.downloaded_videos,
            "platform_id": self.platform_id,
            "affiliate_link": self.affiliate_link,
        }


@dataclass
class BaseSearchParameters:
    """Base search parameters shared across platforms.

    Each platform can extend this class to add platform-specific filters
    while maintaining a common interface for basic search functionality.
    """

    # Price filtering
    min_price: float | None = None
    max_price: float | None = None

    # Quality filtering
    min_rating: float | None = None

    # Brand and category filtering
    brands: list[str] = field(default_factory=list)
    category: str | None = None

    # Shipping options
    free_shipping: bool = False

    # Sort order (platform-specific values)
    sort_order: str = "relevance"

    # Results configuration
    max_results: int = 10
    include_sponsored: bool = False
    skip_unavailable: bool = True

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # All list fields are now initialized with default_factory
        pass

    def validate(self) -> list[str]:
        """Validate search parameters and return list of errors."""
        errors = []

        if self.min_price is not None and self.min_price < 0:
            errors.append("Minimum price cannot be negative")

        if self.max_price is not None and self.max_price < 0:
            errors.append("Maximum price cannot be negative")

        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            errors.append("Minimum price cannot be greater than maximum price")

        if self.min_rating is not None and (self.min_rating < 1 or self.min_rating > 5):
            errors.append("Rating must be between 1 and 5")

        if self.max_results < 1:
            errors.append("Maximum results must be at least 1")

        return errors


@dataclass
class ScrapeResult:
    """Standardized result format for all scraping operations."""

    products: list[BaseProductData]
    platform: Platform
    keyword: str

    # Metadata
    total_found: int = 0
    pages_scraped: int = 1
    success: bool = True
    errors: list[str] = field(default_factory=list)

    # Performance metrics
    duration_seconds: float = 0.0
    products_per_second: float = 0.0

    def __post_init__(self):
        """Initialize default values and calculate metrics."""
        # All list fields are now initialized with default_factory

        if self.products:
            self.total_found = len(self.products)
            if self.duration_seconds > 0:
                self.products_per_second = len(self.products) / self.duration_seconds


class BaseScraper(ABC):
    """Abstract base class for all platform scrapers.

    This defines the common interface that all platform-specific scrapers
    must implement. It ensures consistency across different platforms while
    allowing for platform-specific customizations.
    """

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """Return the platform this scraper handles."""
        pass

    @abstractmethod
    def validate_product_id(self, product_id: str) -> bool:
        """Validate platform-specific product ID format.

        Args:
        ----
            product_id: Platform-specific product identifier

        Returns:
        -------
            True if valid, False otherwise

        """
        pass

    @abstractmethod
    def scrape_products(
        self, keywords: list[str], search_params: BaseSearchParameters | None = None
    ) -> list[BaseProductData]:
        """Scrape products for given keywords.

        Args:
        ----
            keywords: List of search terms or product IDs
            search_params: Optional search filtering parameters

        Returns:
        -------
            List of scraped product data

        """
        pass

    @abstractmethod
    def scrape_single_product(self, product_id: str) -> BaseProductData | None:
        """Scrape a single product by its platform-specific ID.

        Args:
        ----
            product_id: Platform-specific product identifier

        Returns:
        -------
            Product data if found, None otherwise

        """
        pass

    def cleanup(self) -> None:
        """Clean up resources after scraping.

        Default implementation does nothing. Platforms can override
        to clean up browser instances, close connections, etc.
        """
        # Default implementation - platforms can override
        return

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()


class ScraperRegistry:
    """Registry for platform scrapers supporting auto-discovery."""

    _scrapers: dict[Platform, type[BaseScraper]] = {}

    @classmethod
    def register(cls, platform: Platform, scraper_class: type[BaseScraper]):
        """Register a scraper class for a platform."""
        cls._scrapers[platform] = scraper_class

    @classmethod
    def get_scraper_class(cls, platform: Platform) -> type[BaseScraper] | None:
        """Get the scraper class for a platform."""
        return cls._scrapers.get(platform)

    @classmethod
    def get_available_platforms(cls) -> list[Platform]:
        """Get list of platforms with registered scrapers."""
        return list(cls._scrapers.keys())

    @classmethod
    def is_platform_supported(cls, platform: Platform) -> bool:
        """Check if a platform has a registered scraper."""
        return platform in cls._scrapers


def register_scraper(platform: Platform):
    """Decorator to register a scraper class for a platform.

    Usage:
        @register_scraper(Platform.AMAZON)
        class AmazonScraper(BaseScraper):
            ...
    """

    def decorator(scraper_class: type[BaseScraper]):
        ScraperRegistry.register(platform, scraper_class)
        return scraper_class

    return decorator
