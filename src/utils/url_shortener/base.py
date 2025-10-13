"""Base abstract interface for URL shortening services.

This module defines the common interface that all URL shortener providers
must implement, enabling provider-agnostic URL shortening with easy switching
between services.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class URLShortenerProvider(Enum):
    """Supported URL shortening providers."""

    PICSEE = "picsee"
    BITLY = "bitly"
    TINYURL = "tinyurl"


@dataclass
class ShortenedURL:
    """Result of URL shortening operation."""

    original_url: str
    short_url: str
    provider: URLShortenerProvider
    metadata: dict[str, str] | None = None


class URLShortenerError(Exception):
    """Base exception for URL shortener errors."""

    pass


class BaseURLShortener(ABC):
    """Abstract base class for URL shortening services.

    This defines the common interface that all provider-specific implementations
    must follow, ensuring consistency across different URL shortening services.
    """

    @property
    @abstractmethod
    def provider(self) -> URLShortenerProvider:
        """Return the provider this shortener uses."""
        pass

    @abstractmethod
    async def shorten(self, url: str, custom_alias: str | None = None) -> ShortenedURL:
        """Shorten a single URL.

        Args:
        ----
            url: The long URL to shorten
            custom_alias: Optional custom short code (if supported by provider)

        Returns:
        -------
            ShortenedURL object containing the shortened URL and metadata

        Raises:
        ------
            URLShortenerError: If shortening fails

        """
        pass

    @abstractmethod
    async def shorten_bulk(self, urls: list[str]) -> list[ShortenedURL]:
        """Shorten multiple URLs in bulk.

        Args:
        ----
            urls: List of long URLs to shorten

        Returns:
        -------
            List of ShortenedURL objects

        Raises:
        ------
            URLShortenerError: If bulk shortening fails

        """
        pass

    @abstractmethod
    async def validate_api_key(self) -> bool:
        """Validate that the API key is working.

        Returns
        -------
            True if API key is valid, False otherwise

        """
        pass
