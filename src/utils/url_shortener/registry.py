"""Registry and factory for URL shortening providers.

This module provides a registry system for URL shortener providers,
enabling easy provider switching and fallback mechanisms.
"""

import logging
from typing import Any, Callable

import aiohttp

from .base import BaseURLShortener, URLShortenerProvider

logger = logging.getLogger(__name__)


class URLShortenerRegistry:
    """Registry for URL shortening provider implementations."""

    _providers: dict[URLShortenerProvider, type[BaseURLShortener]] = {}

    @classmethod
    def register(
        cls, provider: URLShortenerProvider, shortener_class: type[BaseURLShortener]
    ) -> None:
        """Register a URL shortener implementation.

        Args:
        ----
            provider: The provider enum value
            shortener_class: The shortener class implementing BaseURLShortener

        """
        cls._providers[provider] = shortener_class
        logger.debug(f"Registered URL shortener: {provider.value}")

    @classmethod
    def get_shortener_class(
        cls, provider: URLShortenerProvider
    ) -> type[BaseURLShortener] | None:
        """Get the shortener class for a provider.

        Args:
        ----
            provider: The provider enum value

        Returns:
        -------
            The shortener class or None if not registered

        """
        return cls._providers.get(provider)

    @classmethod
    def get_available_providers(cls) -> list[URLShortenerProvider]:
        """Get list of registered providers.

        Returns
        -------
            List of available provider enum values

        """
        return list(cls._providers.keys())

    @classmethod
    def is_provider_supported(cls, provider: URLShortenerProvider) -> bool:
        """Check if a provider is registered.

        Args:
        ----
            provider: The provider enum value

        Returns:
        -------
            True if provider is registered

        """
        return provider in cls._providers


def register_shortener(
    provider: URLShortenerProvider,
) -> Callable[[type[BaseURLShortener]], type[BaseURLShortener]]:
    """Decorator to register a URL shortener provider.

    Usage:
        @register_shortener(URLShortenerProvider.PICSEE)
        class PicseeURLShortener(BaseURLShortener):
            ...

    Args:
    ----
        provider: The provider enum value

    """

    def decorator(shortener_class: type[BaseURLShortener]) -> type[BaseURLShortener]:
        URLShortenerRegistry.register(provider, shortener_class)
        return shortener_class

    return decorator


def create_url_shortener(
    provider: URLShortenerProvider | str,
    api_key: str,
    session: aiohttp.ClientSession | None = None,
    **kwargs: Any,
) -> BaseURLShortener:
    """Factory function to create URL shortener instances.

    Args:
    ----
        provider: Provider enum or string name
        api_key: API key for the provider
        session: Optional aiohttp session
        **kwargs: Additional provider-specific configuration

    Returns:
    -------
        Configured URL shortener instance

    Raises:
    ------
        ValueError: If provider is not registered or invalid

    """
    if isinstance(provider, str):
        try:
            provider = URLShortenerProvider(provider.lower())
        except ValueError as e:
            available = [
                p.value for p in URLShortenerRegistry.get_available_providers()
            ]
            raise ValueError(
                f"Invalid provider '{provider}'. Available: {available}"
            ) from e

    shortener_class = URLShortenerRegistry.get_shortener_class(provider)
    if not shortener_class:
        available = [p.value for p in URLShortenerRegistry.get_available_providers()]
        raise ValueError(
            f"Provider {provider.value} not registered. Available: {available}"
        )

    return shortener_class(api_key=api_key, session=session, **kwargs)  # type: ignore[call-arg]
