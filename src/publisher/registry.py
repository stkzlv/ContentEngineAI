"""Registry and factory for video publishing providers.

This module provides a registry system for publisher providers,
enabling easy provider switching and dynamic registration.
"""

import logging
from collections.abc import Callable
from typing import Any

import aiohttp

from .base import BasePublisher, PublisherProvider

logger = logging.getLogger(__name__)


class PublisherRegistry:
    """Registry for video publishing provider implementations."""

    _providers: dict[PublisherProvider, type[BasePublisher]] = {}

    @classmethod
    def register(
        cls, provider: PublisherProvider, publisher_class: type[BasePublisher]
    ) -> None:
        """Register a publisher implementation.

        Args:
        ----
            provider: The provider enum value
            publisher_class: The publisher class implementing BasePublisher

        """
        cls._providers[provider] = publisher_class
        logger.debug("Registered publisher: %s", provider.value)

    @classmethod
    def get_publisher_class(
        cls, provider: PublisherProvider
    ) -> type[BasePublisher] | None:
        """Get the publisher class for a provider.

        Args:
        ----
            provider: The provider enum value

        Returns:
        -------
            The publisher class or None if not registered

        """
        return cls._providers.get(provider)

    @classmethod
    def get_available_providers(cls) -> list[PublisherProvider]:
        """Get list of registered providers.

        Returns
        -------
            List of available provider enum values

        """
        return list(cls._providers.keys())

    @classmethod
    def is_provider_supported(cls, provider: PublisherProvider) -> bool:
        """Check if a provider is registered.

        Args:
        ----
            provider: The provider enum value

        Returns:
        -------
            True if provider is registered

        """
        return provider in cls._providers


def register_publisher(
    provider: PublisherProvider,
) -> Callable[[type[BasePublisher]], type[BasePublisher]]:
    """Decorator to register a publisher provider.

    Usage:
        @register_publisher(PublisherProvider.LATE)
        class LatePublisher(BasePublisher):
            ...

    Args:
    ----
        provider: The provider enum value

    """

    def decorator(publisher_class: type[BasePublisher]) -> type[BasePublisher]:
        PublisherRegistry.register(provider, publisher_class)
        return publisher_class

    return decorator


def create_publisher(
    provider: PublisherProvider | str,
    api_key: str,
    session: aiohttp.ClientSession | None = None,
    **kwargs: Any,
) -> BasePublisher:
    """Factory function to create publisher instances.

    Args:
    ----
        provider: Provider enum or string name
        api_key: API key for the provider
        session: Optional aiohttp session
        **kwargs: Additional provider-specific configuration
                 (e.g., vercel_token, timeout, max_retries)

    Returns:
    -------
        Configured publisher instance

    Raises:
    ------
        ValueError: If provider is not registered or invalid

    Example:
    -------
        >>> publisher = create_publisher(
        ...     provider="late",
        ...     api_key="sk_live_...",
        ...     vercel_token="vercel_...",
        ...     timeout=60.0,
        ...     max_retries=3
        ... )
        >>> await publisher.authenticate()

    """
    if isinstance(provider, str):
        try:
            provider = PublisherProvider(provider.lower())
        except ValueError as e:
            available = [p.value for p in PublisherRegistry.get_available_providers()]
            raise ValueError(
                f"Invalid provider '{provider}'. Available: {available}"
            ) from e

    publisher_class = PublisherRegistry.get_publisher_class(provider)
    if not publisher_class:
        available = [p.value for p in PublisherRegistry.get_available_providers()]
        raise ValueError(
            f"Provider {provider.value} not registered. Available: {available}"
        )

    return publisher_class(api_key=api_key, session=session, **kwargs)  # type: ignore[call-arg]
