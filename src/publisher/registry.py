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


def create_publisher_from_config(
    config: Any,
    session: aiohttp.ClientSession | None = None,
    *,
    api_key: str | None = None,
    vercel_token: str | None = None,
) -> BasePublisher:
    """A publisher carrying every setting the loaded config names.

    The one place the settings are handed to the provider. The publisher CLI
    and the global batch each built their own publisher, and the batch's
    copy stopped short: `tiktok_settings` was missing from it for several
    releases, and `timeout` and `max_retries` never reached it at all, so the
    same config file produced different payloads and different retry
    behaviour on the two paths.

    Args:
    ----
        config: A loaded `PublisherConfig`.
        session: An aiohttp session to reuse, or None for the provider's own.
        api_key: A credential read at publish time, in place of the one on
            the config. The batch reads its settings before the run and its
            key from the environment when it publishes, so the config it
            holds may carry a placeholder.
        vercel_token: Likewise for the upload token.

    """
    return create_publisher(
        provider=PublisherProvider(config.provider),
        api_key=api_key if api_key is not None else config.api_key,
        session=session,
        vercel_token=vercel_token if vercel_token is not None else config.vercel_token,
        timeout=config.timeout,
        max_retries=config.max_retries,
        tiktok_settings=config.tiktok_settings,
        first_comment_config=config.first_comment_config,
        synthetic_media_disclosure=config.synthetic_media_disclosure,
    )
