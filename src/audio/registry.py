"""Registry and factory for audio providers."""

import logging
from collections.abc import Callable
from typing import Any

from .base import AudioProvider, BaseAudioProvider

logger = logging.getLogger(__name__)


class AudioProviderRegistry:
    """Registry for audio provider implementations."""

    _providers: dict[AudioProvider, type[BaseAudioProvider]] = {}

    @classmethod
    def register(
        cls, provider: AudioProvider, provider_class: type[BaseAudioProvider]
    ) -> None:
        cls._providers[provider] = provider_class
        logger.debug("Registered audio provider: %s", provider.value)

    @classmethod
    def get(cls, provider: AudioProvider) -> type[BaseAudioProvider] | None:
        return cls._providers.get(provider)

    @classmethod
    def list_available(cls) -> list[AudioProvider]:
        return list(cls._providers.keys())


def register_audio_provider(
    provider: AudioProvider,
) -> Callable[[type[BaseAudioProvider]], type[BaseAudioProvider]]:
    """Decorator to register an audio provider implementation."""

    def decorator(
        provider_class: type[BaseAudioProvider],
    ) -> type[BaseAudioProvider]:
        AudioProviderRegistry.register(provider, provider_class)
        return provider_class

    return decorator


def create_audio_provider(name: str, **kwargs: Any) -> BaseAudioProvider:
    """Factory function to create audio provider instances."""
    try:
        provider = AudioProvider(name.lower())
    except ValueError as e:
        available = [p.value for p in AudioProviderRegistry.list_available()]
        raise ValueError(
            f"Unknown audio provider '{name}'. Available: {available}"
        ) from e

    provider_class = AudioProviderRegistry.get(provider)
    if not provider_class:
        available = [p.value for p in AudioProviderRegistry.list_available()]
        raise ValueError(
            f"Audio provider '{provider.value}' not registered. "
            f"Available: {available}"
        )

    return provider_class(**kwargs)
