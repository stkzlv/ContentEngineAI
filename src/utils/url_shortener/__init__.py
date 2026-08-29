"""URL shortening utilities with multi-provider support.

This package provides a flexible URL shortening system with support for
multiple providers (Picsee.io, Bitly, etc.) through a common interface.

Usage:
    from src.utils.url_shortener import create_url_shortener, URLShortenerProvider

    # Create a Picsee shortener
    shortener = create_url_shortener(
        provider=URLShortenerProvider.PICSEE,
        api_key="your_api_key"
    )

    # Shorten a URL
    result = await shortener.shorten("https://example.com/long-url")
    print(result.short_url)
"""

from .bare import BareURLShortener
from .base import (
    BaseURLShortener,
    ShortenedURL,
    URLShortenerError,
    URLShortenerProvider,
)
from .config import (
    URLShortenerApiSettings,
    URLShortenerIntegrationSettings,
    URLShortenerProviderSettings,
    URLShortenerSettings,
    load_url_shortener_settings,
)
from .picsee import PicseeURLShortener
from .registry import (
    URLShortenerRegistry,
    create_url_shortener,
    register_shortener,
)

register_shortener(URLShortenerProvider.BARE)(BareURLShortener)
register_shortener(URLShortenerProvider.PICSEE)(PicseeURLShortener)

__all__ = [
    "BareURLShortener",
    "BaseURLShortener",
    "ShortenedURL",
    "URLShortenerError",
    "URLShortenerProvider",
    "PicseeURLShortener",
    "URLShortenerRegistry",
    "create_url_shortener",
    "register_shortener",
    "URLShortenerApiSettings",
    "URLShortenerIntegrationSettings",
    "URLShortenerProviderSettings",
    "URLShortenerSettings",
    "load_url_shortener_settings",
]
