"""No-op (bare) URL shortener.

Returns the input URL unchanged. Used as the default when the project doesn't
want a third-party shortener dependency, e.g. when the canonical Amazon URL is
already short enough and the affiliate tag must round-trip untouched.
"""

import logging
from typing import Any

import aiohttp

from .base import BaseURLShortener, ShortenedURL, URLShortenerProvider

logger = logging.getLogger(__name__)


class BareURLShortener(BaseURLShortener):
    """Pass-through shortener. Returns the input URL unchanged.

    No API key, no network calls, no external dependency. The provider exists so
    the rest of the pipeline can treat URL shortening uniformly (everything goes
    through the registry) while shipping a sensible default that doesn't depend
    on a third-party service preserving the affiliate tag in a 302 redirect.
    """

    def __init__(
        self,
        api_key: str | None = None,
        session: aiohttp.ClientSession | None = None,
        **_: Any,
    ) -> None:
        del api_key, session

    @property
    def provider(self) -> URLShortenerProvider:
        return URLShortenerProvider.BARE

    async def shorten(self, url: str, custom_alias: str | None = None) -> ShortenedURL:
        del custom_alias
        return ShortenedURL(
            original_url=url, short_url=url, provider=URLShortenerProvider.BARE
        )

    async def shorten_bulk(self, urls: list[str]) -> list[ShortenedURL]:
        return [
            ShortenedURL(
                original_url=u, short_url=u, provider=URLShortenerProvider.BARE
            )
            for u in urls
        ]

    async def validate_api_key(self) -> bool:
        return True
