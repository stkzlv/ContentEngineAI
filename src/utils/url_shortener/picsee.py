"""Picsee.io URL shortening client implementation.

This module provides integration with the Picsee.io URL shortening service,
supporting single and bulk URL shortening operations.

API Documentation: https://picsee.notion.site/Short-Link-API-Document-PicSee-URL-Shortener-482cd19c0fc94acfbe40ae8fe5d55236
"""

import logging
from typing import Any

import aiohttp

from .base import (
    BaseURLShortener,
    ShortenedURL,
    URLShortenerError,
    URLShortenerProvider,
)

logger = logging.getLogger(__name__)


class PicseeURLShortener(BaseURLShortener):
    """Picsee.io URL shortening client.

    Provides async URL shortening capabilities using the Picsee.io API,
    supporting both single and bulk operations with custom aliases.
    """

    API_BASE_URL = "https://api.pics.ee"
    DEFAULT_TIMEOUT = 30
    MAX_BULK_SIZE = 100

    def __init__(
        self,
        api_key: str,
        session: aiohttp.ClientSession | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        custom_domain: str | None = None,
    ):
        """Initialize Picsee URL shortener.

        Args:
        ----
            api_key: Picsee.io API key
            session: Optional aiohttp session (will create new if None)
            timeout: Request timeout in seconds
            custom_domain: Optional custom branded short domain (BSD)

        """
        self.api_key = api_key
        self._session = session
        self._owns_session = session is None
        self.timeout = timeout
        self.custom_domain = custom_domain

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:  # type: ignore[attr-defined]
            from src.utils.connection_pool import get_http_session

            self._session = await get_http_session()
        return self._session

    def _get_headers(self) -> dict[str, str]:
        """Get API request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @property
    def provider(self) -> URLShortenerProvider:
        """Return the provider type."""
        return URLShortenerProvider.PICSEE

    async def shorten(
        self, url: str, custom_alias: str | None = None
    ) -> ShortenedURL:
        """Shorten a single URL using Picsee.io API.

        Args:
        ----
            url: The long URL to shorten
            custom_alias: Optional custom short code

        Returns:
        -------
            ShortenedURL object with shortened URL

        Raises:
        ------
            URLShortenerError: If API request fails

        """
        session = await self._get_session()

        # Build endpoint with access_token as query parameter
        endpoint = f"{self.API_BASE_URL}/v1/links?access_token={self.api_key}"

        payload: dict[str, Any] = {
            "url": url,
        }

        if custom_alias:
            payload["encodeId"] = custom_alias

        if self.custom_domain:
            payload["domain"] = self.custom_domain

        try:
            async with session.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                data = await response.json()

                # Picsee v1 API returns picseeUrl in data object
                short_url = data.get("data", {}).get("picseeUrl")
                if not short_url:
                    raise URLShortenerError("No short URL in response")

                metadata = {
                    "response": data.get("data", {}),
                }

                logger.info(f"Shortened URL: {url} -> {short_url}")
                return ShortenedURL(
                    original_url=url,
                    short_url=short_url,
                    provider=self.provider,
                    metadata=metadata,
                )

        except aiohttp.ClientError as e:
            logger.error(f"Picsee API request failed: {e}")
            raise URLShortenerError(f"Failed to shorten URL: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error shortening URL: {e}")
            raise URLShortenerError(f"Unexpected error: {e}") from e

    async def shorten_bulk(self, urls: list[str]) -> list[ShortenedURL]:
        """Shorten multiple URLs in bulk.

        Args:
        ----
            urls: List of long URLs to shorten (max 100 per batch)

        Returns:
        -------
            List of ShortenedURL objects

        Raises:
        ------
            URLShortenerError: If bulk operation fails

        """
        if len(urls) > self.MAX_BULK_SIZE:
            raise URLShortenerError(
                f"Bulk size {len(urls)} exceeds maximum {self.MAX_BULK_SIZE}"
            )

        session = await self._get_session()
        endpoint = f"{self.API_BASE_URL}/shortlink/bulk"

        payload: dict[str, Any] = {
            "urls": [{"url": url} for url in urls],
        }

        if self.custom_domain:
            payload["domain"] = self.custom_domain

        try:
            async with session.post(
                endpoint,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout * 2,  # Longer timeout for bulk
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if not data.get("success"):
                    error_msg = data.get("message", "Unknown error")
                    raise URLShortenerError(f"Picsee bulk API error: {error_msg}")

                results = data.get("data", [])
                shortened_urls = []

                for item in results:
                    short_url = item.get("shortLink")
                    original_url = item.get("originalUrl")

                    if short_url and original_url:
                        metadata = {
                            "picsee_id": item.get("id", ""),
                            "created_at": item.get("createdAt", ""),
                        }
                        shortened_urls.append(
                            ShortenedURL(
                                original_url=original_url,
                                short_url=short_url,
                                provider=self.provider,
                                metadata=metadata,
                            )
                        )

                logger.info(f"Bulk shortened {len(shortened_urls)} URLs")
                return shortened_urls

        except aiohttp.ClientError as e:
            logger.error(f"Picsee bulk API request failed: {e}")
            raise URLShortenerError(f"Failed to bulk shorten URLs: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in bulk shortening: {e}")
            raise URLShortenerError(f"Unexpected error: {e}") from e

    async def validate_api_key(self) -> bool:
        """Validate that the API key is working.

        Returns
        -------
            True if API key is valid, False otherwise

        """
        try:
            test_url = "https://example.com"
            await self.shorten(test_url, custom_alias=None)
            return True
        except URLShortenerError:
            return False
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Close HTTP session if owned by this instance."""
        if self._owns_session and self._session and not self._session.closed:  # type: ignore[attr-defined]
            await self._session.close()  # type: ignore[attr-defined]

    async def __aenter__(self) -> "PicseeURLShortener":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()
