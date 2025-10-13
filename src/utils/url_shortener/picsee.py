"""Picsee.io URL shortening client implementation.

This module provides integration with the Picsee.io URL shortening service,
supporting single and bulk URL shortening operations.

API Documentation: https://picsee.notion.site/Short-Link-API-Document-PicSee-URL-Shortener-482cd19c0fc94acfbe40ae8fe5d55236
"""

import asyncio
import logging
import random
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

    def __init__(
        self,
        api_key: str,
        session: aiohttp.ClientSession | None = None,
        timeout: int = 30,
        custom_domain: str | None = None,
        api_base_url: str = "https://api.pics.ee",
        max_bulk_size: int = 100,
        bulk_timeout_multiplier: float = 2.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        retry_backoff_multiplier: float = 2.0,
    ):
        """Initialize Picsee URL shortener.

        Args:
        ----
            api_key: Picsee.io API key
            session: Optional aiohttp session (will create new if None)
            timeout: Request timeout in seconds
            custom_domain: Optional custom branded short domain (BSD)
            api_base_url: Base URL for Picsee API
            max_bulk_size: Maximum URLs per bulk request
            bulk_timeout_multiplier: Multiplier for bulk request timeout
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            retry_backoff_multiplier: Exponential backoff multiplier

        """
        self.api_key = api_key
        self._session = session
        self._owns_session = session is None
        self.timeout = timeout
        self.custom_domain = custom_domain
        self.api_base_url = api_base_url
        self.max_bulk_size = max_bulk_size
        self.bulk_timeout_multiplier = bulk_timeout_multiplier
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_multiplier = retry_backoff_multiplier

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

    async def _retry_with_backoff(self, operation, operation_name: str) -> Any:
        """Retry an async operation with exponential backoff.

        Args:
        ----
            operation: Async callable to retry
            operation_name: Name of operation for logging

        Returns:
        -------
            Result from successful operation

        Raises:
        ------
            URLShortenerError: If all retries exhausted

        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation()
            except (TimeoutError, aiohttp.ClientError) as e:
                last_error = e

                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.retry_backoff_multiplier**attempt)
                    jitter = random.uniform(0.5, 1.5)  # noqa: S311
                    actual_delay = delay * jitter

                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {e}. "
                        f"Retrying in {actual_delay:.2f}s..."
                    )
                    await asyncio.sleep(actual_delay)
                else:
                    logger.error(
                        f"{operation_name} failed after {self.max_retries + 1} attempts"
                    )

        raise URLShortenerError(
            f"{operation_name} failed after {self.max_retries + 1} attempts: "
            f"{last_error}"
        )

    @property
    def provider(self) -> URLShortenerProvider:
        """Return the provider type."""
        return URLShortenerProvider.PICSEE

    async def shorten(self, url: str, custom_alias: str | None = None) -> ShortenedURL:
        """Shorten a single URL using Picsee.io API with retry logic.

        Args:
        ----
            url: The long URL to shorten
            custom_alias: Optional custom short code

        Returns:
        -------
            ShortenedURL object with shortened URL

        Raises:
        ------
            URLShortenerError: If API request fails after retries

        """

        async def _shorten_operation():
            session = await self._get_session()
            endpoint = f"{self.api_base_url}/v1/links?access_token={self.api_key}"

            payload: dict[str, Any] = {"url": url}

            if custom_alias:
                payload["encodeId"] = custom_alias

            if self.custom_domain:
                payload["domain"] = self.custom_domain

            async with session.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                data = await response.json()

                short_url = data.get("data", {}).get("picseeUrl")
                if not short_url:
                    raise URLShortenerError("No short URL in response")

                metadata = {"response": data.get("data", {})}

                logger.info(f"Shortened URL: {url} -> {short_url}")
                return ShortenedURL(
                    original_url=url,
                    short_url=short_url,
                    provider=self.provider,
                    metadata=metadata,
                )

        try:
            result: ShortenedURL = await self._retry_with_backoff(
                _shorten_operation, f"Shorten URL ({url[:50]}...)"
            )
            return result
        except URLShortenerError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error shortening URL: {e}")
            raise URLShortenerError(f"Unexpected error: {e}") from e

    async def shorten_bulk(self, urls: list[str]) -> list[ShortenedURL]:
        """Shorten multiple URLs in bulk with retry logic.

        Args:
        ----
            urls: List of long URLs to shorten (max per batch from config)

        Returns:
        -------
            List of ShortenedURL objects

        Raises:
        ------
            URLShortenerError: If bulk operation fails after retries

        """
        if len(urls) > self.max_bulk_size:
            raise URLShortenerError(
                f"Bulk size {len(urls)} exceeds maximum {self.max_bulk_size}"
            )

        async def _bulk_operation():
            session = await self._get_session()
            endpoint = f"{self.api_base_url}/shortlink/bulk"

            payload: dict[str, Any] = {"urls": [{"url": url} for url in urls]}

            if self.custom_domain:
                payload["domain"] = self.custom_domain

            async with session.post(
                endpoint,
                headers=self._get_headers(),
                json=payload,
                timeout=int(self.timeout * self.bulk_timeout_multiplier),
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

        try:
            result: list[ShortenedURL] = await self._retry_with_backoff(
                _bulk_operation, f"Bulk shorten {len(urls)} URLs"
            )
            return result
        except URLShortenerError:
            raise
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

    async def __aexit__(
        self,
        exc_type: Any,  # noqa: ARG002
        exc_val: Any,  # noqa: ARG002
        exc_tb: Any,  # noqa: ARG002
    ) -> None:
        """Async context manager exit."""
        await self.cleanup()
