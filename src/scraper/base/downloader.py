"""Platform-agnostic download functionality for the multi-platform scraper architecture.

This module provides shared download capabilities that can be used across all platform
scrapers, including media downloads, validation, and file management.
"""

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import requests

from .config import get_config_manager
from .models import Platform
from .utils import exponential_backoff_retry, sanitize_filename


class BaseDownloader:
    """Base downloader class with common functionality for all platforms."""

    def __init__(self, platform: Platform):
        """Initialize downloader for specific platform.

        Args:
        ----
            platform: The platform this downloader is for

        """
        self.platform = platform
        self.config_manager = get_config_manager()
        self.logger = logging.getLogger(__name__)

        # Load download configuration
        global_settings = self.config_manager.get_global_settings()
        self.download_config = global_settings.get("download_config", {})

        # Set timeouts and limits
        self.download_timeout = self.download_config.get("download_timeout", 30)
        self.chunk_size = self.download_config.get("download_chunk_size", 8192)
        self.validation_timeout = self.download_config.get("validation_timeout", 10)

    def get_download_directory(self, product_id: str, media_type: str) -> Path:
        """Get download directory for a product's media.

        Args:
        ----
            product_id: Platform-specific product identifier
            media_type: Type of media ('images' or 'videos')

        Returns:
        -------
            Path object for download directory

        """
        # Get platform media directory
        media_dir = self.config_manager.get_output_path("media", self.platform)

        # Create product-specific subdirectory
        product_dir = media_dir / product_id / media_type
        product_dir.mkdir(parents=True, exist_ok=True)

        return product_dir

    def generate_filename(
        self, url: str, product_id: str, index: int, media_type: str
    ) -> str:
        """Generate filename for downloaded media.

        Args:
        ----
            url: Original URL of the media
            product_id: Platform-specific product identifier
            index: Index of this media item
            media_type: Type of media ('image' or 'video')

        Returns:
        -------
            Generated filename

        """
        # Extract file extension from URL
        parsed_url = urlparse(url)
        path = parsed_url.path
        ext = os.path.splitext(path)[1] or ".jpg"  # Default to .jpg for images

        # Use config manager to generate filename
        filename = self.config_manager.get_filename_pattern(
            media_type,
            asin=product_id,  # Keep 'asin' for backward compatibility
            index=index,
            ext=ext.lstrip("."),
        )

        return sanitize_filename(filename)

    @exponential_backoff_retry
    def validate_url(self, url: str) -> bool:
        """Validate that a URL is accessible.

        Args:
        ----
            url: URL to validate

        Returns:
        -------
            True if URL is accessible, False otherwise

        """
        try:
            # Use HEAD request to check availability
            response = requests.head(
                url,
                timeout=self.validation_timeout,
                allow_redirects=True,
                headers={"User-Agent": self._get_user_agent()},
            )

            # Check if response is successful
            return 200 <= response.status_code < 400

        except Exception as e:
            self.logger.debug(f"URL validation failed for {url}: {e}")
            return False

    @exponential_backoff_retry
    def download_file_sync(self, url: str, filepath: Path) -> bool:
        """Download a file synchronously.

        Args:
        ----
            url: URL to download from
            filepath: Local path to save file

        Returns:
        -------
            True if download successful, False otherwise

        """
        try:
            response = requests.get(
                url,
                timeout=self.download_timeout,
                stream=True,
                headers={"User-Agent": self._get_user_agent()},
            )
            response.raise_for_status()

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Download file in chunks
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)

            # Verify file was created and has content
            if filepath.exists() and filepath.stat().st_size > 0:
                self.logger.debug(f"Downloaded {url} to {filepath}")
                return True
            else:
                self.logger.warning(f"Downloaded file is empty: {filepath}")
                if filepath.exists():
                    filepath.unlink()
                return False

        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            # Clean up partial file
            if filepath.exists():
                with contextlib.suppress(Exception):
                    filepath.unlink()
            return False

    async def download_file_async(
        self, session: aiohttp.ClientSession, url: str, filepath: Path
    ) -> bool:
        """Download a file asynchronously.

        Args:
        ----
            session: Aiohttp session for downloads
            url: URL to download from
            filepath: Local path to save file

        Returns:
        -------
            True if download successful, False otherwise

        """
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=self.download_timeout),
                headers={"User-Agent": self._get_user_agent()},
            ) as response:
                response.raise_for_status()

                # Ensure directory exists
                filepath.parent.mkdir(parents=True, exist_ok=True)

                # Download file
                with open(filepath, "wb") as f:
                    async for chunk in response.content.iter_chunked(self.chunk_size):
                        f.write(chunk)

                # Verify file was created and has content
                if filepath.exists() and filepath.stat().st_size > 0:
                    self.logger.debug(f"Downloaded {url} to {filepath}")
                    return True
                else:
                    self.logger.warning(f"Downloaded file is empty: {filepath}")
                    if filepath.exists():
                        filepath.unlink()
                    return False

        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            # Clean up partial file
            if filepath.exists():
                with contextlib.suppress(Exception):
                    filepath.unlink()
            return False

    def download_media_batch(
        self, urls: list[str], product_id: str, media_type: str
    ) -> list[str]:
        """Download a batch of media files synchronously.

        Args:
        ----
            urls: List of URLs to download
            product_id: Platform-specific product identifier
            media_type: Type of media ('images' or 'videos')

        Returns:
        -------
            List of successfully downloaded file paths

        """
        if not urls:
            return []

        download_dir = self.get_download_directory(product_id, media_type)
        downloaded_files = []

        for i, url in enumerate(urls):
            if not url or not self.validate_url(url):
                continue

            filename = self.generate_filename(url, product_id, i, media_type)
            filepath = download_dir / filename

            # Skip if file already exists
            if filepath.exists() and filepath.stat().st_size > 0:
                downloaded_files.append(str(filepath))
                continue

            # Download file
            if self.download_file_sync(url, filepath):
                downloaded_files.append(str(filepath))

        self.logger.info(
            f"Downloaded {len(downloaded_files)}/{len(urls)} {media_type} "
            f"for {product_id}"
        )

        return downloaded_files

    async def download_media_batch_async(
        self, urls: list[str], product_id: str, media_type: str
    ) -> list[str]:
        """Download a batch of media files asynchronously.

        Args:
        ----
            urls: List of URLs to download
            product_id: Platform-specific product identifier
            media_type: Type of media ('images' or 'videos')

        Returns:
        -------
            List of successfully downloaded file paths

        """
        if not urls:
            return []

        download_dir = self.get_download_directory(product_id, media_type)
        downloaded_files = []

        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent downloads

        async def download_single(i: int, url: str) -> str | None:
            async with semaphore:
                if not url or not self.validate_url(url):
                    return None

                filename = self.generate_filename(url, product_id, i, media_type)
                filepath = download_dir / filename

                # Skip if file already exists
                if filepath.exists() and filepath.stat().st_size > 0:
                    return str(filepath)

                async with aiohttp.ClientSession() as session:
                    if await self.download_file_async(session, url, filepath):
                        return str(filepath)

                return None

        # Create download tasks
        tasks = [download_single(i, url) for i, url in enumerate(urls)]

        # Execute downloads concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful downloads
        for result in results:
            if isinstance(result, str):
                downloaded_files.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Download task failed: {result}")

        self.logger.info(
            f"Downloaded {len(downloaded_files)}/{len(urls)} {media_type} "
            f"for {product_id}"
        )

        return downloaded_files

    def _get_user_agent(self) -> str:
        """Get user agent for downloads.

        Returns
        -------
            User agent string

        """
        try:
            platform_config = self.config_manager.get_platform_config(self.platform)
            headers = platform_config.get("http_headers", {})
            download_headers = headers.get("media_download", {})
            return download_headers.get(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
        except Exception:
            return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def create_downloader(platform: Platform) -> BaseDownloader:
    """Factory function to create a downloader for a platform.

    Args:
    ----
        platform: The platform to create downloader for

    Returns:
    -------
        BaseDownloader instance for the platform

    """
    return BaseDownloader(platform)
