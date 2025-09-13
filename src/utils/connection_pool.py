"""Connection pooling utilities for HTTP requests."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class GlobalConnectionPool:
    """Global connection pool manager for HTTP requests."""

    def __init__(
        self,
        pool_limit: int = 100,
        host_limit: int = 20,
        dns_ttl_sec: int = 300,
        keepalive_timeout_sec: int = 60,
        total_timeout_sec: int = 300,
        connect_timeout_sec: int = 30,
        read_timeout_sec: int = 60,
        cleanup_interval_sec: int = 300,
    ):
        self.pool_limit = pool_limit
        self.host_limit = host_limit
        self.dns_ttl_sec = dns_ttl_sec
        self.keepalive_timeout_sec = keepalive_timeout_sec
        self.total_timeout_sec = total_timeout_sec
        self.connect_timeout_sec = connect_timeout_sec
        self.read_timeout_sec = read_timeout_sec
        self.cleanup_interval_sec = cleanup_interval_sec

        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create the global HTTP session with connection pooling."""
        if self._session is None or self._session.closed:  # type: ignore[attr-defined]
            async with self._session_lock:
                if self._session is None or self._session.closed:  # type: ignore[attr-defined]
                    await self._create_session()

        if self._session is None:
            raise RuntimeError("Failed to create session")
        return self._session

    async def _create_session(self) -> None:
        """Create a new HTTP session with optimized settings."""
        # Configure connection pooling
        connector = aiohttp.TCPConnector(  # type: ignore[attr-defined]
            limit=self.pool_limit,  # Total connection pool size
            limit_per_host=self.host_limit,  # Max connections per host
            ttl_dns_cache=self.dns_ttl_sec,  # DNS cache TTL
            use_dns_cache=True,  # Enable DNS caching
            keepalive_timeout=self.keepalive_timeout_sec,  # Keep-alive timeout
            enable_cleanup_closed=True,  # Clean up closed connections
        )

        # Configure timeouts
        timeout = aiohttp.ClientTimeout(
            total=self.total_timeout_sec,  # Total timeout
            connect=self.connect_timeout_sec,  # Connection timeout
            sock_read=self.read_timeout_sec,  # Socket read timeout
        )

        # Create session
        self._session = aiohttp.ClientSession(  # type: ignore[call-arg]
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "ContentEngineAI/1.0"},
        )

        logger.debug("Created new HTTP session with connection pooling")

        # Schedule cleanup task
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up idle connections."""
        while True:
            try:
                # Configurable cleanup interval
                await asyncio.sleep(self.cleanup_interval_sec)
                if self._session and not self._session.closed:  # type: ignore[attr-defined]
                    # Properly cleanup idle connections without closing the connector
                    if hasattr(self._session.connector, "cleanup"):
                        await self._session.connector.cleanup()  # type: ignore[attr-defined]
                    logger.debug("Cleaned up idle HTTP connections")
            except Exception as e:
                logger.warning(f"Error during connection cleanup: {e}")

    async def close(self) -> None:
        """Close the connection pool and cleanup resources."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            from contextlib import suppress

            with suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        if self._session and not self._session.closed:  # type: ignore[attr-defined]
            await self._session.close()  # type: ignore[attr-defined]
            self._session = None
            logger.debug("Closed HTTP connection pool")


# Global connection pool instance
_global_pool: GlobalConnectionPool | None = None


def initialize_global_pool(
    pool_limit: int = 100,
    host_limit: int = 20,
    dns_ttl_sec: int = 300,
    keepalive_timeout_sec: int = 60,
    total_timeout_sec: int = 300,
    connect_timeout_sec: int = 30,
    read_timeout_sec: int = 60,
    cleanup_interval_sec: int = 300,
) -> None:
    """Initialize the global connection pool with custom settings."""
    global _global_pool
    _global_pool = GlobalConnectionPool(
        pool_limit,
        host_limit,
        dns_ttl_sec,
        keepalive_timeout_sec,
        total_timeout_sec,
        connect_timeout_sec,
        read_timeout_sec,
        cleanup_interval_sec,
    )


async def get_http_session() -> aiohttp.ClientSession:
    """Get the global HTTP session with connection pooling."""
    global _global_pool
    if _global_pool is None:
        _global_pool = GlobalConnectionPool()
    return await _global_pool.get_session()


async def close_global_pool() -> None:
    """Close the global connection pool."""
    global _global_pool
    if _global_pool is not None:
        await _global_pool.close()


async def http_get(
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int | None = None,
):
    """Make an HTTP GET request using the global connection pool.

    Args:
    ----
        url: URL to request
        headers: Optional request headers
        params: Optional query parameters
        timeout: Optional timeout override

    Returns:
    -------
        aiohttp.ClientResponse context manager

    """
    session = await get_http_session()

    # Override timeout if specified
    request_timeout = None
    if timeout is not None:
        request_timeout = aiohttp.ClientTimeout(total=timeout)

    return session.get(  # type: ignore[attr-defined]
        url,
        headers=headers,
        params=params,
        timeout=request_timeout,
    )


async def http_post(
    url: str,
    data: Any = None,
    json: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int | None = None,
) -> aiohttp.ClientResponse:
    """Make an HTTP POST request using the global connection pool.

    Args:
    ----
        url: URL to request
        data: Optional form data
        json: Optional JSON data
        headers: Optional request headers
        timeout: Optional timeout override

    Returns:
    -------
        aiohttp.ClientResponse object

    """
    session = await get_http_session()

    # Override timeout if specified
    request_timeout = None
    if timeout is not None:
        request_timeout = aiohttp.ClientTimeout(total=timeout)

    response = await session.post(  # type: ignore[call-arg,misc]
        url,
        data=data,
        json=json,
        headers=headers,
        timeout=request_timeout,  # type: ignore[arg-type]
    )
    return response  # type: ignore[no-any-return]


class DownloadManager:
    """Manages concurrent downloads with connection pooling."""

    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def download_file(
        self,
        url: str,
        output_path: Path,
        headers: dict[str, str] | None = None,
        chunk_size: int = 8192,
    ) -> bool:
        """Download a file using the connection pool.

        Args:
        ----
            url: URL to download
            output_path: Where to save the file
            headers: Optional request headers
            chunk_size: Size of chunks to read

        Returns:
        -------
            True if successful, False otherwise

        """
        async with self.semaphore:  # Limit concurrent downloads
            try:
                from pathlib import Path

                async with http_get(url, headers=headers) as response:  # type: ignore[attr-defined]
                    response.raise_for_status()

                    # Ensure output directory exists
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Download in chunks
                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            f.write(chunk)

                    logger.debug(f"Downloaded {url} to {output_path}")
                    return True

            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                # Clean up partial file
                if output_path.exists():
                    from contextlib import suppress

                    with suppress(Exception):
                        output_path.unlink()
                return False


# Global download manager
_download_manager: DownloadManager | None = None


def initialize_download_manager(max_concurrent: int = 5) -> None:
    """Initialize the global download manager with custom settings."""
    global _download_manager
    _download_manager = DownloadManager(max_concurrent)


def get_download_manager() -> DownloadManager:
    """Get the global download manager."""
    global _download_manager
    if _download_manager is None:
        _download_manager = DownloadManager()
    return _download_manager
