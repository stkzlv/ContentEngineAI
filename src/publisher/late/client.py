"""Late.dev publisher client implementation.

This module provides the concrete implementation of BasePublisher for the Late.dev
scheduling service, supporting multi-platform video publishing to YouTube, TikTok,
Instagram, and other social media platforms.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
from late import Late

from src.publisher.base import (
    AuthenticationError,
    BasePublisher,
    PublisherProvider,
    PublishError,
    UploadError,
    ValidationError,
)
from src.publisher.registry import register_publisher
from src.video.config.constants import (
    DEFAULT_EXPONENTIAL_BACKOFF_BASE,
)

logger = logging.getLogger(__name__)


@register_publisher(PublisherProvider.LATE)
class LatePublisher(BasePublisher):
    """Late.dev implementation of video publishing service.

    This class provides video publishing capabilities through the Late.dev API,
    supporting:
    - Multiple social media platforms (YouTube, TikTok, Instagram, etc.)
    - Small file uploads (≤4 MB) via direct API
    - Large file uploads (4-500 MB) via Vercel token
    - Immediate and scheduled publishing
    - Multi-platform posts in single operation

    Attributes
    ----------
        client: Late SDK client instance
        vercel_token: Optional Vercel token for large file uploads
        session: Aiohttp session for HTTP requests
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for transient failures

    """

    def __init__(
        self,
        api_key: str,
        vercel_token: str | None = None,
        session: aiohttp.ClientSession | None = None,
        timeout: float = 120.0,  # Configurable via publisher.yaml
        max_retries: int = 3,  # Configurable via publisher.yaml
    ):
        """Initialize Late.dev publisher client.

        Args:
        ----
            api_key: Late.dev API key (required)
            vercel_token: Vercel token for large file uploads (optional)
            session: Aiohttp session for HTTP requests (optional, created if None)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts for transient failures (default: 3)

        Raises:
        ------
            ValidationError: If api_key is empty or invalid format
            ValueError: If timeout or max_retries are invalid

        Example:
        -------
            >>> publisher = LatePublisher(
            ...     api_key="sk_live_abc123",
            ...     vercel_token="vercel_xyz456",
            ...     timeout=60.0,
            ...     max_retries=5
            ... )
            >>> await publisher.authenticate()

        """
        # Validate inputs
        if not api_key or not api_key.strip():
            raise ValidationError("api_key cannot be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        # Store configuration
        self._api_key = api_key
        self.vercel_token = vercel_token
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize Late SDK client
        try:
            self.client = Late(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Late client: {e}")
            raise ValidationError(f"Invalid Late API configuration: {e}") from e

        # HTTP session for additional requests if needed
        self._session = session
        self._should_close_session = session is None

        logger.info(f"Late publisher: {timeout}s timeout, {max_retries} retries")
        logger.debug(
            f"API key: {api_key[:4]}..." if len(api_key) > 4 else "API key set"
        )
        logger.debug(f"Vercel token: {'set' if vercel_token else 'NOT SET'}")

    @property
    def provider(self) -> PublisherProvider:
        """Return the publisher provider.

        Returns
        -------
            PublisherProvider.LATE

        """
        return PublisherProvider.LATE

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session.

        Returns
        -------
            Active aiohttp session

        """
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def _close_session(self):
        """Close aiohttp session if created internally."""
        if self._should_close_session and self._session:
            await self._session.close()
            self._session = None

    async def _retry_with_backoff(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs,
    ):
        """Execute operation with exponential backoff retry logic.

        Implements comprehensive error handling for:
        - Network timeouts (aiohttp.ClientTimeout)
        - Connection errors (aiohttp.ClientConnectionError)
        - Rate limits (429) with retry-after header extraction
        - Server errors (5xx) with exponential backoff
        - Permanent failures (401/403) without retries

        Args:
        ----
            operation: Async function to execute
            operation_name: Name for logging
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
        -------
            Result from operation

        Raises:
        ------
            AuthenticationError: For 401/403 permanent auth failures
            PublishError: For rate limits (429) after retries
            UploadError: For upload-specific failures
            Exception: If operation fails after all retries

        """
        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = await operation(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"{operation_name} succeeded on attempt {attempt}")
                return result

            except aiohttp.ClientResponseError as e:
                # HTTP response errors (4xx, 5xx)
                last_exception = e

                # 401/403: Permanent authentication failures - don't retry
                if e.status in (401, 403):
                    key_prefix = self._api_key[:4]
                    error_msg = (
                        f"{operation_name} auth failed ({e.status}): "
                        f"API key ({key_prefix}...) invalid - {e.message}"
                    )
                    logger.error(error_msg)
                    raise AuthenticationError(error_msg) from e

                # 429: Rate limit - extract retry-after header
                if e.status == 429:
                    retry_after = self._extract_retry_after(e)
                    if attempt < self.max_retries:
                        logger.warning(
                            f"{operation_name} rate limited (429), "
                            f"waiting {retry_after}s before retry "
                            f"(attempt {attempt}/{self.max_retries})"
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        error_msg = (
                            f"{operation_name} rate limit exceeded after "
                            f"{self.max_retries} attempts"
                        )
                        logger.error(error_msg)
                        raise PublishError(error_msg) from e

                # 400, 422: Validation errors - don't retry
                if e.status in (400, 422):
                    error_msg = (
                        f"{operation_name} validation failed (HTTP {e.status}): "
                        f"{e.message}"
                    )
                    logger.error(error_msg)
                    raise ValidationError(error_msg) from e

                # 5xx: Server errors - retry with exponential backoff
                if 500 <= e.status < 600:
                    if attempt < self.max_retries:
                        delay = DEFAULT_EXPONENTIAL_BACKOFF_BASE ** (attempt - 1)
                        logger.warning(
                            f"{operation_name} server error ({e.status}), retry "
                            f"{attempt}/{self.max_retries} in {delay}s: {e.message}"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        error_msg = (
                            f"{operation_name} server error (HTTP {e.status}) after "
                            f"{self.max_retries} attempts: {e.message}"
                        )
                        logger.error(error_msg)

                # Other 4xx errors - log and retry (might be transient)
                if attempt < self.max_retries:
                    delay = DEFAULT_EXPONENTIAL_BACKOFF_BASE ** (attempt - 1)
                    logger.warning(
                        f"{operation_name} failed (HTTP {e.status}), "
                        f"retrying in {delay}s (attempt {attempt}/{self.max_retries}): "
                        f"{e.message}"
                    )
                    await asyncio.sleep(delay)
                else:
                    error_msg = (
                        f"{operation_name} failed (HTTP {e.status}) after "
                        f"{self.max_retries} attempts: {e.message}"
                    )
                    logger.error(error_msg)

            except aiohttp.ClientConnectionError as e:
                # Connection errors (DNS, network unreachable, etc.)
                last_exception = e
                if attempt < self.max_retries:
                    delay = DEFAULT_EXPONENTIAL_BACKOFF_BASE ** (attempt - 1)
                    logger.warning(
                        f"{operation_name} connection error, "
                        f"retry {attempt}/{self.max_retries} in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    error_msg = (
                        f"{operation_name} connection failed after "
                        f"{self.max_retries} retries: {e}"
                    )
                    logger.error(error_msg)

            except TimeoutError as e:
                # Request timeout
                last_exception = e
                if attempt < self.max_retries:
                    delay = DEFAULT_EXPONENTIAL_BACKOFF_BASE ** (attempt - 1)
                    logger.warning(
                        f"{operation_name} timed out after {self.timeout}s, "
                        f"retrying in {delay}s (attempt {attempt}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    error_msg = (
                        f"{operation_name} timed out after {self.max_retries} attempts "
                        f"({self.timeout}s each)"
                    )
                    logger.error(error_msg)

            except aiohttp.ClientError as e:
                # Generic aiohttp errors (catch-all for other client errors)
                last_exception = e
                if attempt < self.max_retries:
                    delay = DEFAULT_EXPONENTIAL_BACKOFF_BASE ** (attempt - 1)
                    logger.warning(
                        f"{operation_name} client error, "
                        f"retry {attempt}/{self.max_retries} in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    retries = self.max_retries
                    error_msg = f"{operation_name} failed after {retries} attempts: {e}"
                    logger.error(error_msg)

            except (ValidationError, AuthenticationError, UploadError, PublishError):
                # Our custom exceptions - don't retry, just propagate
                raise

            except Exception as e:
                # Unexpected errors - log full context and don't retry
                err_type = type(e).__name__
                logger.error(
                    f"{operation_name} unexpected error: {err_type}: {e}",
                    exc_info=True,
                )
                raise

        # All retries exhausted
        if last_exception:
            raise last_exception

    def _extract_retry_after(self, error: aiohttp.ClientResponseError) -> int:
        """Extract retry-after value from rate limit response.

        Args:
        ----
            error: ClientResponseError with status 429

        Returns:
        -------
            Retry delay in seconds (default: 60 if header missing)

        """
        # Try to extract Retry-After header from response
        # Note: Late SDK may not expose response headers directly
        # Default to configured value if not available
        retry_after = 60  # Default retry delay in seconds

        # Check if error has headers attribute
        if hasattr(error, "headers") and error.headers:
            retry_header = error.headers.get("Retry-After") or error.headers.get(
                "retry-after"
            )
            if retry_header:
                try:
                    retry_after = int(retry_header)
                    logger.debug(f"Extracted Retry-After header: {retry_after}s")
                except ValueError:
                    # Retry-After might be HTTP date format, not seconds
                    logger.debug(f"Could not parse Retry-After header: {retry_header}")

        return retry_after

    async def authenticate(self) -> bool:
        """Validate API credentials with Late.dev service.

        Attempts to list accounts as a test of authentication. This verifies that
        the API key is valid and the service is accessible.

        Returns:
        -------
            True if authentication succeeds

        Raises:
        ------
            AuthenticationError: If credentials are invalid or expired
            PublisherError: If authentication request fails

        Example:
        -------
            >>> is_authenticated = await publisher.authenticate()
            >>> if is_authenticated:
            ...     print("Credentials valid")

        """
        logger.info("Authenticating with Late.dev API")

        try:
            # Test authentication by attempting to list accounts
            async def _auth_test():
                # Note: Late SDK methods might be sync or async
                # Check if method is async and call appropriately
                if asyncio.iscoroutinefunction(self.client.accounts.list):
                    accounts = await self.client.accounts.list()
                else:
                    accounts = self.client.accounts.list()
                return accounts

            await self._retry_with_backoff(_auth_test, "Authentication")

            logger.info("Authentication successful")
            return True

        except AttributeError as e:
            error_msg = f"Late SDK missing expected method: {e}"
            logger.error(error_msg)
            raise AuthenticationError(error_msg) from e

        except Exception as e:
            error_msg = f"Authentication failed: {e}"
            logger.error(error_msg)
            # Check for 401/403 status codes that indicate auth failure
            if "401" in str(e) or "403" in str(e) or "Unauthorized" in str(e):
                raise AuthenticationError(
                    f"Invalid or expired API key ({self._api_key[:4]}...): {e}"
                ) from e
            raise AuthenticationError(error_msg) from e

    async def get_accounts(self) -> list[dict[str, str]]:
        """Fetch all connected social media accounts from Late.dev.

        Returns:
        -------
            List of account dictionaries with platform, username, account_id, status

        Raises:
        ------
            AuthenticationError: If not authenticated or credentials expired
            PublisherError: If account fetching fails after retries

        Example:
        -------
            >>> accounts = await publisher.get_accounts()
            >>> for account in accounts:
            ...     print(f"{account['platform']}: @{account['username']}")

        """
        logger.info("Fetching connected accounts from Late.dev")

        try:

            async def _fetch_accounts():
                if asyncio.iscoroutinefunction(self.client.accounts.list):
                    return await self.client.accounts.list()
                else:
                    return self.client.accounts.list()

            raw_accounts = await self._retry_with_backoff(
                _fetch_accounts, "Fetch accounts"
            )

            # Debug: Log raw response structure
            logger.debug(f"Raw accounts type: {type(raw_accounts)}")
            logger.debug(f"Raw accounts value: {raw_accounts}")

            # Parse and normalize account data
            # Late SDK returns AccountsListResponse with .accounts attribute
            accounts_list = (
                raw_accounts.accounts
                if hasattr(raw_accounts, "accounts")
                else raw_accounts
            )

            accounts = []
            for account in accounts_list:
                # Handle both dict and object responses
                if hasattr(account, "platform"):
                    # Pydantic model or object
                    account_dict = {
                        "platform": getattr(account, "platform", "unknown"),
                        "username": getattr(account, "username", "")
                        or getattr(account, "handle", ""),
                        "account_id": getattr(account, "field_id", "")
                        or getattr(account, "id", ""),
                        "status": "active"
                        if getattr(account, "isActive", True)
                        else "inactive",
                        "display_name": getattr(account, "displayName", ""),
                    }
                else:
                    # Dict response
                    account_dict = {
                        "platform": account.get("platform", "unknown"),
                        "username": account.get("username")
                        or account.get("handle", ""),
                        "account_id": account.get("id", ""),
                        "status": account.get("status", "active"),
                        "display_name": account.get("displayName", ""),
                    }
                accounts.append(account_dict)

            logger.info(f"Found {len(accounts)} connected accounts")
            return accounts

        except Exception as e:
            error_msg = f"Failed to fetch accounts: {e}"
            logger.error(error_msg)
            if "401" in str(e) or "403" in str(e):
                raise AuthenticationError(f"Authentication expired: {e}") from e
            raise PublishError(error_msg) from e

    async def list_posts(self, status: str | None = None) -> list[dict[str, Any]]:
        """Fetch all posts from Late.dev, optionally filtered by status.

        Args:
        ----
            status: Optional status filter ('scheduled', 'published', 'failed')

        Returns:
        -------
            List of post dictionaries with id, status, scheduledFor, platforms

        Raises:
        ------
            AuthenticationError: If not authenticated or credentials expired
            PublishError: If post fetching fails after retries

        Example:
        -------
            >>> posts = await publisher.list_posts(status='scheduled')
            >>> for post in posts:
            ...     print(f"{post['id']}: {post['scheduledFor']}")

        """
        logger.info(f"Fetching posts from Late.dev (status={status or 'all'})")

        try:
            # Fetch all posts with pagination
            all_posts_list = []
            page = 1
            page_size = 100  # Max items per page

            while True:

                async def _fetch_posts_page(current_page=page):
                    if asyncio.iscoroutinefunction(self.client.posts.list):
                        return await self.client.posts.list(
                            page=current_page, limit=page_size
                        )
                    else:
                        return self.client.posts.list(
                            page=current_page, limit=page_size
                        )

                raw_posts = await self._retry_with_backoff(
                    _fetch_posts_page, f"Fetch posts page {page}"
                )

                # Parse and normalize post data
                # Late SDK returns PostsListResponse with .posts attribute
                posts_list = (
                    raw_posts.posts if hasattr(raw_posts, "posts") else raw_posts
                )

                if not posts_list:
                    # No more posts
                    break

                all_posts_list.extend(posts_list)

                # Check if we got a full page - if not, we're done
                if len(posts_list) < page_size:
                    break

                page += 1

            logger.debug(
                f"Fetched {len(all_posts_list)} total posts " f"across {page} page(s)"
            )

            posts = []
            for post in all_posts_list:
                # Handle both dict and object responses
                if isinstance(post, dict):
                    # Already a dict
                    post_dict = post
                else:
                    # Pydantic model or object
                    # Late SDK uses field_id instead of id
                    post_id = None
                    if hasattr(post, "id"):
                        post_id = str(post.id)
                    elif hasattr(post, "field_id"):
                        post_id = str(post.field_id)

                    post_dict = {
                        "id": post_id,
                        "status": (
                            str(post.status).split(".")[-1].lower()
                            if hasattr(post, "status")
                            else None
                        ),
                        "scheduledFor": (
                            post.scheduledFor if hasattr(post, "scheduledFor") else None
                        ),
                        "platforms": (
                            [
                                {
                                    "platform": str(p.platform).split(".")[-1].lower(),
                                    "account_id": (
                                        str(p.field_id)
                                        if hasattr(p, "field_id")
                                        else (
                                            str(p.accountId)
                                            if hasattr(p, "accountId")
                                            else None
                                        )
                                    ),
                                }
                                for p in post.platforms
                            ]
                            if hasattr(post, "platforms")
                            else []
                        ),
                    }

                # Apply status filter if provided
                post_status = post_dict.get("status")
                if status is None or post_status == status:
                    posts.append(post_dict)

            logger.info(f"Found {len(posts)} post(s)")
            return posts

        except Exception as e:
            error_msg = f"Failed to fetch posts: {e}"
            logger.error(error_msg)
            if "401" in str(e) or "403" in str(e):
                raise AuthenticationError(f"Authentication expired: {e}") from e
            raise PublishError(error_msg) from e

    async def delete_post(self, post_id: str) -> bool:
        """Delete a post from Late.dev.

        Args:
        ----
            post_id: The ID of the post to delete

        Returns:
        -------
            True if deletion was successful

        Raises:
        ------
            AuthenticationError: If not authenticated or credentials expired
            PublishError: If deletion fails

        Example:
        -------
            >>> success = await publisher.delete_post("6947ea211395270412c251ed")
            >>> if success:
            ...     print("Post deleted")

        """
        logger.info(f"Deleting post: {post_id}")

        try:

            async def _delete_post():
                if asyncio.iscoroutinefunction(self.client.posts.delete):
                    return await self.client.posts.delete(post_id)
                else:
                    return self.client.posts.delete(post_id)

            await self._retry_with_backoff(_delete_post, "Delete post")
            logger.info(f"Post {post_id} deleted successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to delete post {post_id}: {e}"
            logger.error(error_msg)
            if "401" in str(e) or "403" in str(e):
                raise AuthenticationError(f"Authentication expired: {e}") from e
            if "404" in str(e):
                logger.warning(f"Post {post_id} not found (may already be deleted)")
                return True  # Consider it deleted if not found
            raise PublishError(error_msg) from e

    async def upload_media(
        self,
        video_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        """Upload video file to Late.dev.

        Automatically selects upload method based on file size:
        - ≤4 MB: Direct upload via client.media.upload()
        - 4-500 MB: Large file upload via client.media.upload_large()
        - >500 MB: Raises ValidationError

        Args:
        ----
            video_path: Path to video file
            progress_callback: Optional callback for progress updates
                (bytes_uploaded, total_bytes)

        Returns:
        -------
            Media ID string for use in publish()

        Raises:
        ------
            ValidationError: If file doesn't exist, unreadable, or >500 MB
            UploadError: If upload fails after retries

        Example:
        -------
            >>> media_id = await publisher.upload_media(Path("video.mp4"))
            >>> print(f"Uploaded: {media_id}")

        """
        logger.info(f"Uploading video: {video_path}")

        # Comprehensive file validation
        if not video_path.exists():
            raise ValidationError(f"Video file not found: {video_path}")
        if not video_path.is_file():
            raise ValidationError(f"Path is not a file: {video_path}")

        # Check file is readable
        try:
            with open(video_path, "rb") as f:
                # Read first byte to verify access
                f.read(1)
        except PermissionError as e:
            raise ValidationError(
                f"File not readable (permission denied): {video_path}"
            ) from e
        except OSError as e:
            raise ValidationError(f"File not accessible: {video_path} - {e}") from e

        # Validate file size
        try:
            file_size = video_path.stat().st_size
        except OSError as e:
            raise ValidationError(
                f"Cannot determine file size: {video_path} - {e}"
            ) from e

        # Check minimum size (must be non-empty)
        if file_size == 0:
            raise ValidationError(f"Video file is empty: {video_path}")

        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        # Check size limits
        if file_size > 500 * 1024 * 1024:  # 500 MB
            raise ValidationError(
                f"File exceeds Late.dev 500 MB limit: {file_size_mb:.2f} MB"
            )

        # Validate video file extension
        valid_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
        if video_path.suffix.lower() not in valid_extensions:
            logger.warning(
                f"Video file has unusual extension: {video_path.suffix}. "
                f"Expected one of: {', '.join(valid_extensions)}"
            )

        # Track upload progress
        last_logged_progress = 0

        def _log_progress(bytes_uploaded: int, total_bytes: int):
            """Log upload progress every 10%."""
            nonlocal last_logged_progress
            progress_pct = int((bytes_uploaded / total_bytes) * 100)

            # Log every 10% increment
            if progress_pct >= last_logged_progress + 10:
                up_mb = bytes_uploaded / (1024 * 1024)
                tot_mb = total_bytes / (1024 * 1024)
                logger.info(f"Upload: {progress_pct}% ({up_mb:.1f}/{tot_mb:.1f} MB)")
                last_logged_progress = progress_pct

            # Call user-provided callback if any
            if progress_callback:
                progress_callback(bytes_uploaded, total_bytes)

        try:
            # Small file upload (≤4 MB)
            if file_size <= 4 * 1024 * 1024:
                logger.info("Using direct upload for small file")

                async def _upload_small():
                    # Late SDK expects file path, not file object
                    if asyncio.iscoroutinefunction(self.client.media.upload):
                        result = await self.client.media.upload(str(video_path))
                    else:
                        result = self.client.media.upload(str(video_path))
                    # Simulate 100% progress for callback
                    _log_progress(file_size, file_size)
                    return result

                media_response = await self._retry_with_backoff(
                    _upload_small, f"Upload {video_path.name}"
                )

            # Large file upload (>4 MB and ≤500 MB)
            else:
                if not self.vercel_token:
                    raise ValidationError(
                        "Vercel token required for large file uploads (>4 MB)"
                    )

                logger.info("Using large file upload with Vercel token")
                logger.info("Note: Progress tracking may not work for large uploads")

                async def _upload_large():
                    # Late SDK may not support progress callbacks for large uploads
                    logger.info("Starting large file upload...")

                    # Late SDK expects file path, not file object
                    if asyncio.iscoroutinefunction(self.client.media.upload_large):
                        result = await self.client.media.upload_large(
                            str(video_path),
                            vercel_token=self.vercel_token,
                        )
                    else:
                        result = self.client.media.upload_large(
                            str(video_path), vercel_token=self.vercel_token
                        )

                    # Log completion
                    _log_progress(file_size, file_size)
                    return result

                media_response = await self._retry_with_backoff(
                    _upload_large, f"Upload large {video_path.name}"
                )

            # Extract media URL from response
            # Late SDK returns files array, Vercel Blob returns url directly
            media_url = None
            if hasattr(media_response, "url") and media_response.url:
                # Vercel Blob SDK response has url attribute directly
                media_url = str(media_response.url)
            elif hasattr(media_response, "files") and media_response.files:
                # Late SDK returns files array
                first_file = media_response.files[0]
                media_url = str(first_file.url) if hasattr(first_file, "url") else None
            elif isinstance(media_response, dict):
                # Dict response - check for url or files array
                media_url = media_response.get("url")
                if not media_url:
                    files = media_response.get("files", [])
                    if files:
                        media_url = files[0].get("url")

            if not media_url:
                raise UploadError(f"No media URL in response: {media_response}")

            logger.info(f"Upload successful, media URL: {media_url}")
            return media_url

        except ValidationError:
            raise
        except Exception as e:
            error_msg = f"Upload failed for {video_path.name}: {e}"
            logger.error(error_msg)
            raise UploadError(error_msg) from e

    async def publish(
        self,
        media_id: str,
        platforms: list[dict[str, str]],
        content: str | None = None,
        scheduled_time: datetime | None = None,
        platform_contents: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, str | list[str] | datetime | None]:
        """Create and publish/schedule a post via Late.dev.

        For multi-platform posts, the API may return partial success if some platforms
        fail. This method logs platform-specific failures but returns success if at
        least one platform succeeds.

        Args:
        ----
            media_id: Media ID from upload_media()
            platforms: List of dicts with "platform" and "account_id" keys
            content: Post content - used if platform_contents not provided
            scheduled_time: Optional UTC datetime for scheduled publishing
            platform_contents: Optional dict mapping platform name to content dict
                e.g. {"youtube": {"content": "...", "title": "..."}}

        Returns:
        -------
            Dictionary with post_id, status, scheduled_time, published_urls,
            platform_failures (if any platforms failed)

        Raises:
        ------
            ValidationError: If inputs invalid or scheduled_time in past
            PublishError: If post creation fails for ALL platforms

        Example:
        -------
            >>> result = await publisher.publish(
            ...     media_id="https://example.com/video.mp4",
            ...     platforms=[{"platform": "youtube", "account_id": "acc_1"}],
            ...     content="Amazing product! #ad"
            ... )

        """
        platform_names = [p.get("platform", "unknown") for p in platforms]
        logger.info(
            f"Publishing to {len(platforms)} platform(s): {', '.join(platform_names)}"
        )

        # Validate inputs
        if not media_id or not media_id.strip():
            raise ValidationError("media_id cannot be empty")
        if not platforms:
            raise ValidationError("platforms list cannot be empty")
        # Either content or platform_contents must be provided
        if not platform_contents and (not content or not content.strip()):
            raise ValidationError("content or platform_contents must be provided")

        # Validate each platform entry
        for idx, platform in enumerate(platforms):
            if not isinstance(platform, dict):
                raise ValidationError(f"Platform {idx} must be a dictionary")
            if "platform" not in platform:
                raise ValidationError(f"Platform {idx} missing 'platform' key")
            if "account_id" not in platform:
                raise ValidationError(f"Platform {idx} missing 'account_id' key")

        # Validate scheduled time is not in past
        if scheduled_time:
            if scheduled_time.tzinfo is None:
                raise ValidationError("scheduled_time must include timezone info")
            if scheduled_time < datetime.now(UTC):
                raise ValidationError(
                    f"scheduled_time cannot be in past: {scheduled_time}"
                )

        try:
            # Prepare post data for Late SDK
            publish_now = scheduled_time is None

            # Build SDK platforms with platformSpecificData for per-platform content
            sdk_platforms: list[dict[str, object]] = []
            main_content = content  # Default fallback

            for p in platforms:
                platform_name = p["platform"]
                platform_entry: dict[str, object] = {
                    "platform": platform_name,
                    "accountId": p["account_id"],
                }

                # Add platform-specific data if provided
                if platform_contents and platform_name in platform_contents:
                    pc = platform_contents[platform_name]
                    # Use first platform's content as main content
                    if main_content is None:
                        main_content = pc.get("content", "")

                    # Add customContent for this platform
                    platform_entry["customContent"] = pc.get("content", "")

                    # Add platformSpecificData for YouTube title
                    if platform_name == "youtube" and pc.get("title"):
                        platform_entry["platformSpecificData"] = {
                            "title": pc["title"],
                        }

                    # Add TikTok settings
                    if platform_name == "tiktok":
                        platform_entry["platformSpecificData"] = {
                            "tiktokSettings": {
                                "privacy_level": "PUBLIC_TO_EVERYONE",
                                "allow_comment": True,
                                "allow_duet": False,
                                "allow_stitch": False,
                                "content_preview_confirmed": True,
                                "express_consent_given": True,
                            }
                        }

                sdk_platforms.append(platform_entry)

            async def _create_post():
                post_data: dict[str, object] = {
                    "content": main_content,
                    "platforms": sdk_platforms,
                    "media_items": [{"type": "video", "url": media_id}],
                    "publish_now": publish_now,
                }

                # Add TikTok settings at top level (required by API)
                has_tiktok = any(
                    p.get("platform", "").lower() == "tiktok" for p in platforms
                )
                if has_tiktok:
                    post_data["tiktok_settings"] = {
                        "privacyLevel": "PUBLIC_TO_EVERYONE",
                        "mediaType": "video",
                        "commercialContentType": "brand_organic",
                    }

                if scheduled_time:
                    # Convert to UTC ISO format for API
                    post_data["scheduled_for"] = scheduled_time.astimezone(
                        UTC
                    ).isoformat()

                if asyncio.iscoroutinefunction(self.client.posts.create):
                    return await self.client.posts.create(**post_data)
                else:
                    return self.client.posts.create(**post_data)

            post_response = await self._retry_with_backoff(_create_post, "Create post")

            # Parse response (handle both Pydantic model and dict)
            logger.debug(f"Post response type: {type(post_response)}")
            logger.debug(f"Post response: {post_response}")

            # Extract post_id from response (PostCreateResponse has post.field_id)
            post_id = None
            post_obj = None
            if hasattr(post_response, "post") and post_response.post:
                post_obj = post_response.post
                post_id = getattr(post_obj, "field_id", None)
                if not post_id:
                    post_id = getattr(post_obj, "id", None)
            elif hasattr(post_response, "field_id"):
                post_id = post_response.field_id
            elif hasattr(post_response, "id"):
                post_id = post_response.id
            elif isinstance(post_response, dict):
                post = post_response.get("post", {})
                post_id = post.get("_id") or post.get("id", "")
            if not post_id:
                post_id = "unknown"

            # Get status from response or use default
            if post_obj and hasattr(post_obj, "status"):
                status = (
                    str(post_obj.status.value)
                    if hasattr(post_obj.status, "value")
                    else str(post_obj.status)
                )
            else:
                status = "published" if publish_now else "scheduled"

            # Extract URLs from platform results
            published_urls = []
            if post_obj and hasattr(post_obj, "platforms"):
                for p in post_obj.platforms or []:
                    url = getattr(p, "platformPostUrl", None)
                    if url:
                        published_urls.append(str(url))

            # Check for platform-specific failures in response
            platform_failures = []
            platform_results = None
            if hasattr(post_response, "platform_results"):
                platform_results = post_response.platform_results
            elif (
                isinstance(post_response, dict) and "platform_results" in post_response
            ):
                platform_results = post_response.get("platform_results", [])

            if platform_results:
                for platform_result in platform_results:
                    if hasattr(platform_result, "platform"):
                        platform_name = platform_result.platform
                        platform_status = getattr(platform_result, "status", "unknown")
                        error_msg = getattr(platform_result, "error", "Unknown error")
                    else:
                        platform_name = platform_result.get("platform", "unknown")
                        platform_status = platform_result.get("status", "unknown")
                        error_msg = platform_result.get("error", "Unknown error")

                    if platform_status in ("failed", "error"):
                        platform_failures.append(
                            {"platform": platform_name, "error": error_msg}
                        )
                        logger.warning(
                            f"Platform '{platform_name}' failed: {error_msg}"
                        )
                    else:
                        logger.info(f"Platform '{platform_name}' succeeded")

            # Check if response indicates partial failure
            errors = None
            if hasattr(post_response, "errors"):
                errors = post_response.errors
            elif isinstance(post_response, dict) and "errors" in post_response:
                errors = post_response.get("errors", [])

            if errors:
                for error in errors:
                    if hasattr(error, "platform"):
                        platform_name = error.platform
                        error_msg = getattr(error, "message", "Unknown error")
                    else:
                        platform_name = error.get("platform", "unknown")
                        error_msg = error.get("message", "Unknown error")
                    platform_failures.append(
                        {"platform": platform_name, "error": error_msg}
                    )
                    logger.warning(f"Platform '{platform_name}' failed: {error_msg}")

            # Build result
            result = {
                "post_id": post_id,
                "status": status,
                "scheduled_time": scheduled_time,
                "published_urls": published_urls,
            }

            # Add platform failures if any occurred
            if platform_failures:
                result["platform_failures"] = platform_failures
                failed_count = len(platform_failures)
                success_count = len(platforms) - failed_count
                logger.warning(
                    f"Post partial success: {failed_count} failed, {success_count} ok"
                )
            else:
                logger.info(
                    f"Post created successfully: {post_id} (status: {status}) "
                    f"on {len(platforms)} platform(s)"
                )

            # Log published URLs if available
            if published_urls:
                logger.info("Published URLs:")
                for url in published_urls:
                    logger.info(f"  - {url}")
            elif status == "published":
                logger.debug(
                    "No published URLs in API response yet (may take time to propagate)"
                )

            # For scheduled posts, log scheduled time in user's local timezone
            if scheduled_time:
                # Convert to local timezone for user-friendly display
                try:
                    import zoneinfo

                    local_tz = zoneinfo.ZoneInfo(
                        "America/New_York"
                    )  # Default to Eastern
                    local_time = scheduled_time.astimezone(local_tz)
                    logger.info(
                        f"Scheduled for: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                    )
                except Exception:
                    # Fallback if zoneinfo not available
                    time_str = scheduled_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                    logger.info(f"Scheduled for: {time_str}")

            return result

        except ValidationError:
            raise
        except Exception as e:
            error_msg = f"Failed to create post on all platforms: {e}"
            logger.error(error_msg)
            # Log platform details for debugging (without sensitive data)
            logger.error(f"Target platforms: {platform_names}")
            content_len = len(content) if content else 0
            logger.error(f"Content length: {content_len} chars")
            raise PublishError(error_msg) from e

    async def get_status(
        self,
        post_id: str,
        local_timezone: str | None = None,
    ) -> dict[str, str | list | datetime | None]:
        """Fetch post status from Late.dev.

        Args:
        ----
            post_id: Post ID from publish()
            local_timezone: Optional timezone for timestamp conversion
                (e.g., 'America/New_York')

        Returns:
        -------
            Dictionary with post_id, status, scheduled_time, published_time,
            published_urls, error_message, platform_results

        Raises:
        ------
            ValidationError: If post_id is empty
            PublisherError: If status check fails (won't raise, returns error in dict)

        Example:
        -------
            >>> status = await publisher.get_status("post_abc123", "America/Chicago")
            >>> print(f"Status: {status['status']}")
            >>> if status['published_urls']:
            ...     print(f"URLs: {status['published_urls']}")

        """
        if not post_id or not post_id.strip():
            raise ValidationError("post_id cannot be empty")

        logger.info(f"Fetching status for post: {post_id}")

        try:

            async def _get_post():
                if asyncio.iscoroutinefunction(self.client.posts.get):
                    return await self.client.posts.get(post_id)
                else:
                    return self.client.posts.get(post_id)

            post_response = await self._retry_with_backoff(
                _get_post, f"Get status for {post_id}"
            )

            # Handle both Pydantic model and dict responses (like publish() does)
            logger.debug(f"Post response type: {type(post_response)}")

            # Extract post object from response wrapper if present
            post_data = None
            if hasattr(post_response, "post") and post_response.post:
                post_data = post_response.post
            else:
                post_data = post_response

            # Extract status - handle both Pydantic model and dict
            status = "unknown"
            if hasattr(post_data, "status"):
                status_val = post_data.status
                if hasattr(status_val, "value"):
                    status = str(status_val.value)
                else:
                    status = str(status_val)
            elif isinstance(post_data, dict):
                status = post_data.get("status", "unknown")

            # Parse timestamps with optional timezone conversion
            scheduled_time = None
            published_time = None
            scheduled_time_local = None
            published_time_local = None

            # Get scheduled_time from model or dict
            scheduled_time_str = None
            if hasattr(post_data, "scheduledFor"):
                scheduled_time_str = post_data.scheduledFor
            elif hasattr(post_data, "scheduled_time"):
                scheduled_time_str = post_data.scheduled_time
            elif isinstance(post_data, dict):
                scheduled_time_str = post_data.get("scheduled_time") or post_data.get(
                    "scheduledFor"
                )

            if scheduled_time_str:
                try:
                    if isinstance(scheduled_time_str, str):
                        dt = datetime.fromisoformat(scheduled_time_str)
                        scheduled_time = dt.astimezone(UTC)
                    elif isinstance(scheduled_time_str, datetime):
                        scheduled_time = scheduled_time_str.astimezone(UTC)
                except Exception as e:
                    logger.debug(f"Could not parse scheduled_time: {e}")

                # Convert to local timezone if specified
                if scheduled_time and local_timezone:
                    try:
                        import zoneinfo

                        local_tz = zoneinfo.ZoneInfo(local_timezone)
                        scheduled_time_local = scheduled_time.astimezone(local_tz)
                    except Exception as e:
                        logger.debug(
                            f"Could not convert to timezone {local_timezone}: {e}"
                        )

            # Get published_time from model or dict
            published_time_str = None
            if hasattr(post_data, "publishedAt"):
                published_time_str = post_data.publishedAt
            elif hasattr(post_data, "published_time"):
                published_time_str = post_data.published_time
            elif isinstance(post_data, dict):
                published_time_str = post_data.get("published_time") or post_data.get(
                    "publishedAt"
                )

            if published_time_str:
                try:
                    if isinstance(published_time_str, str):
                        dt = datetime.fromisoformat(published_time_str)
                        published_time = dt.astimezone(UTC)
                    elif isinstance(published_time_str, datetime):
                        published_time = published_time_str.astimezone(UTC)
                except Exception as e:
                    logger.debug(f"Could not parse published_time: {e}")

                # Convert to local timezone if specified
                if published_time and local_timezone:
                    try:
                        import zoneinfo

                        local_tz = zoneinfo.ZoneInfo(local_timezone)
                        published_time_local = published_time.astimezone(local_tz)
                    except Exception as e:
                        logger.debug(
                            f"Could not convert to timezone {local_timezone}: {e}"
                        )

            # Extract platform-specific results if available
            platform_results: list[Any] = []
            if hasattr(post_data, "platforms"):
                platform_results = post_data.platforms or []
            elif hasattr(post_data, "platform_results"):
                platform_results = post_data.platform_results or []
            elif isinstance(post_data, dict):
                platform_results = (
                    post_data.get("platform_results")
                    or post_data.get("platforms")
                    or []
                )

            # Extract published URLs from platforms
            published_urls = []
            if hasattr(post_data, "platforms"):
                for p in post_data.platforms or []:
                    url = getattr(p, "platformPostUrl", None)
                    if url:
                        published_urls.append(str(url))
            elif isinstance(post_data, dict):
                published_urls = post_data.get("urls", [])

            # Extract error message
            error_message = None
            if hasattr(post_data, "error"):
                error_message = post_data.error
            elif isinstance(post_data, dict):
                error_message = post_data.get("error")

            status_info = {
                "post_id": post_id,
                "status": status,
                "scheduled_time": scheduled_time,
                "published_time": published_time,
                "scheduled_time_local": scheduled_time_local,
                "published_time_local": published_time_local,
                "published_urls": published_urls,
                "error_message": error_message,
                "platform_results": platform_results,
            }

            # Log status details
            logger.info(f"Post status: {status_info['status']}")
            published_urls_list = status_info["published_urls"]
            if published_urls_list and isinstance(published_urls_list, list):
                logger.debug(f"Published URLs: {len(published_urls_list)} URL(s)")
            if status_info["error_message"]:
                logger.warning(f"Post error: {status_info['error_message']}")
            if platform_results:
                count = len(platform_results)
                logger.debug(f"Platform results available for {count} platform(s)")

            return status_info

        except Exception as e:
            # Don't raise exception for status check failures - continue gracefully
            logger.warning(f"Failed to fetch status for {post_id}: {e}", exc_info=False)
            return {
                "post_id": post_id,
                "status": "unknown",
                "scheduled_time": None,
                "published_time": None,
                "scheduled_time_local": None,
                "published_time_local": None,
                "published_urls": None,
                "error_message": str(e),
                "platform_results": None,
            }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Async context manager exit - cleanup session."""
        await self._close_session()
