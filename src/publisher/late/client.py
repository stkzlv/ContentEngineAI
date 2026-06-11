"""Late.dev publisher client implementation.

This module provides the concrete implementation of BasePublisher for the Late.dev
scheduling service, supporting multi-platform video publishing to YouTube, TikTok,
Instagram, and other social media platforms.
"""

import asyncio
import logging
import zoneinfo
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
from src.publisher.constants import SDK_LIST_PAGE_SIZE
from src.publisher.models import FirstCommentConfig, TikTokContentSettings
from src.publisher.registry import register_publisher
from src.video.config.constants import (
    DEFAULT_EXPONENTIAL_BACKOFF_BASE,
    LATE_DEFAULT_RETRY_AFTER_SEC,
    LATE_DIRECT_UPLOAD_MAX_BYTES,
    LATE_MAX_UPLOAD_SIZE_BYTES,
)

logger = logging.getLogger(__name__)


def _extract_account_id(acc: Any) -> str | None:
    """Pull the account id string from a platform target's accountId.

    The SDK returns accountId as a nested SocialAccount object (``field_id``),
    a dict (``_id``), or already a string depending on the endpoint.
    """
    if acc is None:
        return None
    if isinstance(acc, str):
        return acc
    if isinstance(acc, dict):
        return acc.get("_id") or acc.get("field_id") or acc.get("id")
    return getattr(acc, "field_id", None) or getattr(acc, "_id", None)


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
        tiktok_settings: TikTokContentSettings | None = None,
        first_comment_config: FirstCommentConfig | None = None,
    ):
        """Initialize Late.dev publisher client.

        Args:
        ----
            api_key: Late.dev API key (required)
            vercel_token: Vercel token for large file uploads (optional)
            session: Aiohttp session for HTTP requests (optional, created if None)
            timeout: Request timeout in seconds (default: 120.0)
            max_retries: Maximum retry attempts for transient failures (default: 3)
            tiktok_settings: TikTok content disclosure settings (optional)
            first_comment_config: First-comment config for affiliate links (optional)

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
        self.tiktok_settings = tiktok_settings or TikTokContentSettings()
        self.first_comment_config = first_comment_config or FirstCommentConfig()

        # Initialize Late SDK client
        try:
            self.client = Late(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        except (TypeError, ValueError) as e:
            logger.error("Failed to initialize Late client: %s", e)
            raise ValidationError(f"Invalid Late API configuration: {e}") from e

        # HTTP session for additional requests if needed
        self._session = session
        self._should_close_session = session is None

        logger.info("Late publisher: %ss timeout, %d retries", timeout, max_retries)
        logger.debug(
            "%s", f"API key: {api_key[:4]}..." if len(api_key) > 4 else "API key set"
        )
        vercel_status = "set" if vercel_token else "NOT SET"
        logger.debug("Vercel token: %s", vercel_status)

    async def _call_sdk(self, method: Callable, *args, **kwargs):
        """Call an SDK method, handling both sync and async variants.

        Args:
        ----
            method: SDK method to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
        -------
            Result from the SDK method

        """
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        return method(*args, **kwargs)

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
                    logger.info("%s succeeded on attempt %d", operation_name, attempt)
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
                            "%s rate limited (429), waiting %ds before retry "
                            "(attempt %d/%d)",
                            operation_name,
                            retry_after,
                            attempt,
                            self.max_retries,
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
                            "%s server error (%d), retry %d/%d in %ds: %s",
                            operation_name,
                            e.status,
                            attempt,
                            self.max_retries,
                            delay,
                            e.message,
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
                        "%s failed (HTTP %d), retrying in %ds " "(attempt %d/%d): %s",
                        operation_name,
                        e.status,
                        delay,
                        attempt,
                        self.max_retries,
                        e.message,
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
                        "%s connection error, retry %d/%d in %ds: %s",
                        operation_name,
                        attempt,
                        self.max_retries,
                        delay,
                        e,
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
                        "%s timed out after %ss, retrying in %ds " "(attempt %d/%d)",
                        operation_name,
                        self.timeout,
                        delay,
                        attempt,
                        self.max_retries,
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
                        "%s client error, retry %d/%d in %ds: %s",
                        operation_name,
                        attempt,
                        self.max_retries,
                        delay,
                        e,
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
                    "%s unexpected error: %s: %s",
                    operation_name,
                    err_type,
                    e,
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
        retry_after = LATE_DEFAULT_RETRY_AFTER_SEC

        # Check if error has headers attribute
        if hasattr(error, "headers") and error.headers:
            retry_header = error.headers.get("Retry-After") or error.headers.get(
                "retry-after"
            )
            if retry_header:
                try:
                    retry_after = int(retry_header)
                    logger.debug("Extracted Retry-After header: %ds", retry_after)
                except ValueError:
                    # Retry-After might be HTTP date format, not seconds
                    logger.debug("Could not parse Retry-After header: %s", retry_header)

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
                return await self._call_sdk(self.client.accounts.list)

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
                return await self._call_sdk(self.client.accounts.list)

            raw_accounts = await self._retry_with_backoff(
                _fetch_accounts, "Fetch accounts"
            )

            # Debug: Log raw response structure
            logger.debug("Raw accounts type: %s", type(raw_accounts))
            logger.debug("Raw accounts value: %s", raw_accounts)

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
                    # Pydantic model or object. Late SDK returns platform as
                    # a Platform5 enum; unwrap to its string value so callers
                    # get a plain "instagram"/"tiktok"/"youtube" string.
                    platform = getattr(account, "platform", "unknown")
                    if hasattr(platform, "value"):
                        platform = platform.value
                    account_dict = {
                        "platform": platform,
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

            logger.info("Found %d connected accounts", len(accounts))
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
        logger.info("Fetching posts from Late.dev (status=%s)", status or "all")

        try:
            # Fetch all posts with pagination
            all_posts_list = []
            page = 1
            page_size = SDK_LIST_PAGE_SIZE

            while True:

                async def _fetch_posts_page(current_page=page):
                    return await self._call_sdk(
                        self.client.posts.list,
                        page=current_page,
                        limit=page_size,
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
                "Fetched %d total posts across %d page(s)", len(all_posts_list), page
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

            logger.info("Found %d post(s)", len(posts))
            return posts

        except Exception as e:
            error_msg = f"Failed to fetch posts: {e}"
            logger.error(error_msg)
            if "401" in str(e) or "403" in str(e):
                raise AuthenticationError(f"Authentication expired: {e}") from e
            raise PublishError(error_msg) from e

    async def get_post_platforms(self, post_id: str) -> list[dict[str, Any]]:
        """Return normalized per-platform targets for a post.

        Each item: ``{platform, platform_post_id, account_id, status}``. Used to
        find the platform post id and account needed to read inbox comments.
        ``platform_post_id`` is None until the platform actually publishes.
        """

        async def _get_post():
            return await self._call_sdk(self.client.posts.get, post_id)

        resp = await self._retry_with_backoff(_get_post, f"Get platforms for {post_id}")
        post_data = resp.post if getattr(resp, "post", None) else resp
        raw = getattr(post_data, "platforms", None)
        if raw is None and isinstance(post_data, dict):
            raw = post_data.get("platforms")
        out: list[dict[str, Any]] = []
        for p in raw or []:
            if isinstance(p, dict):
                platform = str(p.get("platform", "")).split(".")[-1].lower()
                ppid = p.get("platformPostId")
                acc = p.get("accountId")
                status = p.get("status")
            else:
                platform = str(getattr(p, "platform", "")).split(".")[-1].lower()
                ppid = getattr(p, "platformPostId", None)
                acc = getattr(p, "accountId", None)
                raw_status = getattr(p, "status", None)
                status = str(raw_status).split(".")[-1].lower() if raw_status else None
            out.append(
                {
                    "platform": platform,
                    "platform_post_id": ppid,
                    "account_id": _extract_account_id(acc),
                    "status": status,
                }
            )
        return out

    async def get_unpublished_media_urls(self) -> set[str]:
        """Return media URLs referenced by posts that aren't fully published.

        Used by blob retention to protect uploads that the scheduling service
        still needs (scheduled/pending/draft/failed/partial posts). The
        normalized ``list_posts`` drops mediaItems, so this reads the raw
        paginated SDK response.
        """
        urls: set[str] = set()
        page = 1
        while True:

            async def _fetch(current_page=page):
                return await self._call_sdk(
                    self.client.posts.list, page=current_page, limit=SDK_LIST_PAGE_SIZE
                )

            resp = await self._retry_with_backoff(_fetch, f"List posts page {page}")
            data = (
                resp.model_dump(by_alias=True, mode="json")
                if hasattr(resp, "model_dump")
                else resp
            )
            posts = data.get("posts") or [] if isinstance(data, dict) else []
            for post in posts:
                status = str(post.get("status") or "").lower()
                if status == "published":
                    continue
                for item in post.get("mediaItems") or []:
                    url = item.get("url") if isinstance(item, dict) else None
                    if url:
                        urls.add(str(url))
            if len(posts) < SDK_LIST_PAGE_SIZE:
                return urls
            page += 1

    async def get_post_comments(
        self, platform_post_id: str, account_id: str, limit: int = 25
    ) -> list[dict[str, Any]]:
        """Return inbox comments on a published post (list of dicts).

        Each comment has ``message`` and a ``from`` object with ``isOwner``,
        which flags the account owner's own comment (our first comment).
        """

        async def _get():
            return await self._call_sdk(
                self.client.comments.get_inbox_post_comments,
                post_id=platform_post_id,
                account_id=account_id,
                limit=limit,
            )

        resp = await self._retry_with_backoff(
            _get, f"Get comments for {platform_post_id}"
        )
        if isinstance(resp, dict):
            return resp.get("comments") or []
        return getattr(resp, "comments", None) or []

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
        logger.info("Deleting post: %s", post_id)

        try:

            async def _delete_post():
                return await self._call_sdk(self.client.posts.delete, post_id)

            await self._retry_with_backoff(_delete_post, "Delete post")
            logger.info("Post %s deleted successfully", post_id)
            return True

        except Exception as e:
            error_msg = f"Failed to delete post {post_id}: {e}"
            logger.error(error_msg)
            if "401" in str(e) or "403" in str(e):
                raise AuthenticationError(f"Authentication expired: {e}") from e
            if "404" in str(e):
                logger.warning("Post %s not found (may already be deleted)", post_id)
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
        logger.info("Uploading video: %s", video_path)

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
        logger.info("File size: %.2f MB", file_size_mb)

        # Check size limits
        if file_size > LATE_MAX_UPLOAD_SIZE_BYTES:
            max_mb = LATE_MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
            raise ValidationError(
                f"File exceeds Late.dev {max_mb:.0f} MB limit: {file_size_mb:.2f} MB"
            )

        # Validate video file extension
        valid_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
        if video_path.suffix.lower() not in valid_extensions:
            extensions_str = ", ".join(valid_extensions)
            logger.warning(
                "Video file has unusual extension: %s. Expected one of: %s",
                video_path.suffix,
                extensions_str,
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
                logger.info("Upload: %d%% (%.1f/%.1f MB)", progress_pct, up_mb, tot_mb)
                last_logged_progress = progress_pct

            # Call user-provided callback if any
            if progress_callback:
                progress_callback(bytes_uploaded, total_bytes)

        try:
            # Small file upload (≤4 MB) - uses direct Late API
            if file_size <= LATE_DIRECT_UPLOAD_MAX_BYTES:
                logger.info("Using direct upload for small file")

                async def _upload_small():
                    result = await self._call_sdk(
                        self.client.media.upload, str(video_path)
                    )
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
                    logger.info("Starting large file upload...")
                    result = await self._call_sdk(
                        self.client.media.upload_large,
                        str(video_path),
                        vercel_token=self.vercel_token,
                    )
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

            logger.info("Upload successful, media URL: %s", media_url)
            return media_url

        except ValidationError:
            raise
        except Exception as e:
            error_msg = f"Upload failed for {video_path.name}: {e}"
            logger.error(error_msg)
            raise UploadError(error_msg) from e

    def _build_sdk_platforms(
        self,
        platforms: list[dict[str, str]],
        content: str | None,
        platform_contents: dict[str, dict[str, str]] | None,
    ) -> tuple[list[dict[str, object]], str | None]:
        """Build SDK platform entries with per-platform content and TikTok settings.

        Returns
        -------
            Tuple of (sdk_platforms list, resolved main_content)

        """
        sdk_platforms: list[dict[str, object]] = []
        main_content = content

        for p in platforms:
            platform_name = p["platform"]
            platform_entry: dict[str, object] = {
                "platform": platform_name,
                "accountId": p["account_id"],
            }

            if platform_contents and platform_name in platform_contents:
                pc = platform_contents[platform_name]
                if main_content is None:
                    main_content = pc.get("content", "")
                platform_entry["customContent"] = pc.get("content", "")

                if platform_name == "youtube":
                    yt_psd: dict[str, object] = {"containsSyntheticMedia": True}
                    if pc.get("title"):
                        yt_psd["title"] = pc["title"]
                    platform_entry["platformSpecificData"] = yt_psd
                if platform_name == "tiktok":
                    platform_entry["platformSpecificData"] = {
                        "tiktokSettings": self.tiktok_settings.to_sdk_dict()
                    }

                # Attach first comment via platformSpecificData
                first_comment = pc.get("first_comment")
                if first_comment and platform_name != "tiktok":
                    raw_psd = platform_entry.get("platformSpecificData")
                    if isinstance(raw_psd, dict):
                        raw_psd["firstComment"] = first_comment
                    else:
                        platform_entry["platformSpecificData"] = {
                            "firstComment": first_comment,
                        }

            # Always disclose AI-generated content on YouTube, even without
            # platform-specific content (YouTube policy requires the flag).
            if (
                platform_name == "youtube"
                and "platformSpecificData" not in platform_entry
            ):
                platform_entry["platformSpecificData"] = {
                    "containsSyntheticMedia": True,
                }

            # Add TikTok settings even without platform-specific content
            if (
                platform_name == "tiktok"
                and "platformSpecificData" not in platform_entry
            ):
                platform_entry["platformSpecificData"] = {
                    "tiktokSettings": self.tiktok_settings.to_sdk_dict()
                }

            sdk_platforms.append(platform_entry)

        return sdk_platforms, main_content

    def _parse_publish_response(
        self,
        post_response: Any,
        publish_now: bool,
    ) -> dict[str, Any]:
        """Parse post creation response into standardized result dict.

        Returns
        -------
            Dict with post_id, status, published_urls keys

        """
        logger.debug("Post response type: %s", type(post_response))
        logger.debug("Post response: %s", post_response)

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

        if post_obj and hasattr(post_obj, "status"):
            status = (
                str(post_obj.status.value)
                if hasattr(post_obj.status, "value")
                else str(post_obj.status)
            )
        else:
            status = "published" if publish_now else "scheduled"

        published_urls = []
        if post_obj and hasattr(post_obj, "platforms"):
            for p in post_obj.platforms or []:
                url = getattr(p, "platformPostUrl", None)
                if url:
                    published_urls.append(str(url))

        return {
            "post_id": post_id,
            "status": status,
            "published_urls": published_urls,
        }

    def _extract_platform_failures(
        self,
        post_response: Any,
    ) -> list[dict[str, str]]:
        """Extract platform-specific failures from post response.

        Returns
        -------
            List of dicts with platform and error keys

        """
        failures: list[dict[str, str]] = []

        # Check platform_results attribute
        platform_results = None
        if hasattr(post_response, "platform_results"):
            platform_results = post_response.platform_results
        elif isinstance(post_response, dict) and "platform_results" in post_response:
            platform_results = post_response.get("platform_results", [])

        if platform_results:
            for pr in platform_results:
                if hasattr(pr, "platform"):
                    name = pr.platform
                    pr_status = getattr(pr, "status", "unknown")
                    error = getattr(pr, "error", "Unknown error")
                else:
                    name = pr.get("platform", "unknown")
                    pr_status = pr.get("status", "unknown")
                    error = pr.get("error", "Unknown error")

                if pr_status in ("failed", "error"):
                    failures.append({"platform": name, "error": error})
                    logger.warning("Platform '%s' failed: %s", name, error)
                else:
                    logger.info("Platform '%s' succeeded", name)

        # Check errors attribute
        errors = None
        if hasattr(post_response, "errors"):
            errors = post_response.errors
        elif isinstance(post_response, dict) and "errors" in post_response:
            errors = post_response.get("errors", [])

        if errors:
            for err in errors:
                if hasattr(err, "platform"):
                    name = err.platform
                    msg = getattr(err, "message", "Unknown error")
                else:
                    name = err.get("platform", "unknown")
                    msg = err.get("message", "Unknown error")
                failures.append({"platform": name, "error": msg})
                logger.warning("Platform '%s' failed: %s", name, msg)

        return failures

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
        platforms_str = ", ".join(platform_names)
        logger.info("Publishing to %d platform(s): %s", len(platforms), platforms_str)

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
            publish_now = scheduled_time is None
            sdk_platforms, main_content = self._build_sdk_platforms(
                platforms, content, platform_contents
            )

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
                    post_data["tiktok_settings"] = (
                        self.tiktok_settings.to_top_level_dict()
                    )

                if scheduled_time:
                    # Convert to UTC ISO format for API
                    post_data["scheduled_for"] = scheduled_time.astimezone(
                        UTC
                    ).isoformat()

                return await self._call_sdk(self.client.posts.create, **post_data)

            post_response = await self._retry_with_backoff(_create_post, "Create post")

            # Parse response and extract failures
            parsed = self._parse_publish_response(post_response, publish_now)
            post_id = parsed["post_id"]
            status = parsed["status"]
            published_urls = parsed["published_urls"]
            platform_failures = self._extract_platform_failures(post_response)

            # Build result
            result = {
                "post_id": post_id,
                "status": status,
                "scheduled_time": scheduled_time,
                "published_urls": published_urls,
            }

            if platform_failures:
                result["platform_failures"] = platform_failures
                failed_count = len(platform_failures)
                success_count = len(platforms) - failed_count
                logger.warning(
                    "Post partial success: %d failed, %d ok",
                    failed_count,
                    success_count,
                )
            else:
                logger.info(
                    "Post created successfully: %s (status: %s) on %d platform(s)",
                    post_id,
                    status,
                    len(platforms),
                )

            # Log published URLs if available
            if published_urls:
                logger.info("Published URLs:")
                for url in published_urls:
                    logger.info("  - %s", url)
            elif status == "published":
                logger.debug(
                    "No published URLs in API response yet (may take time to propagate)"
                )

            # For scheduled posts, log scheduled time in user's local timezone
            if scheduled_time:
                # Convert to local timezone for user-friendly display
                try:
                    local_tz = zoneinfo.ZoneInfo(
                        "America/New_York"
                    )  # Default to Eastern
                    local_time = scheduled_time.astimezone(local_tz)
                    time_str = local_time.strftime("%Y-%m-%d %H:%M:%S %Z")
                    logger.info("Scheduled for: %s", time_str)
                except (ImportError, KeyError):
                    # Fallback if zoneinfo not available
                    time_str = scheduled_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                    logger.info("Scheduled for: %s", time_str)

            return result

        except ValidationError:
            raise
        except Exception as e:
            error_msg = f"Failed to create post on all platforms: {e}"
            logger.error(error_msg)
            # Log platform details for debugging (without sensitive data)
            logger.error("Target platforms: %s", platform_names)
            content_len = len(content) if content else 0
            logger.error("Content length: %d chars", content_len)
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

        logger.info("Fetching status for post: %s", post_id)

        try:

            async def _get_post():
                return await self._call_sdk(self.client.posts.get, post_id)

            post_response = await self._retry_with_backoff(
                _get_post, f"Get status for {post_id}"
            )

            # Handle both Pydantic model and dict responses (like publish() does)
            logger.debug("Post response type: %s", type(post_response))

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
                except (ValueError, TypeError) as e:
                    logger.debug("Could not parse scheduled_time: %s", e)

                # Convert to local timezone if specified
                if scheduled_time and local_timezone:
                    try:
                        local_tz = zoneinfo.ZoneInfo(local_timezone)
                        scheduled_time_local = scheduled_time.astimezone(local_tz)
                    except (KeyError, ImportError) as e:
                        logger.debug(
                            "Could not convert to timezone %s: %s", local_timezone, e
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
                except (ValueError, TypeError) as e:
                    logger.debug("Could not parse published_time: %s", e)

                # Convert to local timezone if specified
                if published_time and local_timezone:
                    try:
                        local_tz = zoneinfo.ZoneInfo(local_timezone)
                        published_time_local = published_time.astimezone(local_tz)
                    except (KeyError, ImportError) as e:
                        logger.debug(
                            "Could not convert to timezone %s: %s", local_timezone, e
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
            logger.info("Post status: %s", status_info["status"])
            published_urls_list = status_info["published_urls"]
            if published_urls_list and isinstance(published_urls_list, list):
                logger.debug("Published URLs: %d URL(s)", len(published_urls_list))
            if status_info["error_message"]:
                logger.warning("Post error: %s", status_info["error_message"])
            if platform_results:
                count = len(platform_results)
                logger.debug("Platform results available for %d platform(s)", count)

            return status_info

        except Exception as e:
            # Don't raise exception for status check failures - continue gracefully
            logger.warning(
                "Failed to fetch status for %s: %s", post_id, e, exc_info=False
            )
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
