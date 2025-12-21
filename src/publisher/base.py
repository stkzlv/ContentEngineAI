"""Base abstract interface for video publishing services.

This module defines the common interface that all publisher providers
must implement, enabling provider-agnostic video publishing with easy switching
between scheduling services (Late.dev, Buffer, Hootsuite, etc.).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class PublisherProvider(Enum):
    """Supported video publishing providers."""

    LATE = "late"
    BUFFER = "buffer"
    HOOTSUITE = "hootsuite"
    LATER = "later"


class PublisherError(Exception):
    """Base exception for publisher errors."""

    pass


class AuthenticationError(PublisherError):
    """Raised when authentication fails (invalid API key, expired token, etc.)."""

    pass


class UploadError(PublisherError):
    """Raised when video upload fails."""

    pass


class PublishError(PublisherError):
    """Raised when post creation/publishing fails."""

    pass


class ValidationError(PublisherError):
    """Raised when input validation fails."""

    pass


class BasePublisher(ABC):
    """Abstract base class for video publishing services.

    This defines the common interface that all provider-specific implementations
    must follow, ensuring consistency across different publishing/scheduling services.

    All publishers follow a common workflow:
    1. Authenticate with provider API
    2. Discover connected social media accounts
    3. Upload video file to provider
    4. Create post with metadata and schedule
    5. Check post status and retrieve published URLs
    """

    @property
    @abstractmethod
    def provider(self) -> PublisherProvider:
        """Return the provider this publisher uses.

        Returns:
        -------
            PublisherProvider enum value identifying this provider

        Example:
        -------
            >>> publisher.provider
            PublisherProvider.LATE

        """
        pass

    @abstractmethod
    async def authenticate(self) -> bool:
        """Validate API credentials and establish connection.

        This method should verify that the provided API key/credentials are valid
        and the service is accessible. It should be called before any other operations.

        Returns:
        -------
            True if authentication succeeds, False otherwise

        Raises:
        ------
            AuthenticationError: If credentials are invalid or expired
            PublisherError: If authentication request fails due to network issues

        Example:
        -------
            >>> is_authenticated = await publisher.authenticate()
            >>> if is_authenticated:
            ...     print("Ready to publish")

        Implementation Notes
        --------------------
            - Verify API key format before making API call
            - Cache authentication status to avoid repeated calls
            - Log authentication failures with actionable error messages
            - Do not expose API keys in error messages (show first 4 chars only)

        """
        pass

    @abstractmethod
    async def get_accounts(self) -> list[dict[str, str]]:
        """Fetch all connected social media accounts from the provider.

        Returns list of accounts with platform type, username/handle, and account ID.
        This information is used to map target platforms to provider account IDs.

        Returns:
        -------
            List of account dictionaries, each containing:
                - platform: Platform type (e.g., "youtube", "tiktok", "instagram")
                - username: Account username or handle
                - account_id: Provider's internal account identifier
                - status: Account status (e.g., "active", "disconnected")

        Raises:
        ------
            AuthenticationError: If not authenticated or credentials expired
            PublisherError: If account fetching fails after retries

        Example:
        -------
            >>> accounts = await publisher.get_accounts()
            >>> for account in accounts:
            ...     print(f"{account['platform']}: @{account['username']}")
            youtube: @MyChannel
            tiktok: @mychannel
            instagram: @my.channel

        Implementation Notes
        --------------------
            - Retry up to 3 times on transient failures (network timeout, 5xx)
            - Return empty list if no accounts connected
            - Include account status to indicate disconnected accounts
            - Cache results for short duration to reduce API calls

        """
        pass

    @abstractmethod
    async def upload_media(
        self,
        video_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        """Upload video file to the publishing provider.

        Handles both small and large file uploads based on provider limits.
        Returns media ID that can be used in subsequent publish() calls.

        Args:
        ----
            video_path: Path to the video file to upload
            progress_callback: Optional callback function for progress updates
                               Called with (bytes_uploaded, total_bytes) for large files

        Returns:
        -------
            Media ID (string) assigned by the provider for the uploaded video

        Raises:
        ------
            ValidationError: If file doesn't exist, is unreadable, or exceeds
                size limits
            UploadError: If upload fails after retries
            PublisherError: If provider API returns unexpected error

        Example:
        -------
            >>> def progress(uploaded, total):
            ...     pct = (uploaded / total) * 100
            ...     print(f"Upload: {pct:.1f}%")
            >>> media_id = await publisher.upload_media(
            ...     Path("outputs/B0ASIN/video.mp4"),
            ...     progress_callback=progress
            ... )
            Upload: 25.0%
            Upload: 50.0%
            Upload: 75.0%
            Upload: 100.0%
            >>> print(f"Media ID: {media_id}")

        Implementation Notes
        --------------------
            - Validate file exists and is readable before upload attempt
            - Check file size against provider limits (e.g., Late: 500MB max)
            - Use direct upload for small files (e.g., ≤4MB)
            - Use chunked/resumable upload for large files (e.g., >4MB)
            - Stream file in chunks (32KB) to minimize memory usage
            - Retry up to 3 times with exponential backoff (2s, 4s, 8s)
            - Call progress_callback every 10% completion for large files
            - Log upload start and completion with file size

        """
        pass

    @abstractmethod
    async def publish(
        self,
        media_id: str,
        platforms: list[dict[str, str]],
        content: str,
        scheduled_time: datetime | None = None,
        platform_contents: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, str | list[str] | datetime | None]:
        """Create and publish/schedule a post to social media platforms.

        Args:
        ----
            media_id: Media ID returned from upload_media()
            platforms: List of platform dicts with "platform" and "account_id" keys
                      Example: [{"platform": "youtube", "account_id": "acc_123"}]
            content: Post content (title/description/caption with metadata)
            scheduled_time: Optional datetime for scheduled publishing (UTC)
                           If None, publishes immediately
            platform_contents: Optional per-platform content dict mapping platform name
                             to content dict with "content" and optional "title" keys

        Returns:
        -------
            Dictionary containing:
                - post_id: Provider's post ID (string)
                - status: Post status (e.g., "scheduled", "published", "failed")
                - scheduled_time: Scheduled time if applicable (datetime or None)
                - published_urls: List of published post URLs (if immediate publish)

        Raises:
        ------
            ValidationError: If media_id invalid, platforms empty, or
                scheduled_time in past
            PublishError: If post creation fails
            PublisherError: If API request fails

        Example:
        -------
            >>> result = await publisher.publish(
            ...     media_id="media_abc123",
            ...     platforms=[
            ...         {"platform": "youtube", "account_id": "acc_yt1"},
            ...         {"platform": "tiktok", "account_id": "acc_tt1"}
            ...     ],
            ...     content="Check out this product! #ad",
            ...     scheduled_time=datetime(2025, 12, 20, 14, 0, tzinfo=timezone.utc)
            ... )
            >>> print(f"Post scheduled: {result['post_id']}")

        Implementation Notes
        --------------------
            - Validate scheduled_time is not in the past (if provided)
            - Convert all datetimes to UTC for provider API
            - Create separate platform objects for multi-platform posts
            - Set publish_now=True if scheduled_time is None
            - Return post ID immediately after creation
            - For immediate posts, attempt to fetch published URLs
            - Log platform-specific errors (e.g., TikTok upload failed)
            - Continue multi-platform post even if one platform fails

        """
        pass

    @abstractmethod
    async def get_status(self, post_id: str) -> dict[str, str | list | datetime | None]:
        """Fetch the current status of a published or scheduled post.

        Args:
        ----
            post_id: Post ID returned from publish()

        Returns:
        -------
            Dictionary containing:
                - post_id: Post ID (string)
                - status: Current status (e.g., "scheduled", "published", "failed")
                - scheduled_time: Scheduled publish time if applicable (datetime)
                - published_time: Actual publish time if published (datetime)
                - published_urls: List of platform URLs if published (list[str])
                - error_message: Error description if failed (string or None)

        Raises:
        ------
            ValidationError: If post_id is invalid or empty
            PublisherError: If status check fails

        Example:
        -------
            >>> status = await publisher.get_status("post_abc123")
            >>> if status["status"] == "published":
            ...     print(f"Published at: {status['published_time']}")
            ...     for url in status["published_urls"]:
            ...         print(f"  - {url}")
            Published at: 2025-12-16 14:30:00+00:00
              - https://youtube.com/watch?v=abc123
              - https://tiktok.com/@user/video/123456

        Implementation Notes
        --------------------
            - Return cached status if recently checked (< 30 seconds ago)
            - Convert all timestamps to UTC datetime objects
            - Extract published URLs from provider response
            - Return None for fields not available from provider
            - Log errors but don't raise exception if status check fails
            - Include error_message field if post failed

        """
        pass

    @abstractmethod
    async def list_posts(self, status: str | None = None) -> list[dict[str, Any]]:
        """Fetch all posts from the provider, optionally filtered by status.

        Args:
        ----
            status: Optional status filter ('scheduled', 'published', 'failed')
                   If None, returns all posts

        Returns:
        -------
            List of post dictionaries, each containing:
                - id: Post ID (string)
                - status: Post status (string)
                - scheduledFor: Scheduled time if applicable (datetime or None)
                - platforms: List of platform dicts with platform name

        Raises:
        ------
            AuthenticationError: If not authenticated or credentials expired
            PublisherError: If post listing fails

        Example:
        -------
            >>> posts = await publisher.list_posts(status='scheduled')
            >>> for post in posts:
            ...     print(f"{post['id']}: {post['scheduledFor']}")
            post_123: 2025-12-21 10:00:00+00:00
            post_456: 2025-12-22 10:00:00+00:00

        Implementation Notes
        --------------------
            - Return posts in reverse chronological order (newest first)
            - Include all relevant post metadata
            - Filter by status if provided
            - Handle pagination if provider supports it
            - Convert all timestamps to UTC datetime objects

        """
        pass

    @abstractmethod
    async def delete_post(self, post_id: str) -> bool:
        """Delete a post from the provider.

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
            >>> success = await publisher.delete_post("post_123")
            >>> if success:
            ...     print("Post deleted")

        Implementation Notes
        --------------------
            - Return True even if post doesn't exist (idempotent)
            - Log all delete operations for audit trail
            - Handle rate limiting with exponential backoff

        """
        pass
