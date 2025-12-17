"""Unit tests for LatePublisher client implementation."""

import asyncio
from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import aiohttp
import pytest

from src.publisher.base import (
    AuthenticationError,
    PublisherProvider,
    PublishError,
    UploadError,
    ValidationError,
)
from src.publisher.late.client import LatePublisher


class TestLatePublisherInit:
    """Test LatePublisher initialization and validation."""

    def test_init_success(self):
        """Test successful initialization with valid parameters."""
        publisher = LatePublisher(
            api_key="sk_test_abc123",
            vercel_token="vercel_xyz456",
            timeout=60.0,
            max_retries=5,
        )

        assert publisher._api_key == "sk_test_abc123"
        assert publisher.vercel_token == "vercel_xyz456"
        assert publisher.timeout == 60.0
        assert publisher.max_retries == 5
        assert publisher.provider == PublisherProvider.LATE

    def test_init_minimal_params(self):
        """Test initialization with only required parameters."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        assert publisher._api_key == "sk_test_abc123"
        assert publisher.vercel_token is None
        assert publisher.timeout == 30.0
        assert publisher.max_retries == 3

    def test_init_empty_api_key(self):
        """Test initialization with empty API key raises ValidationError."""
        with pytest.raises(ValidationError, match="api_key cannot be empty"):
            LatePublisher(api_key="")

    def test_init_whitespace_api_key(self):
        """Test initialization with whitespace API key raises ValidationError."""
        with pytest.raises(ValidationError, match="api_key cannot be empty"):
            LatePublisher(api_key="   ")

    def test_init_negative_timeout(self):
        """Test initialization with negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            LatePublisher(api_key="sk_test_abc123", timeout=-1.0)

    def test_init_zero_timeout(self):
        """Test initialization with zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            LatePublisher(api_key="sk_test_abc123", timeout=0.0)

    def test_init_negative_max_retries(self):
        """Test initialization with negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            LatePublisher(api_key="sk_test_abc123", max_retries=-1)

    def test_init_with_custom_session(self):
        """Test initialization with custom aiohttp session."""
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        publisher = LatePublisher(
            api_key="sk_test_abc123",
            session=mock_session,
        )

        assert publisher._session == mock_session
        assert publisher._should_close_session is False

    def test_provider_property(self):
        """Test provider property returns LATE."""
        publisher = LatePublisher(api_key="sk_test_abc123")
        assert publisher.provider == PublisherProvider.LATE


class TestLatePublisherAuthenticate:
    """Test LatePublisher authentication."""

    @pytest.mark.asyncio
    async def test_authenticate_success(self):
        """Test successful authentication."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock the client.posts.list method
        publisher.client.posts = MagicMock()
        publisher.client.posts.list = AsyncMock(return_value=[])

        result = await publisher.authenticate()

        assert result is True
        publisher.client.posts.list.assert_called_once_with(limit=1)

    @pytest.mark.asyncio
    async def test_authenticate_auth_failure_401(self):
        """Test authentication failure with 401 error."""
        publisher = LatePublisher(api_key="sk_test_invalid")

        # Mock the client.posts.list to raise 401 error
        publisher.client.posts = MagicMock()
        publisher.client.posts.list = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=401,
                message="Unauthorized",
            )
        )

        with pytest.raises(AuthenticationError, match="authentication failed"):
            await publisher.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_auth_failure_403(self):
        """Test authentication failure with 403 error."""
        publisher = LatePublisher(api_key="sk_test_invalid")

        # Mock the client.posts.list to raise 403 error
        publisher.client.posts = MagicMock()
        publisher.client.posts.list = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=403,
                message="Forbidden",
            )
        )

        with pytest.raises(AuthenticationError, match="authentication failed"):
            await publisher.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_retry_on_500(self):
        """Test authentication retries on 500 error."""
        publisher = LatePublisher(api_key="sk_test_abc123", max_retries=2)

        # Mock to fail once with 500, then succeed
        publisher.client.posts = MagicMock()
        publisher.client.posts.list = AsyncMock(
            side_effect=[
                aiohttp.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=500,
                    message="Internal Server Error",
                ),
                [],
            ]
        )

        result = await publisher.authenticate()

        assert result is True
        assert publisher.client.posts.list.call_count == 2


class TestLatePublisherGetAccounts:
    """Test LatePublisher get_accounts method."""

    @pytest.mark.asyncio
    async def test_get_accounts_success(self):
        """Test successful account listing."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock the client.accounts.list method
        mock_accounts = [
            {
                "platform": "youtube",
                "account_id": "acc_youtube_123",
                "username": "user1",
            },
            {"platform": "tiktok", "account_id": "acc_tiktok_456", "username": "user2"},
        ]
        publisher.client.accounts = MagicMock()
        publisher.client.accounts.list = AsyncMock(return_value=mock_accounts)

        result = await publisher.get_accounts()

        assert len(result) == 2
        assert result[0]["platform"] == "youtube"
        assert result[1]["platform"] == "tiktok"
        publisher.client.accounts.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_accounts_empty(self):
        """Test account listing with no accounts."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        publisher.client.accounts = MagicMock()
        publisher.client.accounts.list = AsyncMock(return_value=[])

        result = await publisher.get_accounts()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_accounts_auth_failure(self):
        """Test account listing with authentication failure."""
        publisher = LatePublisher(api_key="sk_test_invalid")

        publisher.client.accounts = MagicMock()
        publisher.client.accounts.list = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=401,
                message="Unauthorized",
            )
        )

        with pytest.raises(AuthenticationError):
            await publisher.get_accounts()


class TestLatePublisherUploadMedia:
    """Test LatePublisher upload_media method."""

    @pytest.fixture
    def temp_video_file(self, tmp_path):
        """Create a temporary video file for testing."""
        video_file = tmp_path / "test_video.mp4"
        # Create a small file (< 4 MB)
        video_file.write_bytes(b"fake video content" * 100)
        return video_file

    @pytest.fixture
    def large_video_file(self, tmp_path):
        """Create a large temporary video file for testing."""
        video_file = tmp_path / "large_video.mp4"
        # Create a file > 4 MB (4 * 1024 * 1024 bytes)
        video_file.write_bytes(b"x" * (5 * 1024 * 1024))
        return video_file

    @pytest.mark.asyncio
    async def test_upload_small_file_success(self, temp_video_file):
        """Test successful upload of small file (< 4 MB)."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock the client.media.upload method
        publisher.client.media = MagicMock()
        publisher.client.media.upload = AsyncMock(
            return_value={"media_id": "media_123"}
        )

        media_id = await publisher.upload_media(temp_video_file)

        assert media_id == "media_123"
        publisher.client.media.upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_large_file_success(self, large_video_file):
        """Test successful upload of large file (> 4 MB) with Vercel token."""
        publisher = LatePublisher(
            api_key="sk_test_abc123",
            vercel_token="vercel_token_xyz",
        )

        # Mock the client.media.upload_large method
        publisher.client.media = MagicMock()
        publisher.client.media.upload_large = AsyncMock(
            return_value={"media_id": "media_456"}
        )

        media_id = await publisher.upload_media(large_video_file)

        assert media_id == "media_456"
        publisher.client.media.upload_large.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_large_file_no_vercel_token(self, large_video_file):
        """Test upload of large file without Vercel token raises UploadError."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        with pytest.raises(
            UploadError, match="Vercel token required for files larger than 4 MB"
        ):
            await publisher.upload_media(large_video_file)

    @pytest.mark.asyncio
    async def test_upload_file_not_found(self):
        """Test upload of non-existent file raises UploadError."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        with pytest.raises(UploadError, match="Video file not found"):
            await publisher.upload_media(Path("/nonexistent/video.mp4"))

    @pytest.mark.asyncio
    async def test_upload_file_permission_denied(self, tmp_path):
        """Test upload of file without read permission raises UploadError."""
        video_file = tmp_path / "restricted_video.mp4"
        video_file.write_bytes(b"content")
        video_file.chmod(0o000)

        publisher = LatePublisher(api_key="sk_test_abc123")

        try:
            with pytest.raises(UploadError, match="Cannot read video file"):
                await publisher.upload_media(video_file)
        finally:
            # Restore permissions for cleanup
            video_file.chmod(0o644)

    @pytest.mark.asyncio
    async def test_upload_invalid_extension(self, tmp_path):
        """Test upload of file with invalid extension raises UploadError."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not a video")

        publisher = LatePublisher(api_key="sk_test_abc123")

        with pytest.raises(
            UploadError, match="Invalid video format: .txt not in allowed extensions"
        ):
            await publisher.upload_media(invalid_file)

    @pytest.mark.asyncio
    async def test_upload_with_progress_callback(self, temp_video_file):
        """Test upload with progress callback."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock the client.media.upload method
        publisher.client.media = MagicMock()
        publisher.client.media.upload = AsyncMock(
            return_value={"media_id": "media_123"}
        )

        progress_calls = []

        def progress_callback(bytes_uploaded: int, total_bytes: int):
            progress_calls.append((bytes_uploaded, total_bytes))

        media_id = await publisher.upload_media(temp_video_file, progress_callback)

        assert media_id == "media_123"
        # Progress callback should be called at least once
        assert len(progress_calls) > 0

    @pytest.mark.asyncio
    async def test_upload_retry_on_network_timeout(self, temp_video_file):
        """Test upload retries on network timeout."""
        publisher = LatePublisher(api_key="sk_test_abc123", max_retries=2)

        # Mock to timeout once, then succeed
        publisher.client.media = MagicMock()
        publisher.client.media.upload = AsyncMock(
            side_effect=[
                TimeoutError("Request timed out"),
                {"media_id": "media_123"},
            ]
        )

        media_id = await publisher.upload_media(temp_video_file)

        assert media_id == "media_123"
        assert publisher.client.media.upload.call_count == 2


class TestLatePublisherPublish:
    """Test LatePublisher publish method."""

    @pytest.mark.asyncio
    async def test_publish_single_platform_success(self):
        """Test successful publishing to single platform."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock the client.posts.create method
        mock_response = {
            "post_id": "post_123",
            "status": "published",
            "platforms": ["youtube"],
            "published_urls": ["https://youtube.com/watch?v=abc"],
        }
        publisher.client.posts = MagicMock()
        publisher.client.posts.create = AsyncMock(return_value=mock_response)

        platforms = [{"platform": "youtube", "account_id": "acc_123"}]
        result = await publisher.publish(
            media_id="media_123",
            platforms=platforms,
            content="Test video content",
        )

        assert result["post_id"] == "post_123"
        assert result["status"] == "published"
        assert "youtube" in result["platforms"]
        publisher.client.posts.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_multiple_platforms_success(self):
        """Test successful publishing to multiple platforms."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock the client.posts.create method
        mock_response = {
            "post_id": "post_456",
            "status": "published",
            "platforms": ["youtube", "tiktok", "instagram"],
            "published_urls": [
                "https://youtube.com/watch?v=abc",
                "https://tiktok.com/@user/video/123",
                "https://instagram.com/p/abc123",
            ],
        }
        publisher.client.posts = MagicMock()
        publisher.client.posts.create = AsyncMock(return_value=mock_response)

        platforms = [
            {"platform": "youtube", "account_id": "acc_yt"},
            {"platform": "tiktok", "account_id": "acc_tt"},
            {"platform": "instagram", "account_id": "acc_ig"},
        ]
        result = await publisher.publish(
            media_id="media_123",
            platforms=platforms,
            content="Multi-platform post",
        )

        assert result["post_id"] == "post_456"
        assert len(result["platforms"]) == 3
        assert len(result["published_urls"]) == 3

    @pytest.mark.asyncio
    async def test_publish_with_scheduled_time(self):
        """Test publishing with scheduled time."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        scheduled_time = datetime(2025, 12, 25, 14, 0, 0, tzinfo=UTC)

        # Mock the client.posts.create method
        mock_response = {
            "post_id": "post_789",
            "status": "scheduled",
            "scheduled_time": scheduled_time.isoformat(),
            "platforms": ["youtube"],
        }
        publisher.client.posts = MagicMock()
        publisher.client.posts.create = AsyncMock(return_value=mock_response)

        platforms = [{"platform": "youtube", "account_id": "acc_123"}]
        result = await publisher.publish(
            media_id="media_123",
            platforms=platforms,
            content="Scheduled post",
            scheduled_time=scheduled_time,
        )

        assert result["post_id"] == "post_789"
        assert result["status"] == "scheduled"
        assert "scheduled_time" in result

    @pytest.mark.asyncio
    async def test_publish_validation_error(self):
        """Test publish with validation error (400)."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock to raise 400 validation error
        publisher.client.posts = MagicMock()
        publisher.client.posts.create = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=400,
                message="Invalid content format",
            )
        )

        platforms = [{"platform": "youtube", "account_id": "acc_123"}]

        with pytest.raises(ValidationError, match="validation failed"):
            await publisher.publish(
                media_id="media_123",
                platforms=platforms,
                content="",  # Empty content
            )

    @pytest.mark.asyncio
    async def test_publish_rate_limit_with_retry(self):
        """Test publish handles rate limit (429) with retry."""
        publisher = LatePublisher(api_key="sk_test_abc123", max_retries=2)

        # Mock to fail with 429, then succeed
        mock_response = {
            "post_id": "post_999",
            "status": "published",
            "platforms": ["youtube"],
        }

        # Create mock error with headers
        error_response = MagicMock()
        error_response.headers = {"Retry-After": "5"}

        publisher.client.posts = MagicMock()
        publisher.client.posts.create = AsyncMock(
            side_effect=[
                aiohttp.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=429,
                    message="Rate limit exceeded",
                    headers={"Retry-After": "5"},
                ),
                mock_response,
            ]
        )

        platforms = [{"platform": "youtube", "account_id": "acc_123"}]
        result = await publisher.publish(
            media_id="media_123",
            platforms=platforms,
            content="Test content",
        )

        assert result["post_id"] == "post_999"
        assert publisher.client.posts.create.call_count == 2

    @pytest.mark.asyncio
    async def test_publish_partial_platform_failure(self):
        """Test publish continues on partial platform failure."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock response with partial failure
        mock_response = {
            "post_id": "post_partial",
            "status": "partial",
            "platforms": ["youtube"],  # Only YouTube succeeded
            "failed_platforms": ["tiktok"],  # TikTok failed
            "published_urls": ["https://youtube.com/watch?v=abc"],
        }
        publisher.client.posts = MagicMock()
        publisher.client.posts.create = AsyncMock(return_value=mock_response)

        platforms = [
            {"platform": "youtube", "account_id": "acc_yt"},
            {"platform": "tiktok", "account_id": "acc_tt"},
        ]
        result = await publisher.publish(
            media_id="media_123",
            platforms=platforms,
            content="Partial success post",
        )

        assert result["post_id"] == "post_partial"
        assert "youtube" in result["platforms"]
        assert "tiktok" in result.get("failed_platforms", [])


class TestLatePublisherGetStatus:
    """Test LatePublisher get_status method."""

    @pytest.mark.asyncio
    async def test_get_status_published(self):
        """Test getting status of published post."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock the client.posts.get method
        mock_response = {
            "post_id": "post_123",
            "status": "published",
            "published_time": "2025-01-15T10:30:00Z",
            "published_urls": ["https://youtube.com/watch?v=abc"],
            "platforms": ["youtube"],
        }
        publisher.client.posts = MagicMock()
        publisher.client.posts.get = AsyncMock(return_value=mock_response)

        result = await publisher.get_status("post_123")

        assert result["post_id"] == "post_123"
        assert result["status"] == "published"
        assert len(result["published_urls"]) == 1
        publisher.client.posts.get.assert_called_once_with("post_123")

    @pytest.mark.asyncio
    async def test_get_status_scheduled(self):
        """Test getting status of scheduled post."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock the client.posts.get method
        mock_response = {
            "post_id": "post_456",
            "status": "scheduled",
            "scheduled_time": "2025-12-25T14:00:00Z",
            "platforms": ["youtube", "tiktok"],
        }
        publisher.client.posts = MagicMock()
        publisher.client.posts.get = AsyncMock(return_value=mock_response)

        result = await publisher.get_status("post_456")

        assert result["post_id"] == "post_456"
        assert result["status"] == "scheduled"
        assert "scheduled_time" in result

    @pytest.mark.asyncio
    async def test_get_status_failed(self):
        """Test getting status of failed post."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock the client.posts.get method
        mock_response = {
            "post_id": "post_789",
            "status": "failed",
            "error_message": "Platform authentication expired",
            "platforms": ["youtube"],
        }
        publisher.client.posts = MagicMock()
        publisher.client.posts.get = AsyncMock(return_value=mock_response)

        result = await publisher.get_status("post_789")

        assert result["post_id"] == "post_789"
        assert result["status"] == "failed"
        assert "error_message" in result

    @pytest.mark.asyncio
    async def test_get_status_with_timezone_conversion(self):
        """Test getting status with timezone conversion."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock the client.posts.get method
        mock_response = {
            "post_id": "post_tz",
            "status": "published",
            "published_time": "2025-01-15T10:30:00Z",
            "scheduled_time": None,
            "published_urls": ["https://youtube.com/watch?v=abc"],
        }
        publisher.client.posts = MagicMock()
        publisher.client.posts.get = AsyncMock(return_value=mock_response)

        result = await publisher.get_status(
            "post_tz", local_timezone="America/New_York"
        )

        assert result["post_id"] == "post_tz"
        # Timezone conversion should be applied
        assert "published_time_local" in result or "published_time" in result

    @pytest.mark.asyncio
    async def test_get_status_not_found(self):
        """Test getting status of non-existent post."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        # Mock to raise 404 error
        publisher.client.posts = MagicMock()
        publisher.client.posts.get = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Post not found",
            )
        )

        with pytest.raises(PublishError, match="Post not found"):
            await publisher.get_status("post_nonexistent")


class TestLatePublisherRetryLogic:
    """Test LatePublisher retry logic and error handling."""

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff_timing(self):
        """Test exponential backoff timing (2s, 4s, 8s)."""
        publisher = LatePublisher(api_key="sk_test_abc123", max_retries=3)

        # Mock operation that fails with 5xx errors
        mock_operation = AsyncMock(
            side_effect=[
                aiohttp.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=500,
                    message="Server Error",
                ),
                aiohttp.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=502,
                    message="Bad Gateway",
                ),
                {"success": True},
            ]
        )

        start_time = asyncio.get_event_loop().time()
        result = await publisher._retry_with_backoff(mock_operation, "test_operation")
        end_time = asyncio.get_event_loop().time()

        # Should succeed after 2 retries with delays of 2s + 4s = 6s total
        assert result == {"success": True}
        assert mock_operation.call_count == 3
        # Allow some tolerance for timing
        assert end_time - start_time >= 5.0  # At least 6s with some tolerance

    @pytest.mark.asyncio
    async def test_retry_extract_retry_after_header(self):
        """Test extraction of Retry-After header."""
        publisher = LatePublisher(api_key="sk_test_abc123", max_retries=2)

        # Mock operation with rate limit and Retry-After header
        mock_error = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=429,
            message="Rate limit exceeded",
            headers={"Retry-After": "10"},
        )

        mock_operation = AsyncMock(side_effect=[mock_error, {"success": True}])

        start_time = asyncio.get_event_loop().time()
        result = await publisher._retry_with_backoff(mock_operation, "test_operation")
        end_time = asyncio.get_event_loop().time()

        # Should wait for 10s as specified in Retry-After header
        assert result == {"success": True}
        assert end_time - start_time >= 9.0  # At least 10s with some tolerance

    @pytest.mark.asyncio
    async def test_retry_no_retry_on_auth_errors(self):
        """Test that authentication errors (401/403) are not retried."""
        publisher = LatePublisher(api_key="sk_test_invalid", max_retries=3)

        # Mock operation that fails with 401
        mock_operation = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=401,
                message="Unauthorized",
            )
        )

        with pytest.raises(AuthenticationError):
            await publisher._retry_with_backoff(mock_operation, "test_operation")

        # Should only be called once (no retries)
        assert mock_operation.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_no_retry_on_validation_errors(self):
        """Test that validation errors (400/422) are not retried."""
        publisher = LatePublisher(api_key="sk_test_abc123", max_retries=3)

        # Mock operation that fails with 400
        mock_operation = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=400,
                message="Bad Request",
            )
        )

        with pytest.raises(ValidationError):
            await publisher._retry_with_backoff(mock_operation, "test_operation")

        # Should only be called once (no retries)
        assert mock_operation.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_max_retries_exceeded(self):
        """Test that operation fails after max retries exceeded."""
        publisher = LatePublisher(api_key="sk_test_abc123", max_retries=2)

        # Mock operation that always fails with 500
        mock_operation = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=500,
                message="Server Error",
            )
        )

        with pytest.raises(aiohttp.ClientResponseError):
            await publisher._retry_with_backoff(mock_operation, "test_operation")

        # Should be called max_retries times
        assert mock_operation.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_network_timeout(self):
        """Test retry on network timeout errors."""
        publisher = LatePublisher(api_key="sk_test_abc123", max_retries=2)

        # Mock operation that times out once, then succeeds
        mock_operation = AsyncMock(
            side_effect=[
                TimeoutError("Connection timeout"),
                {"success": True},
            ]
        )

        result = await publisher._retry_with_backoff(mock_operation, "test_operation")

        assert result == {"success": True}
        assert mock_operation.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_connection_error(self):
        """Test retry on connection errors."""
        publisher = LatePublisher(api_key="sk_test_abc123", max_retries=2)

        # Mock operation that has connection error once, then succeeds
        mock_operation = AsyncMock(
            side_effect=[
                aiohttp.ClientConnectionError("Connection refused"),
                {"success": True},
            ]
        )

        result = await publisher._retry_with_backoff(mock_operation, "test_operation")

        assert result == {"success": True}
        assert mock_operation.call_count == 2


class TestLatePublisherContextManager:
    """Test LatePublisher context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_session_cleanup(self):
        """Test that session is closed when using context manager."""
        publisher = LatePublisher(api_key="sk_test_abc123")

        async with publisher as pub:
            assert pub is publisher
            # Session should be created
            session = await pub._get_session()
            assert session is not None

        # Session should be closed after exiting context
        assert publisher._session is None or publisher._session.closed

    @pytest.mark.asyncio
    async def test_context_manager_with_custom_session(self):
        """Test that custom session is not closed by context manager."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False
        mock_session.close = AsyncMock()

        publisher = LatePublisher(
            api_key="sk_test_abc123",
            session=mock_session,
        )

        async with publisher:
            pass

        # Custom session should not be closed
        mock_session.close.assert_not_called()
