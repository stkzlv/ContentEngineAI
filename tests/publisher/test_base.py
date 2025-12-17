"""Unit tests for BasePublisher abstract interface."""

from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.publisher.base import (
    AuthenticationError,
    BasePublisher,
    PublisherError,
    PublisherProvider,
    PublishError,
    UploadError,
    ValidationError,
)


class TestPublisherProvider:
    """Test PublisherProvider enum."""

    def test_provider_values(self):
        """Test that provider enum has expected values."""
        assert PublisherProvider.LATE.value == "late"
        assert PublisherProvider.BUFFER.value == "buffer"
        assert PublisherProvider.HOOTSUITE.value == "hootsuite"
        assert PublisherProvider.LATER.value == "later"

    def test_provider_from_string(self):
        """Test creating provider from string value."""
        provider = PublisherProvider("late")
        assert provider == PublisherProvider.LATE

    def test_provider_invalid_string(self):
        """Test creating provider from invalid string raises ValueError."""
        with pytest.raises(ValueError):
            PublisherProvider("invalid_provider")


class TestPublisherExceptions:
    """Test custom exception hierarchy."""

    def test_publisher_error_base(self):
        """Test PublisherError base exception."""
        error = PublisherError("Base error message")
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)

    def test_authentication_error_inherits_publisher_error(self):
        """Test AuthenticationError inherits from PublisherError."""
        error = AuthenticationError("Auth failed")
        assert isinstance(error, PublisherError)
        assert isinstance(error, Exception)
        assert str(error) == "Auth failed"

    def test_upload_error_inherits_publisher_error(self):
        """Test UploadError inherits from PublisherError."""
        error = UploadError("Upload failed")
        assert isinstance(error, PublisherError)
        assert str(error) == "Upload failed"

    def test_publish_error_inherits_publisher_error(self):
        """Test PublishError inherits from PublisherError."""
        error = PublishError("Publish failed")
        assert isinstance(error, PublisherError)
        assert str(error) == "Publish failed"

    def test_validation_error_inherits_publisher_error(self):
        """Test ValidationError inherits from PublisherError."""
        error = ValidationError("Invalid input")
        assert isinstance(error, PublisherError)
        assert str(error) == "Invalid input"


class TestBasePublisher:
    """Test BasePublisher abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BasePublisher cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BasePublisher()

    def test_must_implement_authenticate(self):
        """Test that concrete class must implement authenticate()."""

        class IncompletePublisher(BasePublisher):
            """Publisher missing authenticate method."""

            @property
            def provider(self):
                return PublisherProvider.LATE

            async def get_accounts(self):
                pass

            async def upload_media(self, video_path, progress_callback=None):
                pass

            async def publish(self, media_id, platforms, content, scheduled_time=None):
                pass

            async def get_status(self, post_id):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompletePublisher()

    def test_must_implement_get_accounts(self):
        """Test that concrete class must implement get_accounts()."""

        class IncompletePublisher(BasePublisher):
            """Publisher missing get_accounts method."""

            @property
            def provider(self):
                return PublisherProvider.LATE

            async def authenticate(self):
                pass

            async def upload_media(self, video_path, progress_callback=None):
                pass

            async def publish(self, media_id, platforms, content, scheduled_time=None):
                pass

            async def get_status(self, post_id):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompletePublisher()

    def test_must_implement_upload_media(self):
        """Test that concrete class must implement upload_media()."""

        class IncompletePublisher(BasePublisher):
            """Publisher missing upload_media method."""

            @property
            def provider(self):
                return PublisherProvider.LATE

            async def authenticate(self):
                pass

            async def get_accounts(self):
                pass

            async def publish(self, media_id, platforms, content, scheduled_time=None):
                pass

            async def get_status(self, post_id):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompletePublisher()

    def test_must_implement_publish(self):
        """Test that concrete class must implement publish()."""

        class IncompletePublisher(BasePublisher):
            """Publisher missing publish method."""

            @property
            def provider(self):
                return PublisherProvider.LATE

            async def authenticate(self):
                pass

            async def get_accounts(self):
                pass

            async def upload_media(self, video_path, progress_callback=None):
                pass

            async def get_status(self, post_id):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompletePublisher()

    def test_must_implement_get_status(self):
        """Test that concrete class must implement get_status()."""

        class IncompletePublisher(BasePublisher):
            """Publisher missing get_status method."""

            @property
            def provider(self):
                return PublisherProvider.LATE

            async def authenticate(self):
                pass

            async def get_accounts(self):
                pass

            async def upload_media(self, video_path, progress_callback=None):
                pass

            async def publish(self, media_id, platforms, content, scheduled_time=None):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompletePublisher()

    def test_concrete_implementation_can_be_instantiated(self):
        """Test that complete concrete implementation can be instantiated."""

        class ConcretePublisher(BasePublisher):
            """Complete publisher implementation."""

            @property
            def provider(self):
                return PublisherProvider.LATE

            async def authenticate(self):
                return True

            async def get_accounts(self):
                return []

            async def upload_media(self, video_path, progress_callback=None):
                return "media_123"

            async def publish(self, media_id, platforms, content, scheduled_time=None):
                return {
                    "post_id": "post_123",
                    "status": "published",
                    "scheduled_time": scheduled_time,
                    "published_urls": [],
                }

            async def get_status(self, post_id):
                return {
                    "post_id": post_id,
                    "status": "published",
                    "scheduled_time": None,
                    "published_time": datetime.now(UTC),
                    "published_urls": [],
                    "error_message": None,
                }

        # Should not raise
        publisher = ConcretePublisher()
        assert publisher.provider == PublisherProvider.LATE

    @pytest.mark.asyncio
    async def test_concrete_implementation_methods_work(self):
        """Test that concrete implementation methods can be called."""

        class ConcretePublisher(BasePublisher):
            """Complete publisher implementation."""

            @property
            def provider(self):
                return PublisherProvider.LATE

            async def authenticate(self):
                return True

            async def get_accounts(self):
                return [{"platform": "youtube", "account_id": "acc_123"}]

            async def upload_media(self, video_path, progress_callback=None):
                return "media_123"

            async def publish(self, media_id, platforms, content, scheduled_time=None):
                return {
                    "post_id": "post_123",
                    "status": "published",
                    "scheduled_time": scheduled_time,
                    "published_urls": ["https://youtube.com/watch?v=abc"],
                }

            async def get_status(self, post_id):
                return {
                    "post_id": post_id,
                    "status": "published",
                    "scheduled_time": None,
                    "published_time": datetime.now(UTC),
                    "published_urls": ["https://youtube.com/watch?v=abc"],
                    "error_message": None,
                }

        publisher = ConcretePublisher()

        # Test authenticate
        is_authenticated = await publisher.authenticate()
        assert is_authenticated is True

        # Test get_accounts
        accounts = await publisher.get_accounts()
        assert len(accounts) == 1
        assert accounts[0]["platform"] == "youtube"

        # Test upload_media
        media_id = await publisher.upload_media(Path("test.mp4"))
        assert media_id == "media_123"

        # Test publish
        result = await publisher.publish(
            media_id="media_123",
            platforms=[{"platform": "youtube", "account_id": "acc_123"}],
            content="Test content",
        )
        assert result["post_id"] == "post_123"
        assert result["status"] == "published"

        # Test get_status
        status = await publisher.get_status("post_123")
        assert status["post_id"] == "post_123"
        assert status["status"] == "published"
