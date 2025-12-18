"""Unit tests for publisher registry and factory."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from src.publisher.base import BasePublisher, PublisherProvider, ValidationError
from src.publisher.registry import (
    PublisherRegistry,
    create_publisher,
    register_publisher,
)


class TestPublisherRegistry:
    """Test PublisherRegistry functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        PublisherRegistry._providers.clear()

    def test_register_publisher_class(self):
        """Test registering a publisher class."""

        class TestPublisher(BasePublisher):
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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        PublisherRegistry.register(PublisherProvider.LATE, TestPublisher)

        assert PublisherProvider.LATE in PublisherRegistry._providers
        assert PublisherRegistry._providers[PublisherProvider.LATE] == TestPublisher

    def test_get_provider_class(self):
        """Test retrieving a registered provider class."""

        class TestPublisher(BasePublisher):
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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        PublisherRegistry.register(PublisherProvider.LATE, TestPublisher)

        retrieved_class = PublisherRegistry.get_publisher_class(PublisherProvider.LATE)
        assert retrieved_class == TestPublisher

    def test_get_provider_class_not_registered(self):
        """Test retrieving unregistered provider returns None."""
        result = PublisherRegistry.get_publisher_class(PublisherProvider.BUFFER)
        assert result is None

    def test_is_provider_supported(self):
        """Test checking if provider is supported."""

        class TestPublisher(BasePublisher):
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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        PublisherRegistry.register(PublisherProvider.LATE, TestPublisher)

        assert PublisherRegistry.is_provider_supported(PublisherProvider.LATE) is True
        assert (
            PublisherRegistry.is_provider_supported(PublisherProvider.BUFFER) is False
        )

    def test_list_providers(self):
        """Test listing all registered providers."""

        class TestPublisher1(BasePublisher):
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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        class TestPublisher2(BasePublisher):
            @property
            def provider(self):
                return PublisherProvider.BUFFER

            async def authenticate(self):
                return True

            async def get_accounts(self):
                return []

            async def upload_media(self, video_path, progress_callback=None):
                return "media_456"

            async def publish(self, media_id, platforms, content, scheduled_time=None):
                return {"post_id": "post_456", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        PublisherRegistry.register(PublisherProvider.LATE, TestPublisher1)
        PublisherRegistry.register(PublisherProvider.BUFFER, TestPublisher2)

        providers = PublisherRegistry.get_available_providers()
        assert len(providers) == 2
        assert PublisherProvider.LATE in providers
        assert PublisherProvider.BUFFER in providers


class TestRegisterPublisherDecorator:
    """Test @register_publisher decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        PublisherRegistry._providers.clear()

    def test_decorator_registers_class(self):
        """Test that @register_publisher decorator registers the class."""

        @register_publisher(PublisherProvider.LATE)
        class TestPublisher(BasePublisher):
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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        assert PublisherRegistry.is_provider_supported(PublisherProvider.LATE) is True
        assert (
            PublisherRegistry.get_publisher_class(PublisherProvider.LATE)
            == TestPublisher
        )

    def test_decorator_returns_class(self):
        """Test that @register_publisher decorator returns the class."""

        @register_publisher(PublisherProvider.LATE)
        class TestPublisher(BasePublisher):
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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        # Class should still be usable
        assert TestPublisher is not None
        instance = TestPublisher()
        assert instance.provider == PublisherProvider.LATE

    def test_decorator_multiple_providers(self):
        """Test registering multiple providers with decorator."""

        @register_publisher(PublisherProvider.LATE)
        class LatePublisher(BasePublisher):
            @property
            def provider(self):
                return PublisherProvider.LATE

            async def authenticate(self):
                return True

            async def get_accounts(self):
                return []

            async def upload_media(self, video_path, progress_callback=None):
                return "media_late"

            async def publish(self, media_id, platforms, content, scheduled_time=None):
                return {"post_id": "late_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        @register_publisher(PublisherProvider.BUFFER)
        class BufferPublisher(BasePublisher):
            @property
            def provider(self):
                return PublisherProvider.BUFFER

            async def authenticate(self):
                return True

            async def get_accounts(self):
                return []

            async def upload_media(self, video_path, progress_callback=None):
                return "media_buffer"

            async def publish(self, media_id, platforms, content, scheduled_time=None):
                return {"post_id": "buffer_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        assert PublisherRegistry.is_provider_supported(PublisherProvider.LATE) is True
        assert PublisherRegistry.is_provider_supported(PublisherProvider.BUFFER) is True
        assert (
            PublisherRegistry.get_publisher_class(PublisherProvider.LATE)
            == LatePublisher
        )
        assert (
            PublisherRegistry.get_publisher_class(PublisherProvider.BUFFER)
            == BufferPublisher
        )


class TestCreatePublisher:
    """Test create_publisher factory function."""

    def setup_method(self):
        """Clear registry before each test."""
        PublisherRegistry._providers.clear()

    def test_create_publisher_with_enum(self):
        """Test creating publisher with PublisherProvider enum."""

        @register_publisher(PublisherProvider.LATE)
        class TestPublisher(BasePublisher):
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.kwargs = kwargs

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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        publisher = create_publisher(PublisherProvider.LATE, api_key="test_key_123")

        assert isinstance(publisher, TestPublisher)
        assert publisher.api_key == "test_key_123"

    def test_create_publisher_with_string(self):
        """Test creating publisher with string provider."""

        @register_publisher(PublisherProvider.LATE)
        class TestPublisher(BasePublisher):
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key

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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        publisher = create_publisher("late", api_key="test_key_123")

        assert isinstance(publisher, TestPublisher)
        assert publisher.api_key == "test_key_123"

    def test_create_publisher_with_kwargs(self):
        """Test creating publisher with additional kwargs."""

        @register_publisher(PublisherProvider.LATE)
        class TestPublisher(BasePublisher):
            def __init__(self, api_key, vercel_token=None, timeout=30.0, **kwargs):
                self.api_key = api_key
                self.vercel_token = vercel_token
                self.timeout = timeout

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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        publisher = create_publisher(
            PublisherProvider.LATE,
            api_key="test_key_123",
            vercel_token="vercel_token_456",  # noqa: S106
            timeout=60.0,
        )

        assert publisher.api_key == "test_key_123"
        assert publisher.vercel_token == "vercel_token_456"  # noqa: S105
        assert publisher.timeout == 60.0

    def test_create_publisher_unregistered_provider(self):
        """Test creating publisher with unregistered provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider .* not registered"):
            create_publisher(PublisherProvider.BUFFER, api_key="test_key")

    def test_create_publisher_invalid_provider_string(self):
        """Test creating publisher with invalid provider string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid provider.*invalid_provider"):
            create_publisher("invalid_provider", api_key="test_key")

    def test_create_publisher_with_session(self):
        """Test creating publisher with aiohttp session."""

        @register_publisher(PublisherProvider.LATE)
        class TestPublisher(BasePublisher):
            def __init__(self, api_key, session=None, **kwargs):
                self.api_key = api_key
                self.session = session

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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        publisher = create_publisher(
            PublisherProvider.LATE,
            api_key="test_key_123",
            session=mock_session,
        )

        assert publisher.session == mock_session

    def test_factory_isolation(self):
        """Test that factory creates independent instances."""

        @register_publisher(PublisherProvider.LATE)
        class TestPublisher(BasePublisher):
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.state = []

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
                return {"post_id": "post_123", "status": "published"}

            async def get_status(self, post_id):
                return {"post_id": post_id, "status": "published"}

        publisher1 = create_publisher(PublisherProvider.LATE, api_key="key_1")
        publisher2 = create_publisher(PublisherProvider.LATE, api_key="key_2")

        # Instances should be independent
        publisher1.state.append("item1")
        assert len(publisher1.state) == 1
        assert len(publisher2.state) == 0
        assert publisher1.api_key == "key_1"
        assert publisher2.api_key == "key_2"
