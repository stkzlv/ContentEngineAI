"""Video publishing and scheduling system.

This module provides a unified interface for publishing videos to social media
platforms via third-party scheduling services (Late.dev, Buffer, Hootsuite, etc.).

Example Usage
-------------
    >>> from src.publisher import create_publisher, PublisherProvider
    >>> publisher = create_publisher(
    ...     provider=PublisherProvider.LATE,
    ...     api_key="sk_live_...",
    ...     vercel_token="vercel_..."
    ... )
    >>> await publisher.authenticate()
    >>> accounts = await publisher.get_accounts()
    >>> media_id = await publisher.upload_media(Path("video.mp4"))
    >>> result = await publisher.publish(
    ...     media_id=media_id,
    ...     platforms=[{"platform": "youtube", "account_id": "acc_123"}],
    ...     content="Amazing product review! #ad"
    ... )

"""

# Import provider implementations to trigger registration
from . import late  # noqa: F401
from .base import (
    AuthenticationError,
    BasePublisher,
    PublisherError,
    PublisherProvider,
    PublishError,
    UploadError,
    ValidationError,
)
from .models import (
    BatchPublishSummary,
    Platform,
    PublisherConfig,
    PublishMetadata,
    PublishResult,
    PublishStatus,
)
from .registry import (
    PublisherRegistry,
    create_publisher,
    register_publisher,
)

__all__ = [
    # Base classes and exceptions
    "BasePublisher",
    "PublisherProvider",
    "PublisherError",
    "AuthenticationError",
    "UploadError",
    "PublishError",
    "ValidationError",
    # Data models
    "PublishResult",
    "PublishMetadata",
    "PublisherConfig",
    "BatchPublishSummary",
    "PublishStatus",
    "Platform",
    # Registry and factory
    "PublisherRegistry",
    "create_publisher",
    "register_publisher",
]
