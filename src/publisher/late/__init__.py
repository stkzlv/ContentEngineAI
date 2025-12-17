"""Late.dev publisher implementation.

This module provides the Late.dev-specific implementation of the BasePublisher
interface, enabling video publishing to YouTube, TikTok, Instagram, and other
platforms via the Late.dev scheduling service.
"""

from .client import LatePublisher

__all__ = [
    "LatePublisher",
]
