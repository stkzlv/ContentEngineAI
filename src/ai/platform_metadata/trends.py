"""Trend-aware hashtag generation for platform metadata.

This module provides integration with trend providers to suggest current
trending hashtags for YouTube, TikTok, and Instagram, improving content
discoverability. It includes a caching layer to minimize API calls and
graceful fallback mechanisms.
"""

import abc
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Dict, List, Optional

from src.ai.platform_metadata.models import PlatformFallbackTags, TrendSettings

logger = logging.getLogger(__name__)


@dataclass
class TrendingTags:
    """Container for platform-specific trending hashtags.

    Attributes
    ----------
        platform: Platform identifier ("youtube", "tiktok", "instagram")
        tags: List of trending hashtags (including # prefix)
        fetched_at: Timestamp when trends were fetched

    """

    platform: str
    tags: List[str]
    fetched_at: datetime


class TrendProvider(abc.ABC):
    """Abstract base class for trend data providers."""

    @abc.abstractmethod
    async def get_trending_tags(self, platform: str) -> List[str]:
        """Fetch trending hashtags for a specific platform.

        Args:
        ----
            platform: Platform identifier

        Returns:
        -------
            List of trending hashtags

        """
        pass


class StaticTrendProvider(TrendProvider):
    """Fallback trend provider using configurable high-performance tags.

    Used when external APIs are unavailable or as a baseline.
    Tags are configured via TrendSettings.fallback_tags in ai_services.yaml.
    """

    def __init__(self, fallback_tags: PlatformFallbackTags | None = None):
        """Initialize with configurable fallback tags.

        Args:
        ----
            fallback_tags: Platform-specific fallback tags from config.
                           If None, uses PlatformFallbackTags defaults.

        """
        self.fallback_tags = fallback_tags or PlatformFallbackTags()

    async def get_trending_tags(self, platform: str) -> List[str]:
        """Return configurable fallback trends for the platform."""
        return getattr(self.fallback_tags, platform, [])


class TrendCache:
    """Simple in-memory cache for trending hashtags with TTL support.

    Prevents redundant external API calls within the TTL window.
    """

    def __init__(self, ttl_hours: int):
        """Initialize trend cache.

        Args:
        ----
            ttl_hours: Time-to-live for cache entries in hours

        """
        self._cache: Dict[str, TrendingTags] = {}
        self.ttl = timedelta(hours=ttl_hours)

    def get(self, platform: str) -> Optional[List[str]]:
        """Retrieve cached trends if valid and not expired.

        Args:
        ----
            platform: Platform identifier

        Returns:
        -------
            List of tags or None if not found/expired

        """
        if platform not in self._cache:
            return None

        entry = self._cache[platform]
        # Check if entry is expired
        if datetime.now(UTC) - entry.fetched_at.replace(tzinfo=UTC) > self.ttl:
            logger.debug(f"Trend cache expired for {platform}")
            del self._cache[platform]
            return None

        return entry.tags

    def set(self, platform: str, tags: List[str]):
        """Store trends in cache with current timestamp.

        Args:
        ----
            platform: Platform identifier
            tags: List of trending hashtags

        """
        self._cache[platform] = TrendingTags(platform, tags, datetime.now(UTC))


class TrendAwareHashtagGenerator:
    """Orchestrates trend-aware hashtag generation and merging.

    Integrates providers, caching, and merging logic to enhance generated
    metadata with relevant trending tags.

    Example usage:
        generator = TrendAwareHashtagGenerator(settings)
        hashtags = await generator.merge_trending_tags("tiktok", ["#ad", "#cool"])
    """

    def __init__(self, settings: TrendSettings):
        """Initialize trend-aware generator.

        Args:
        ----
            settings: Trend generation configuration

        """
        self.settings = settings
        self.cache = TrendCache(settings.cache_ttl_hours)
        self.provider = self._get_provider(settings.provider)

    def _get_provider(self, name: str) -> TrendProvider:
        """Create provider instance by name.

        Args:
        ----
            name: Provider identifier

        Returns:
        -------
            TrendProvider instance

        """
        if name == "static":
            return StaticTrendProvider(self.settings.fallback_tags)
        # Add more providers here (e.g., "tiktok_api", "google_trends")
        # Default to static with configured fallback tags
        return StaticTrendProvider(self.settings.fallback_tags)

    async def merge_trending_tags(
        self, platform: str, existing_tags: List[str]
    ) -> List[str]:
        """Fetch and merge trending tags with existing ones.

        Respects max_trending_tags limit and prevents duplicates.

        Args:
        ----
            platform: Platform identifier
            existing_tags: List of tags generated by LLM

        Returns:
        -------
            Enhanced list of hashtags

        """
        if not self.settings.enabled:
            return existing_tags

        # Try to get from cache first
        trending = self.cache.get(platform)

        if trending is None:
            try:
                logger.info(f"Fetching trending tags for {platform}...")
                trending = await self.provider.get_trending_tags(platform)
                self.cache.set(platform, trending)
            except Exception as e:
                logger.error(f"Failed to fetch trends for {platform}: {e}")
                return existing_tags

        if not trending:
            return existing_tags

        # Case-insensitive comparison for duplicates
        existing_lower = {t.lower() for t in existing_tags}
        
        # Filter out tags already present and pick up to limit
        tags_to_add = []
        for tag in trending:
            if tag.lower() not in existing_lower:
                tags_to_add.append(tag)
                if len(tags_to_add) >= self.settings.max_trending_tags:
                    break

        if not tags_to_add:
            return existing_tags

        logger.debug(f"Adding trending tags to {platform}: {tags_to_add}")
        return existing_tags + tags_to_add
