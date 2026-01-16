"""Unit tests for trend-aware hashtag generation."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.platform_metadata.models import PlatformMetadata, TrendSettings
from src.ai.platform_metadata.trends import (
    StaticTrendProvider,
    TrendAwareHashtagGenerator,
    TrendCache,
    TrendingTags,
)


class TestTrendCache:
    """Test TrendCache class."""

    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        cache = TrendCache(ttl_hours=1)
        tags = ["#test1", "#test2"]

        cache.set("youtube", tags)
        cached = cache.get("youtube")

        assert cached == tags

    def test_cache_miss(self):
        """Test cache miss for unknown platform."""
        cache = TrendCache(ttl_hours=1)
        assert cache.get("unknown") is None

    def test_cache_expiration(self):
        """Test cache entry expiration based on TTL."""
        cache = TrendCache(ttl_hours=1)
        tags = ["#test"]

        # Manually set an expired entry
        past = datetime.now(UTC) - timedelta(hours=2)
        cache._cache["youtube"] = TrendingTags("youtube", tags, past)

        assert cache.get("youtube") is None
        assert "youtube" not in cache._cache

    def test_cache_not_expired(self):
        """Test cache entry not expired within TTL."""
        cache = TrendCache(ttl_hours=5)
        tags = ["#test"]

        # Set entry 2 hours ago (TTL is 5)
        past = datetime.now(UTC) - timedelta(hours=2)
        cache._cache["youtube"] = TrendingTags("youtube", tags, past)

        assert cache.get("youtube") == tags


class TestStaticTrendProvider:
    """Test StaticTrendProvider class."""

    @pytest.mark.asyncio
    async def test_get_trending_tags(self):
        """Test getting static trends for known platforms."""
        provider = StaticTrendProvider()

        youtube_tags = await provider.get_trending_tags("youtube")
        assert "#Shorts" in youtube_tags
        assert len(youtube_tags) > 0

        tiktok_tags = await provider.get_trending_tags("tiktok")
        assert "#ForYou" in tiktok_tags

        unknown_tags = await provider.get_trending_tags("unknown")
        assert unknown_tags == []


class TestTrendAwareHashtagGenerator:
    """Test TrendAwareHashtagGenerator class."""

    @pytest.fixture
    def settings(self):
        """Create trend settings."""
        return TrendSettings(
            enabled=True,
            provider="static",
            cache_ttl_hours=1,
            max_trending_tags=2,
        )

    @pytest.fixture
    def generator(self, settings):
        """Create trend-aware generator."""
        return TrendAwareHashtagGenerator(settings)

    @pytest.mark.asyncio
    async def test_merge_trending_tags_enabled(self, generator):
        """Test merging tags when enabled."""
        existing = ["#ad", "#product"]

        # Static youtube trends: ["#Shorts", "#Trending", "#Tech", ...]
        merged = await generator.merge_trending_tags("youtube", existing)

        assert len(merged) == len(existing) + 2
        assert "#ad" in merged
        assert "#product" in merged
        assert "#Shorts" in merged
        assert "#Trending" in merged

    @pytest.mark.asyncio
    async def test_merge_trending_tags_disabled(self, settings):
        """Test merging tags when disabled."""
        settings.enabled = False
        generator = TrendAwareHashtagGenerator(settings)
        existing = ["#ad"]

        merged = await generator.merge_trending_tags("youtube", existing)

        assert merged == existing

    @pytest.mark.asyncio
    async def test_merge_trending_tags_limit(self, generator, settings):
        """Test respecting max_trending_tags limit."""
        settings.max_trending_tags = 1
        existing = ["#ad"]

        merged = await generator.merge_trending_tags("youtube", existing)

        assert len(merged) == 2  # 1 original + 1 trending
        assert "#Shorts" in merged
        assert "#Trending" not in merged

    @pytest.mark.asyncio
    async def test_merge_trending_tags_no_duplicates(self, generator):
        """Test preventing duplicate tags (case-insensitive)."""
        # Static trends include #Shorts
        existing = ["#ad", "#shorts"]  # lowercase version

        merged = await generator.merge_trending_tags("youtube", existing)

        # Should skip #Shorts because #shorts exists, and take next 2
        assert len(merged) == len(existing) + 2
        assert "#Shorts" not in merged[len(existing) :]
        assert "#Trending" in merged
        assert "#Tech" in merged

    @pytest.mark.asyncio
    async def test_merge_trending_tags_caching(self, generator):
        """Test that trends are cached after first fetch."""
        existing = ["#ad"]

        with patch.object(
            StaticTrendProvider,
            "get_trending_tags",
            wraps=StaticTrendProvider().get_trending_tags,
        ) as mock_get:
            # First call - should call provider
            await generator.merge_trending_tags("youtube", existing)
            assert mock_get.call_count == 1

            # Second call - should use cache
            await generator.merge_trending_tags("youtube", existing)
            assert mock_get.call_count == 1

    @pytest.mark.asyncio
    async def test_merge_trending_tags_api_failure(self, generator):
        """Test graceful handling of provider failures."""
        existing = ["#ad"]

        # Mock provider to raise exception
        generator.provider.get_trending_tags = AsyncMock(
            side_effect=Exception("API Error")
        )

        merged = await generator.merge_trending_tags("youtube", existing)

        # Should return original tags
        assert merged == existing


class TestFactoryTrendIntegration:
    """Test integration of trends in PlatformMetadataFactory."""

    @pytest.mark.asyncio
    async def test_factory_applies_trends(self):
        """Test that generate_multi_platform applies trends."""
        from src.ai.platform_metadata import PlatformMetadataFactory

        product = MagicMock()
        product.asin = "B0TEST"

        mock_meta = PlatformMetadata.create(
            platform="youtube",
            description="test",
            hashtags=["#ad"],
            keywords=[],
            product_id="B0TEST",
        )

        # Mock generator to return one result
        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(return_value=mock_meta)

        trend_settings = TrendSettings(enabled=True, max_trending_tags=1)
        trend_gen = TrendAwareHashtagGenerator(trend_settings)

        with patch.object(PlatformMetadataFactory, "create", return_value=mock_gen):
            results = await PlatformMetadataFactory.generate_multi_platform(
                product=product,
                settings=MagicMock(),
                secrets={},
                session=MagicMock(),
                platform_settings={"youtube": {}},
                intermediate_paths={},
                trend_generator=trend_gen,
            )

            assert results["youtube"] is not None
            # Original #ad + 1 trending tag (#Shorts)
            assert len(results["youtube"].hashtags) == 2
            assert "#Shorts" in results["youtube"].hashtags

    @pytest.mark.asyncio
    async def test_factory_no_trends_when_none_provided(self):
        """Test that trends are not applied if trend_generator is None."""
        from src.ai.platform_metadata import PlatformMetadataFactory

        product = MagicMock()
        product.asin = "B0TEST"

        mock_meta = PlatformMetadata.create(
            platform="youtube",
            description="test",
            hashtags=["#ad"],
            keywords=[],
            product_id="B0TEST",
        )

        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(return_value=mock_meta)

        with patch.object(PlatformMetadataFactory, "create", return_value=mock_gen):
            results = await PlatformMetadataFactory.generate_multi_platform(
                product=product,
                settings=MagicMock(),
                secrets={},
                session=MagicMock(),
                platform_settings={"youtube": {}},
                intermediate_paths={},
                trend_generator=None,
            )

            assert results["youtube"] is not None
            assert len(results["youtube"].hashtags) == 1
            assert results["youtube"].hashtags == ["#ad"]
