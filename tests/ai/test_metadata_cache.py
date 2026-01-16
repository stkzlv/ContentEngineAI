"""Unit tests for platform metadata caching."""

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.ai.platform_metadata.cache import CacheEntry, MetadataCache
from src.ai.platform_metadata.models import MetadataCacheSettings, PlatformMetadata


@dataclass
class MockProduct:
    """Mock product data for testing."""

    asin: str = "B0TESTASIN"
    title: str = "Test Product Title"
    description: str = "Test product description"
    price: str = "29.99"
    brand: str = "TestBrand"


class TestMetadataCacheSettings:
    """Test MetadataCacheSettings Pydantic model."""

    def test_default_settings(self):
        """Test default cache settings."""
        settings = MetadataCacheSettings()

        assert settings.enabled is True
        assert settings.ttl_hours == 24
        assert settings.cache_dir == ".cache/platform_metadata"
        assert settings.max_entries == 1000

    def test_custom_settings(self):
        """Test custom cache settings."""
        settings = MetadataCacheSettings(
            enabled=False,
            ttl_hours=48,
            cache_dir=".custom_cache",
            max_entries=500,
        )

        assert settings.enabled is False
        assert settings.ttl_hours == 48
        assert settings.cache_dir == ".custom_cache"
        assert settings.max_entries == 500

    def test_ttl_validation_min(self):
        """Test TTL minimum validation."""
        with pytest.raises(ValueError):
            MetadataCacheSettings(ttl_hours=0)

    def test_ttl_validation_max(self):
        """Test TTL maximum validation."""
        with pytest.raises(ValueError):
            MetadataCacheSettings(ttl_hours=1000)


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test Title",
            description="Test description",
            hashtags=["#test", "#ad"],
            keywords=["test"],
            product_id="B0TESTASIN",
        )

        entry = CacheEntry(
            metadata=metadata,
            product_hash="abcd1234",
            created_at="2025-01-15T12:00:00+00:00",
            expires_at="2025-01-16T12:00:00+00:00",
        )

        assert entry.metadata.platform == "youtube"
        assert entry.product_hash == "abcd1234"
        assert entry.created_at == "2025-01-15T12:00:00+00:00"
        assert entry.expires_at == "2025-01-16T12:00:00+00:00"

    def test_cache_entry_is_expired_false(self):
        """Test cache entry not expired."""
        metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test",
            description="Test",
            hashtags=["#ad"],
            keywords=[],
            product_id="B0TEST",
        )

        future = datetime.now(UTC) + timedelta(hours=1)
        entry = CacheEntry(
            metadata=metadata,
            product_hash="abcd1234",
            created_at=datetime.now(UTC).isoformat(),
            expires_at=future.isoformat(),
        )

        assert entry.is_expired() is False

    def test_cache_entry_is_expired_true(self):
        """Test cache entry expired."""
        metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test",
            description="Test",
            hashtags=["#ad"],
            keywords=[],
            product_id="B0TEST",
        )

        past = datetime.now(UTC) - timedelta(hours=1)
        entry = CacheEntry(
            metadata=metadata,
            product_hash="abcd1234",
            created_at=(past - timedelta(hours=24)).isoformat(),
            expires_at=past.isoformat(),
        )

        assert entry.is_expired() is True

    def test_cache_entry_to_dict(self):
        """Test cache entry serialization."""
        metadata = PlatformMetadata.create(
            platform="tiktok",
            description="Test caption",
            hashtags=["#test", "#ad"],
            keywords=["test"],
            product_id="B0TESTASIN",
        )

        entry = CacheEntry(
            metadata=metadata,
            product_hash="abcd1234",
            created_at="2025-01-15T12:00:00+00:00",
            expires_at="2025-01-16T12:00:00+00:00",
        )

        result = entry.to_dict()

        assert isinstance(result, dict)
        assert result["product_hash"] == "abcd1234"
        assert result["created_at"] == "2025-01-15T12:00:00+00:00"
        assert result["expires_at"] == "2025-01-16T12:00:00+00:00"
        assert result["metadata"]["platform"] == "tiktok"

    def test_cache_entry_from_dict(self):
        """Test cache entry deserialization."""
        data = {
            "metadata": {
                "platform": "instagram",
                "title": None,
                "description": "Test caption",
                "hashtags": ["#test", "#ad"],
                "keywords": ["test"],
                "character_counts": {"description": 12},
                "generated_at": "2025-01-15T12:00:00+00:00",
                "product_id": "B0TESTASIN",
                "validation_status": "valid",
                "validation_messages": [],
            },
            "product_hash": "efgh5678",
            "created_at": "2025-01-15T12:00:00+00:00",
            "expires_at": "2025-01-16T12:00:00+00:00",
        }

        entry = CacheEntry.from_dict(data)

        assert entry.metadata.platform == "instagram"
        assert entry.product_hash == "efgh5678"
        assert entry.created_at == "2025-01-15T12:00:00+00:00"


class TestMetadataCache:
    """Test MetadataCache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def cache_settings(self, temp_cache_dir):
        """Create cache settings with temp directory."""
        return MetadataCacheSettings(
            enabled=True,
            ttl_hours=24,
            cache_dir=temp_cache_dir,
            max_entries=100,
        )

    @pytest.fixture
    def cache(self, cache_settings):
        """Create cache instance."""
        return MetadataCache(cache_settings, project_root=Path("/"))

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return PlatformMetadata.create(
            platform="youtube",
            title="Test Product Review",
            description="Great product for testing!",
            hashtags=["#Shorts", "#test", "#ad"],
            keywords=["test product", "review"],
            product_id="B0TESTASIN",
        )

    @pytest.fixture
    def sample_product(self):
        """Create sample product."""
        return MockProduct()

    def test_cache_disabled(self, temp_cache_dir):
        """Test cache operations when disabled."""
        settings = MetadataCacheSettings(enabled=False, cache_dir=temp_cache_dir)
        cache = MetadataCache(settings, project_root=Path("/"))

        metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test",
            description="Test",
            hashtags=["#ad"],
            keywords=[],
            product_id="B0TEST",
        )

        # All operations should be no-ops when disabled
        assert cache.set(metadata) is False
        assert cache.get("B0TEST", "youtube") is None
        assert cache.invalidate("B0TEST") == 0
        assert cache.clear() == 0

    def test_cache_set_and_get(self, cache, sample_metadata, sample_product):
        """Test basic set and get operations."""
        # Set metadata in cache
        result = cache.set(sample_metadata, sample_product)
        assert result is True

        # Get metadata from cache
        cached = cache.get("B0TESTASIN", "youtube", sample_product)
        assert cached is not None
        assert cached.platform == "youtube"
        assert cached.title == "Test Product Review"
        assert cached.product_id == "B0TESTASIN"

    def test_cache_miss_not_found(self, cache):
        """Test cache miss when entry doesn't exist."""
        result = cache.get("NONEXISTENT", "youtube")
        assert result is None

    def test_cache_miss_expired(self, cache, sample_metadata, sample_product):
        """Test cache miss when entry is expired."""
        # Create cache with very short TTL
        settings = MetadataCacheSettings(
            enabled=True,
            ttl_hours=1,
            cache_dir=str(cache.cache_dir),
            max_entries=100,
        )
        short_ttl_cache = MetadataCache(settings, project_root=Path("/"))

        # Set metadata
        short_ttl_cache.set(sample_metadata, sample_product)

        # Manually expire the entry by modifying the file
        cache_path = short_ttl_cache._get_cache_path("B0TESTASIN", "youtube")
        with cache_path.open("r") as f:
            data = json.load(f)

        # Set expires_at to past
        past = datetime.now(UTC) - timedelta(hours=1)
        data["expires_at"] = past.isoformat()

        with cache_path.open("w") as f:
            json.dump(data, f)

        # Should return None for expired entry
        result = short_ttl_cache.get("B0TESTASIN", "youtube", sample_product)
        assert result is None

        # Expired file should be removed
        assert not cache_path.exists()

    def test_cache_miss_product_changed(self, cache, sample_metadata, sample_product):
        """Test cache miss when product data has changed."""
        # Set metadata with original product
        cache.set(sample_metadata, sample_product)

        # Create modified product
        modified_product = MockProduct(
            asin="B0TESTASIN",
            title="Modified Title",  # Changed
            description="Test product description",
            price="29.99",
            brand="TestBrand",
        )

        # Should return None because product hash doesn't match
        result = cache.get("B0TESTASIN", "youtube", modified_product)
        assert result is None

    def test_cache_invalidate_single_platform(
        self, cache, sample_metadata, sample_product
    ):
        """Test invalidating cache for single platform."""
        cache.set(sample_metadata, sample_product)

        # Verify it's cached
        assert cache.get("B0TESTASIN", "youtube", sample_product) is not None

        # Invalidate
        count = cache.invalidate("B0TESTASIN", "youtube")
        assert count == 1

        # Should be gone
        assert cache.get("B0TESTASIN", "youtube", sample_product) is None

    def test_cache_invalidate_all_platforms(self, cache, sample_product):
        """Test invalidating cache for all platforms."""
        # Create metadata for multiple platforms
        for platform in ["youtube", "tiktok", "instagram"]:
            metadata = PlatformMetadata.create(
                platform=platform,
                title="Test" if platform == "youtube" else None,
                description="Test caption",
                hashtags=["#ad"],
                keywords=[],
                product_id="B0TESTASIN",
            )
            cache.set(metadata, sample_product)

        # Invalidate all platforms
        count = cache.invalidate("B0TESTASIN")
        assert count == 3

        # All should be gone
        for platform in ["youtube", "tiktok", "instagram"]:
            assert cache.get("B0TESTASIN", platform, sample_product) is None

    def test_cache_clear(self, cache, sample_product):
        """Test clearing entire cache."""
        # Add multiple entries
        for i in range(5):
            metadata = PlatformMetadata.create(
                platform="youtube",
                title=f"Test {i}",
                description="Test",
                hashtags=["#ad"],
                keywords=[],
                product_id=f"B0TEST{i}",
            )
            cache.set(metadata, sample_product)

        # Clear cache
        count = cache.clear()
        assert count == 5

        # Verify empty
        stats = cache.get_stats()
        assert stats["total_entries"] == 0

    def test_cache_stats(self, cache, sample_product):
        """Test cache statistics."""
        # Empty cache stats
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["expired_entries"] == 0
        assert stats["enabled"] is True

        # Add entries
        for i in range(3):
            metadata = PlatformMetadata.create(
                platform="youtube",
                title=f"Test {i}",
                description="Test",
                hashtags=["#ad"],
                keywords=[],
                product_id=f"B0TEST{i}",
            )
            cache.set(metadata, sample_product)

        # Check stats
        stats = cache.get_stats()
        assert stats["total_entries"] == 3
        assert stats["expired_entries"] == 0

    def test_cache_corruption_handling(self, cache):
        """Test graceful handling of corrupted cache files."""
        product_id = "B0CORRUPT"
        platform = "youtube"

        # Create corrupted cache file
        cache_path = cache._get_cache_path(product_id, platform)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            f.write("{ invalid json }")

        # Should return None and remove corrupted file
        result = cache.get(product_id, platform)
        assert result is None
        assert not cache_path.exists()

    def test_cache_corruption_missing_fields(self, cache):
        """Test handling of cache file with missing fields."""
        product_id = "B0MISSING"
        platform = "youtube"

        # Create cache file with missing fields
        cache_path = cache._get_cache_path(product_id, platform)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump({"metadata": {}, "product_hash": "test"}, f)

        # Should return None and remove invalid file
        result = cache.get(product_id, platform)
        assert result is None
        assert not cache_path.exists()

    def test_compute_product_hash_deterministic(self, cache, sample_product):
        """Test that product hash is deterministic."""
        hash1 = cache.compute_product_hash(sample_product)
        hash2 = cache.compute_product_hash(sample_product)

        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA-256

    def test_compute_product_hash_changes_with_data(self, cache):
        """Test that product hash changes when data changes."""
        product1 = MockProduct(title="Original Title")
        product2 = MockProduct(title="Modified Title")

        hash1 = cache.compute_product_hash(product1)
        hash2 = cache.compute_product_hash(product2)

        assert hash1 != hash2

    def test_cache_path_sanitization(self, cache):
        """Test that product IDs are sanitized for filesystem."""
        # Product ID with special characters
        path = cache._get_cache_path("B0TEST/ID:123", "youtube")

        # Should not contain special characters
        assert "/" not in path.name
        assert ":" not in path.name
        assert path.name == "B0TEST_ID_123_youtube.json"

    def test_max_entries_enforcement(self, temp_cache_dir):
        """Test that max_entries limit is enforced."""
        settings = MetadataCacheSettings(
            enabled=True,
            ttl_hours=24,
            cache_dir=temp_cache_dir,
            max_entries=3,
        )
        cache = MetadataCache(settings, project_root=Path("/"))

        # Add more entries than max_entries
        for i in range(5):
            metadata = PlatformMetadata.create(
                platform="youtube",
                title=f"Test {i}",
                description="Test",
                hashtags=["#ad"],
                keywords=[],
                product_id=f"B0TEST{i:03d}",
            )
            cache.set(metadata)

        # Should have at most max_entries
        stats = cache.get_stats()
        assert stats["total_entries"] <= 3


class TestMetadataCacheIntegration:
    """Integration tests for cache with factory."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_cache_settings_in_platform_metadata_settings(self):
        """Test that cache settings are part of PlatformMetadataSettings."""
        from src.ai.platform_metadata.models import PlatformMetadataSettings

        settings = PlatformMetadataSettings()

        assert hasattr(settings, "cache")
        assert isinstance(settings.cache, MetadataCacheSettings)
        assert settings.cache.enabled is True
        assert settings.cache.ttl_hours == 24
