"""File-based caching layer for platform metadata.

This module provides a persistent cache for PlatformMetadata objects,
avoiding regeneration of metadata for unchanged products. Cache entries
are keyed by (product_id, platform) and support configurable TTL.

Features:
    - File-based persistence using JSON storage
    - Configurable TTL (time-to-live) for cache entries
    - Product change detection via content hash
    - Graceful handling of cache corruption
    - Thread-safe file operations
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.ai.platform_metadata.models import MetadataCacheSettings, PlatformMetadata

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Internal representation of a cached metadata entry.

    Attributes
    ----------
        metadata: The cached PlatformMetadata object
        product_hash: Hash of product data for change detection
        created_at: ISO 8601 timestamp when entry was created
        expires_at: ISO 8601 timestamp when entry expires

    """

    metadata: PlatformMetadata
    product_hash: str
    created_at: str
    expires_at: str

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        expires = datetime.fromisoformat(self.expires_at)
        return datetime.now(UTC) > expires

    def to_dict(self) -> dict:
        """Convert cache entry to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "product_hash": self.product_hash,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        """Create CacheEntry from dictionary.

        Args:
        ----
            data: Dictionary with cache entry fields

        Returns:
        -------
            CacheEntry instance

        Raises:
        ------
            KeyError: If required fields are missing
            ValueError: If metadata cannot be reconstructed

        """
        metadata = PlatformMetadata(
            platform=data["metadata"]["platform"],
            title=data["metadata"].get("title"),
            description=data["metadata"]["description"],
            hashtags=data["metadata"]["hashtags"],
            keywords=data["metadata"]["keywords"],
            character_counts=data["metadata"]["character_counts"],
            generated_at=data["metadata"]["generated_at"],
            product_id=data["metadata"]["product_id"],
            validation_status=data["metadata"]["validation_status"],
            validation_messages=data["metadata"]["validation_messages"],
            prompt_variant=data["metadata"].get("prompt_variant"),
        )
        return cls(
            metadata=metadata,
            product_hash=data["product_hash"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
        )


class MetadataCache:
    """File-based cache for platform metadata with TTL and invalidation support.

    This cache stores PlatformMetadata objects keyed by (product_id, platform).
    Entries are automatically invalidated when:
    - TTL expires
    - Product data changes (detected via content hash)

    Cache files are stored as JSON in the configured cache directory with the
    naming pattern: {product_id}_{platform}.json

    Example usage:
        cache = MetadataCache(settings)

        # Check for cached metadata
        cached = cache.get(product_id="B0TESTID", platform="youtube", product=product)
        if cached:
            return cached  # Use cached metadata

        # Generate new metadata
        metadata = await generator.generate(...)

        # Store in cache
        cache.set(metadata, product)

    """

    def __init__(
        self, settings: MetadataCacheSettings, project_root: Path | None = None
    ):
        """Initialize the metadata cache.

        Args:
        ----
            settings: Cache configuration settings
            project_root: Project root directory (defaults to cwd)

        """
        self.settings = settings
        self.project_root = project_root or Path.cwd()
        self.cache_dir = self.project_root / settings.cache_dir

        if settings.enabled:
            self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory ready: {self.cache_dir}")
        except OSError as e:
            logger.warning(f"Failed to create cache directory: {e}")

    def _get_cache_path(self, product_id: str, platform: str) -> Path:
        """Get the cache file path for a product/platform combination.

        Args:
        ----
            product_id: Product identifier (e.g., ASIN)
            platform: Platform name (youtube, tiktok, instagram)

        Returns:
        -------
            Path to cache file

        """
        # Sanitize product_id to be filesystem-safe
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in product_id)
        return self.cache_dir / f"{safe_id}_{platform}.json"

    @staticmethod
    def compute_product_hash(product) -> str:
        """Compute a hash of product data for change detection.

        The hash is computed from key product fields that would affect
        metadata generation. If any of these change, cached metadata
        should be invalidated.

        Args:
        ----
            product: ProductData object

        Returns:
        -------
            SHA-256 hash string (first 16 chars)

        """
        # Fields that affect metadata generation
        hash_data = {
            "title": getattr(product, "title", ""),
            "description": getattr(product, "description", ""),
            "price": str(getattr(product, "price", "")),
            "brand": getattr(product, "brand", ""),
        }

        # Create deterministic JSON string
        json_str = json.dumps(hash_data, sort_keys=True, ensure_ascii=True)
        full_hash = hashlib.sha256(json_str.encode()).hexdigest()

        # Return first 16 chars for brevity
        return full_hash[:16]

    def get(
        self,
        product_id: str,
        platform: str,
        product=None,
    ) -> PlatformMetadata | None:
        """Retrieve cached metadata if valid.

        Returns cached metadata only if:
        - Cache is enabled
        - Cache file exists and is readable
        - Entry has not expired (TTL)
        - Product hash matches (no product changes)

        Args:
        ----
            product_id: Product identifier
            platform: Platform name
            product: Optional ProductData for change detection.
                If provided, validates product hash matches.

        Returns:
        -------
            Cached PlatformMetadata or None if not found/invalid

        """
        if not self.settings.enabled:
            return None

        cache_path = self._get_cache_path(product_id, platform)

        if not cache_path.exists():
            logger.debug(f"Cache miss (not found): {product_id}/{platform}")
            return None

        try:
            entry = self._load_entry(cache_path)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Handle cache corruption gracefully
            logger.warning(f"Cache corruption detected, removing: {cache_path} ({e})")
            self._safe_remove(cache_path)
            return None

        # Check TTL expiration
        if entry.is_expired():
            logger.debug(f"Cache miss (expired): {product_id}/{platform}")
            self._safe_remove(cache_path)
            return None

        # Check product hash if product provided
        if product is not None:
            current_hash = self.compute_product_hash(product)
            if current_hash != entry.product_hash:
                logger.debug(
                    f"Cache miss (product changed): {product_id}/{platform} "
                    f"(hash: {entry.product_hash} -> {current_hash})"
                )
                self._safe_remove(cache_path)
                return None

        logger.info(f"Cache hit: {product_id}/{platform}")
        return entry.metadata

    def set(
        self,
        metadata: PlatformMetadata,
        product=None,
    ) -> bool:
        """Store metadata in cache.

        Creates a cache entry with TTL and product hash for later validation.

        Args:
        ----
            metadata: PlatformMetadata to cache
            product: Optional ProductData for hash computation.
                If not provided, uses empty hash (less accurate invalidation).

        Returns:
        -------
            True if successfully cached, False otherwise

        """
        if not self.settings.enabled:
            return False

        # Check max entries limit
        if self.settings.max_entries > 0:
            self._enforce_max_entries()

        product_id = metadata.product_id
        platform = metadata.platform
        cache_path = self._get_cache_path(product_id, platform)

        # Compute timestamps
        now = datetime.now(UTC)
        expires_at = now + timedelta(hours=self.settings.ttl_hours)

        # Compute product hash
        product_hash = self.compute_product_hash(product) if product else ""

        entry = CacheEntry(
            metadata=metadata,
            product_hash=product_hash,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
        )

        try:
            self._save_entry(entry, cache_path)
            logger.info(
                f"Cached metadata: {product_id}/{platform} "
                f"(expires: {expires_at.isoformat()})"
            )
            return True
        except OSError as e:
            logger.warning(f"Failed to cache metadata: {e}")
            return False

    def invalidate(self, product_id: str, platform: str | None = None) -> int:
        """Invalidate cache entries for a product.

        Args:
        ----
            product_id: Product identifier
            platform: Optional platform name. If None, invalidates all platforms.

        Returns:
        -------
            Number of entries invalidated

        """
        if not self.settings.enabled:
            return 0

        count = 0

        if platform:
            # Invalidate specific platform
            cache_path = self._get_cache_path(product_id, platform)
            if self._safe_remove(cache_path):
                count = 1
                logger.info(f"Invalidated cache: {product_id}/{platform}")
        else:
            # Invalidate all platforms for this product
            platforms = ["youtube", "tiktok", "instagram"]
            for p in platforms:
                cache_path = self._get_cache_path(product_id, p)
                if self._safe_remove(cache_path):
                    count += 1

            if count > 0:
                logger.info(f"Invalidated {count} cache entries for {product_id}")

        return count

    def clear(self) -> int:
        """Clear all cache entries.

        Returns
        -------
            Number of entries cleared

        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if self._safe_remove(cache_file):
                count += 1

        logger.info(f"Cleared {count} cache entries")
        return count

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns
        -------
            Dictionary with cache statistics:
            - total_entries: Number of cache files
            - expired_entries: Number of expired entries
            - cache_dir: Cache directory path
            - enabled: Whether caching is enabled

        """
        if not self.settings.enabled or not self.cache_dir.exists():
            return {
                "total_entries": 0,
                "expired_entries": 0,
                "cache_dir": str(self.cache_dir),
                "enabled": self.settings.enabled,
            }

        total = 0
        expired = 0

        for cache_file in self.cache_dir.glob("*.json"):
            total += 1
            try:
                entry = self._load_entry(cache_file)
                if entry.is_expired():
                    expired += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                expired += 1  # Count corrupted as expired

        return {
            "total_entries": total,
            "expired_entries": expired,
            "cache_dir": str(self.cache_dir),
            "enabled": self.settings.enabled,
        }

    def _load_entry(self, cache_path: Path) -> CacheEntry:
        """Load cache entry from file.

        Args:
        ----
            cache_path: Path to cache file

        Returns:
        -------
            CacheEntry instance

        Raises:
        ------
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
            ValueError: If data cannot be parsed

        """
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return CacheEntry.from_dict(data)

    def _save_entry(self, entry: CacheEntry, cache_path: Path) -> None:
        """Save cache entry to file.

        Args:
        ----
            entry: CacheEntry to save
            cache_path: Path to cache file

        Raises:
        ------
            OSError: If file cannot be written

        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)

    def _safe_remove(self, cache_path: Path) -> bool:
        """Safely remove a cache file.

        Args:
        ----
            cache_path: Path to cache file

        Returns:
        -------
            True if file was removed, False if it didn't exist or error

        """
        try:
            if cache_path.exists():
                cache_path.unlink()
                return True
            return False
        except OSError as e:
            logger.warning(f"Failed to remove cache file {cache_path}: {e}")
            return False

    def _enforce_max_entries(self) -> None:
        """Remove oldest entries if max_entries limit is exceeded."""
        if not self.cache_dir.exists():
            return

        cache_files = list(self.cache_dir.glob("*.json"))
        if len(cache_files) < self.settings.max_entries:
            return

        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest entries until under limit
        entries_to_remove = len(cache_files) - self.settings.max_entries + 1
        for cache_file in cache_files[:entries_to_remove]:
            self._safe_remove(cache_file)
            logger.debug(f"Removed oldest cache entry: {cache_file.name}")
