"""Persistent caching and connection pooling utilities."""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic caching
T = TypeVar("T")


class PersistentCache:
    """Persistent file-based cache for API responses and metadata."""

    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, key: str) -> str:
        """Generate a cache key hash from the input string."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.json"

    def get(self, key: str) -> Any | None:
        """Get cached value if it exists and hasn't expired."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with cache_path.open("r") as f:
                cached_data = json.load(f)

            # Check expiration
            cached_time = cached_data.get("timestamp", 0)
            if time.time() - cached_time > self.ttl_seconds:
                cache_path.unlink(missing_ok=True)
                return None

            return cached_data.get("data")

        except Exception as e:
            logger.warning(f"Error reading cache file {cache_path}: {e}")
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache with timestamp."""
        cache_path = self._get_cache_path(key)

        try:
            cached_data = {"timestamp": time.time(), "data": value}

            with cache_path.open("w") as f:
                json.dump(cached_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Error writing to cache file {cache_path}: {e}")

    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error removing cache file {cache_file}: {e}")


# Global instances
_cache_dir = Path.home() / ".cache" / "content-engine-ai"
_media_cache = PersistentCache(_cache_dir / "media_metadata", ttl_seconds=86400)  # 24h


def cache_media_metadata(file_path: Path, metadata: dict[str, Any]) -> None:
    """Cache media file metadata."""
    cache_key = f"media:{file_path}:{file_path.stat().st_mtime}"
    _media_cache.set(cache_key, metadata)


def get_cached_media_metadata(file_path: Path) -> dict[str, Any] | None:
    """Get cached media file metadata."""
    if not file_path.exists():
        return None
    cache_key = f"media:{file_path}:{file_path.stat().st_mtime}"
    return _media_cache.get(cache_key)
