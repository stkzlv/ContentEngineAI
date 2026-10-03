"""Tests for metadata loading utilities."""

import json
from pathlib import Path

import pytest

from src.publisher.metadata import (
    _load_from_json,
    load_platform_metadata,
)
from src.publisher.models import Platform


@pytest.fixture
def product_dir(tmp_path):
    """Create product directory structure."""
    product = tmp_path / "B0TEST001"
    product.mkdir()
    return product


@pytest.fixture
def metadata_json():
    """Standard metadata JSON content."""
    return {
        "title": "Amazing Product Review",
        "description": "Check out this product! #shorts #review",
        "hashtags": ["#shorts", "review", "tech"],
        "keywords": ["product", "review"],
    }


class TestLoadPlatformMetadata:
    """Test load_platform_metadata function."""

    def test_loads_from_unified_json(self, product_dir, metadata_json):
        """Loads metadata from metadata.json (unified mode)."""
        json_path = product_dir / "metadata.json"
        json_path.write_text(json.dumps(metadata_json))

        result = load_platform_metadata(
            "B0TEST001", Platform.YOUTUBE, product_dir.parent
        )

        assert result is not None
        assert result.title == "Amazing Product Review"
        assert "Check out this product!" in result.description
        assert result.product_id == "B0TEST001"

    def test_loads_from_platform_specific_json(self, product_dir, metadata_json):
        """Falls back to metadata_youtube.json if unified not found."""
        json_path = product_dir / "metadata_youtube.json"
        json_path.write_text(json.dumps(metadata_json))

        result = load_platform_metadata(
            "B0TEST001", Platform.YOUTUBE, product_dir.parent
        )

        assert result is not None
        assert result.title == "Amazing Product Review"

    def test_returns_none_when_no_files(self, product_dir):
        """Returns None when no metadata files found."""
        result = load_platform_metadata(
            "B0TEST001", Platform.YOUTUBE, product_dir.parent
        )

        assert result is None

    def test_accepts_string_platform(self, product_dir, metadata_json):
        """Accepts platform as string, converts to enum."""
        json_path = product_dir / "metadata.json"
        json_path.write_text(json.dumps(metadata_json))

        result = load_platform_metadata("B0TEST001", "youtube", product_dir.parent)

        assert result is not None
        assert result.platform == Platform.YOUTUBE

    def test_invalid_string_platform_returns_none(self, product_dir):
        """Returns None for invalid platform string."""
        result = load_platform_metadata("B0TEST001", "fakePlatform", product_dir.parent)

        assert result is None

    def test_accepts_string_outputs_dir(self, product_dir, metadata_json):
        """Accepts outputs_dir as string."""
        json_path = product_dir / "metadata.json"
        json_path.write_text(json.dumps(metadata_json))

        result = load_platform_metadata(
            "B0TEST001", Platform.YOUTUBE, str(product_dir.parent)
        )

        assert result is not None

    def test_prefers_unified_over_platform_specific(self, product_dir):
        """Prefers metadata.json over metadata_youtube.json."""
        unified = product_dir / "metadata.json"
        unified.write_text(
            json.dumps({"title": "Unified", "description": "Unified desc"})
        )

        platform = product_dir / "metadata_youtube.json"
        platform.write_text(
            json.dumps({"title": "Platform", "description": "Platform desc"})
        )

        result = load_platform_metadata(
            "B0TEST001", Platform.YOUTUBE, product_dir.parent
        )

        assert result is not None
        assert result.title == "Unified"


class TestLoadFromJson:
    """Test _load_from_json helper."""

    def test_missing_file_returns_none(self, tmp_path):
        """Returns None for nonexistent file."""
        result = _load_from_json(
            tmp_path / "nonexistent.json", Platform.YOUTUBE, "B0TEST001"
        )

        assert result is None

    def test_invalid_json_returns_none(self, tmp_path):
        """Returns None for malformed JSON."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json")

        result = _load_from_json(bad_json, Platform.YOUTUBE, "B0TEST001")

        assert result is None

    def test_missing_description_returns_none(self, tmp_path):
        """Returns None when description is empty."""
        no_desc = tmp_path / "no_desc.json"
        no_desc.write_text(json.dumps({"title": "Title", "description": ""}))

        result = _load_from_json(no_desc, Platform.YOUTUBE, "B0TEST001")

        assert result is None

    def test_strips_hashtags_from_description(self, tmp_path):
        """Trailing hashtags are stripped from description."""
        json_path = tmp_path / "meta.json"
        json_path.write_text(
            json.dumps(
                {
                    "title": "Title",
                    "description": "Great product! #shorts #review",
                }
            )
        )

        result = _load_from_json(json_path, Platform.YOUTUBE, "B0TEST001")

        assert result is not None
        assert result.description == "Great product!"

    def test_normalizes_hashtags(self, tmp_path):
        """Hashtag # prefix is stripped."""
        json_path = tmp_path / "meta.json"
        json_path.write_text(
            json.dumps(
                {
                    "title": "Title",
                    "description": "Test",
                    "hashtags": ["#shorts", "review", "#tech"],
                }
            )
        )

        result = _load_from_json(json_path, Platform.YOUTUBE, "B0TEST001")

        assert result is not None
        assert result.hashtags == ["shorts", "review", "tech"]

    def test_empty_hashtags_and_keywords_defaults(self, tmp_path):
        """Empty hashtags and keywords default to empty lists."""
        json_path = tmp_path / "meta.json"
        json_path.write_text(json.dumps({"title": "Title", "description": "Test desc"}))

        result = _load_from_json(json_path, Platform.YOUTUBE, "B0TEST001")

        assert result is not None
        assert result.hashtags == []
        assert result.keywords == []
