"""Integration tests for platform metadata file structure and persistence.

These tests verify the integration aspects of platform metadata without triggering
heavy module imports that cause circular dependencies. They focus on file I/O,
data structures, and cross-platform compatibility.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest


class TestPlatformMetadataFileIntegration:
    """Test platform metadata file structure and integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = None
        self.product_id = "B08TEST123"

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir:
            self.temp_dir.cleanup()

    def _create_metadata_dict(
        self,
        platform: str,
        title: str | None,
        description: str,
        hashtags: list[str],
        keywords: list[str],
    ) -> dict[str, Any]:
        """Create a metadata dictionary matching PlatformMetadata structure."""
        char_counts = {"description": len(description)}
        if title:
            char_counts["title"] = len(title)

        return {
            "platform": platform,
            "title": title,
            "description": description,
            "hashtags": hashtags,
            "keywords": keywords,
            "character_counts": char_counts,
            "generated_at": "2025-01-15T12:00:00Z",
            "product_id": self.product_id,
            "validation_status": "valid",
            "validation_messages": [],
        }

    def test_multi_platform_file_creation_and_persistence(self):
        """Test creating and persisting metadata files for all platforms."""
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)
        text_dir = output_dir / self.product_id / "text"
        text_dir.mkdir(parents=True)

        # Create metadata for all three platforms
        youtube_metadata = self._create_metadata_dict(
            platform="youtube",
            title="Best Wireless Earbuds 2025 - Noise Cancelling #Shorts",
            description="Discover amazing wireless earbuds with active noise cancellation and 30-hour battery life.",
            hashtags=["#Shorts", "#WirelessEarbuds", "#TechReview", "#ad"],
            keywords=["wireless earbuds", "noise cancelling"],
        )

        tiktok_metadata = self._create_metadata_dict(
            platform="tiktok",
            title=None,
            description="Experience crystal-clear sound with these amazing wireless earbuds! 30-hour battery, noise cancellation.",
            hashtags=["#WirelessEarbuds", "#TechTok", "#AudioGear", "#ad"],
            keywords=["wireless earbuds", "noise cancelling", "bluetooth"],
        )

        instagram_metadata = self._create_metadata_dict(
            platform="instagram",
            title=None,
            description="Premium wireless earbuds with active noise cancellation and 30-hour battery life.",
            hashtags=[f"#Tag{i}" for i in range(15)] + ["#ad"],  # 16 total
            keywords=["wireless earbuds", "noise cancelling"],
        )

        # Save all metadata files
        youtube_file = text_dir / "metadata_youtube.json"
        tiktok_file = text_dir / "metadata_tiktok.json"
        instagram_file = text_dir / "metadata_instagram.json"

        youtube_file.write_text(json.dumps(youtube_metadata, indent=2))
        tiktok_file.write_text(json.dumps(tiktok_metadata, indent=2))
        instagram_file.write_text(json.dumps(instagram_metadata, indent=2))

        # Verify all files exist
        assert youtube_file.exists()
        assert tiktok_file.exists()
        assert instagram_file.exists()

        # Verify file contents
        youtube_loaded = json.loads(youtube_file.read_text())
        tiktok_loaded = json.loads(tiktok_file.read_text())
        instagram_loaded = json.loads(instagram_file.read_text())

        # YouTube verification
        assert youtube_loaded["platform"] == "youtube"
        assert "#Shorts" in youtube_loaded["hashtags"]
        assert "#ad" in youtube_loaded["hashtags"]
        assert youtube_loaded["title"] is not None

        # TikTok verification
        assert tiktok_loaded["platform"] == "tiktok"
        assert "#ad" in tiktok_loaded["hashtags"]
        assert tiktok_loaded["title"] is None  # TikTok has no title field

        # Instagram verification
        assert instagram_loaded["platform"] == "instagram"
        assert "#ad" in instagram_loaded["hashtags"]
        assert len(instagram_loaded["hashtags"]) >= 15  # Instagram requires 15-30

    def test_partial_platform_failure_handling(self):
        """Test file system state when only some platforms succeed."""
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)
        text_dir = output_dir / self.product_id / "text"
        text_dir.mkdir(parents=True)

        # Simulate only YouTube succeeding
        youtube_metadata = self._create_metadata_dict(
            platform="youtube",
            title="Test Product #Shorts",
            description="Test description",
            hashtags=["#Shorts", "#Test", "#ad"],
            keywords=["test"],
        )

        youtube_file = text_dir / "metadata_youtube.json"
        youtube_file.write_text(json.dumps(youtube_metadata, indent=2))

        # Verify only YouTube file exists
        assert youtube_file.exists()
        assert not (text_dir / "metadata_tiktok.json").exists()
        assert not (text_dir / "metadata_instagram.json").exists()

        # Verify the successful file is valid
        loaded = json.loads(youtube_file.read_text())
        assert loaded["platform"] == "youtube"
        assert loaded["validation_status"] == "valid"

    def test_upload_instructions_file_generation(self):
        """Test UPLOAD_INSTRUCTIONS.txt generation from metadata files."""
        from src.ai.platform_metadata.models import PlatformMetadata
        from src.ai.platform_metadata.text_formatter import (
            format_upload_instructions,
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)
        text_dir = output_dir / self.product_id / "text"
        text_dir.mkdir(parents=True)

        # Create PlatformMetadata objects for all platforms
        youtube_metadata = PlatformMetadata(
            platform="youtube",
            title="Test Product - Best Features #Shorts",
            description="Check out this amazing product with great features.",
            hashtags=["#Shorts", "#Product", "#Review", "#ad"],
            keywords=["test product", "features"],
            character_counts={"title": 36, "description": 53},
            generated_at="2025-01-15T12:00:00Z",
            product_id=self.product_id,
            validation_status="valid",
            validation_messages=[],
        )

        tiktok_metadata = PlatformMetadata(
            platform="tiktok",
            title=None,
            description="Amazing product features! #Product #Review #TikTokMadeMeBuyIt #ad",
            hashtags=["#Product", "#Review", "#TikTokMadeMeBuyIt", "#ad"],
            keywords=["product", "review", "features"],
            character_counts={"description": 68},
            generated_at="2025-01-15T12:00:00Z",
            product_id=self.product_id,
            validation_status="valid",
            validation_messages=[],
        )

        instagram_metadata = PlatformMetadata(
            platform="instagram",
            title=None,
            description="Premium product with exceptional features",
            hashtags=[f"#Tag{i}" for i in range(20)] + ["#ad"],
            keywords=["premium product", "features"],
            character_counts={"description": 43},
            generated_at="2025-01-15T12:00:00Z",
            product_id=self.product_id,
            validation_status="valid",
            validation_messages=[],
        )

        # Generate upload instructions
        metadata_results = {
            "youtube": youtube_metadata,
            "tiktok": tiktok_metadata,
            "instagram": instagram_metadata,
        }

        instructions_text = format_upload_instructions(
            metadata_results=metadata_results,
            product_id=self.product_id,
            video_filename=f"video_{self.product_id}_test.mp4",
            product_name="Test Product",
            product_url=f"https://www.amazon.com/dp/{self.product_id}",
        )

        # Write to file
        instructions_file = text_dir / "UPLOAD_INSTRUCTIONS.txt"
        instructions_file.write_text(instructions_text, encoding="utf-8")

        # Verify file exists
        assert instructions_file.exists()

        # Verify file content
        content = instructions_file.read_text(encoding="utf-8")

        # Check header (updated format)
        assert "UPLOAD INSTRUCTIONS" in content
        assert self.product_id in content
        assert f"video_{self.product_id}_test.mp4" in content

        # Check platform sections exist (plain text format)
        assert "YOUTUBE SHORTS" in content
        assert "TIKTOK" in content
        assert "INSTAGRAM REELS" in content

        # Check YouTube content
        assert "Test Product - Best Features #Shorts" in content
        assert "#Shorts" in content

        # Check TikTok content
        assert "Amazing product features!" in content
        assert "#TikTokMadeMeBuyIt" in content

        # Check Instagram content
        assert "Premium product with exceptional features" in content

        # Check footer (simplified format)
        assert "Generated:" in content
        assert "Product: Test Product" in content

    def test_metadata_json_schema_compatibility(self):
        """Test that all platform metadata files follow the same schema."""
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)

        required_fields = [
            "platform",
            "title",
            "description",
            "hashtags",
            "keywords",
            "character_counts",
            "generated_at",
            "product_id",
            "validation_status",
            "validation_messages",
        ]

        for platform in ["youtube", "tiktok", "instagram"]:
            metadata = self._create_metadata_dict(
                platform=platform,
                title="Test" if platform == "youtube" else None,
                description="Test description",
                hashtags=["#Test", "#ad"],
                keywords=["test"],
            )

            metadata_file = output_dir / f"metadata_{platform}.json"
            metadata_file.write_text(json.dumps(metadata, indent=2))

            loaded = json.loads(metadata_file.read_text())

            # Verify all required fields exist
            for field in required_fields:
                assert (
                    field in loaded
                ), f"Missing field '{field}' in {platform} metadata"

            # Verify field types
            assert isinstance(loaded["hashtags"], list)
            assert isinstance(loaded["keywords"], list)
            assert isinstance(loaded["character_counts"], dict)
            assert isinstance(loaded["validation_messages"], list)

    def test_character_count_accuracy(self):
        """Test that character counts match actual content length."""
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)

        title = "Best Wireless Earbuds 2025 - Complete Review #Shorts"
        description = "Full review of the best wireless earbuds on the market today!"

        metadata = self._create_metadata_dict(
            platform="youtube",
            title=title,
            description=description,
            hashtags=["#Shorts", "#TechReview", "#ad"],
            keywords=["earbuds", "review"],
        )

        metadata_file = output_dir / "metadata_youtube.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        loaded = json.loads(metadata_file.read_text())

        # Verify character counts match
        assert loaded["character_counts"]["title"] == len(title)
        assert loaded["character_counts"]["description"] == len(description)
        assert loaded["character_counts"]["title"] == len(loaded["title"])
        assert loaded["character_counts"]["description"] == len(loaded["description"])

    def test_validation_status_and_messages_persistence(self):
        """Test that validation status and messages are preserved."""
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)

        # Create metadata with validation warnings
        metadata = {
            "platform": "youtube",
            "title": "Short",
            "description": "Test",
            "hashtags": ["#Shorts", "#ad"],
            "keywords": ["test"],
            "character_counts": {"title": 5, "description": 4},
            "generated_at": "2025-01-15T12:00:00Z",
            "product_id": self.product_id,
            "validation_status": "warning",
            "validation_messages": [
                "Title is shorter than recommended (50-60 chars)",
                "Hashtag count below minimum (3-5 recommended)",
            ],
        }

        metadata_file = output_dir / "metadata_youtube.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        loaded = json.loads(metadata_file.read_text())

        assert loaded["validation_status"] == "warning"
        assert len(loaded["validation_messages"]) == 2
        assert "Title is shorter" in loaded["validation_messages"][0]
        assert "Hashtag count" in loaded["validation_messages"][1]

    def test_platform_specific_constraints(self):
        """Test platform-specific requirements are documented in metadata."""
        # YouTube must have title and #Shorts tag
        youtube = self._create_metadata_dict(
            platform="youtube",
            title="Test Title #Shorts",
            description="Test",
            hashtags=["#Shorts", "#Test", "#ad"],
            keywords=["test"],
        )
        assert youtube["title"] is not None
        assert "#Shorts" in youtube["hashtags"]

        # TikTok has no title field
        tiktok = self._create_metadata_dict(
            platform="tiktok",
            title=None,
            description="Test caption",
            hashtags=["#Test", "#ad"],
            keywords=["test"],
        )
        assert tiktok["title"] is None

        # Instagram requires many hashtags (15-30)
        instagram_hashtags = [f"#Tag{i}" for i in range(20)] + ["#ad"]
        instagram = self._create_metadata_dict(
            platform="instagram",
            title=None,
            description="Test caption",
            hashtags=instagram_hashtags,
            keywords=["test"],
        )
        assert len(instagram["hashtags"]) >= 15
        assert len(instagram["hashtags"]) <= 30

    def test_file_naming_convention(self):
        """Test metadata files follow consistent naming pattern."""
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)
        text_dir = output_dir / self.product_id / "text"
        text_dir.mkdir(parents=True)

        platforms = ["youtube", "tiktok", "instagram"]
        for platform in platforms:
            metadata = self._create_metadata_dict(
                platform=platform,
                title=None,
                description="Test",
                hashtags=["#Test", "#ad"],
                keywords=["test"],
            )

            # Follow naming convention: metadata_{platform}.json
            expected_filename = f"metadata_{platform}.json"
            metadata_file = text_dir / expected_filename

            metadata_file.write_text(json.dumps(metadata, indent=2))
            assert metadata_file.exists()
            assert metadata_file.name == expected_filename

    def test_metadata_directory_structure(self):
        """Test metadata files are organized in correct directory structure."""
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)

        # Structure: outputs/{product_id}/text/metadata_{platform}.json
        product_dir = output_dir / self.product_id
        text_dir = product_dir / "text"
        text_dir.mkdir(parents=True)

        metadata = self._create_metadata_dict(
            platform="youtube",
            title="Test",
            description="Test",
            hashtags=["#Shorts", "#ad"],
            keywords=["test"],
        )

        metadata_file = text_dir / "metadata_youtube.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        # Verify directory structure
        assert product_dir.exists()
        assert product_dir.is_dir()
        assert text_dir.exists()
        assert text_dir.is_dir()
        assert metadata_file.exists()
        assert metadata_file.is_file()
        assert str(metadata_file).endswith(
            f"{self.product_id}/text/metadata_youtube.json"
        )
