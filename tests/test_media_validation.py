"""Tests for media validation logic with profile awareness."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.video.producer import validate_media_requirements
from src.video.stock_media import StockMediaInfo
from src.video.video_config import VideoProfile


class TestProfileAwareValidation:
    """Test profile-aware media validation logic."""

    def create_mock_profile(self, use_scraped_videos=True, description="Test Profile"):
        """Create a mock video profile for testing."""
        profile = Mock(spec=VideoProfile)
        profile.use_scraped_videos = use_scraped_videos
        profile.description = description
        return profile

    def create_stock_images(self, count=0):
        """Create mock stock image items."""
        return [
            StockMediaInfo(
                source="pexels",
                type="image",
                url=f"https://example.com/image{i}.jpg",
                author=f"Author {i}",
                path=Path(f"/mock/path/image{i}.jpg"),
            )
            for i in range(count)
        ]

    def create_stock_videos(self, count=0):
        """Create mock stock video items."""
        return [
            StockMediaInfo(
                source="pexels",
                type="video",
                url=f"https://example.com/video{i}.mp4",
                author=f"Author {i}",
                path=Path(f"/mock/path/video{i}.mp4"),
            )
            for i in range(count)
        ]

    @pytest.mark.unit
    def test_profile_excludes_videos_needs_5_images(self):
        """Test validation when profile excludes videos and needs 5+ images."""
        profile = self.create_mock_profile(use_scraped_videos=False)
        scraped_images: list = []
        scraped_videos: list = []
        stock_media = self.create_stock_images(4)  # Only 4 images, need 5

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile
        )

        assert not is_valid
        assert (
            "Profile excludes videos, need at least 5 images but only have 4" in reason
        )

    @pytest.mark.unit
    def test_profile_excludes_videos_with_enough_images(self):
        """Test validation passes when profile excludes videos but has 5+ images."""
        profile = self.create_mock_profile(use_scraped_videos=False)
        scraped_images = [Path("/mock/scraped1.jpg")]
        scraped_videos: list = []
        stock_media = self.create_stock_images(4)  # 1 scraped + 4 stock = 5 images

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile
        )

        assert is_valid
        assert "Media validation passed: 5 images, 0 videos, 4 stock items" in reason

    @pytest.mark.unit
    def test_profile_allows_videos_with_videos_and_images(self):
        """Test validation when profile allows videos and has both videos and images."""
        profile = self.create_mock_profile(use_scraped_videos=True)
        scraped_images = [Path("/mock/scraped1.jpg")]
        scraped_videos = [Path("/mock/scraped1.mp4")]
        stock_media = self.create_stock_images(1)  # 2 images + 1 video = 3 total

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile
        )

        assert is_valid
        assert "Media validation passed: 2 images, 1 videos, 1 stock items" in reason

    @pytest.mark.unit
    def test_profile_allows_videos_but_no_videos_found(self):
        """Test validation when profile allows videos but none found, needs
        5+ images.
        """
        profile = self.create_mock_profile(use_scraped_videos=True)
        scraped_images: list = []
        scraped_videos: list = []
        stock_media = self.create_stock_images(
            4
        )  # Only 4 images, need 5 when no videos

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile
        )

        assert not is_valid
        assert "No videos found, need at least 5 images but only have 4" in reason

    @pytest.mark.unit
    def test_mixed_stock_media_types(self):
        """Test validation with mixed stock media types (images and videos)."""
        profile = self.create_mock_profile(use_scraped_videos=True)
        scraped_images: list = []
        scraped_videos: list = []
        # Create mixed stock media: 2 images + 1 video = 3 total
        stock_media = self.create_stock_images(2) + self.create_stock_videos(1)

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile
        )

        assert is_valid
        assert "Media validation passed: 2 images, 1 videos, 3 stock items" in reason

    @pytest.mark.unit
    def test_insufficient_total_media(self):
        """Test validation fails when total media is below minimum (3)."""
        profile = self.create_mock_profile(use_scraped_videos=True)
        scraped_images = [Path("/mock/scraped1.jpg")]
        scraped_videos: list = []
        stock_media = self.create_stock_images(1)  # Only 2 total, need 3 minimum

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile
        )

        assert not is_valid
        assert "Insufficient total media: 2 items (minimum 3)" in reason

    @pytest.mark.unit
    def test_has_videos_but_insufficient_images(self):
        """Test validation when has videos but insufficient images for
        video+image mix.
        """
        profile = self.create_mock_profile(use_scraped_videos=True)
        scraped_images: list = []  # 0 images, need at least 2 when have videos
        scraped_videos = [Path("/mock/scraped1.mp4")]
        stock_media = self.create_stock_videos(2)  # 3 videos total, 0 images

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile
        )

        assert not is_valid
        assert "Have 3 video(s) but only 0 image(s), need at least 2" in reason

    @pytest.mark.unit
    def test_profile_without_use_scraped_videos_attribute(self):
        """Test validation with profile that doesn't have use_scraped_videos
        attribute.
        """
        profile = Mock()
        # Intentionally don't set use_scraped_videos to test getattr default
        scraped_images: list = []
        scraped_videos: list = []
        stock_media = self.create_stock_images(5)  # 5 images should pass

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile
        )

        assert is_valid  # Should default to True and pass validation

    @pytest.mark.unit
    def test_validation_success_message_format(self):
        """Test that success message format is correct with different media counts."""
        profile = self.create_mock_profile(use_scraped_videos=True)
        scraped_images = [Path("/mock/scraped1.jpg"), Path("/mock/scraped2.jpg")]
        scraped_videos = [Path("/mock/scraped1.mp4")]
        stock_media = self.create_stock_images(2) + self.create_stock_videos(1)

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile
        )

        assert is_valid
        # Should count: 2 scraped + 2 stock = 4 images, 1 scraped + 1 stock = 2 videos
        assert "Media validation passed: 4 images, 2 videos, 3 stock items" in reason

    @pytest.mark.unit
    def test_empty_stock_media_list(self):
        """Test validation with empty stock media list."""
        profile = self.create_mock_profile(use_scraped_videos=False)
        scraped_images = [Path(f"/mock/scraped{i}.jpg") for i in range(5)]
        scraped_videos: list = []
        stock_media: list = []  # Empty stock media

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile
        )

        assert is_valid
        assert "Media validation passed: 5 images, 0 videos, 0 stock items" in reason
