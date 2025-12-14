"""Unit tests for platform metadata models."""

import pytest
from pydantic import ValidationError

# Import directly from models.py to avoid circular import issues with __init__.py
from src.ai.platform_metadata import models


class TestPlatformMetadata:
    """Test models.PlatformMetadata dataclass."""

    def test_platform_metadata_valid_youtube(self):
        """Test valid YouTube metadata creation."""
        metadata = models.PlatformMetadata(
            platform="youtube",
            title="Best Wireless Earbuds Under $50",
            description="Looking for affordable wireless earbuds?",
            hashtags=["#Shorts", "#WirelessEarbuds", "#BudgetTech"],
            keywords=["wireless earbuds", "budget tech", "affordable earbuds"],
            character_counts={"title": 35, "description": 42},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0ASIN123",
            validation_status="valid",
            validation_messages=[],
        )

        assert metadata.platform == "youtube"
        assert metadata.title == "Best Wireless Earbuds Under $50"
        assert metadata.description == "Looking for affordable wireless earbuds?"
        assert len(metadata.hashtags) == 3
        assert len(metadata.keywords) == 3
        assert metadata.character_counts["title"] == 35
        assert metadata.character_counts["description"] == 42
        assert metadata.product_id == "B0ASIN123"
        assert metadata.validation_status == "valid"
        assert metadata.validation_messages == []

    def test_platform_metadata_valid_tiktok(self):
        """Test valid TikTok metadata creation (no title)."""
        metadata = models.PlatformMetadata(
            platform="tiktok",
            description="Best wireless earbuds under $50 with amazing sound quality",
            hashtags=["#WirelessEarbuds", "#BudgetTech", "#TechReview"],
            keywords=["wireless earbuds under 50", "budget tech"],
            character_counts={"description": 63},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0ASIN123",
            validation_status="valid",
            validation_messages=[],
            title=None,
        )

        assert metadata.platform == "tiktok"
        assert metadata.title is None
        assert metadata.description == "Best wireless earbuds under $50 with amazing sound quality"
        assert "title" not in metadata.character_counts
        assert metadata.validation_status == "valid"

    def test_platform_metadata_to_dict(self):
        """Test metadata serialization to dictionary."""
        metadata = models.PlatformMetadata.create(
            platform="instagram",
            description="Game changer alert 🔥",
            hashtags=["#WirelessEarbuds", "#TechGadgets"],
            keywords=["wireless earbuds", "budget tech"],
            product_id="B0ASIN123",
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["platform"] == "instagram"
        assert result["description"] == "Game changer alert 🔥"
        assert result["hashtags"] == ["#WirelessEarbuds", "#TechGadgets"]
        assert result["keywords"] == ["wireless earbuds", "budget tech"]
        assert result["product_id"] == "B0ASIN123"
        assert "generated_at" in result
        assert "validation_status" in result
        assert "character_counts" in result

    def test_platform_metadata_create_factory_with_title(self):
        """Test factory method with title (YouTube)."""
        metadata = models.PlatformMetadata.create(
            platform="youtube",
            title="Best Earbuds 2025",
            description="Looking for affordable wireless earbuds?",
            hashtags=["#Shorts", "#Tech"],
            keywords=["wireless earbuds", "budget"],
            product_id="B0ASIN123",
        )

        assert metadata.platform == "youtube"
        assert metadata.title == "Best Earbuds 2025"
        assert metadata.character_counts["title"] == len("Best Earbuds 2025")
        assert metadata.character_counts["description"] == len("Looking for affordable wireless earbuds?")
        assert metadata.validation_status == "valid"
        assert metadata.validation_messages == []
        assert metadata.generated_at is not None

    def test_platform_metadata_create_factory_without_title(self):
        """Test factory method without title (TikTok/Instagram)."""
        metadata = models.PlatformMetadata.create(
            platform="tiktok",
            description="Best wireless earbuds under $50",
            hashtags=["#WirelessEarbuds", "#BudgetTech"],
            keywords=["wireless earbuds"],
            product_id="B0ASIN123",
        )

        assert metadata.title is None
        assert "title" not in metadata.character_counts
        assert metadata.character_counts["description"] == len("Best wireless earbuds under $50")

    def test_platform_metadata_create_with_validation_messages(self):
        """Test factory method with validation warnings."""
        metadata = models.PlatformMetadata.create(
            platform="youtube",
            description="Short desc",
            hashtags=["#Tech"],
            keywords=["tech"],
            product_id="B0ASIN123",
            validation_status="warning",
            validation_messages=["Title length below recommended minimum"],
        )

        assert metadata.validation_status == "warning"
        assert len(metadata.validation_messages) == 1
        assert "Title length below recommended minimum" in metadata.validation_messages

    def test_platform_metadata_immutable(self):
        """Test that metadata is immutable (frozen dataclass)."""
        metadata = models.PlatformMetadata.create(
            platform="youtube",
            description="Test",
            hashtags=["#Tech"],
            keywords=["tech"],
            product_id="B0ASIN123",
        )

        with pytest.raises(AttributeError):
            metadata.platform = "tiktok"

        with pytest.raises(AttributeError):
            metadata.validation_status = "error"

    def test_platform_metadata_empty_lists(self):
        """Test metadata with empty hashtags and keywords."""
        metadata = models.PlatformMetadata.create(
            platform="youtube",
            description="Test description",
            hashtags=[],
            keywords=[],
            product_id="B0ASIN123",
        )

        assert metadata.hashtags == []
        assert metadata.keywords == []
        assert metadata.validation_status == "valid"


class TestYouTubePlatformSettings:
    """Test models.YouTubePlatformSettings model."""

    def test_youtube_settings_valid(self):
        """Test valid YouTube settings creation."""
        settings = models.YouTubePlatformSettings(
            enabled=True,
            title_length_max=60,
            description_length_max=5000,
            hashtag_count_min=3,
            hashtag_count_max=5,
            include_shorts_tag=True,
            seo_keywords=True,
        )

        assert settings.enabled is True
        assert settings.title_length_max == 60
        assert settings.description_length_max == 5000
        assert settings.hashtag_count_min == 3
        assert settings.hashtag_count_max == 5
        assert settings.include_shorts_tag is True
        assert settings.seo_keywords is True

    def test_youtube_settings_defaults(self):
        """Test YouTube settings with default values."""
        settings = models.YouTubePlatformSettings()

        assert settings.enabled is True
        assert settings.title_length_max == 60
        assert settings.description_length_max == 5000
        assert settings.hashtag_count_min == 3
        assert settings.hashtag_count_max == 5
        assert settings.include_shorts_tag is True
        assert settings.seo_keywords is True

    def test_youtube_settings_invalid_title_length_max_too_high(self):
        """Test YouTube settings with title length exceeding maximum."""
        with pytest.raises(ValidationError) as exc_info:
            models.YouTubePlatformSettings(title_length_max=101)

        assert "title_length_max" in str(exc_info.value)

    def test_youtube_settings_invalid_title_length_max_too_low(self):
        """Test YouTube settings with title length below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            models.YouTubePlatformSettings(title_length_max=0)

        assert "title_length_max" in str(exc_info.value)

    def test_youtube_settings_invalid_description_length_max(self):
        """Test YouTube settings with description length exceeding limit."""
        with pytest.raises(ValidationError) as exc_info:
            models.YouTubePlatformSettings(description_length_max=5001)

        assert "description_length_max" in str(exc_info.value)

    def test_youtube_settings_invalid_hashtag_count_min(self):
        """Test YouTube settings with invalid hashtag minimum."""
        with pytest.raises(ValidationError) as exc_info:
            models.YouTubePlatformSettings(hashtag_count_min=11)

        assert "hashtag_count_min" in str(exc_info.value)

    def test_youtube_settings_invalid_hashtag_count_max(self):
        """Test YouTube settings with invalid hashtag maximum."""
        with pytest.raises(ValidationError) as exc_info:
            models.YouTubePlatformSettings(hashtag_count_max=16)

        assert "hashtag_count_max" in str(exc_info.value)

    def test_youtube_settings_edge_case_min_values(self):
        """Test YouTube settings with minimum allowed values."""
        settings = models.YouTubePlatformSettings(
            title_length_max=1,
            description_length_max=1,
            hashtag_count_min=0,
            hashtag_count_max=1,
        )

        assert settings.title_length_max == 1
        assert settings.description_length_max == 1
        assert settings.hashtag_count_min == 0
        assert settings.hashtag_count_max == 1

    def test_youtube_settings_edge_case_max_values(self):
        """Test YouTube settings with maximum allowed values."""
        settings = models.YouTubePlatformSettings(
            title_length_max=100,
            description_length_max=5000,
            hashtag_count_min=10,
            hashtag_count_max=15,
        )

        assert settings.title_length_max == 100
        assert settings.description_length_max == 5000
        assert settings.hashtag_count_min == 10
        assert settings.hashtag_count_max == 15


class TestTikTokPlatformSettings:
    """Test models.TikTokPlatformSettings model."""

    def test_tiktok_settings_valid(self):
        """Test valid TikTok settings creation."""
        settings = models.TikTokPlatformSettings(
            enabled=True,
            caption_length_optimal=150,
            caption_length_max=2200,
            hashtag_count_min=3,
            hashtag_count_max=5,
            seo_focused=True,
            avoid_generic_tags=["foryoupage", "fyp", "viral"],
        )

        assert settings.enabled is True
        assert settings.caption_length_optimal == 150
        assert settings.caption_length_max == 2200
        assert settings.hashtag_count_min == 3
        assert settings.hashtag_count_max == 5
        assert settings.seo_focused is True
        assert len(settings.avoid_generic_tags) == 3

    def test_tiktok_settings_defaults(self):
        """Test TikTok settings with default values."""
        settings = models.TikTokPlatformSettings()

        assert settings.enabled is True
        assert settings.caption_length_optimal == 150
        assert settings.caption_length_max == 2200
        assert settings.hashtag_count_min == 3
        assert settings.hashtag_count_max == 5
        assert settings.seo_focused is True
        assert "foryoupage" in settings.avoid_generic_tags
        assert "fyp" in settings.avoid_generic_tags
        assert "viral" in settings.avoid_generic_tags

    def test_tiktok_settings_invalid_caption_length_optimal_too_low(self):
        """Test TikTok settings with optimal caption length below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            models.TikTokPlatformSettings(caption_length_optimal=49)

        assert "caption_length_optimal" in str(exc_info.value)

    def test_tiktok_settings_invalid_caption_length_optimal_too_high(self):
        """Test TikTok settings with optimal caption length exceeding maximum."""
        with pytest.raises(ValidationError) as exc_info:
            models.TikTokPlatformSettings(caption_length_optimal=301)

        assert "caption_length_optimal" in str(exc_info.value)

    def test_tiktok_settings_invalid_caption_length_max(self):
        """Test TikTok settings with caption max length exceeding limit."""
        with pytest.raises(ValidationError) as exc_info:
            models.TikTokPlatformSettings(caption_length_max=2201)

        assert "caption_length_max" in str(exc_info.value)

    def test_tiktok_settings_invalid_hashtag_count_min(self):
        """Test TikTok settings with invalid hashtag minimum."""
        with pytest.raises(ValidationError) as exc_info:
            models.TikTokPlatformSettings(hashtag_count_min=11)

        assert "hashtag_count_min" in str(exc_info.value)

    def test_tiktok_settings_invalid_hashtag_count_max(self):
        """Test TikTok settings with invalid hashtag maximum."""
        with pytest.raises(ValidationError) as exc_info:
            models.TikTokPlatformSettings(hashtag_count_max=11)

        assert "hashtag_count_max" in str(exc_info.value)

    def test_tiktok_settings_edge_case_min_values(self):
        """Test TikTok settings with minimum allowed values."""
        settings = models.TikTokPlatformSettings(
            caption_length_optimal=50,
            caption_length_max=1,
            hashtag_count_min=0,
            hashtag_count_max=1,
        )

        assert settings.caption_length_optimal == 50
        assert settings.caption_length_max == 1
        assert settings.hashtag_count_min == 0
        assert settings.hashtag_count_max == 1

    def test_tiktok_settings_edge_case_max_values(self):
        """Test TikTok settings with maximum allowed values."""
        settings = models.TikTokPlatformSettings(
            caption_length_optimal=300,
            caption_length_max=2200,
            hashtag_count_min=10,
            hashtag_count_max=10,
        )

        assert settings.caption_length_optimal == 300
        assert settings.caption_length_max == 2200
        assert settings.hashtag_count_min == 10
        assert settings.hashtag_count_max == 10

    def test_tiktok_settings_custom_avoid_tags(self):
        """Test TikTok settings with custom avoid_generic_tags."""
        settings = models.TikTokPlatformSettings(
            avoid_generic_tags=["custom1", "custom2"]
        )

        assert len(settings.avoid_generic_tags) == 2
        assert "custom1" in settings.avoid_generic_tags
        assert "custom2" in settings.avoid_generic_tags


class TestInstagramPlatformSettings:
    """Test models.InstagramPlatformSettings model."""

    def test_instagram_settings_valid(self):
        """Test valid Instagram settings creation."""
        settings = models.InstagramPlatformSettings(
            enabled=True,
            caption_style="seo",
            caption_length_short=15,
            caption_length_seo=200,
            hashtag_count_min=15,
            hashtag_count_max=30,
            emoji_enabled=True,
        )

        assert settings.enabled is True
        assert settings.caption_style == "seo"
        assert settings.caption_length_short == 15
        assert settings.caption_length_seo == 200
        assert settings.hashtag_count_min == 15
        assert settings.hashtag_count_max == 30
        assert settings.emoji_enabled is True

    def test_instagram_settings_defaults(self):
        """Test Instagram settings with default values."""
        settings = models.InstagramPlatformSettings()

        assert settings.enabled is True
        assert settings.caption_style == "seo"
        assert settings.caption_length_short == 15
        assert settings.caption_length_seo == 200
        assert settings.hashtag_count_min == 15
        assert settings.hashtag_count_max == 30
        assert settings.emoji_enabled is True

    def test_instagram_settings_caption_style_short(self):
        """Test Instagram settings with 'short' caption style."""
        settings = models.InstagramPlatformSettings(caption_style="short")

        assert settings.caption_style == "short"

    def test_instagram_settings_invalid_caption_style(self):
        """Test Instagram settings with invalid caption style."""
        with pytest.raises(ValidationError) as exc_info:
            models.InstagramPlatformSettings(caption_style="medium")

        assert "caption_style" in str(exc_info.value)

    def test_instagram_settings_invalid_caption_length_short_too_low(self):
        """Test Instagram settings with short caption length below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            models.InstagramPlatformSettings(caption_length_short=4)

        assert "caption_length_short" in str(exc_info.value)

    def test_instagram_settings_invalid_caption_length_short_too_high(self):
        """Test Instagram settings with short caption length exceeding maximum."""
        with pytest.raises(ValidationError) as exc_info:
            models.InstagramPlatformSettings(caption_length_short=31)

        assert "caption_length_short" in str(exc_info.value)

    def test_instagram_settings_invalid_caption_length_seo_too_low(self):
        """Test Instagram settings with SEO caption length below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            models.InstagramPlatformSettings(caption_length_seo=49)

        assert "caption_length_seo" in str(exc_info.value)

    def test_instagram_settings_invalid_caption_length_seo_too_high(self):
        """Test Instagram settings with SEO caption length exceeding maximum."""
        with pytest.raises(ValidationError) as exc_info:
            models.InstagramPlatformSettings(caption_length_seo=301)

        assert "caption_length_seo" in str(exc_info.value)

    def test_instagram_settings_invalid_hashtag_count_min(self):
        """Test Instagram settings with invalid hashtag minimum."""
        with pytest.raises(ValidationError) as exc_info:
            models.InstagramPlatformSettings(hashtag_count_min=4)

        assert "hashtag_count_min" in str(exc_info.value)

    def test_instagram_settings_invalid_hashtag_count_max(self):
        """Test Instagram settings with invalid hashtag maximum."""
        with pytest.raises(ValidationError) as exc_info:
            models.InstagramPlatformSettings(hashtag_count_max=31)

        assert "hashtag_count_max" in str(exc_info.value)

    def test_instagram_settings_edge_case_min_values(self):
        """Test Instagram settings with minimum allowed values."""
        settings = models.InstagramPlatformSettings(
            caption_length_short=5,
            caption_length_seo=50,
            hashtag_count_min=5,
            hashtag_count_max=15,
        )

        assert settings.caption_length_short == 5
        assert settings.caption_length_seo == 50
        assert settings.hashtag_count_min == 5
        assert settings.hashtag_count_max == 15

    def test_instagram_settings_edge_case_max_values(self):
        """Test Instagram settings with maximum allowed values."""
        settings = models.InstagramPlatformSettings(
            caption_length_short=30,
            caption_length_seo=300,
            hashtag_count_min=20,
            hashtag_count_max=30,
        )

        assert settings.caption_length_short == 30
        assert settings.caption_length_seo == 300
        assert settings.hashtag_count_min == 20
        assert settings.hashtag_count_max == 30


class TestPlatformMetadataSettings:
    """Test models.PlatformMetadataSettings model."""

    def test_platform_metadata_settings_valid(self):
        """Test valid platform metadata settings creation."""
        settings = models.PlatformMetadataSettings(
            enabled=True,
            target_platform="multi",
            youtube=models.YouTubePlatformSettings(),
            tiktok=models.TikTokPlatformSettings(),
            instagram=models.InstagramPlatformSettings(),
        )

        assert settings.enabled is True
        assert settings.target_platform == "multi"
        assert isinstance(settings.youtube, models.YouTubePlatformSettings)
        assert isinstance(settings.tiktok, models.TikTokPlatformSettings)
        assert isinstance(settings.instagram, models.InstagramPlatformSettings)

    def test_platform_metadata_settings_defaults(self):
        """Test platform metadata settings with default values."""
        settings = models.PlatformMetadataSettings()

        assert settings.enabled is True
        assert settings.target_platform == "multi"
        assert isinstance(settings.youtube, models.YouTubePlatformSettings)
        assert isinstance(settings.tiktok, models.TikTokPlatformSettings)
        assert isinstance(settings.instagram, models.InstagramPlatformSettings)
        assert settings.youtube.enabled is True
        assert settings.tiktok.enabled is True
        assert settings.instagram.enabled is True

    def test_platform_metadata_settings_target_platform_youtube(self):
        """Test platform metadata settings with YouTube target."""
        settings = models.PlatformMetadataSettings(target_platform="youtube")

        assert settings.target_platform == "youtube"

    def test_platform_metadata_settings_target_platform_tiktok(self):
        """Test platform metadata settings with TikTok target."""
        settings = models.PlatformMetadataSettings(target_platform="tiktok")

        assert settings.target_platform == "tiktok"

    def test_platform_metadata_settings_target_platform_instagram(self):
        """Test platform metadata settings with Instagram target."""
        settings = models.PlatformMetadataSettings(target_platform="instagram")

        assert settings.target_platform == "instagram"

    def test_platform_metadata_settings_invalid_target_platform(self):
        """Test platform metadata settings with invalid target platform."""
        with pytest.raises(ValidationError) as exc_info:
            models.PlatformMetadataSettings(target_platform="facebook")

        assert "target_platform" in str(exc_info.value)

    def test_platform_metadata_settings_nested_youtube_custom(self):
        """Test platform metadata settings with custom YouTube settings."""
        settings = models.PlatformMetadataSettings(
            youtube=models.YouTubePlatformSettings(
                title_length_max=50,
                hashtag_count_max=3,
            )
        )

        assert settings.youtube.title_length_max == 50
        assert settings.youtube.hashtag_count_max == 3

    def test_platform_metadata_settings_nested_tiktok_custom(self):
        """Test platform metadata settings with custom TikTok settings."""
        settings = models.PlatformMetadataSettings(
            tiktok=models.TikTokPlatformSettings(
                caption_length_optimal=200,
                seo_focused=False,
            )
        )

        assert settings.tiktok.caption_length_optimal == 200
        assert settings.tiktok.seo_focused is False

    def test_platform_metadata_settings_nested_instagram_custom(self):
        """Test platform metadata settings with custom Instagram settings."""
        settings = models.PlatformMetadataSettings(
            instagram=models.InstagramPlatformSettings(
                caption_style="short",
                emoji_enabled=False,
            )
        )

        assert settings.instagram.caption_style == "short"
        assert settings.instagram.emoji_enabled is False

    def test_platform_metadata_settings_disabled(self):
        """Test platform metadata settings with global disable."""
        settings = models.PlatformMetadataSettings(enabled=False)

        assert settings.enabled is False
        # Nested settings should still have their defaults
        assert settings.youtube.enabled is True
        assert settings.tiktok.enabled is True
        assert settings.instagram.enabled is True

    def test_platform_metadata_settings_individual_platform_disabled(self):
        """Test platform metadata settings with individual platform disabled."""
        settings = models.PlatformMetadataSettings(
            youtube=models.YouTubePlatformSettings(enabled=False),
            tiktok=models.TikTokPlatformSettings(enabled=False),
        )

        assert settings.youtube.enabled is False
        assert settings.tiktok.enabled is False
        assert settings.instagram.enabled is True
