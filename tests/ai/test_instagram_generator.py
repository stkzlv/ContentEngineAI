"""Unit tests for Instagram metadata generator.

Tests cover LLM response mocking, validation rules, error handling,
and Instagram-specific requirements (dual caption styles, 15-30 hashtag count).
"""

from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

# Import generator and models using safe method to avoid circular imports
from src.ai.platform_metadata import models
from src.ai.platform_metadata.instagram import InstagramMetadataGenerator


# Mock classes to avoid circular import with VideoConfig
class MockLLMSettings(NamedTuple):
    """Mock LLM settings for testing."""

    api_key_env_var: str = "OPENROUTER_API_KEY"
    models: list = ["anthropic/claude-3.5-sonnet"]
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: int = 30
    max_retries: int = 3
    auto_select_free_model: bool = False


class MockProductData(NamedTuple):
    """Mock product data for testing."""

    asin: str
    title: str
    description: str
    url: str
    affiliate_link: str | None = None
    shortened_affiliate_link: str | None = None


@pytest.fixture
def instagram_settings_short():
    """Instagram platform settings fixture with short caption style."""
    return {
        "caption_style": "short",
        "caption_length_seo": 200,
        "hashtag_count_min": 15,
        "hashtag_count_max": 30,
        "emoji_enabled": True,
    }


@pytest.fixture
def instagram_settings_seo():
    """Instagram platform settings fixture with SEO caption style."""
    return {
        "caption_style": "seo",
        "caption_length_seo": 200,
        "hashtag_count_min": 15,
        "hashtag_count_max": 30,
        "emoji_enabled": True,
    }


@pytest.fixture
def instagram_settings_no_emoji():
    """Instagram platform settings fixture with emojis disabled."""
    return {
        "caption_style": "seo",
        "caption_length_seo": 200,
        "hashtag_count_min": 15,
        "hashtag_count_max": 30,
        "emoji_enabled": False,
    }


@pytest.fixture
def product_data():
    """Sample product data fixture."""
    return MockProductData(
        asin="B0TEST789",
        title="Portable Phone Charger 20000mAh Power Bank",
        description="High-capacity portable charger with fast charging, 3 USB ports, LED display, and compact design. Perfect for travel and emergencies.",
        url="https://amazon.com/dp/B0TEST789",
        affiliate_link="https://amazon.com/dp/B0TEST789?tag=stealtech06-20",
        shortened_affiliate_link="https://stte.psee.io/test789",
    )


@pytest.fixture
def llm_settings():
    """LLM settings fixture."""
    return MockLLMSettings()


@pytest.fixture
def mock_session():
    """Mock aiohttp session."""
    return AsyncMock(spec=aiohttp.ClientSession)


@pytest.fixture
def secrets():
    """API secrets fixture."""
    return {"OPENROUTER_API_KEY": "test_api_key_99999"}


class TestInstagramMetadataGenerator:
    """Test suite for InstagramMetadataGenerator."""

    @pytest.mark.asyncio
    async def test_successful_generation_short_style(
        self,
        instagram_settings_short,
        product_data,
        llm_settings,
        secrets,
        mock_session,
    ):
        """Test successful Instagram metadata generation with short caption style."""
        generator = InstagramMetadataGenerator(instagram_settings_short)

        # Mock LLM response with short caption
        mock_llm_response = """CAPTION: Never die again 🔋⚡

HASHTAGS: #PortableCharger #PowerBank #TravelEssentials #TechGadgets #PhoneAccessories #TravelTech #TechMustHaves #OnTheGo #BatteryPack #FastCharging #TravelGear #TechReview #GadgetLover #ProductReview #PhoneTech #TechFinds #EmergencyPower

KEYWORDS: portable charger, power bank, travel essentials, phone accessories"""

        # Mock the LLM API calls
        with (
            patch(
                "src.ai.platform_metadata.utilities.fetch_and_select_model",
                return_value=None,
            ),
            patch(
                "src.ai.platform_metadata.utilities.call_llm_api_with_retry",
                return_value=mock_llm_response,
            ),
        ):
            metadata = await generator.generate(
                product_data,
                llm_settings,
                secrets,
                mock_session,
                {},
                debug_mode=False,
            )

        # Verify metadata was generated
        assert metadata is not None
        assert metadata.platform == "instagram"
        assert metadata.title is None  # Instagram doesn't use titles
        assert metadata.description == "Never die again 🔋⚡"
        assert metadata.product_id == "B0TEST789"
        assert metadata.validation_status == "valid"

        # Verify #ad was auto-added
        assert "#ad" in metadata.hashtags

        # Verify hashtag count (17 original + #ad = 18 total, within 15-30 range)
        assert 15 <= len(metadata.hashtags) <= 30

    @pytest.mark.asyncio
    async def test_successful_generation_seo_style(
        self, instagram_settings_seo, product_data, llm_settings, secrets, mock_session
    ):
        """Test successful Instagram metadata generation with SEO caption style."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        # Mock LLM response with SEO caption
        mock_llm_response = """CAPTION: 20000mAh portable charger that charges your iPhone 5X - perfect for travel, camping, and emergencies. Fast charging with 3 ports 📱💪

HASHTAGS: #PortableCharger #PowerBank #TravelEssentials #TechGadgets #PhoneAccessories #TravelTech #FastCharging #BatteryPack #CampingGear #TechReview #EmergencyPrep #PhoneTech #MobileAccessories #TechMustHaves #ProductReview #GadgetReview #iPhoneAccessories #TechFinds #OnTheGo #TravelGear

KEYWORDS: portable charger 20000mah, phone power bank, travel tech, fast charging"""

        # Mock the LLM generation
        with (
            patch(
                "src.ai.platform_metadata.utilities.fetch_and_select_model",
                return_value=None,
            ),
            patch(
                "src.ai.platform_metadata.utilities.call_llm_api_with_retry",
                return_value=mock_llm_response,
            ),
        ):
            metadata = await generator.generate(
                product_data,
                llm_settings,
                secrets,
                mock_session,
                {},
                debug_mode=False,
            )

        # Verify metadata was generated
        assert metadata is not None
        assert metadata.platform == "instagram"
        assert "20000mAh portable charger" in metadata.description
        assert len(metadata.description) <= 200  # SEO style max length

    @pytest.mark.asyncio
    async def test_missing_api_key(
        self,
        instagram_settings_seo,
        product_data,
        llm_settings,
        mock_session,
    ):
        """Test generation fails when API key is missing."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        # Empty secrets (no API key)
        empty_secrets: dict[str, str] = {}

        metadata = await generator.generate(
            product_data,
            llm_settings,
            empty_secrets,
            mock_session,
            {},
            debug_mode=False,
        )

        # Should return None when API key missing
        assert metadata is None

    @pytest.mark.asyncio
    async def test_llm_generation_failure(
        self,
        instagram_settings_seo,
        product_data,
        llm_settings,
        secrets,
        mock_session,
    ):
        """Test generation fails when all LLM models fail."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        # Mock all models failing
        with (
            patch(
                "src.ai.platform_metadata.utilities.fetch_and_select_model",
                return_value=None,
            ),
            patch(
                "src.ai.platform_metadata.utilities.call_llm_api_with_retry",
                side_effect=Exception("API error"),
            ),
        ):
            metadata = await generator.generate(
                product_data,
                llm_settings,
                secrets,
                mock_session,
                {},
                debug_mode=False,
            )

        # Should return None when all models fail
        assert metadata is None

    @pytest.mark.asyncio
    async def test_parse_failure_missing_caption(
        self,
        instagram_settings_seo,
        product_data,
        llm_settings,
        secrets,
        mock_session,
    ):
        """Test generation fails when LLM response missing required CAPTION field."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        # Mock malformed LLM response (missing CAPTION)
        mock_llm_response = """HASHTAGS: #Tech #Review

KEYWORDS: tech, review"""

        with (
            patch(
                "src.ai.platform_metadata.utilities.fetch_and_select_model",
                return_value=None,
            ),
            patch(
                "src.ai.platform_metadata.utilities.call_llm_api_with_retry",
                return_value=mock_llm_response,
            ),
        ):
            metadata = await generator.generate(
                product_data,
                llm_settings,
                secrets,
                mock_session,
                {},
                debug_mode=False,
            )

        # Should return None when parsing fails
        assert metadata is None

    @pytest.mark.asyncio
    async def test_caption_truncation_short_style(
        self,
        instagram_settings_short,
        product_data,
        llm_settings,
        secrets,
        mock_session,
    ):
        """Test short caption is truncated to 5 words if exceeds limit."""
        generator = InstagramMetadataGenerator(instagram_settings_short)

        # Mock LLM response with too many words for short style
        mock_llm_response = """CAPTION: This is way too many words for a short caption

HASHTAGS: #Test #Tech #Review #One #Two #Three #Four #Five #Six #Seven #Eight #Nine #Ten #Eleven #Twelve #Thirteen #Fourteen #Fifteen

KEYWORDS: test"""

        with (
            patch(
                "src.ai.platform_metadata.utilities.fetch_and_select_model",
                return_value=None,
            ),
            patch(
                "src.ai.platform_metadata.utilities.call_llm_api_with_retry",
                return_value=mock_llm_response,
            ),
        ):
            metadata = await generator.generate(
                product_data,
                llm_settings,
                secrets,
                mock_session,
                {},
                debug_mode=False,
            )

        # Caption should be truncated to 5 words
        assert metadata is not None
        word_count = len(metadata.description.split())
        assert word_count == 5

    @pytest.mark.asyncio
    async def test_ad_tag_auto_add(
        self,
        instagram_settings_seo,
        product_data,
        llm_settings,
        secrets,
        mock_session,
    ):
        """Test #ad tag is automatically added for advertising disclosure."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        # Mock LLM response without #ad
        mock_llm_response = """CAPTION: Great portable charger

HASHTAGS: #PortableCharger #PowerBank #TravelEssentials #TechGadgets #PhoneAccessories #TravelTech #FastCharging #BatteryPack #TechReview #PhoneTech #MobileAccessories #TechMustHaves #ProductReview #GadgetReview #TechFinds

KEYWORDS: portable charger, power bank"""

        with (
            patch(
                "src.ai.platform_metadata.utilities.fetch_and_select_model",
                return_value=None,
            ),
            patch(
                "src.ai.platform_metadata.utilities.call_llm_api_with_retry",
                return_value=mock_llm_response,
            ),
        ):
            metadata = await generator.generate(
                product_data,
                llm_settings,
                secrets,
                mock_session,
                {},
                debug_mode=False,
            )

        # #ad should be auto-added
        assert metadata is not None
        assert "#ad" in metadata.hashtags

    def test_validate_success_seo_style(self, instagram_settings_seo):
        """Test validation passes for valid Instagram metadata with SEO caption."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        metadata = models.PlatformMetadata(
            platform="instagram",
            title=None,
            description="Portable charger 20000mAh with fast charging and 3 USB ports - perfect for travel and emergencies 📱⚡",  # 110 chars
            hashtags=[f"#Tag{i}" for i in range(15)]
            + ["#ad"],  # 15 tags + #ad = 16 total
            keywords=["portable charger", "power bank"],
            character_counts={"title": 0, "description": 110},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST789",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert is_valid
        assert error_msg == ""

    def test_validate_success_short_style(self, instagram_settings_short):
        """Test validation passes for valid Instagram metadata with short caption."""
        generator = InstagramMetadataGenerator(instagram_settings_short)

        metadata = models.PlatformMetadata(
            platform="instagram",
            title=None,
            description="Never die again 🔋",  # 4 words
            hashtags=[f"#Tag{i}" for i in range(15)] + ["#ad"],
            keywords=["portable charger"],
            character_counts={"title": 0, "description": 19},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST789",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert is_valid
        assert error_msg == ""

    def test_validate_platform_mismatch(self, instagram_settings_seo):
        """Test validation fails for wrong platform."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        metadata = models.PlatformMetadata(
            platform="youtube",  # Wrong platform!
            title="Test",
            description="Test description",
            hashtags=[f"#Tag{i}" for i in range(15)] + ["#ad"],
            keywords=["test"],
            character_counts={"title": 4, "description": 16},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST789",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Platform mismatch" in error_msg

    def test_validate_short_caption_too_many_words(self, instagram_settings_short):
        """Test validation fails when short caption exceeds 5 words."""
        generator = InstagramMetadataGenerator(instagram_settings_short)

        metadata = models.PlatformMetadata(
            platform="instagram",
            title=None,
            description="This is way too many words for short style",  # 9 words
            hashtags=[f"#Tag{i}" for i in range(15)] + ["#ad"],
            keywords=["test"],
            character_counts={"title": 0, "description": 48},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST789",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "too many words" in error_msg

    def test_validate_too_few_hashtags(self, instagram_settings_seo):
        """Test validation fails with fewer than 15 hashtags."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        metadata = models.PlatformMetadata(
            platform="instagram",
            title=None,
            description="Test caption",
            hashtags=["#Tag1", "#Tag2", "#Tag3", "#ad"],  # Only 4 hashtags
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST789",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Too few hashtags" in error_msg
        assert "15-30 hashtags" in error_msg

    def test_validate_too_many_hashtags(self, instagram_settings_seo):
        """Test validation fails with more than 30 hashtags."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        metadata = models.PlatformMetadata(
            platform="instagram",
            title=None,
            description="Test caption",
            hashtags=[f"#Tag{i}" for i in range(35)],  # 35 hashtags (> 30 max)
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST789",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Too many hashtags" in error_msg

    def test_validate_missing_ad_tag(self, instagram_settings_seo):
        """Test validation fails when #ad tag is missing."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        metadata = models.PlatformMetadata(
            platform="instagram",
            title=None,
            description="Test caption",
            hashtags=[f"#Tag{i}" for i in range(15)],  # 15 tags but missing #ad
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST789",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Missing required #ad hashtag" in error_msg

    def test_determine_caption_style_short(self, instagram_settings_short):
        """Test caption style determination returns 'short'."""
        generator = InstagramMetadataGenerator(instagram_settings_short)

        style = generator._determine_caption_style()

        assert style == "short"

    def test_determine_caption_style_seo(self, instagram_settings_seo):
        """Test caption style determination returns 'seo'."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        style = generator._determine_caption_style()

        assert style == "seo"

    def test_determine_caption_style_invalid_defaults_to_seo(self):
        """Test invalid caption style defaults to 'seo'."""
        settings = {
            "caption_style": "invalid_style",
            "caption_length_seo": 200,
            "hashtag_count_min": 15,
            "hashtag_count_max": 30,
        }
        generator = InstagramMetadataGenerator(settings)

        style = generator._determine_caption_style()

        assert style == "seo"

    def test_validate_caption_style_short_truncation(self, instagram_settings_short):
        """Test short caption is truncated to 5 words."""
        generator = InstagramMetadataGenerator(instagram_settings_short)

        long_caption = "One two three four five six seven eight nine ten"
        truncated = generator._validate_caption_style(long_caption, "short")

        word_count = len(truncated.split())
        assert word_count == 5
        assert truncated == "One two three four five"

    def test_validate_caption_style_seo_truncation(self, instagram_settings_seo):
        """Test SEO caption is truncated to max length."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        long_caption = "A" * 250  # Exceeds 200 char limit
        truncated = generator._validate_caption_style(long_caption, "seo")

        assert len(truncated) <= 200

    def test_parse_llm_response_success(self, instagram_settings_seo):
        """Test successful parsing of LLM response."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        response = """CAPTION: Great portable charger

HASHTAGS: #PortableCharger #PowerBank #TechGadgets #PhoneAccessories #TravelTech #FastCharging #BatteryPack #TechReview #PhoneTech #MobileAccessories #TechMustHaves #ProductReview #GadgetReview #TechFinds #Travel

KEYWORDS: portable charger, power bank, fast charging"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        caption, hashtags, keywords = parsed
        assert caption == "Great portable charger"
        assert len(hashtags) == 15
        assert keywords == ["portable charger", "power bank", "fast charging"]

    def test_parse_llm_response_missing_hashtags(self, instagram_settings_seo):
        """Test parsing succeeds even without optional HASHTAGS field."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        response = """CAPTION: Test caption

KEYWORDS: test, keywords"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        caption, hashtags, keywords = parsed
        assert caption == "Test caption"
        assert hashtags == []
        assert keywords == ["test", "keywords"]

    def test_parse_llm_response_hashtags_without_hash(self, instagram_settings_seo):
        """Test parsing adds # prefix to hashtags if missing."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        response = """CAPTION: Test

HASHTAGS: Tech Review Gadgets Test1 Test2 Test3 Test4 Test5 Test6 Test7 Test8 Test9 Test10 Test11 Test12

KEYWORDS: test"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        _, hashtags, _ = parsed
        # All hashtags should have # prefix added
        assert all(tag.startswith("#") for tag in hashtags)

    def test_parse_llm_response_missing_caption_fails(self, instagram_settings_seo):
        """Test parsing fails when required CAPTION field is missing."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        response = """HASHTAGS: #Test

KEYWORDS: test"""

        parsed = generator._parse_llm_response(response)

        # Should return None when caption missing
        assert parsed is None

    def test_parse_llm_response_multiline_caption(self, instagram_settings_seo):
        """Test parsing handles multi-line captions correctly."""
        generator = InstagramMetadataGenerator(instagram_settings_seo)

        response = """CAPTION: Line 1 of caption
Line 2 of caption

HASHTAGS: #Test #Tech #Review #One #Two #Three #Four #Five #Six #Seven #Eight #Nine #Ten #Eleven #Fifteen

KEYWORDS: test"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        caption, _, _ = parsed
        # Should capture all lines
        assert "Line 1" in caption
        assert "Line 2" in caption
