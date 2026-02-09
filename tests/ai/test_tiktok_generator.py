"""Unit tests for TikTok metadata generator.

Tests cover LLM response mocking, validation rules, error handling,
and TikTok-specific requirements (generic hashtag blacklist, caption length).
"""

from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

# Import generator and models using safe method to avoid circular imports
from src.ai.platform_metadata import models
from src.ai.platform_metadata.tiktok import TikTokMetadataGenerator


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
def tiktok_settings():
    """TikTok platform settings fixture."""
    return {
        "caption_length_optimal": 300,
        "caption_length_max": 2200,
        "hashtag_count_min": 3,
        "hashtag_count_max": 5,
        "avoid_generic_tags": ["fyp", "foryoupage", "viral"],
    }


@pytest.fixture
def product_data():
    """Sample product data fixture."""
    return MockProductData(
        asin="B0TEST456",
        title="Smart Fitness Tracker Watch",
        description="Advanced fitness tracker with heart rate monitoring, sleep tracking, GPS, and 10-day battery life. Water-resistant design perfect for athletes.",
        url="https://amazon.com/dp/B0TEST456",
        affiliate_link="https://amazon.com/dp/B0TEST456?tag=test-20",
        shortened_affiliate_link="https://stte.psee.io/test456",
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
    return {"OPENROUTER_API_KEY": "test_api_key_67890"}


class TestTikTokMetadataGenerator:
    """Test suite for TikTokMetadataGenerator."""

    @pytest.mark.asyncio
    async def test_successful_generation(
        self, tiktok_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test successful TikTok metadata generation with valid LLM response."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        # Mock LLM response
        mock_llm_response = """CAPTION: Smart fitness tracker with heart rate monitor, sleep tracking, GPS, and 10-day battery life. Water-resistant design perfect for runners and athletes. Track your health goals with precision.

HASHTAGS: #FitnessTracker #SmartWatch #HealthTech

KEYWORDS: fitness tracker, smart watch, heart rate monitor, GPS watch, health tech"""

        with patch(
            "src.ai.platform_metadata.tiktok.generate_with_llm",
            return_value=mock_llm_response,
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
        assert metadata.platform == "tiktok"
        assert metadata.title is None  # TikTok doesn't use titles
        assert "heart rate monitor" in metadata.description
        assert metadata.product_id == "B0TEST456"
        assert metadata.validation_status == "valid"

        # Verify #ad was auto-added
        assert "#ad" in metadata.hashtags

        # Verify hashtag count includes original + #ad
        assert len(metadata.hashtags) == 4  # 3 original + #ad

    @pytest.mark.asyncio
    async def test_missing_api_key(
        self, tiktok_settings, product_data, llm_settings, mock_session
    ):
        """Test generation fails when API key is missing."""
        generator = TikTokMetadataGenerator(tiktok_settings)

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
        self, tiktok_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test generation fails when LLM returns empty response."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        # Mock empty LLM response
        with patch(
            "src.ai.platform_metadata.tiktok.generate_with_llm", return_value=None
        ):
            metadata = await generator.generate(
                product_data,
                llm_settings,
                secrets,
                mock_session,
                {},
                debug_mode=False,
            )

        # Should return None when LLM fails
        assert metadata is None

    @pytest.mark.asyncio
    async def test_parse_failure_missing_caption(
        self, tiktok_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test generation fails when LLM response missing required CAPTION field."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        # Mock malformed LLM response (missing CAPTION)
        mock_llm_response = """HASHTAGS: #Tech #Review

KEYWORDS: fitness, tech"""

        with patch(
            "src.ai.platform_metadata.tiktok.generate_with_llm",
            return_value=mock_llm_response,
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
    async def test_caption_length_truncation(
        self, tiktok_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test caption is truncated when exceeding max length."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        # Create very long caption (>2200 chars)
        long_caption = "A" * 2500

        mock_llm_response = f"""CAPTION: {long_caption}

HASHTAGS: #Tech

KEYWORDS: test"""

        with patch(
            "src.ai.platform_metadata.tiktok.generate_with_llm",
            return_value=mock_llm_response,
        ):
            metadata = await generator.generate(
                product_data,
                llm_settings,
                secrets,
                mock_session,
                {},
                debug_mode=False,
            )

        # Caption should be truncated to 2200 chars
        assert metadata is not None
        assert len(metadata.description) <= 2200

    @pytest.mark.asyncio
    async def test_ad_tag_auto_add(
        self, tiktok_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test #ad tag is automatically added for advertising disclosure."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        # Mock LLM response without #ad
        mock_llm_response = """CAPTION: Great fitness tracker for serious athletes

HASHTAGS: #FitnessTracker #SmartWatch #HealthTech

KEYWORDS: fitness, health"""

        with patch(
            "src.ai.platform_metadata.tiktok.generate_with_llm",
            return_value=mock_llm_response,
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

    def test_validate_success(self, tiktok_settings):
        """Test validation passes for valid TikTok metadata."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        metadata = models.PlatformMetadata(
            platform="tiktok",
            title=None,
            description="Smart fitness tracker with GPS and heart rate monitoring for athletes and runners.",  # 95 chars
            hashtags=["#FitnessTracker", "#SmartWatch", "#HealthTech", "#ad"],
            keywords=["fitness tracker", "smart watch"],
            character_counts={"title": 0, "description": 95},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert is_valid
        assert error_msg == ""

    def test_validate_platform_mismatch(self, tiktok_settings):
        """Test validation fails for wrong platform."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        metadata = models.PlatformMetadata(
            platform="youtube",  # Wrong platform!
            title="Test",
            description="Test description",
            hashtags=["#Test", "#Tech", "#Review", "#ad"],
            keywords=["test"],
            character_counts={"title": 4, "description": 16},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Platform mismatch" in error_msg

    def test_validate_caption_too_long(self, tiktok_settings):
        """Test validation fails when caption exceeds max length."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        long_caption = "A" * 2500  # Exceeds 2200 char max

        metadata = models.PlatformMetadata(
            platform="tiktok",
            title=None,
            description=long_caption,
            hashtags=["#Test", "#Tech", "#Review", "#ad"],
            keywords=["test"],
            character_counts={"title": 0, "description": 2500},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Caption too long" in error_msg

    def test_validate_too_few_hashtags(self, tiktok_settings):
        """Test validation fails with fewer than 3 hashtags."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        metadata = models.PlatformMetadata(
            platform="tiktok",
            title=None,
            description="Test caption",
            hashtags=["#Test", "#ad"],  # Only 2 hashtags
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Too few hashtags" in error_msg

    def test_validate_too_many_hashtags(self, tiktok_settings):
        """Test validation fails with more than 5 hashtags."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        metadata = models.PlatformMetadata(
            platform="tiktok",
            title=None,
            description="Test caption",
            hashtags=[
                "#Test",
                "#Tech",
                "#Review",
                "#Fitness",
                "#Health",
                "#ad",
            ],  # 6 hashtags
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Too many hashtags" in error_msg

    def test_validate_blacklisted_hashtag_fyp(self, tiktok_settings):
        """Test validation fails when generic #fyp hashtag is used."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        metadata = models.PlatformMetadata(
            platform="tiktok",
            title=None,
            description="Test caption",
            hashtags=["#fyp", "#FitnessTracker", "#Tech", "#ad"],  # #fyp is blacklisted
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Generic hashtags found" in error_msg
        assert "#fyp" in error_msg

    def test_validate_blacklisted_hashtag_foryoupage(self, tiktok_settings):
        """Test validation fails when generic #foryoupage hashtag is used."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        metadata = models.PlatformMetadata(
            platform="tiktok",
            title=None,
            description="Test caption",
            hashtags=[
                "#foryoupage",
                "#FitnessTracker",
                "#Tech",
                "#ad",
            ],  # #foryoupage is blacklisted
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Generic hashtags found" in error_msg
        assert "#foryoupage" in error_msg

    def test_validate_blacklisted_hashtag_viral(self, tiktok_settings):
        """Test validation fails when generic #viral hashtag is used."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        metadata = models.PlatformMetadata(
            platform="tiktok",
            title=None,
            description="Test caption",
            hashtags=[
                "#viral",
                "#FitnessTracker",
                "#Tech",
                "#ad",
            ],  # #viral is blacklisted
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Generic hashtags found" in error_msg
        assert "#viral" in error_msg

    def test_validate_blacklisted_hashtag_case_insensitive(self, tiktok_settings):
        """Test validation catches blacklisted hashtags regardless of case."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        metadata = models.PlatformMetadata(
            platform="tiktok",
            title=None,
            description="Test caption",
            hashtags=[
                "#FYP",  # Uppercase version of blacklisted tag
                "#FitnessTracker",
                "#Tech",
                "#ad",
            ],
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Generic hashtags found" in error_msg

    def test_validate_multiple_blacklisted_hashtags(self, tiktok_settings):
        """Test validation reports all blacklisted hashtags found."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        metadata = models.PlatformMetadata(
            platform="tiktok",
            title=None,
            description="Test caption",
            hashtags=[
                "#fyp",
                "#viral",
                "#ad",
            ],  # Two blacklisted tags
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Generic hashtags found" in error_msg
        assert "#fyp" in error_msg or "#viral" in error_msg

    def test_validate_missing_ad_tag(self, tiktok_settings):
        """Test validation fails when #ad tag is missing."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        metadata = models.PlatformMetadata(
            platform="tiktok",
            title=None,
            description="Test caption",
            hashtags=["#FitnessTracker", "#SmartWatch", "#HealthTech"],  # Missing #ad
            keywords=["test"],
            character_counts={"title": 0, "description": 12},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST456",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Missing required #ad hashtag" in error_msg

    def test_parse_llm_response_success(self, tiktok_settings):
        """Test successful parsing of LLM response."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        response = """CAPTION: Smart fitness tracker with GPS and heart rate monitoring

HASHTAGS: #FitnessTracker #SmartWatch #HealthTech

KEYWORDS: fitness tracker, GPS, heart rate"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        caption, hashtags, keywords = parsed
        assert caption == "Smart fitness tracker with GPS and heart rate monitoring"
        assert hashtags == ["#FitnessTracker", "#SmartWatch", "#HealthTech"]
        assert keywords == ["fitness tracker", "GPS", "heart rate"]

    def test_parse_llm_response_missing_hashtags(self, tiktok_settings):
        """Test parsing succeeds even without optional HASHTAGS field."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        response = """CAPTION: Test caption

KEYWORDS: test, keywords"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        caption, hashtags, keywords = parsed
        assert caption == "Test caption"
        assert hashtags == []  # Empty list when no hashtags
        assert keywords == ["test", "keywords"]

    def test_parse_llm_response_missing_keywords(self, tiktok_settings):
        """Test parsing succeeds even without optional KEYWORDS field."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        response = """CAPTION: Test caption

HASHTAGS: #Tech #Review"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        caption, hashtags, keywords = parsed
        assert caption == "Test caption"
        assert hashtags == ["#Tech", "#Review"]
        assert keywords == []  # Empty list when no keywords

    def test_parse_llm_response_hashtags_without_hash(self, tiktok_settings):
        """Test parsing adds # prefix to hashtags if missing."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        response = """CAPTION: Test

HASHTAGS: Fitness SmartWatch HealthTech

KEYWORDS: test"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        _, hashtags, _ = parsed
        # All hashtags should have # prefix added
        assert hashtags == ["#Fitness", "#SmartWatch", "#HealthTech"]

    def test_parse_llm_response_missing_caption_fails(self, tiktok_settings):
        """Test parsing fails when required CAPTION field is missing."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        response = """HASHTAGS: #Test

KEYWORDS: test"""

        parsed = generator._parse_llm_response(response)

        # Should return None when caption missing
        assert parsed is None

    def test_parse_llm_response_multiline_caption(self, tiktok_settings):
        """Test parsing handles multi-line captions correctly."""
        generator = TikTokMetadataGenerator(tiktok_settings)

        response = """CAPTION: Line 1 of caption
Line 2 of caption
Line 3 of caption

HASHTAGS: #Test

KEYWORDS: test"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        caption, _, _ = parsed
        # Should capture all lines
        assert "Line 1" in caption
        assert "Line 2" in caption
        assert "Line 3" in caption
