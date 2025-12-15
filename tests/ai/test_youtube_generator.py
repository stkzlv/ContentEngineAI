"""Unit tests for YouTube metadata generator.

Tests cover LLM response mocking, validation rules, error handling,
and YouTube-specific requirements (#Shorts tag, title length, hashtag count).
"""

from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

# Import generator and models using safe method to avoid circular imports
from src.ai.platform_metadata import models
from src.ai.platform_metadata.youtube import YouTubeMetadataGenerator


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
def youtube_settings():
    """YouTube platform settings fixture."""
    return {
        "title_length_max": 60,
        "description_length_max": 5000,
        "hashtag_count_min": 3,
        "hashtag_count_max": 5,
        "include_shorts_tag": True,
    }


@pytest.fixture
def product_data():
    """Sample product data fixture."""
    return MockProductData(
        asin="B0TEST123",
        title="Wireless Earbuds with Noise Cancellation",
        description="Premium wireless earbuds with active noise cancellation, 30-hour battery life, and crystal-clear sound quality. Perfect for workouts, commutes, and travel.",
        url="https://amazon.com/dp/B0TEST123",
        affiliate_link="https://amazon.com/dp/B0TEST123?tag=stealtech06-20",
        shortened_affiliate_link="https://stte.psee.io/test123",
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
    return {"OPENROUTER_API_KEY": "test_api_key_12345"}


class TestYouTubeMetadataGenerator:
    """Test suite for YouTubeMetadataGenerator."""

    @pytest.mark.asyncio
    async def test_successful_generation(
        self, youtube_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test successful YouTube metadata generation with valid LLM response."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        # Mock LLM response
        mock_llm_response = """TITLE: Best Wireless Earbuds 2025 - Noise Cancelling Review

DESCRIPTION: Discover the best wireless earbuds with active noise cancellation in 2025. These premium earbuds deliver crystal-clear sound quality, 30-hour battery life, and comfortable all-day wear. Perfect for workouts, commutes, and travel. In-depth review of features, sound quality, battery performance, and value for money.

HASHTAGS: #WirelessEarbuds #NoiseCancelling #TechReview

KEYWORDS: wireless earbuds 2025, noise cancelling earbuds, best earbuds, premium audio, tech review"""

        with patch(
            "src.ai.platform_metadata.youtube.generate_with_llm",
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
        assert metadata.platform == "youtube"
        assert metadata.title == "Best Wireless Earbuds 2025 - Noise Cancelling Review"
        assert "active noise cancellation" in metadata.description
        assert metadata.product_id == "B0TEST123"
        assert metadata.validation_status == "valid"

        # Verify #Shorts was auto-added
        assert "#Shorts" in metadata.hashtags

        # Verify #ad was auto-added
        assert "#ad" in metadata.hashtags

        # Verify hashtag count includes original + auto-added
        assert len(metadata.hashtags) == 5  # 3 original + #Shorts + #ad

    @pytest.mark.asyncio
    async def test_missing_api_key(
        self, youtube_settings, product_data, llm_settings, mock_session
    ):
        """Test generation fails when API key is missing."""
        generator = YouTubeMetadataGenerator(youtube_settings)

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
        self, youtube_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test generation fails when LLM returns empty response."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        # Mock empty LLM response
        with patch(
            "src.ai.platform_metadata.youtube.generate_with_llm", return_value=None
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
    async def test_parse_failure_missing_title(
        self, youtube_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test generation fails when LLM response missing required TITLE field."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        # Mock malformed LLM response (missing TITLE)
        mock_llm_response = """DESCRIPTION: Great earbuds

HASHTAGS: #Tech #Review

KEYWORDS: earbuds, tech"""

        with patch(
            "src.ai.platform_metadata.youtube.generate_with_llm",
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
    async def test_title_length_truncation(
        self, youtube_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test title is truncated when exceeding max length."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        # Mock LLM response with very long title (>60 chars)
        mock_llm_response = """TITLE: This is an extremely long title that definitely exceeds the sixty character maximum limit for YouTube

DESCRIPTION: Test description

HASHTAGS: #Tech

KEYWORDS: test"""

        with patch(
            "src.ai.platform_metadata.youtube.generate_with_llm",
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

        # Title should be truncated to 60 chars
        assert metadata is not None
        assert len(metadata.title) <= 60

    @pytest.mark.asyncio
    async def test_description_length_truncation(
        self, youtube_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test description is truncated when exceeding max length."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        # Create very long description (>5000 chars)
        long_description = "A" * 5500

        mock_llm_response = f"""TITLE: Test Title

DESCRIPTION: {long_description}

HASHTAGS: #Tech

KEYWORDS: test"""

        with patch(
            "src.ai.platform_metadata.youtube.generate_with_llm",
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

        # Description should be truncated to 5000 chars
        assert metadata is not None
        assert len(metadata.description) <= 5000

    @pytest.mark.asyncio
    async def test_shorts_tag_auto_add(
        self, youtube_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test #Shorts tag is automatically added when not present."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        # Mock LLM response without #Shorts
        mock_llm_response = """TITLE: Test Title About Earbuds

DESCRIPTION: Test description

HASHTAGS: #Tech #Review #Gadgets

KEYWORDS: test, earbuds"""

        with patch(
            "src.ai.platform_metadata.youtube.generate_with_llm",
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

        # #Shorts should be auto-added
        assert metadata is not None
        assert "#Shorts" in metadata.hashtags
        # Should have 3 original + #Shorts + #ad = 5 total
        assert len(metadata.hashtags) == 5

    @pytest.mark.asyncio
    async def test_ad_tag_auto_add(
        self, youtube_settings, product_data, llm_settings, secrets, mock_session
    ):
        """Test #ad tag is automatically added for advertising disclosure."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        # Mock LLM response without #ad
        mock_llm_response = """TITLE: Test Title

DESCRIPTION: Test description

HASHTAGS: #Tech #Review #Gadgets

KEYWORDS: test"""

        with patch(
            "src.ai.platform_metadata.youtube.generate_with_llm",
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

    def test_validate_success(self, youtube_settings):
        """Test validation passes for valid YouTube metadata."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        metadata = models.PlatformMetadata(
            platform="youtube",
            title="Best Wireless Earbuds 2025 - Top Picks & Reviews",  # 50 chars
            description="Check out the best wireless earbuds of 2025.",
            hashtags=["#Shorts", "#WirelessEarbuds", "#Tech", "#ad"],
            keywords=["wireless earbuds", "tech review"],
            character_counts={"title": 50, "description": 49},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST123",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert is_valid
        assert error_msg == ""

    def test_validate_platform_mismatch(self, youtube_settings):
        """Test validation fails for wrong platform."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        metadata = models.PlatformMetadata(
            platform="tiktok",  # Wrong platform!
            title="Test",
            description="Test",
            hashtags=["#Shorts", "#Test", "#Tech", "#ad"],
            keywords=["test"],
            character_counts={"title": 4, "description": 4},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST123",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Platform mismatch" in error_msg

    def test_validate_title_too_long(self, youtube_settings):
        """Test validation fails when title exceeds max length."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        long_title = "A" * 65  # Exceeds 60 char max

        metadata = models.PlatformMetadata(
            platform="youtube",
            title=long_title,
            description="Test",
            hashtags=["#Shorts", "#Test", "#Tech", "#ad"],
            keywords=["test"],
            character_counts={"title": 65, "description": 4},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST123",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Title too long" in error_msg

    def test_validate_description_too_long(self, youtube_settings):
        """Test validation fails when description exceeds max length."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        long_description = "A" * 5500  # Exceeds 5000 char max

        metadata = models.PlatformMetadata(
            platform="youtube",
            title="Test Title For YouTube Shorts Video About Earbuds",  # 52 chars
            description=long_description,
            hashtags=["#Shorts", "#Test", "#Tech", "#ad"],
            keywords=["test"],
            character_counts={"title": 52, "description": 5500},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST123",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Description too long" in error_msg

    def test_validate_too_few_hashtags(self, youtube_settings):
        """Test validation fails with fewer than 3 hashtags."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        metadata = models.PlatformMetadata(
            platform="youtube",
            title="Test Title For YouTube Shorts Video About Earbuds",
            description="Test description",
            hashtags=["#Shorts", "#ad"],  # Only 2 hashtags
            keywords=["test"],
            character_counts={"title": 52, "description": 16},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST123",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Too few hashtags" in error_msg

    def test_validate_too_many_hashtags(self, youtube_settings):
        """Test validation fails with more than 5 hashtags."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        metadata = models.PlatformMetadata(
            platform="youtube",
            title="Test Title For YouTube Shorts Video About Earbuds",
            description="Test description",
            hashtags=[
                "#Shorts",
                "#Tech",
                "#Review",
                "#Gadgets",
                "#Amazon",
                "#ad",
            ],  # 6 hashtags
            keywords=["test"],
            character_counts={"title": 52, "description": 16},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST123",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Too many hashtags" in error_msg

    def test_validate_missing_shorts_tag(self, youtube_settings):
        """Test validation fails when #Shorts tag is missing."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        metadata = models.PlatformMetadata(
            platform="youtube",
            title="Test Title For YouTube Shorts Video About Earbuds",
            description="Test description",
            hashtags=["#Tech", "#Review", "#Gadgets", "#ad"],  # Missing #Shorts
            keywords=["test"],
            character_counts={"title": 52, "description": 16},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST123",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Missing required #Shorts hashtag" in error_msg

    def test_validate_missing_ad_tag(self, youtube_settings):
        """Test validation fails when #ad tag is missing."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        metadata = models.PlatformMetadata(
            platform="youtube",
            title="Test Title For YouTube Shorts Video About Earbuds",
            description="Test description",
            hashtags=["#Shorts", "#Tech", "#Review", "#Gadgets"],  # Missing #ad
            keywords=["test"],
            character_counts={"title": 52, "description": 16},
            generated_at="2025-01-15T12:00:00Z",
            product_id="B0TEST123",
            validation_status="pending",
            validation_messages=[],
        )

        is_valid, error_msg = generator.validate(metadata)

        assert not is_valid
        assert "Missing required #ad hashtag" in error_msg

    def test_parse_llm_response_success(self, youtube_settings):
        """Test successful parsing of LLM response."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        response = """TITLE: Best Wireless Earbuds 2025

DESCRIPTION: Top picks for wireless earbuds

HASHTAGS: #Tech #Review #Earbuds

KEYWORDS: wireless earbuds, tech review, 2025"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        title, description, hashtags, keywords = parsed
        assert title == "Best Wireless Earbuds 2025"
        assert description == "Top picks for wireless earbuds"
        assert hashtags == ["#Tech", "#Review", "#Earbuds"]
        assert keywords == ["wireless earbuds", "tech review", "2025"]

    def test_parse_llm_response_missing_hashtags(self, youtube_settings):
        """Test parsing succeeds even without optional HASHTAGS field."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        response = """TITLE: Test Title

DESCRIPTION: Test description

KEYWORDS: test, keywords"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        title, description, hashtags, keywords = parsed
        assert title == "Test Title"
        assert description == "Test description"
        assert hashtags == []  # Empty list when no hashtags
        assert keywords == ["test", "keywords"]

    def test_parse_llm_response_missing_keywords(self, youtube_settings):
        """Test parsing succeeds even without optional KEYWORDS field."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        response = """TITLE: Test Title

DESCRIPTION: Test description

HASHTAGS: #Tech #Review"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        title, description, hashtags, keywords = parsed
        assert title == "Test Title"
        assert description == "Test description"
        assert hashtags == ["#Tech", "#Review"]
        assert keywords == []  # Empty list when no keywords

    def test_parse_llm_response_hashtags_without_hash(self, youtube_settings):
        """Test parsing adds # prefix to hashtags if missing."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        response = """TITLE: Test

DESCRIPTION: Test

HASHTAGS: Tech Review Earbuds

KEYWORDS: test"""

        parsed = generator._parse_llm_response(response)

        assert parsed is not None
        _, _, hashtags, _ = parsed
        # All hashtags should have # prefix added
        assert hashtags == ["#Tech", "#Review", "#Earbuds"]

    def test_parse_llm_response_missing_title_fails(self, youtube_settings):
        """Test parsing fails when required TITLE field is missing."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        response = """DESCRIPTION: Test

HASHTAGS: #Test

KEYWORDS: test"""

        parsed = generator._parse_llm_response(response)

        # Should return None when title missing
        assert parsed is None

    def test_parse_llm_response_missing_description_fails(self, youtube_settings):
        """Test parsing fails when required DESCRIPTION field is missing."""
        generator = YouTubeMetadataGenerator(youtube_settings)

        response = """TITLE: Test

HASHTAGS: #Test

KEYWORDS: test"""

        parsed = generator._parse_llm_response(response)

        # Should return None when description missing
        assert parsed is None
