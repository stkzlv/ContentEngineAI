"""Unit tests for platform metadata text formatter."""

from datetime import UTC, datetime

import pytest

from src.ai.platform_metadata.models import PlatformMetadata
from src.ai.platform_metadata.text_formatter import format_upload_instructions


@pytest.fixture
def sample_youtube_metadata():
    """Sample YouTube metadata for testing."""
    return PlatformMetadata(
        platform="youtube",
        title="Smart Posture Corrector - Fix Hunchback Fast",
        description="Improve your posture with our smart corrector.",
        hashtags=["#Shorts", "#PostureCorrector", "#HealthTech", "#ad"],
        keywords=["smart posture corrector", "back pain relief"],
        character_counts={"title": 44, "description": 50},
        generated_at=datetime.now(UTC).isoformat(),
        product_id="B0DNTC69V6",
        validation_status="valid",
        validation_messages=[],
    )


@pytest.fixture
def sample_tiktok_metadata():
    """Sample TikTok metadata for testing."""
    return PlatformMetadata(
        platform="tiktok",
        title=None,
        description="Smart posture corrector with vibration reminder. #PostureCorrector #HealthTech #ad",
        hashtags=["#PostureCorrector", "#HealthTech", "#ad"],
        keywords=["posture corrector", "vibration reminder"],
        character_counts={"description": 85},
        generated_at=datetime.now(UTC).isoformat(),
        product_id="B0DNTC69V6",
        validation_status="valid",
        validation_messages=[],
    )


@pytest.fixture
def sample_instagram_metadata():
    """Sample Instagram metadata for testing."""
    return PlatformMetadata(
        platform="instagram",
        title=None,
        description="Smart Posture Corrector - Improve Your Back Posture",
        hashtags=["#SmartPostureCorrector", "#PostureCorrection", "#BackPain", "#ad"],
        keywords=["smart posture corrector", "back posture"],
        character_counts={"description": 50},
        generated_at=datetime.now(UTC).isoformat(),
        product_id="B0DNTC69V6",
        validation_status="valid",
        validation_messages=[],
    )


def test_format_upload_instructions_all_platforms(
    sample_youtube_metadata, sample_tiktok_metadata, sample_instagram_metadata
):
    """Test upload instructions with all platforms."""
    metadata_results = {
        "youtube": sample_youtube_metadata,
        "tiktok": sample_tiktok_metadata,
        "instagram": sample_instagram_metadata,
    }

    result = format_upload_instructions(
        metadata_results=metadata_results,
        product_id="B0DNTC69V6",
        video_filename="video_B0DNTC69V6.mp4",
        product_name="Smart Posture Corrector",
        product_url="https://www.amazon.com/dp/B0DNTC69V6",
    )

    # Header
    assert "UPLOAD INSTRUCTIONS - B0DNTC69V6" in result
    assert "Video: video_B0DNTC69V6.mp4" in result
    assert "URL: https://www.amazon.com/dp/B0DNTC69V6" in result

    # YouTube section
    assert "YOUTUBE SHORTS" in result
    assert "Title:" in result
    assert sample_youtube_metadata.title in result
    assert "Description:" in result
    assert "Hashtags:" in result
    assert "#Shorts #PostureCorrector #HealthTech #ad" in result

    # TikTok section
    assert "TIKTOK" in result
    assert "Caption (includes hashtags):" in result
    assert sample_tiktok_metadata.description in result

    # Instagram section
    assert "INSTAGRAM REELS" in result
    assert "Caption:" in result
    assert sample_instagram_metadata.description in result
    assert "#SmartPostureCorrector #PostureCorrection #BackPain #ad" in result

    # Footer
    assert "Generated:" in result
    assert "Product: Smart Posture Corrector" in result


def test_format_upload_instructions_youtube_only(sample_youtube_metadata):
    """Test upload instructions with only YouTube metadata."""
    metadata_results = {
        "youtube": sample_youtube_metadata,
        "tiktok": None,
        "instagram": None,
    }

    result = format_upload_instructions(
        metadata_results=metadata_results,
        product_id="B0TEST",
        video_filename="video_test.mp4",
    )

    assert "YOUTUBE SHORTS" in result
    assert "TIKTOK" not in result
    assert "INSTAGRAM REELS" not in result


def test_format_upload_instructions_no_url(sample_youtube_metadata):
    """Test upload instructions without product URL."""
    metadata_results = {"youtube": sample_youtube_metadata}

    result = format_upload_instructions(
        metadata_results=metadata_results,
        product_id="B0TEST",
        video_filename="video_test.mp4",
    )

    assert "UPLOAD INSTRUCTIONS - B0TEST" in result
    assert "URL:" not in result


def test_format_upload_instructions_empty_metadata():
    """Test upload instructions with no metadata."""
    result = format_upload_instructions(
        metadata_results={},
        product_id="B0EMPTY",
        video_filename="video_empty.mp4",
    )

    assert "UPLOAD INSTRUCTIONS - B0EMPTY" in result
    assert "Video: video_empty.mp4" in result
    # No platform sections
    assert "YOUTUBE SHORTS" not in result
    assert "TIKTOK" not in result
    assert "INSTAGRAM REELS" not in result
