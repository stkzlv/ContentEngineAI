"""Unit tests for platform metadata text formatter."""

from datetime import UTC, datetime

import pytest

from src.ai.platform_metadata.models import PlatformMetadata
from src.ai.platform_metadata.text_formatter import (
    _format_instagram_section,
    _format_metadata_table,
    _format_tiktok_section,
    _format_upload_checklist,
    _format_youtube_section,
    _get_validation_icon,
    format_upload_instructions,
)


@pytest.fixture
def sample_youtube_metadata():
    """Sample YouTube metadata for testing."""
    return PlatformMetadata(
        platform="youtube",
        title="Smart Posture Corrector - Fix Hunchback Fast",
        description="Improve your posture with our smart corrector featuring vibration reminders and adjustable angles.",
        hashtags=["#Shorts", "#PostureCorrector", "#BackPain", "#HealthTech", "#ad"],
        keywords=[
            "smart posture corrector",
            "hunchback correction",
            "back pain relief",
            "vibration reminder",
            "posture improvement",
        ],
        character_counts={"title": 44, "description": 105},
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
        description="Upgraded smart posture corrector with vibration reminder and adjustable angle. #PostureCorrector #HealthTech #BackHealth #SmartGadgets #ad",
        hashtags=[
            "#PostureCorrector",
            "#HealthTech",
            "#BackHealth",
            "#SmartGadgets",
            "#ad",
        ],
        keywords=[
            "posture corrector",
            "vibration reminder",
            "adjustable angle",
            "hunchback correction",
            "back health",
            "smart gadgets",
            "wellness tech",
        ],
        character_counts={"description": 140},
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
        description="Smart Posture Corrector with Vibration Reminder - Improve Your Back Posture",
        hashtags=[
            "#SmartPostureCorrector",
            "#PostureCorrection",
            "#BackPainRelief",
            "#VibrationReminder",
            "#HunchbackCorrector",
            "#PostureImprovement",
            "#BackSupport",
            "#SpineHealth",
            "#ErgonomicDesign",
            "#WellnessTech",
            "#HealthGadgets",
            "#FitnessEssentials",
            "#PostureCorrectorForMen",
            "#PostureCorrectorForWomen",
            "#BackPostureCorrector",
            "#AdjustablePostureCorrector",
            "#IntelligentPostureCorrector",
            "#PostureMonitor",
            "#HunchbackSolution",
            "#BackPainSolutions",
            "#ad",
            "#PostureCorrectionTips",
            "#HealthyBack",
            "#SpineCare",
        ],
        keywords=[
            "smart posture corrector",
            "vibration reminder",
            "back pain relief",
            "hunchback corrector",
            "posture improvement",
            "back support",
            "spine health",
            "ergonomic design",
            "wellness tech",
            "health gadgets",
            "fitness essentials",
            "adjustable posture corrector",
        ],
        character_counts={"description": 75},
        generated_at=datetime.now(UTC).isoformat(),
        product_id="B0DNTC69V6",
        validation_status="valid",
        validation_messages=[],
    )


def test_get_validation_icon():
    """Test validation status icon mapping."""
    assert _get_validation_icon("valid") == "✅"
    assert _get_validation_icon("warning") == "⚠️"
    assert _get_validation_icon("error") == "❌"
    assert _get_validation_icon("unknown") == "❓"


def test_format_youtube_section(sample_youtube_metadata):
    """Test YouTube Shorts section formatting."""
    result = _format_youtube_section(sample_youtube_metadata)

    assert "📋 TITLE (copy below):" in result
    assert sample_youtube_metadata.title in result
    assert "📄 DESCRIPTION (copy below):" in result
    assert sample_youtube_metadata.description in result
    assert "🏷️ HASHTAGS (add to description or tags field):" in result
    assert "#Shorts #PostureCorrector #BackPain #HealthTech #ad" in result
    assert "⚙️ SETTINGS:" in result
    assert "- Video Type: Shorts" in result


def test_format_tiktok_section(sample_tiktok_metadata):
    """Test TikTok section formatting."""
    result = _format_tiktok_section(sample_tiktok_metadata)

    assert "📝 CAPTION (copy entire block below - hashtags included):" in result
    assert sample_tiktok_metadata.description in result
    assert "⚙️ SETTINGS:" in result
    assert "- Allow Comments: Yes" in result
    assert "- Allow Duet: Yes" in result
    assert "- Allow Stitch: Yes" in result


def test_format_instagram_section(sample_instagram_metadata):
    """Test Instagram Reels section formatting."""
    result = _format_instagram_section(sample_instagram_metadata)

    assert "📝 CAPTION (copy below):" in result
    assert sample_instagram_metadata.description in result
    assert "🏷️ HASHTAGS (paste in caption OR first comment):" in result
    assert "#SmartPostureCorrector" in result
    assert "💡 TIP: Instagram allows up to 30 hashtags." in result
    assert "Use all 24 for maximum reach." in result
    assert "⚙️ SETTINGS:" in result


def test_format_metadata_table_all_platforms(
    sample_youtube_metadata, sample_tiktok_metadata, sample_instagram_metadata
):
    """Test metadata summary table with all platforms."""
    metadata_results = {
        "youtube": sample_youtube_metadata,
        "tiktok": sample_tiktok_metadata,
        "instagram": sample_instagram_metadata,
    }

    result = _format_metadata_table(metadata_results)

    assert (
        "Platform     | Status | Title Length | Description | Hashtags | Keywords"
        in result
    )
    assert "YouTube      | ✅ Valid" in result
    assert "44 chars" in result
    assert "105 chars" in result
    assert "5 tags" in result
    assert "5 terms" in result
    assert "TikTok       | ✅ Valid" in result
    assert "Instagram    | ✅ Valid" in result


def test_format_metadata_table_partial_failure(sample_youtube_metadata):
    """Test metadata table with some platforms missing."""
    metadata_results = {
        "youtube": sample_youtube_metadata,
        "tiktok": None,
        "instagram": None,
    }

    result = _format_metadata_table(metadata_results)

    assert "YouTube      | ✅ Valid" in result
    assert "TikTok       | ❌ Failed" in result
    assert "Instagram    | ❌ Failed" in result


def test_format_upload_checklist_all_platforms(
    sample_youtube_metadata, sample_tiktok_metadata, sample_instagram_metadata
):
    """Test upload checklist with all platforms."""
    metadata_results = {
        "youtube": sample_youtube_metadata,
        "tiktok": sample_tiktok_metadata,
        "instagram": sample_instagram_metadata,
    }
    video_filename = "video_B0DNTC69V6_slideshow_images4.mp4"

    result = _format_upload_checklist(video_filename, metadata_results)

    assert "□ YouTube Shorts:" in result
    assert f"  □ Upload {video_filename}" in result
    assert "  □ Paste title" in result
    assert "□ TikTok:" in result
    assert "  □ Paste caption (hashtags included)" in result
    assert "□ Instagram Reels:" in result
    assert "  □ Add all 24 hashtags (caption or first comment)" in result


def test_format_upload_checklist_partial_platforms(sample_youtube_metadata):
    """Test upload checklist with only YouTube."""
    metadata_results = {
        "youtube": sample_youtube_metadata,
        "tiktok": None,
        "instagram": None,
    }
    video_filename = "video_test.mp4"

    result = _format_upload_checklist(video_filename, metadata_results)

    assert "□ YouTube Shorts:" in result
    assert "□ TikTok:" not in result
    assert "□ Instagram Reels:" not in result


def test_format_upload_instructions_complete(
    sample_youtube_metadata, sample_tiktok_metadata, sample_instagram_metadata
):
    """Test complete upload instructions generation."""
    metadata_results = {
        "youtube": sample_youtube_metadata,
        "tiktok": sample_tiktok_metadata,
        "instagram": sample_instagram_metadata,
    }
    product_id = "B0DNTC69V6"
    video_filename = "video_B0DNTC69V6_slideshow_images4.mp4"
    product_name = "Caliora Smart Posture Corrector"

    product_url = "https://www.amazon.com/dp/B0DNTC69V6"

    result = format_upload_instructions(
        metadata_results=metadata_results,
        product_id=product_id,
        video_filename=video_filename,
        product_name=product_name,
        product_url=product_url,
    )

    # Header
    assert "READY-TO-POST SOCIAL MEDIA CONTENT" in result
    assert f"Product: {product_id}" in result
    assert f"Video: {video_filename}" in result
    assert f"URL: {product_url}" in result
    assert "📱 ALL PLATFORMS: Upload the same video file to each platform" in result

    # Platform sections
    assert "🎬 YOUTUBE SHORTS" in result
    assert "🎵 TIKTOK" in result
    assert "📷 INSTAGRAM REELS" in result

    # Summary and checklist
    assert "📊 METADATA SUMMARY" in result
    assert "✅ UPLOAD CHECKLIST" in result
    assert "📌 NOTES" in result

    # Notes content
    assert "All metadata optimized for each platform's 2025 algorithm" in result
    assert "All include #ad for FTC compliance" in result
    assert f"Product: {product_name} ({product_id})" in result


def test_format_upload_instructions_minimal(sample_youtube_metadata):
    """Test upload instructions with minimal metadata."""
    metadata_results = {
        "youtube": sample_youtube_metadata,
        "tiktok": None,
        "instagram": None,
    }
    product_id = "B0TEST"
    video_filename = "video_test.mp4"

    result = format_upload_instructions(
        metadata_results=metadata_results,
        product_id=product_id,
        video_filename=video_filename,
    )

    # Should still have header and YouTube section
    assert "READY-TO-POST SOCIAL MEDIA CONTENT" in result
    assert "🎬 YOUTUBE SHORTS" in result

    # TikTok and Instagram sections should be missing
    assert "🎵 TIKTOK" not in result
    assert "📷 INSTAGRAM REELS" not in result

    # Should still have table and notes
    assert "📊 METADATA SUMMARY" in result
    assert "📌 NOTES" in result


def test_format_upload_instructions_validation_warning():
    """Test upload instructions with validation warnings."""
    warning_metadata = PlatformMetadata(
        platform="youtube",
        title="Test Title",
        description="Test description",
        hashtags=["#Shorts", "#ad"],
        keywords=["test"],
        character_counts={"title": 10, "description": 16},
        generated_at=datetime.now(UTC).isoformat(),
        product_id="B0TEST",
        validation_status="warning",
        validation_messages=["Too few hashtags: 2 (min 3)"],
    )

    metadata_results = {"youtube": warning_metadata, "tiktok": None, "instagram": None}

    result = format_upload_instructions(
        metadata_results=metadata_results,
        product_id="B0TEST",
        video_filename="video_test.mp4",
    )

    # Should show warning icon in table
    assert "⚠️" in result
    assert "Warning" in result
