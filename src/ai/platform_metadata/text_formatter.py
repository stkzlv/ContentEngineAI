"""Format platform metadata into human-readable upload instructions.

This module converts PlatformMetadata objects into ready-to-copy text files
for manual social media posting.
"""

from datetime import UTC, datetime

from src.ai.platform_metadata.models import PlatformMetadata


def format_upload_instructions(
    metadata_results: dict[str, PlatformMetadata | None],
    product_id: str,
    video_filename: str,
    product_name: str = "Product",
    product_url: str | None = None,
) -> str:
    """Generate complete UPLOAD_INSTRUCTIONS.txt content.

    Args:
    ----
        metadata_results: Dict mapping platform names to PlatformMetadata objects
        product_id: Product identifier (ASIN)
        video_filename: Name of the generated video file
        product_name: Human-readable product name
        product_url: Full product URL for reference

    Returns:
    -------
        Formatted string ready to write to UPLOAD_INSTRUCTIONS.txt

    """
    sections = []

    # Header
    sections.append("=" * 80)
    sections.append(" " * 20 + "READY-TO-POST SOCIAL MEDIA CONTENT")
    sections.append(f"{' ' * 24}Product: {product_id}")
    sections.append(f"{' ' * 20}Video: {video_filename}")
    if product_url:
        sections.append(f"{' ' * 20}URL: {product_url}")
    sections.append("=" * 80)
    sections.append("")
    sections.append("📱 ALL PLATFORMS: Upload the same video file to each platform")
    sections.append("")

    # YouTube Shorts section
    youtube_metadata = metadata_results.get("youtube")
    if youtube_metadata:
        sections.append("")
        sections.append("━" * 80)
        sections.append("🎬 YOUTUBE SHORTS")
        sections.append("━" * 80)
        sections.append("")
        sections.append(_format_youtube_section(youtube_metadata))

    # TikTok section
    tiktok_metadata = metadata_results.get("tiktok")
    if tiktok_metadata:
        sections.append("")
        sections.append("━" * 80)
        sections.append("🎵 TIKTOK")
        sections.append("━" * 80)
        sections.append("")
        sections.append(_format_tiktok_section(tiktok_metadata))

    # Instagram Reels section
    instagram_metadata = metadata_results.get("instagram")
    if instagram_metadata:
        sections.append("")
        sections.append("━" * 80)
        sections.append("📷 INSTAGRAM REELS")
        sections.append("━" * 80)
        sections.append("")
        sections.append(_format_instagram_section(instagram_metadata))

    # Metadata summary table
    sections.append("")
    sections.append("━" * 80)
    sections.append("📊 METADATA SUMMARY")
    sections.append("━" * 80)
    sections.append("")
    sections.append(_format_metadata_table(metadata_results))

    # Upload checklist
    sections.append("")
    sections.append("━" * 80)
    sections.append("✅ UPLOAD CHECKLIST")
    sections.append("━" * 80)
    sections.append("")
    sections.append(_format_upload_checklist(video_filename, metadata_results))

    # Notes
    sections.append("")
    sections.append("━" * 80)
    sections.append("📌 NOTES")
    sections.append("━" * 80)
    sections.append("")
    sections.append("• All metadata optimized for each platform's 2025 algorithm")
    sections.append("• YouTube: SEO-focused with #Shorts for discovery")
    sections.append(
        "• TikTok: Search-optimized with niche hashtags (avoids generic #fyp)"
    )
    sections.append("• Instagram: Maximum hashtag strategy for broad reach")
    sections.append("• All include #ad for FTC compliance")
    sections.append("")
    sections.append(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d')}")
    sections.append(f"Product: {product_name} ({product_id})")
    sections.append("")

    return "\n".join(sections)


def _format_youtube_section(metadata: PlatformMetadata) -> str:
    """Format YouTube Shorts section."""
    lines = []

    # Title
    lines.append("📋 TITLE (copy below):")
    lines.append(metadata.title or "")
    lines.append("")

    # Description
    lines.append("📄 DESCRIPTION (copy below):")
    lines.append(metadata.description)
    lines.append("")

    # Hashtags
    hashtags_str = " ".join(metadata.hashtags)
    lines.append("🏷️ HASHTAGS (add to description or tags field):")
    lines.append(hashtags_str)
    lines.append("")

    # Settings
    lines.append("⚙️ SETTINGS:")
    lines.append("- Video Type: Shorts")
    lines.append("- Visibility: Public")
    lines.append("- Category: Science & Technology")
    lines.append("- Age Restriction: No")
    lines.append("")

    return "\n".join(lines)


def _format_tiktok_section(metadata: PlatformMetadata) -> str:
    """Format TikTok section."""
    lines = []

    # Caption with hashtags embedded
    lines.append("📝 CAPTION (copy entire block below - hashtags included):")
    lines.append(metadata.description)
    lines.append("")

    # Settings
    lines.append("⚙️ SETTINGS:")
    lines.append("- Allow Comments: Yes")
    lines.append("- Allow Duet: Yes")
    lines.append("- Allow Stitch: Yes")
    lines.append("- Visibility: Public")
    lines.append("")

    return "\n".join(lines)


def _format_instagram_section(metadata: PlatformMetadata) -> str:
    """Format Instagram Reels section."""
    lines = []

    # Caption (without hashtags)
    lines.append("📝 CAPTION (copy below):")
    lines.append(metadata.description)
    lines.append("")

    # Hashtags (separate from caption)
    hashtags_str = " ".join(metadata.hashtags)
    lines.append("🏷️ HASHTAGS (paste in caption OR first comment):")
    lines.append(hashtags_str)
    lines.append("")

    # Tip
    hashtag_count = len(metadata.hashtags)
    lines.append(
        f"💡 TIP: Instagram allows up to 30 hashtags. "
        f"Use all {hashtag_count} for maximum reach."
    )
    lines.append("")

    # Settings
    lines.append("⚙️ SETTINGS:")
    lines.append("- Format: Reels")
    lines.append("- Cover: Auto-select best frame")
    lines.append("- Allow Comments: Yes")
    lines.append("- Share to Feed: Yes")
    lines.append("")

    return "\n".join(lines)


def _format_metadata_table(metadata_results: dict[str, PlatformMetadata | None]) -> str:
    """Generate summary table with character counts and validation status."""
    lines = []

    # Platform name mapping for proper casing
    platform_names = {
        "youtube": "YouTube",
        "tiktok": "TikTok",
        "instagram": "Instagram",
    }

    # Table header
    lines.append(
        "Platform     | Status | Title Length | Description | Hashtags | Keywords"
    )
    lines.append(
        "-------------|--------|--------------|-------------|----------|----------"
    )

    # Table rows
    for platform in ["youtube", "tiktok", "instagram"]:
        metadata = metadata_results.get(platform)
        platform_display = platform_names[platform].ljust(12)

        if metadata:
            status_icon = _get_validation_icon(metadata.validation_status)
            title_len = (
                f"{metadata.character_counts.get('title', 0)} chars"
                if metadata.title
                else "N/A"
            )
            desc_len = f"{metadata.character_counts.get('description', 0)} chars"
            hashtag_count = f"{len(metadata.hashtags)} tags"
            keyword_count = f"{len(metadata.keywords)} terms"

            status_display = f"{status_icon} {metadata.validation_status.title()}"
            status_display = status_display.ljust(6)
            title_display = title_len.ljust(12)
            desc_display = desc_len.ljust(11)
            hashtag_display = hashtag_count.ljust(8)

            lines.append(
                f"{platform_display} | {status_display} | "
                f"{title_display} | {desc_display} | "
                f"{hashtag_display} | {keyword_count}"
            )
        else:
            lines.append(
                f"{platform_display} | ❌ Failed | "
                "N/A          | N/A         | N/A      | N/A"
            )

    lines.append("")
    return "\n".join(lines)


def _format_upload_checklist(
    video_filename: str, metadata_results: dict[str, PlatformMetadata | None]
) -> str:
    """Generate platform-specific upload checklists."""
    lines = []

    # YouTube checklist
    if metadata_results.get("youtube"):
        lines.append("□ YouTube Shorts:")
        lines.append(f"  □ Upload {video_filename}")
        lines.append("  □ Paste title")
        lines.append("  □ Paste description")
        lines.append("  □ Add hashtags")
        lines.append('  □ Select "Shorts" format')
        lines.append("  □ Publish as Public")
        lines.append("")

    # TikTok checklist
    if metadata_results.get("tiktok"):
        lines.append("□ TikTok:")
        lines.append(f"  □ Upload {video_filename}")
        lines.append("  □ Paste caption (hashtags included)")
        lines.append("  □ Enable comments/duet/stitch")
        lines.append("  □ Publish as Public")
        lines.append("")

    # Instagram checklist
    instagram_metadata = metadata_results.get("instagram")
    if instagram_metadata:
        lines.append("□ Instagram Reels:")
        lines.append(f"  □ Upload {video_filename}")
        lines.append("  □ Paste caption")
        hashtag_count = len(instagram_metadata.hashtags)
        lines.append(f"  □ Add all {hashtag_count} hashtags (caption or first comment)")
        lines.append("  □ Select cover frame")
        lines.append("  □ Share to Feed")
        lines.append("  □ Publish")
        lines.append("")

    return "\n".join(lines)


def _get_validation_icon(status: str) -> str:
    """Convert validation status to emoji."""
    status_icons = {
        "valid": "✅",
        "warning": "⚠️",
        "error": "❌",
    }
    return status_icons.get(status.lower(), "❓")
