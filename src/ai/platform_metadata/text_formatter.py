"""Format platform metadata into upload instructions for manual posting fallback."""

from datetime import UTC, datetime

from src.ai.platform_metadata.models import PlatformMetadata


def format_upload_instructions(
    metadata_results: dict[str, PlatformMetadata | None],
    product_id: str,
    video_filename: str,
    product_name: str = "Product",
    product_url: str | None = None,
) -> str:
    """Generate UPLOAD_INSTRUCTIONS.txt for manual posting fallback.

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
    lines = [
        "=" * 60,
        f"UPLOAD INSTRUCTIONS - {product_id}",
        "=" * 60,
        f"Video: {video_filename}",
    ]
    if product_url:
        lines.append(f"URL: {product_url}")
    lines.append("")

    # YouTube
    youtube = metadata_results.get("youtube")
    if youtube:
        lines.extend(
            [
                "-" * 60,
                "YOUTUBE SHORTS",
                "-" * 60,
                "",
                "Title:",
                youtube.title or "",
                "",
                "Description:",
                youtube.description,
                "",
                "Hashtags:",
                " ".join(youtube.hashtags),
                "",
            ]
        )

    # TikTok
    tiktok = metadata_results.get("tiktok")
    if tiktok:
        lines.extend(
            [
                "-" * 60,
                "TIKTOK",
                "-" * 60,
                "",
                "Caption (includes hashtags):",
                tiktok.description,
                "",
            ]
        )

    # Instagram
    instagram = metadata_results.get("instagram")
    if instagram:
        lines.extend(
            [
                "-" * 60,
                "INSTAGRAM REELS",
                "-" * 60,
                "",
                "Caption:",
                instagram.description,
                "",
                "Hashtags:",
                " ".join(instagram.hashtags),
                "",
            ]
        )

    lines.extend(
        [
            "-" * 60,
            f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
            f"Product: {product_name}",
        ]
    )

    return "\n".join(lines)
