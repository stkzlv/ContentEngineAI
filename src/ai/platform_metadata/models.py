"""Platform-specific metadata models for YouTube, TikTok, and Instagram.

This module defines data structures for platform-optimized metadata generation,
including Pydantic configuration models and immutable metadata dataclasses.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class PlatformMetadata:
    """Immutable platform-specific metadata for video content.

    This dataclass stores generated metadata optimized for a specific platform
    (YouTube, TikTok, or Instagram) including titles, descriptions, hashtags,
    and validation status.

    Attributes
    ----------
        platform: Platform identifier ("youtube", "tiktok", "instagram")
        title: Platform-specific title (YouTube only, None for others)
        description: Platform-optimized description or caption
        hashtags: List of platform-optimized hashtags
        keywords: SEO keywords (YouTube primary, TikTok secondary)
        character_counts: Character count tracking ({"title": 58, "description": 487})
        generated_at: ISO 8601 timestamp of generation
        product_id: Product identifier (ASIN or similar)
        validation_status: Validation result ("valid", "warning", "error")
        validation_messages: List of validation details/warnings
        prompt_variant: A/B test variant name used for generation (for analytics)

    """

    platform: str
    description: str
    hashtags: list[str]
    keywords: list[str]
    character_counts: dict[str, int]
    generated_at: str
    product_id: str
    validation_status: str
    validation_messages: list[str]
    title: str | None = None
    prompt_variant: str | None = None

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for JSON serialization.

        Returns
        -------
            Dictionary representation of metadata with all fields.

        """
        return {
            "platform": self.platform,
            "title": self.title,
            "description": self.description,
            "hashtags": self.hashtags,
            "keywords": self.keywords,
            "character_counts": self.character_counts,
            "generated_at": self.generated_at,
            "product_id": self.product_id,
            "validation_status": self.validation_status,
            "validation_messages": self.validation_messages,
            "prompt_variant": self.prompt_variant,
        }

    @classmethod
    def create(
        cls,
        platform: str,
        description: str,
        hashtags: list[str],
        keywords: list[str],
        product_id: str,
        title: str | None = None,
        validation_status: str = "valid",
        validation_messages: list[str] | None = None,
        prompt_variant: str | None = None,
    ) -> "PlatformMetadata":
        """Factory method to create PlatformMetadata with auto-generated fields.

        Args:
        ----
            platform: Platform identifier
            description: Platform-specific description
            hashtags: List of hashtags
            keywords: List of SEO keywords
            product_id: Product identifier
            title: Optional title (YouTube only)
            validation_status: Validation status (default: "valid")
            validation_messages: Optional validation messages
            prompt_variant: A/B test variant name (for analytics)

        Returns:
        -------
            New PlatformMetadata instance with calculated character counts and
            timestamp.

        """
        character_counts = {"description": len(description)}
        if title:
            character_counts["title"] = len(title)

        return cls(
            platform=platform,
            title=title,
            description=description,
            hashtags=hashtags,
            keywords=keywords,
            character_counts=character_counts,
            generated_at=datetime.now(UTC).isoformat(),
            product_id=product_id,
            validation_status=validation_status,
            validation_messages=validation_messages or [],
            prompt_variant=prompt_variant,
        )


class YouTubePlatformSettings(BaseModel):
    """YouTube-specific platform settings with character limits and hashtag rules.

    Configures YouTube Shorts optimization including title length, description limits,
    hashtag counts, and SEO keyword requirements.
    """

    enabled: bool = Field(True, description="Enable YouTube metadata generation")
    title_length_max: int = Field(
        60,
        ge=1,
        le=100,
        description="Maximum title length (recommended 50-60 for SEO)",
    )
    description_length_max: int = Field(
        5000, ge=1, le=5000, description="Maximum description length (YouTube limit)"
    )
    hashtag_count_min: int = Field(
        3, ge=0, le=10, description="Minimum number of hashtags"
    )
    hashtag_count_max: int = Field(
        5, ge=1, le=15, description="Maximum number of hashtags"
    )
    include_shorts_tag: bool = Field(
        True, description="Automatically include #Shorts hashtag for vertical videos"
    )
    seo_keywords: bool = Field(
        True, description="Enable SEO keyword optimization in title and description"
    )


class TikTokPlatformSettings(BaseModel):
    """TikTok-specific platform settings optimized for search discoverability.

    Configures TikTok caption optimization with character limits, hashtag strategy,
    and SEO-focused content requirements.
    """

    enabled: bool = Field(True, description="Enable TikTok metadata generation")
    caption_length_optimal: int = Field(
        150,
        ge=50,
        le=300,
        description="Optimal caption length for engagement (100-300 chars)",
    )
    caption_length_max: int = Field(
        2200, ge=1, le=2200, description="Maximum caption length (TikTok limit)"
    )
    hashtag_count_min: int = Field(
        3, ge=0, le=10, description="Minimum number of hashtags"
    )
    hashtag_count_max: int = Field(
        5, ge=1, le=10, description="Maximum number of hashtags"
    )
    seo_focused: bool = Field(
        True,
        description="Use SEO-focused exact search phrases vs creative captions",
    )
    avoid_generic_tags: list[str] = Field(
        default=["foryoupage", "fyp", "viral", "foryou"],
        description="Generic hashtags to avoid (prefer niche-specific tags)",
    )


class InstagramPlatformSettings(BaseModel):
    """Instagram Reels-specific platform settings with dual caption styles.

    Configures Instagram Reels optimization supporting both ultra-short (3-5 words)
    and SEO-descriptive (100-240 chars) caption styles with extensive hashtag usage.
    """

    enabled: bool = Field(True, description="Enable Instagram metadata generation")
    caption_style: str = Field(
        "seo",
        pattern="^(short|seo)$",
        description="Caption style: 'short' (3-5 words) or 'seo' (100-240 chars)",
    )
    caption_length_short: int = Field(
        15, ge=5, le=30, description="Character limit for 'short' caption style"
    )
    caption_length_seo: int = Field(
        240,
        ge=50,
        le=300,
        description=(
            "Character limit for 'seo' caption style "
            "(room for body + mirrored closing line)"
        ),
    )
    hashtag_count_min: int = Field(
        15, ge=5, le=20, description="Minimum number of hashtags"
    )
    hashtag_count_max: int = Field(
        30, ge=15, le=30, description="Maximum number of hashtags (Instagram limit)"
    )
    emoji_enabled: bool = Field(
        True, description="Allow emoji usage in captions and hashtags"
    )


class MetadataCacheSettings(BaseModel):
    """Configuration settings for metadata caching.

    Attributes
    ----------
        enabled: Enable/disable caching globally
        ttl_hours: Time-to-live for cache entries in hours
        cache_dir: Directory for cache storage (relative to project root)
        max_entries: Maximum number of cache entries (0 = unlimited)

    """

    enabled: bool = Field(True, description="Enable metadata caching")
    ttl_hours: int = Field(
        24,
        ge=1,
        le=720,  # Max 30 days
        description="Cache entry TTL in hours (1-720)",
    )
    cache_dir: str = Field(
        ".cache/platform_metadata",
        description="Cache directory path (relative to project root)",
    )
    max_entries: int = Field(
        1000,
        ge=0,
        description="Maximum cache entries (0 = unlimited)",
    )


class BatchGenerationSettings(BaseModel):
    """Configuration settings for batch metadata generation.

    Controls concurrent processing of multiple products with rate limiting
    and progress tracking.

    Attributes
    ----------
        enabled: Enable/disable batch generation
        max_concurrent: Maximum concurrent product generations (1-20)
        log_progress: Enable progress logging with [N/total] format

    """

    enabled: bool = Field(True, description="Enable batch metadata generation")
    max_concurrent: int = Field(
        3,
        ge=1,
        le=20,
        description="Maximum concurrent product generations (1-20)",
    )
    log_progress: bool = Field(
        True,
        description="Log progress with [N/total] format",
    )


class ExportSettings(BaseModel):
    """Configuration settings for metadata export.

    Controls export formats, encoding, and platform-specific options.

    Attributes
    ----------
        enabled: Enable/disable export functionality
        default_format: Default export format (json, csv, youtube_csv,
            tiktok, instagram)
        youtube_category: Default YouTube category ID (22 = People & Blogs)
        youtube_privacy: Default privacy setting for YouTube exports
        csv_encoding: CSV file encoding (utf-8-sig for Excel compatibility)
        json_indent: JSON indentation spaces for pretty-printing
        youtube_title_fallback_length: Max length when using description as title

    """

    enabled: bool = Field(True, description="Enable metadata export functionality")
    default_format: str = Field(
        "json",
        pattern="^(json|csv|youtube_csv|tiktok|instagram)$",
        description="Default export format",
    )
    youtube_category: str = Field(
        "22",
        description="Default YouTube category ID (22 = People & Blogs)",
    )
    youtube_privacy: str = Field(
        "private",
        pattern="^(private|public|unlisted)$",
        description="Default privacy setting for YouTube exports",
    )
    csv_encoding: str = Field(
        "utf-8-sig",
        description="CSV encoding (utf-8-sig for Excel, utf-8 for others)",
    )
    json_indent: int = Field(
        2,
        ge=0,
        le=8,
        description="JSON indentation spaces (0 for compact)",
    )
    youtube_title_fallback_length: int = Field(
        60,
        ge=10,
        le=100,
        description="Max title length when using description as fallback",
    )


class PlatformFallbackTags(BaseModel):
    """Fallback trending tags per platform when external APIs unavailable.

    Used as baseline high-performance tags when trend providers fail.
    """

    youtube: list[str] = Field(
        default=["#Shorts", "#Trending", "#Tech", "#Gadgets", "#Review"],
        description="Fallback trending tags for YouTube",
    )
    tiktok: list[str] = Field(
        default=["#ForYou", "#Viral", "#LifeHack", "#Shopping", "#TechTok"],
        description="Fallback trending tags for TikTok",
    )
    instagram: list[str] = Field(
        default=["#Reels", "#InstaGood", "#Innovation", "#Gadget", "#TechLife"],
        description="Fallback trending tags for Instagram",
    )


class TrendSettings(BaseModel):
    """Configuration for trend-aware hashtag generation.

    Integrates with trend APIs to suggest current hashtags for better discovery.

    Attributes
    ----------
        enabled: Enable trend-aware hashtags globally
        provider: Trend provider identifier ("static", "mock", "external")
        cache_ttl_hours: Time-to-live for cached trends in hours
        max_trending_tags: Maximum number of trending tags to add per platform
        fallback_tags: Per-platform fallback tags when APIs unavailable

    """

    enabled: bool = Field(False, description="Enable trend-aware hashtags")
    provider: str = Field("static", description="Trend provider identifier")
    cache_ttl_hours: int = Field(
        4,
        ge=1,
        le=24,
        description="Time-to-live for cached trends (1-24h)",
    )
    max_trending_tags: int = Field(
        2,
        ge=1,
        le=5,
        description="Max trending tags to add (respects platform limits)",
    )
    fallback_tags: PlatformFallbackTags = Field(
        default_factory=PlatformFallbackTags,
        description="Per-platform fallback trending tags",
    )


class PlatformMetadataSettings(BaseModel):
    """Top-level platform metadata configuration for multi-platform optimization.

    Aggregates settings for YouTube, TikTok, and Instagram with platform targeting
    control. Enables/disables platform-specific metadata generation globally.

    Attributes
    ----------
        enabled: Global enable/disable for platform-specific metadata
        target_platform: Target platform(s) - "youtube", "tiktok",
            "instagram", or "multi"
        youtube: YouTube-specific settings
        tiktok: TikTok-specific settings
        instagram: Instagram-specific settings
        cache: Metadata caching settings
        batch: Batch generation settings
        export: Export settings
        trends: Trend-aware hashtag settings

    """

    enabled: bool = Field(
        True, description="Enable platform-specific metadata generation"
    )
    target_platform: str = Field(
        "multi",
        pattern="^(youtube|tiktok|instagram|multi)$",
        description=(
            "Target platform: 'youtube', 'tiktok', 'instagram', or 'multi' for all"
        ),
    )
    youtube: YouTubePlatformSettings = Field(
        default_factory=lambda: YouTubePlatformSettings(),  # type: ignore[call-arg]
        description="YouTube platform settings",
    )
    tiktok: TikTokPlatformSettings = Field(
        default_factory=lambda: TikTokPlatformSettings(),  # type: ignore[call-arg]
        description="TikTok platform settings",
    )
    instagram: InstagramPlatformSettings = Field(
        default_factory=lambda: InstagramPlatformSettings(),  # type: ignore[call-arg]
        description="Instagram platform settings",
    )
    cache: MetadataCacheSettings = Field(
        default_factory=lambda: MetadataCacheSettings(),  # type: ignore[call-arg]
        description="Metadata caching settings",
    )
    batch: BatchGenerationSettings = Field(
        default_factory=lambda: BatchGenerationSettings(),  # type: ignore[call-arg]
        description="Batch metadata generation settings",
    )
    export: ExportSettings = Field(
        default_factory=lambda: ExportSettings(),  # type: ignore[call-arg]
        description="Metadata export settings",
    )
    trends: TrendSettings = Field(
        default_factory=lambda: TrendSettings(),  # type: ignore[call-arg]
        description="Trend-aware hashtag settings",
    )
