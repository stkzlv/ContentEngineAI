"""Data models for video publishing and scheduling.

This module defines the core data structures used across all publisher
implementations, providing type-safe representations of publish results,
metadata, configuration, and batch summaries.
"""

import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.video.config.constants import LATE_API_KEY_MIN_LENGTH


class PublishStatus(Enum):
    """Status of a publishing operation."""

    PENDING = "pending"
    UPLOADING = "uploading"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"


class Platform(Enum):
    """Supported social media platforms."""

    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"


DEFAULT_PLATFORMS = [Platform.YOUTUBE, Platform.TIKTOK, Platform.INSTAGRAM]

# Platform-specific content limits for validation
PLATFORM_LIMITS: dict[Platform, dict[str, int | tuple[int, int]]] = {
    Platform.YOUTUBE: {"title": 100, "description": 5000, "hashtags": (3, 15)},
    Platform.TIKTOK: {"description": 2200, "hashtags": (3, 5)},
    Platform.INSTAGRAM: {"description": 2200, "hashtags": (5, 30)},
}

_ELLIPSIS = "..."


def _trim_on_word_boundary(text: str, limit: int) -> str:
    """Trim text to at most `limit` chars, breaking on a word boundary.

    Reserves three chars for an ellipsis. Falls back to a hard cut when no
    whitespace exists in the budgeted range.
    """
    if len(text) <= limit:
        return text
    budget = limit - len(_ELLIPSIS)
    if budget <= 0:
        return text[:limit]
    head = text[:budget]
    cut = head.rfind(" ")
    if cut == -1:
        return head + _ELLIPSIS
    return head[:cut].rstrip() + _ELLIPSIS


@dataclass(frozen=True)
class PublishResult:
    """Result of a video publishing operation.

    This immutable data structure represents the outcome of publishing a video
    to one or more social media platforms via a scheduling service.

    Attributes
    ----------
        post_id: Unique identifier assigned by the publishing provider
        status: Current status of the post (scheduled, published, failed, etc.)
        platforms: List of platforms the post was published to
        scheduled_time: When the post is scheduled to go live (None if immediate)
        published_urls: List of direct URLs to published posts (empty if scheduled)
        error_message: Error description if status is FAILED (None otherwise)
        metadata: Additional provider-specific data

    """

    post_id: str
    status: PublishStatus
    platforms: tuple[Platform, ...]
    scheduled_time: datetime | None = None
    published_urls: tuple[str, ...] = field(default_factory=tuple)
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation."""
        # Validate post_id is not empty
        if not self.post_id or not self.post_id.strip():
            raise ValueError("post_id cannot be empty")

        # Validate platforms list is not empty
        if not self.platforms:
            raise ValueError("platforms cannot be empty")

        # Validate error_message is provided for FAILED status
        if self.status == PublishStatus.FAILED and not self.error_message:
            raise ValueError("error_message required when status is FAILED")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
            Dictionary representation with serializable values

        """
        return {
            "post_id": self.post_id,
            "status": self.status.value,
            "platforms": [p.value for p in self.platforms],
            "scheduled_time": (
                self.scheduled_time.isoformat() if self.scheduled_time else None
            ),
            "published_urls": list(self.published_urls),
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class AffiliateDisclosureConfig:
    """Configuration for the affiliate program literal-phrase disclosure.

    Renders the configured literal phrase in the caption body after the
    standard ``disclosure`` (``#ad``) line. Driven from
    ``config/publisher.yaml::affiliate_disclosure`` so non-Amazon programs
    (ShareASale, Impact, eBay Partner Network) can plug in their own phrases.

    Disabled by default. The phrase asserts membership of a named affiliate
    program, so it must only be emitted while that membership is actually
    active: claiming it otherwise misstates a material connection. Defaulting
    to on meant an unconfigured install published an Amazon Associates claim
    on every caption, and the config loader falls back to these defaults when
    the YAML section is missing or empty.

    ``phrase`` and ``program`` keep the Amazon values so enabling the feature
    needs one line rather than three.
    """

    enabled: bool = False
    phrase: str = "As an Amazon Associate I earn from qualifying purchases"
    program: str = "amazon"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "enabled": self.enabled,
            "phrase": self.phrase,
            "program": self.program,
        }


@dataclass
class AnalyticsConfig:
    """Configuration for the per-post analytics sweep.

    Driven from ``config/publisher.yaml::analytics``. The sweep reads this
    rather than taking the size from its command line, so the manual run and
    the scheduled one cannot disagree about how many posts to measure.

    ``limit`` must exceed the number of posts published inside the provider's
    timeline retention horizon, which is roughly five weeks. Below that, the
    oldest still-reachable figures are skipped on every sweep and then expire
    unrecorded, which is silent: a short sweep looks exactly like a complete
    one. At one post a day the horizon holds about 35 posts, so the shipped
    50 leaves headroom; a faster cadence needs a larger value.
    """

    limit: int = 50

    def __post_init__(self):
        """Reject a limit that would measure nothing, or break the slice.

        The type check is not pedantry: the value reaches ``posts[:limit]``,
        so a YAML float passes a bare ``< 1`` test and then raises
        ``TypeError: slice indices must be integers`` mid-sweep, once a day,
        where nobody is watching. ``bool`` is excluded because it is an int
        subclass and ``posts[:True]`` silently measures one post.
        """
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError(
                f"analytics.limit must be a whole number, got {self.limit!r}"
            )
        if self.limit < 1:
            raise ValueError(f"analytics.limit must be at least 1, got {self.limit}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {"limit": self.limit}


# The caption-leading disclosure. Named so `load_platform_metadata` can pass
# it explicitly at construction time -- the guard that strips disclosure
# tokens runs in `__post_init__`, so anything set afterwards is invisible to it.
DEFAULT_DISCLOSURE = "#ad"


def strip_disclosure_tokens(
    description: str,
    hashtags: list[str],
    disclosure: str = DEFAULT_DISCLOSURE,
) -> tuple[str, list[str]]:
    """Remove disclosure tokens from a caption body.

    Shared rather than a `PublishMetadata` method because both callers need it
    at a different moment. On the object it runs in `__post_init__`, only for
    a render with nothing to disclose. `schedule auto` runs it ahead of
    construction and for every render, because the leading line is the sole
    disclosure there and a second copy the model left in the body would sit
    below the fold saying the same thing twice.

    Covers `#ad` and the configured token. The prompts write `#ad` whatever
    the publisher is configured to say, and on a Spanish render those are two
    different strings.

    The body is edited only for a standalone `#token`; anything else is the
    model's prose. The whitespace repair runs only where a token was actually
    removed, so a caption that never contained one is returned untouched --
    French spacing before `!` and `?`, deliberate ellipses and double spaces
    all survive.
    """
    tokens = {"ad", disclosure.lstrip("#").lower()} - {""}

    kept = [tag for tag in hashtags if tag.lstrip("#").lower() not in tokens]

    for token in tokens:
        # Word-bounded so `#advice` and `#adapter` are left alone. The
        # trailing horizontal space is consumed with the token so the repair
        # below has nothing to do in the common case.
        description, count = re.subn(
            rf"(?<!\w)#{re.escape(token)}\b[ \t]*",
            "",
            description,
            flags=re.IGNORECASE,
        )
        if count:
            description = re.sub(r"[ \t]+([.,!?])", r"\1", description)
    return description.strip(), kept


@dataclass
class PublishMetadata:
    """Platform-specific metadata for video publishing.

    Contains the content and metadata needed to create a post on social media
    platforms. This data is typically loaded from platform-optimized metadata
    files generated by the AI system (v0.17.0+).

    Attributes
    ----------
        platform: Target social media platform
        title: Video title (YouTube only, None for other platforms)
        description: Full description or caption text
        hashtags: List of hashtags (without # prefix)
        keywords: List of keywords for platform algorithms
        character_counts: Character counts for validation

    """

    platform: Platform
    title: str | None
    description: str
    hashtags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    character_counts: dict[str, int] = field(default_factory=dict)
    product_id: str | None = None
    disclosure: str = DEFAULT_DISCLOSURE
    affiliate_disclosure: str | None = None
    # Whether the render has a material connection to disclose, as recorded by
    # the producer. Carried explicitly rather than inferred from an empty
    # `disclosure`, which a caller may clear for its own reasons: the affiliate
    # program phrase asserts membership and must key off the same decision the
    # `#ad` gate uses, not off a side effect of it.
    carries_affiliate_content: bool = True

    def __post_init__(self):
        """Post-initialization validation and normalization."""
        # Validate description is not empty
        if not self.description or not self.description.strip():
            raise ValueError(f"description cannot be empty for {self.platform.value}")

        # Validate YouTube has title
        if self.platform == Platform.YOUTUBE and not self.title:
            raise ValueError("YouTube posts require a title")

        # Normalize hashtags (remove # prefix if present)
        self.hashtags = [
            tag.lstrip("#") if tag.startswith("#") else tag for tag in self.hashtags
        ]

        # Drop hashtags that duplicate the disclosure (it leads the caption now).
        # Compare case-insensitively against the bare token (without #).
        if self.disclosure:
            disc_token = self.disclosure.lstrip("#").lower()
            self.hashtags = [t for t in self.hashtags if t.lower() != disc_token]

        # A render with no material connection must not carry a disclosure
        # token at all, wherever it came from. The caption prompts instruct
        # the model to write `#ad` and demonstrate it in every example, and
        # they are not told whether this render has an affiliate link.
        #
        # Until now it was removed by two accidents: the trailing-hashtag rule
        # in `load_platform_metadata`, written for legacy metadata, and the
        # dedup above, which only matches while `disclosure` still holds its
        # `#ad` default. Neither survives a non-trailing token or a disclosure
        # configured to a language variant.
        if not self.carries_affiliate_content:
            self._strip_disclosure_tokens()

        # Calculate character counts if not provided
        if not self.character_counts:
            self.character_counts = {"description": len(self.description)}
            if self.title:
                self.character_counts["title"] = len(self.title)

    def _strip_disclosure_tokens(self) -> None:
        """Remove disclosure tokens from a render that has nothing to disclose."""
        self.description, self.hashtags = strip_disclosure_tokens(
            self.description, self.hashtags, self.disclosure
        )

    def validate_limits(self) -> tuple[bool, str]:
        """Validate content against platform-specific character limits.

        Returns
        -------
            Tuple of (is_valid, message)
                - is_valid: True if all limits respected, False otherwise
                - message: "Content within limits" if valid, error desc if invalid

        """
        platform_limits = PLATFORM_LIMITS.get(self.platform)
        if not platform_limits:
            return True, "Content within limits"

        # Validate title length (YouTube only)
        title_limit = platform_limits.get("title")
        if (
            isinstance(title_limit, int)
            and self.title
            and len(self.title) > title_limit
        ):
            return (
                False,
                f"Title exceeds {title_limit} chars (got {len(self.title)})",
            )

        # Validate description length
        desc_limit = platform_limits.get("description")
        if isinstance(desc_limit, int) and len(self.description) > desc_limit:
            desc_len = len(self.description)
            return (
                False,
                f"Description exceeds {desc_limit} chars (got {desc_len})",
            )

        # Validate hashtag count
        hashtag_limits = platform_limits.get("hashtags")
        if isinstance(hashtag_limits, tuple) and self.hashtags:
            min_tags, max_tags = hashtag_limits
            tag_count = len(self.hashtags)
            if tag_count < min_tags:
                return (
                    False,
                    f"Hashtags: {min_tags}-{max_tags} required (got {tag_count})",
                )
            if tag_count > max_tags:
                return (
                    False,
                    f"Hashtags: {min_tags}-{max_tags} required (got {tag_count})",
                )

        return True, "Content within limits"

    def clamp_to_limits(self) -> tuple[str, ...]:
        """Trim title and description to platform limits on word boundaries.

        Returns the names of fields that were trimmed (empty tuple when
        nothing changed). Updates ``character_counts`` so downstream
        consumers see the new lengths.
        """
        trimmed: list[str] = []
        limits = PLATFORM_LIMITS.get(self.platform, {})

        title_lim = limits.get("title")
        if isinstance(title_lim, int) and self.title and len(self.title) > title_lim:
            self.title = _trim_on_word_boundary(self.title, title_lim)
            trimmed.append("title")

        desc_lim = limits.get("description")
        if isinstance(desc_lim, int) and len(self.description) > desc_lim:
            self.description = _trim_on_word_boundary(self.description, desc_lim)
            trimmed.append("description")

        if trimmed:
            self.character_counts["description"] = len(self.description)
            if self.title:
                self.character_counts["title"] = len(self.title)

        return tuple(trimmed)

    def format_content(self) -> str:
        """Format content for posting: disclosure, description, hashtags, product_id.

        Disclosure leads the caption on a line of its own so it sits above the
        '...more' fold on Instagram and TikTok and satisfies the FTC requirement
        that the disclosure appears before any other text or hashtags.

        Returns
        -------
            Formatted content string ready for publishing

        """
        parts = []

        # Disclosure leads the caption (FTC: clear and conspicuous, top of caption).
        # Gated on the recorded decision as well as the string: a caller that
        # sets `carries_affiliate_content=False` without also blanking the
        # field would otherwise lead with a disclosure while the body it just
        # had stripped says there is nothing to disclose.
        if self.disclosure and self.carries_affiliate_content:
            parts.append(self.disclosure)

        # Affiliate program literal phrase (Amazon Associates requirement).
        # Sits between the #ad disclosure and the description so both are
        # visible above the "more" fold on Instagram/TikTok.
        if self.affiliate_disclosure:
            parts.append(self.affiliate_disclosure)

        # Description only - title is handled separately by platform APIs
        parts.append(self.description)

        # Collect all hashtags including product_id
        all_hashtags = list(self.hashtags) if self.hashtags else []
        if self.product_id and self.product_id not in all_hashtags:
            all_hashtags.append(self.product_id)

        if all_hashtags:
            hashtag_str = " ".join(f"#{tag}" for tag in all_hashtags)
            parts.append(hashtag_str)

        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
            Dictionary representation with serializable values

        """
        return {
            "platform": self.platform.value,
            "title": self.title,
            "description": self.description,
            "hashtags": self.hashtags,
            "keywords": self.keywords,
            "character_counts": self.character_counts,
            "product_id": self.product_id,
            "disclosure": self.disclosure,
            "affiliate_disclosure": self.affiliate_disclosure,
        }


@dataclass
class AccountConfig:
    """Configuration for a single Late.dev account.

    Stores credentials and metadata for a named account, enabling
    multi-account support where products can be routed to different
    Late.dev accounts.

    Attributes
    ----------
        name: Unique account identifier (e.g., "main", "overflow")
        api_key: API key for this account
        vercel_token: Vercel token for large file uploads (optional)
        description: Human-readable description of the account
        default_platforms: Platform-specific defaults for this account

    """

    name: str
    api_key: str
    vercel_token: str | None = None
    description: str = ""
    default_platforms: list[Platform] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization validation."""
        # Validate name
        if not self.name or not self.name.strip():
            raise ValueError("Account name cannot be empty")

        # Validate API key
        if not self.api_key or not self.api_key.strip():
            raise ValueError(f"api_key cannot be empty for account '{self.name}'")

        # Validate API key format (basic check)
        if len(self.api_key) < LATE_API_KEY_MIN_LENGTH:
            raise ValueError(
                f"Invalid API key format for account '{self.name}': "
                f"must be at least {LATE_API_KEY_MIN_LENGTH} characters"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
            Dictionary representation (excludes sensitive credentials)

        """
        return {
            "name": self.name,
            "api_key": f"{self.api_key[:4]}..." if self.api_key else None,
            "vercel_token": (
                f"{self.vercel_token[:4]}..." if self.vercel_token else None
            ),
            "description": self.description,
            "default_platforms": [p.value for p in self.default_platforms],
        }


def _as_platform(value: "Platform | str") -> "Platform":
    """Return `value` as a Platform, naming the field when it is not one.

    An unknown name is refused rather than dropped. The environment-variable
    path warns and discards, which means publishing to fewer platforms than
    the operator configured without saying so.
    """
    if isinstance(value, Platform):
        return value
    try:
        return Platform(str(value).lower())
    except ValueError as exc:
        known = ", ".join(p.value for p in Platform)
        raise ValueError(
            f"unknown platform {value!r}; expected one of: {known}"
        ) from exc


@dataclass
class PublisherConfig:
    """Configuration for publisher behavior and credentials.

    This configuration is loaded using three-tier precedence:
    CLI arguments > environment variables > YAML config file

    Supports both single-account (legacy) and multi-account modes:
    - Single-account: api_key at root level (backward compatible)
    - Multi-account: accounts dict with named accounts

    Attributes
    ----------
        provider: Publishing service to use (late, buffer, etc.)
        api_key: API key for the publishing service (active account)
        vercel_token: Vercel token for large file uploads (active account)
        accounts: Named accounts for multi-account support
        active_account: Name of the currently active account
        default_platforms: Default platforms to publish to if not specified
        immediate_publish: Publish immediately vs scheduled
        privacy_settings: Platform-specific privacy levels
        max_retries: Maximum retry attempts for failed operations
        timeout: Request timeout in seconds
        stagger_delay_min: Minimum delay between batch posts (seconds)
        stagger_delay_max: Maximum delay between batch posts (seconds)
        schedule_config: Configuration for scheduling and validation
        cleanup_config: Configuration for post-publication cleanup

    """

    provider: str
    api_key: str
    vercel_token: str | None = None
    accounts: dict[str, AccountConfig] = field(default_factory=dict)
    active_account: str | None = None
    default_platforms: list[Platform] = field(default_factory=list)
    immediate_publish: bool = False
    schedule_time: str | None = None
    privacy_settings: dict[Platform, str] = field(default_factory=dict)
    max_retries: int = 3
    timeout: float = 120.0
    stagger_delay_min: int = 30
    stagger_delay_max: int = 60
    schedule_config: "ScheduleConfig" = field(default_factory=lambda: ScheduleConfig())
    cleanup_config: "CleanupConfig" = field(default_factory=lambda: CleanupConfig())
    link_in_bio_config: "LinkInBioConfig" = field(
        default_factory=lambda: LinkInBioConfig()
    )
    tiktok_settings: "TikTokContentSettings" = field(
        default_factory=lambda: TikTokContentSettings()
    )
    first_comment_config: "FirstCommentConfig" = field(
        default_factory=lambda: FirstCommentConfig()
    )
    blob_retention_config: "BlobRetentionConfig" = field(
        default_factory=lambda: BlobRetentionConfig()
    )
    delivery_sweep_config: "DeliverySweepConfig" = field(
        default_factory=lambda: DeliverySweepConfig()
    )
    affiliate_disclosure_config: "AffiliateDisclosureConfig" = field(
        default_factory=lambda: AffiliateDisclosureConfig()
    )
    analytics_config: "AnalyticsConfig" = field(
        default_factory=lambda: AnalyticsConfig()
    )
    # YouTube's altered-or-synthetic-content disclosure. Off by default
    # because the policy targets realistic material that could mislead about
    # real people or events, and explicitly excludes AI narration, AI scripts
    # and stock footage -- which is what this pipeline renders. Turn it on for
    # output that does meet the bar: AI-generated music, or AI-generated
    # footage of a real place. Both are properties of what a provider returns,
    # so this can become true without any change here.
    synthetic_media_disclosure: bool = False
    use_platform_specific_content: bool = False
    # Per-platform video profile routing (Phase 1.3).
    # Maps platform name (lowercased: "tiktok"/"youtube"/"instagram") to a
    # video profile name from config/video_production.yaml::video_profiles.
    # When set, the publisher prefers `video_<asin>_<profile>.mp4` for that
    # platform; when unset, falls back to the first `video_<asin>_*.mp4` in
    # the product directory (legacy behaviour).
    profiles: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation."""
        # Validate provider
        if not self.provider or not self.provider.strip():
            raise ValueError("provider cannot be empty")

        # Validate API key
        if not self.api_key or not self.api_key.strip():
            raise ValueError("api_key cannot be empty")

        # Validate numeric constraints
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.stagger_delay_min < 0:
            raise ValueError("stagger_delay_min must be non-negative")
        if self.stagger_delay_max < self.stagger_delay_min:
            raise ValueError("stagger_delay_max must be >= stagger_delay_min")

        # Set default platforms if empty
        if not self.default_platforms:
            self.default_platforms = list(DEFAULT_PLATFORMS)

        # The loader hands YAML through to the dataclass unconverted, so these
        # two arrive as plain strings while the annotations say Platform. That
        # is not cosmetic: `to_dict` and every caller reading `.value` off a
        # platform raised AttributeError on any config loaded from the shipped
        # file, and mypy could not see it because it believes the annotation.
        # Coerce here rather than in the loader, so a config built directly in
        # Python is the same shape as one read from disk.
        self.default_platforms = [_as_platform(p) for p in self.default_platforms]
        self.privacy_settings = {
            _as_platform(p): v for p, v in self.privacy_settings.items()
        }

    def get_account(self, name: str | None = None) -> AccountConfig | None:
        """Get account configuration by name.

        Args:
        ----
            name: Account name to retrieve (uses active_account if None)

        Returns:
        -------
            AccountConfig if found, None otherwise

        """
        account_name = name or self.active_account
        if not account_name:
            return None
        return self.accounts.get(account_name)

    def list_accounts(self) -> list[str]:
        """List all configured account names.

        Returns
        -------
            List of account names

        """
        return list(self.accounts.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
            Dictionary representation (excludes sensitive credentials)

        """
        return {
            "provider": self.provider,
            "api_key": f"{self.api_key[:4]}..." if self.api_key else None,
            "vercel_token": (
                f"{self.vercel_token[:4]}..." if self.vercel_token else None
            ),
            "accounts": {name: acc.to_dict() for name, acc in self.accounts.items()},
            "active_account": self.active_account,
            "default_platforms": [p.value for p in self.default_platforms],
            "immediate_publish": self.immediate_publish,
            "privacy_settings": {p.value: v for p, v in self.privacy_settings.items()},
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "stagger_delay_min": self.stagger_delay_min,
            "stagger_delay_max": self.stagger_delay_max,
            "schedule_config": self.schedule_config.to_dict(),
            "cleanup_config": self.cleanup_config.to_dict(),
            "affiliate_disclosure_config": self.affiliate_disclosure_config.to_dict(),
            "analytics_config": self.analytics_config.to_dict(),
        }


@dataclass
class BatchPublishSummary:
    """Summary report of batch publishing operation.

    Provides comprehensive overview of batch publishing results including
    success/failure counts per platform and detailed error information.

    Attributes
    ----------
        total_videos: Total number of videos processed
        successful: Number of successfully published/scheduled videos
        failed: Number of failed publishes
        skipped: Number of skipped videos (e.g., missing metadata)
        platform_results: Per-platform success/fail counts
        errors: List of error messages with video identifiers
        duration_seconds: Total batch processing time

    """

    total_videos: int
    successful: int
    failed: int
    skipped: int
    platform_results: dict[Platform, dict[str, int]] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def __post_init__(self):
        """Post-initialization validation."""
        # Validate counts are non-negative
        if self.total_videos < 0:
            raise ValueError("total_videos must be non-negative")
        if self.successful < 0:
            raise ValueError("successful must be non-negative")
        if self.failed < 0:
            raise ValueError("failed must be non-negative")
        if self.skipped < 0:
            raise ValueError("skipped must be non-negative")

        # Validate total matches sum of outcomes
        if self.total_videos != (self.successful + self.failed + self.skipped):
            raise ValueError(
                "total_videos must equal sum of successful + failed + skipped"
            )

    def add_platform_result(self, platform: Platform, success: bool):
        """Add a result for a specific platform.

        Args:
        ----
            platform: Platform that was published to
            success: Whether the publish succeeded

        """
        if platform not in self.platform_results:
            self.platform_results[platform] = {"successful": 0, "failed": 0}

        if success:
            self.platform_results[platform]["successful"] += 1
        else:
            self.platform_results[platform]["failed"] += 1

    def add_error(self, video_id: str, error_message: str):
        """Add an error to the summary.

        Args:
        ----
            video_id: Identifier of the video that failed
            error_message: Description of the error

        """
        self.errors.append({"video_id": video_id, "error": error_message})

    def get_success_rate(self) -> float:
        """Calculate overall success rate.

        Returns
        -------
            Success rate as percentage (0.0 to 100.0)

        """
        if self.total_videos == 0:
            return 0.0
        return (self.successful / self.total_videos) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
            Dictionary representation with all summary data

        """
        return {
            "total_videos": self.total_videos,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": f"{self.get_success_rate():.1f}%",
            "platform_results": {
                p.value: counts for p, counts in self.platform_results.items()
            },
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class RecurringSlot:
    """Recurring time slot for automated scheduling.

    Defines a repeating weekly time slot when videos should be published.
    Used by the auto-scheduling system to assign videos to predefined slots.

    Attributes
    ----------
        day_of_week: Day name (monday, tuesday, etc.) - lowercase
        time: Time in HH:MM:SS format (24-hour)
        timezone: IANA timezone name (e.g., "UTC", "America/New_York")

    """

    day_of_week: str
    time: str
    timezone: str

    def __post_init__(self):
        """Post-initialization validation."""
        # Validate day_of_week
        valid_days = {
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        }
        if self.day_of_week.lower() not in valid_days:
            raise ValueError(
                f"day_of_week must be one of {valid_days}, got '{self.day_of_week}'"
            )

        # Validate time format (HH:MM:SS)
        if not self.time or len(self.time.split(":")) != 3:
            raise ValueError(f"time must be in HH:MM:SS format, got '{self.time}'")

        try:
            hours, minutes, seconds = map(int, self.time.split(":"))
            if not (0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59):
                raise ValueError
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"time must have valid HH:MM:SS values, got '{self.time}'"
            ) from err

        # Validate timezone is not empty
        if not self.timezone or not self.timezone.strip():
            raise ValueError("timezone cannot be empty")

    def next_occurrence(self, after: datetime) -> datetime:
        """Calculate next occurrence of this slot after given datetime.

        Args:
        ----
            after: Reference datetime to calculate from (timezone-aware)

        Returns:
        -------
            Next occurrence of this slot as timezone-aware datetime

        Raises:
        ------
            ValueError: If after is timezone-naive or timezone is invalid

        Example:
        -------
            >>> slot = RecurringSlot("monday", "10:00:00", "UTC")
            >>> from datetime import datetime, UTC
            >>> after = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)  # Wednesday
            >>> next_time = slot.next_occurrence(after)
            >>> # Returns next Monday at 10:00 UTC

        """
        # Validate after has timezone
        if after.tzinfo is None:
            raise ValueError("after datetime must be timezone-aware")

        # Parse slot time
        hour, minute, second = map(int, self.time.split(":"))

        # Get timezone
        try:
            tz = ZoneInfo(self.timezone)
        except (KeyError, ImportError) as e:
            raise ValueError(f"Invalid timezone '{self.timezone}': {e}") from e

        # Convert after to slot timezone
        after_local = after.astimezone(tz)

        # Map day names to weekday numbers (0=Monday, 6=Sunday)
        day_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        target_weekday = day_map[self.day_of_week.lower()]

        # Calculate days until target weekday
        current_weekday = after_local.weekday()
        days_ahead = target_weekday - current_weekday

        # If target day is today, check if time has passed
        if days_ahead == 0:
            slot_time_today = after_local.replace(
                hour=hour, minute=minute, second=second, microsecond=0
            )
            if after_local >= slot_time_today:
                # Time has passed today, go to next week
                days_ahead = 7
        elif days_ahead < 0:
            # Target day is earlier in week, go to next week
            days_ahead += 7

        # Calculate next occurrence
        next_date = after_local.date() + timedelta(days=days_ahead)
        next_datetime = datetime(
            next_date.year,
            next_date.month,
            next_date.day,
            hour,
            minute,
            second,
            tzinfo=tz,
        )

        return next_datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
            Dictionary representation with all slot data

        """
        return {
            "day_of_week": self.day_of_week,
            "time": self.time,
            "timezone": self.timezone,
        }


@dataclass
class ScheduleEntry:
    """Scheduled post entry in the publishing calendar.

    Represents a single video scheduled for publication to one or more
    platforms at a specific time. Tracks status throughout the lifecycle.

    Attributes
    ----------
        product_id: Unique identifier of the product/video
        scheduled_time: When the post should be published (timezone-aware)
        platforms: List of platforms to publish to
        post_id: Provider-assigned post ID (None until scheduled)
        status: Current status (pending, scheduled, published, failed)
        created_at: When this schedule entry was created (timezone-aware)
        slot_index: Index of recurring slot used (None if manual schedule)

    """

    product_id: str
    scheduled_time: datetime
    platforms: list[Platform]
    post_id: str | None = None
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    slot_index: int | None = None

    def __post_init__(self):
        """Post-initialization validation."""
        # Validate product_id
        if not self.product_id or not self.product_id.strip():
            raise ValueError("product_id cannot be empty")

        # Validate scheduled_time has timezone
        if self.scheduled_time.tzinfo is None:
            raise ValueError("scheduled_time must include timezone information")

        # Validate platforms list
        if not self.platforms:
            raise ValueError("platforms cannot be empty")

        # Validate status
        valid_statuses = {"pending", "scheduled", "published", "failed", "partial"}
        if self.status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got '{self.status}'"
            )

        # Validate slot_index if provided
        if self.slot_index is not None and self.slot_index < 0:
            raise ValueError("slot_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
            Dictionary representation with all entry data

        """
        return {
            "product_id": self.product_id,
            "scheduled_time": self.scheduled_time.isoformat(),
            "platforms": [p.value for p in self.platforms],
            "post_id": self.post_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "slot_index": self.slot_index,
        }


@dataclass
class ConflictResolution:
    """Result of conflict resolution when scheduling fails.

    Contains the original conflict details and suggested alternative slots
    sorted by time proximity to the user's preferred time.

    Attributes
    ----------
        original_time: The originally requested schedule time
        conflict_reason: Description of why the original time failed
        alternatives: List of alternative slots sorted by proximity
        auto_resolved: Whether conflict was auto-resolved (first alternative used)
        resolved_time: The time that was actually used (if auto-resolved)

    """

    original_time: datetime
    conflict_reason: str
    alternatives: list[datetime] = field(default_factory=list)
    auto_resolved: bool = False
    resolved_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_time": self.original_time.isoformat(),
            "conflict_reason": self.conflict_reason,
            "alternatives": [t.isoformat() for t in self.alternatives],
            "auto_resolved": self.auto_resolved,
            "resolved_time": self.resolved_time.isoformat()
            if self.resolved_time
            else None,
        }


@dataclass
class ScheduleConfig:
    """Configuration for schedule validation and behavior.

    Controls scheduling rules, conflict prevention, and constraints
    for the auto-scheduling system.

    Attributes
    ----------
        enabled: Whether recurring schedule is enabled
        slots: List of recurring time slots for auto-scheduling
        min_post_spacing_hours: Minimum hours between posts on same platform
        prevent_duplicates: Reject duplicate schedules (same product+platform+time)
        allow_past_schedules: Allow scheduling posts in the past
        max_posts_per_day: Maximum posts allowed per day (0 = unlimited)
        timezone: Default timezone for schedule operations
        use_platform_specific_content: Create separate posts per platform
            with optimized metadata
        conflict_alternatives_count: Number of alternatives to suggest on conflict

    """

    enabled: bool = False
    slots: list[RecurringSlot] = field(default_factory=list)
    min_post_spacing_hours: int = 2
    prevent_duplicates: bool = True
    allow_past_schedules: bool = False
    max_posts_per_day: int = 10
    timezone: str = "UTC"
    use_platform_specific_content: bool = False
    conflict_alternatives_count: int = 5

    def __post_init__(self):
        """Post-initialization validation."""
        # Validate min_post_spacing_hours
        if self.min_post_spacing_hours < 0:
            raise ValueError("min_post_spacing_hours must be non-negative")

        # Validate max_posts_per_day
        if self.max_posts_per_day < 0:
            raise ValueError("max_posts_per_day must be non-negative")

        # Validate timezone
        if not self.timezone or not self.timezone.strip():
            raise ValueError("timezone cannot be empty")

        # Validate conflict_alternatives_count
        if self.conflict_alternatives_count < 1:
            raise ValueError("conflict_alternatives_count must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
            Dictionary representation with all config data

        """
        return {
            "enabled": self.enabled,
            "slots": [
                {
                    "day_of_week": slot.day_of_week,
                    "time": slot.time,
                    "timezone": slot.timezone,
                }
                for slot in self.slots
            ],
            "min_post_spacing_hours": self.min_post_spacing_hours,
            "prevent_duplicates": self.prevent_duplicates,
            "allow_past_schedules": self.allow_past_schedules,
            "max_posts_per_day": self.max_posts_per_day,
            "timezone": self.timezone,
            "conflict_alternatives_count": self.conflict_alternatives_count,
        }


@dataclass
class CleanupConfig:
    """Configuration for post-publication cleanup behavior.

    Controls when and how product directories are removed after successful
    publication, with safety features to prevent accidental data loss.

    Attributes
    ----------
        enabled: Enable automatic cleanup after successful publish
        verify_before_delete: Query API to confirm publication before cleanup
        require_all_platforms: Only cleanup if published to ALL platforms
        archive_before_delete: Create ZIP archive before deletion
        archive_dir: Directory to store archives
        keep_published_days: Days to wait before cleanup (0 = immediate)
        preserve_metadata: Keep metadata JSON files when cleaning
        preserve_logs: Keep log files when cleaning
        settle_timeout_sec: Seconds to wait for a platform still publishing to
            reach a final status. 0 or less checks once and gives up, which is
            what an immediate publish always used to do.
        settle_initial_delay_sec: Delay before the second status check. Each
            later delay doubles, and the last one is trimmed so the delays sum
            to settle_timeout_sec. 0 or less disables waiting, like a zero
            timeout.

    """

    enabled: bool = True
    verify_before_delete: bool = True
    require_all_platforms: bool = True
    archive_before_delete: bool = False
    archive_dir: Path = field(default_factory=lambda: Path("outputs/archive"))
    keep_published_days: int = 0
    preserve_metadata: bool = False
    preserve_logs: bool = True
    settle_timeout_sec: float = 300.0
    settle_initial_delay_sec: float = 30.0

    def __post_init__(self):
        """Post-initialization validation."""
        # Validate keep_published_days
        if self.keep_published_days < 0:
            raise ValueError("keep_published_days must be non-negative")

        # Neither settle field raises. The cleanup section is parsed inside a
        # `except (ValueError, TypeError)` that falls back to a whole default
        # `CleanupConfig`, so one rejected key discards the operator's
        # `enabled: false`, `archive_before_delete: true` and
        # `keep_published_days` along with it -- turning a typo in a wait into
        # immediate unarchived deletion on an install that had cleanup off.
        # A non-positive value in either field means "do not wait", which is
        # the reading `settle_timeout_sec: 0` already has, and
        # `_settle_delays` enforces it.

        # Validate archive_dir if archiving enabled
        if self.archive_before_delete and not self.archive_dir:
            raise ValueError("archive_dir required when archive_before_delete is True")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
            Dictionary representation with all config data

        """
        return {
            "enabled": self.enabled,
            "verify_before_delete": self.verify_before_delete,
            "require_all_platforms": self.require_all_platforms,
            "archive_before_delete": self.archive_before_delete,
            "archive_dir": str(self.archive_dir),
            "keep_published_days": self.keep_published_days,
            "preserve_metadata": self.preserve_metadata,
            "preserve_logs": self.preserve_logs,
            "settle_timeout_sec": self.settle_timeout_sec,
            "settle_initial_delay_sec": self.settle_initial_delay_sec,
        }


@dataclass
class LinkInBioConfig:
    """Configuration for link-in-bio integration.

    Attributes
    ----------
        enabled: Enable link-in-bio updates after publish
        provider: Provider name (lnkbio, linktree, etc.)
        max_links: Maximum links on bio page (oldest rotated out)
        max_title_length: Maximum characters for link title

    """

    enabled: bool = True
    provider: str = "lnkbio"
    max_links: int = 0
    max_title_length: int = 80

    def __post_init__(self):
        if self.max_links < 0:
            raise ValueError("max_links must be non-negative")
        if self.max_title_length < 10:
            raise ValueError("max_title_length must be at least 10")


@dataclass
class TikTokContentSettings:
    """TikTok API content disclosure and interaction settings.

    Attributes
    ----------
        privacy_level: TikTok privacy setting
        allow_comment: Allow comments on posts
        allow_duet: Allow duets
        allow_stitch: Allow stitches
        commercial_content_type: Content disclosure type
        is_brand_organic_post: Whether post is brand organic
        content_preview_confirmed: User confirmed content preview
        express_consent_given: User gave express consent
        video_made_with_ai: Declare TikTok's AI-generated-content label

    """

    privacy_level: str = "PUBLIC_TO_EVERYONE"
    allow_comment: bool = True
    allow_duet: bool = False
    allow_stitch: bool = False
    commercial_content_type: str = "brand_organic"
    is_brand_organic_post: bool = True
    content_preview_confirmed: bool = True
    express_consent_given: bool = True
    # TikTok's AI-generated-content label. On by default, unlike YouTube's
    # synthetic-media disclosure, because the two platforms draw the line in
    # different places and this pipeline lands on opposite sides of it.
    # TikTok requires the label for AI-generated speech and says so
    # explicitly, extending it to AI voiceover even when the footage is real;
    # every render here carries an AI TTS voiceover. YouTube lists cloning
    # one's own voice for voiceover as *not* requiring disclosure, which is
    # why `synthetic_media_disclosure` defaults off.
    #
    # Disclosing is also the cheaper error. TikTok reads C2PA credentials and
    # auto-labels undisclosed AI content, and an auto-flag suppresses
    # distribution; self-disclosure keeps reach. Enforcement escalates from a
    # warning to a posting restriction to a ban.
    video_made_with_ai: bool = True

    def for_render(self, carries_affiliate_content: bool) -> "TikTokContentSettings":
        """Return the settings this particular render should declare.

        The configured values describe a promotional post. A render with no
        material connection -- a topic video with no affiliate link -- is not
        commercial content, and TikTok has a value for that. Declaring it as
        brand-organic instead tells viewers the creator is promoting their own
        business, which is simply untrue.

        `none` is used rather than omitting the settings: absence is
        indistinguishable from a payload that forgot them.

        `content_preview_confirmed` and `express_consent_given` are
        deliberately left alone. They look like part of the commercial-content
        flow, and the worry was that sending them alongside "not commercial"
        would be rejected the way a missing disclosure option is. It is not:
        TikTok's Content Sharing Guidelines make both unconditional
        requirements of the Direct Post API -- "API Clients must only start
        sending content materials to TikTok after the user has expressly
        consent to the upload", and "API Clients should display a preview of
        the to-be-posted content" -- for every post, commercial or not. The
        rejection they were confused with fires only when the disclosure
        toggle is ON and neither option is chosen, which `none` plus
        `is_brand_organic_post=False` is the opposite of.
        """
        if carries_affiliate_content:
            return self
        return replace(
            self,
            commercial_content_type="none",
            is_brand_organic_post=False,
        )

    def to_sdk_dict(self) -> dict[str, object]:
        """Convert to dict format expected by Late SDK."""
        return {
            "privacy_level": self.privacy_level,
            "allow_comment": self.allow_comment,
            "allow_duet": self.allow_duet,
            "allow_stitch": self.allow_stitch,
            "commercial_content_type": self.commercial_content_type,
            "is_brand_organic_post": self.is_brand_organic_post,
            "content_preview_confirmed": self.content_preview_confirmed,
            "express_consent_given": self.express_consent_given,
        }

    def to_platform_data(self) -> dict[str, object]:
        """Fields that belong beside `tiktokSettings`, not inside it.

        The SDK types `platformSpecificData` as a flat `TikTokPlatformData`
        and models no `tiktokSettings` key at all; the nested block this
        project sends is a legacy shape the API still accepts. So a field the
        SDK does model is sent where the SDK models it, rather than guessed
        into the nested block where nothing says it would be read.
        """
        return {"videoMadeWithAi": self.video_made_with_ai}

    def to_top_level_dict(self) -> dict[str, str]:
        """Convert to top-level tiktok_settings format."""
        return {
            "privacyLevel": self.privacy_level,
            "mediaType": "video",
            "commercialContentType": self.commercial_content_type,
        }


@dataclass
class FirstCommentConfig:
    """Configuration for first-comment publishing on supported platforms.

    Moves affiliate links out of post captions into the first comment,
    avoiding algorithm penalties for outbound links in descriptions.

    Attributes
    ----------
        enabled: Enable first-comment publishing
        platforms: Map of platform name to comment template string.
            Placeholders: {affiliate_link}, {hashtags}, {product_title},
            {closing_line}. Only the placeholders a template uses are
            required, so a script-derived template still renders for a
            product with no affiliate link.
        move_hashtags_to_comment: Move hashtags from caption to comment

    """

    enabled: bool = False
    platforms: dict[str, str] = field(default_factory=dict)
    move_hashtags_to_comment: bool = False


@dataclass
class BlobRetentionConfig:
    """Retention policy for the Vercel Blob upload store.

    Large video uploads are staged in the user's Blob store and fetched by
    the scheduling service at publish time; without retention the store
    grows until the free tier pauses access and uploads start failing.
    Blobs referenced by not-yet-published posts are always kept, regardless
    of this policy.

    Attributes
    ----------
        enabled: Apply retention after publish runs
        max_age_days: Delete blobs older than this many days
        max_total_mb: After the age sweep, trim oldest-first to this total

    """

    enabled: bool = False
    max_age_days: int = 30
    max_total_mb: int = 500


@dataclass
class DeliverySweepConfig:
    """Sweep recent posts for silently-failed platform legs after a publish.

    The scheduler reports a post accepted, not delivered. A leg that fails at
    publish time leaves the post ``partial`` with no alert, and the recovery
    (``posts.retry``) expires once the CDN copy is gone. Sweeping a trailing
    window after every publish run catches the previous runs' posts, which
    have since fired; the post just created is still pending and cannot be
    judged yet.

    Attributes
    ----------
        enabled: Run the sweep after publish runs
        limit: How many recent posts to inspect

    """

    enabled: bool = True
    limit: int = 25
