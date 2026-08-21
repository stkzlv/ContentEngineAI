# src/video/producer/utils.py
"""Producer utility functions for logging and validation."""

import difflib
import logging
import random
from pathlib import Path

from src.utils import ensure_dirs_exist
from src.utils.logging_setup import setup_debug_logging
from src.video.config import VideoConfig

logger = logging.getLogger(__name__)

# Profiles excluded from random selection. `base` is the inheritance template
# other profiles extend, not a render target, so random batches shouldn't pick
# it. `slideshow_stock` renders entirely from stock media, which is right for a
# topic and wrong for a scraped product, whose own imagery it would ignore.
# Both stay usable via an explicit --profile / --batch-profile.
EXCLUDED_RANDOM_PROFILES = frozenset({"base", "slideshow_stock"})


def setup_logging(config: VideoConfig, debug_mode: bool = False) -> Path:
    """Set up logging to both console and file.

    Args:
    ----
        config: Video configuration containing log directory path
        debug_mode: Whether to enable debug logging

    Returns:
    -------
        Path to the log file

    """
    # Create log directory
    log_dir = config.general_video_producer_log_dir_path
    ensure_dirs_exist(log_dir)

    # Use fixed log filename that gets overwritten on each run
    log_file = log_dir / "producer.log"

    # Use centralized logging setup
    setup_debug_logging(
        log_file=log_file,
        debug_mode=debug_mode,
        verbose=True,  # Producer uses verbose format by default
        component_name="VideoProducer",
    )

    log_level_name = logging.getLevelName(logging.DEBUG if debug_mode else logging.INFO)
    logger.info(f"Logging configured - Level: {log_level_name}, File: {log_file}")
    return log_file


def validate_media_requirements(
    scraped_images: list,
    scraped_videos: list,
    stock_media: list,
    profile,
    config: VideoConfig,
) -> tuple[bool, str]:
    """Validate if gathered media meets minimum requirements for video creation.

    Implements Requirement 8: Media Validation and Error Handling
    - Enables video-first profiles when ≥1 video available
    - Falls back to image-only processing when no videos
    - Logs clear messages about mode selection

    Args:
    ----
        scraped_images: List of scraped image paths
        scraped_videos: List of scraped video paths
        stock_media: List of stock media items
        profile: Video profile configuration object
        config: Video configuration containing media requirements

    Returns:
    -------
        Tuple of (is_valid: bool, reason: str)

    """
    scraped_image_count = len(scraped_images)
    scraped_video_count = len(scraped_videos)

    # Count stock media by type
    stock_image_count = sum(
        1 for item in stock_media if getattr(item, "type", "image") == "image"
    )
    stock_video_count = sum(
        1 for item in stock_media if getattr(item, "type", "image") == "video"
    )

    # Total counts including stock media
    total_image_count = scraped_image_count + stock_image_count
    total_video_count = scraped_video_count + stock_video_count
    total_media = total_image_count + total_video_count

    uses_scraped_videos = getattr(profile, "use_scraped_videos", True)
    video_assembly_mode = (
        getattr(profile, "video_assembly_mode", None)
        or config.video_settings.video_assembly_mode
    )

    # Get media requirements from config (must match scraper validation)
    MIN_TOTAL_MEDIA = config.video_settings.min_total_media
    MIN_IMAGES_IF_NO_VIDEO = config.video_settings.min_images_if_no_video
    MIN_IMAGES_WITH_VIDEO = config.video_settings.min_images_with_video

    # Check basic minimum
    if total_media < MIN_TOTAL_MEDIA:
        msg = (
            f"Insufficient total media: {total_media} items "
            f"(minimum {MIN_TOTAL_MEDIA})"
        )
        return (False, msg)

    # Determine mode selection based on video availability (Requirement 8.1, 8.2)
    is_video_first_profile = uses_scraped_videos and video_assembly_mode is not None
    has_videos = total_video_count > 0

    # If profile doesn't use videos, need more images for slideshow
    if (
        not uses_scraped_videos or total_video_count == 0
    ) and total_image_count < MIN_IMAGES_IF_NO_VIDEO:
        if not uses_scraped_videos:
            msg = (
                f"Profile excludes videos, need {MIN_IMAGES_IF_NO_VIDEO} images "
                f"but only have {total_image_count}"
            )
        else:
            # No videos available - graceful fallback (Requirement 8.2)
            if is_video_first_profile:
                logger.info(
                    "Video-first profile selected but no videos available - "
                    "falling back to image-only processing"
                )
            msg = (
                f"No videos found, need at least {MIN_IMAGES_IF_NO_VIDEO} images "
                f"but only have {total_image_count}"
            )
        return (False, msg)

    # If we have videos, we can work with fewer images
    if total_video_count > 0 and total_image_count < MIN_IMAGES_WITH_VIDEO:
        msg = (
            f"Have {total_video_count} video(s) but only {total_image_count} "
            f"image(s), need at least {MIN_IMAGES_WITH_VIDEO}"
        )
        return (False, msg)

    # Warn for borderline cases but allow processing
    if total_media == MIN_TOTAL_MEDIA:
        logger.warning(
            f"Minimal media count ({total_media}) - video quality may be limited"
        )

    # Log mode selection (Requirement 8.5 - clear logging)
    if is_video_first_profile and has_videos:
        logger.info(
            f"Video-first mode enabled: {total_video_count} video(s) available "
            f"for assembly mode '{video_assembly_mode}'"
        )
    elif has_videos and not uses_scraped_videos:
        logger.info(
            f"Profile configured to exclude videos - using image-only processing "
            f"({total_video_count} video(s) available but ignored)"
        )
    elif has_videos:
        logger.info(
            f"Video processing enabled: {total_video_count} video(s) available "
            f"(legacy mode - no assembly mode configured)"
        )
    else:
        logger.info(
            f"Image-only processing: {total_image_count} image(s) available "
            f"(no videos found)"
        )

    msg = (
        f"Media validation passed: {total_image_count} images, "
        f"{total_video_count} videos, "
        f"{len(stock_media)} stock items"
    )
    return (True, msg)


# Profile Selection Utilities (Requirement 2.3, 3.3)


class ProfileUsageTracker:
    """Tracks profile usage statistics across batch processing.

    Maintains counts of how many times each profile was selected
    and provides formatted summary output.
    """

    def __init__(self) -> None:
        """Initialize empty usage tracker."""
        self._usage_counts: dict[str, int] = {}

    def record_usage(self, profile_name: str) -> None:
        """Record that a profile was used.

        Args:
        ----
            profile_name: Name of the profile that was selected

        """
        self._usage_counts[profile_name] = self._usage_counts.get(profile_name, 0) + 1

    def get_counts(self) -> dict[str, int]:
        """Get usage counts for all profiles.

        Returns
        -------
            Dictionary mapping profile names to usage counts

        """
        return self._usage_counts.copy()

    def format_summary(self) -> str:
        """Format usage statistics as a human-readable summary.

        Returns
        -------
            Formatted string showing profile distribution

        """
        if not self._usage_counts:
            return "No profile usage recorded"

        total = sum(self._usage_counts.values())
        lines = ["Profile Distribution:"]

        for profile_name, count in sorted(
            self._usage_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / total) * 100
            lines.append(f"  - {profile_name}: {count} ({percentage:.1f}%)")

        return "\n".join(lines)


def validate_profiles(profile_names: list[str], config: VideoConfig) -> None:
    """Validate that all provided profiles exist in the configuration.

    Args:
    ----
        profile_names: List of profile names to validate
        config: VideoConfig instance

    Raises:
    ------
        ValueError: If any profile does not exist, includes suggestions.

    """
    available = list(config.video_profiles.keys())
    invalid_profiles = [p for p in profile_names if p not in config.video_profiles]

    if invalid_profiles:
        error_parts = []
        for p in invalid_profiles:
            suggestions = difflib.get_close_matches(p, available, n=3, cutoff=0.6)
            part = f"'{p}'"
            if suggestions:
                part += f" (did you mean: {', '.join(suggestions)}?)"
            error_parts.append(part)

        msg = f"Invalid profile(s): {', '.join(error_parts)}."
        msg += f"\nAvailable profiles: {', '.join(available)}"
        raise ValueError(msg)


def select_profile_for_product(
    product_id: str,
    profile_pool: list[str],
    config: VideoConfig,
) -> str:
    """Select a profile for a product using deterministic random selection.

    Uses the product ID as a seed to ensure the same product always
    gets the same profile (deterministic behavior for reproducibility).

    Args:
    ----
        product_id: Unique product identifier (ASIN or similar)
        profile_pool: List of available profile names to choose from
        config: VideoConfig instance for profile validation

    Returns:
    -------
        Selected profile name

    Raises:
    ------
        ValueError: If profile_pool is empty or contains invalid profiles

    """
    if not profile_pool:
        raise ValueError("Profile pool cannot be empty")

    # Validate all profiles exist in config
    validate_profiles(profile_pool, config)

    # Use hash of product ID for deterministic seeding
    seed = hash(product_id)
    rng = random.Random(seed)  # noqa: S311

    # Select profile deterministically
    selected = rng.choice(profile_pool)

    return selected


def load_profile_pool(
    cli_pool: list[str] | None,
    yaml_pool: list[str] | None,
    config: VideoConfig,
) -> list[str]:
    """Load profile pool with CLI > YAML > all profiles precedence.

    Implements 3-tier configuration precedence:
    1. CLI --profile-pool argument (highest priority)
    2. YAML profile_pool configuration
    3. All available profiles from config (lowest priority)

    Args:
    ----
        cli_pool: Profile pool from CLI --profile-pool argument
        yaml_pool: Profile pool from YAML configuration
        config: VideoConfig instance containing available profiles

    Returns:
    -------
        List of profile names to use for random selection

    Raises:
    ------
        ValueError: If any profiles in pool are invalid

    """
    # Apply precedence: CLI > YAML > all profiles
    if cli_pool is not None:
        pool = cli_pool
    elif yaml_pool is not None and len(yaml_pool) > 0:
        pool = yaml_pool
    else:
        # Default to all available profiles, minus non-render templates.
        pool = [p for p in config.video_profiles if p not in EXCLUDED_RANDOM_PROFILES]

    # Validate all profiles exist
    validate_profiles(pool, config)

    return pool
