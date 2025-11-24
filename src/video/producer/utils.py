# src/video/producer/utils.py
"""Producer utility functions for logging and validation."""
import logging
from pathlib import Path

from src.utils import ensure_dirs_exist
from src.utils.logging_setup import setup_debug_logging
from src.video.video_config import VideoConfig

logger = logging.getLogger(__name__)


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
