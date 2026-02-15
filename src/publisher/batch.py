"""Batch publishing orchestrator for multiple videos.

This module provides batch publishing capabilities, scanning the outputs directory
for completed videos, loading their metadata, and publishing them to social media
platforms with progress tracking and error handling.
"""

import asyncio
import logging
import random
import time
from pathlib import Path

from src.publisher.base import BasePublisher, PublishError
from src.publisher.constants import DEFAULT_OUTPUTS_DIR
from src.publisher.metadata import load_platform_metadata
from src.publisher.models import (
    DEFAULT_PLATFORMS,
    BatchPublishSummary,
    Platform,
    PublishStatus,
)
from src.publisher.tracking import (
    add_to_retry_queue,
    get_retry_queue,
    remove_from_retry_queue,
)
from src.video.config.constants import LATE_DEFAULT_RETRY_AFTER_SEC

logger = logging.getLogger(__name__)


class BatchPublisher:
    """Orchestrates batch publishing of multiple videos.

    Scans outputs directory for completed videos, loads platform-specific metadata,
    and publishes to social media platforms with staggered delays, progress tracking,
    and comprehensive error handling.

    Attributes
    ----------
        publisher: BasePublisher implementation (e.g., LatePublisher)
        outputs_dir: Directory containing product outputs
        platforms: List of target platforms
        stagger_delay_min: Minimum delay between posts (seconds)
        stagger_delay_max: Maximum delay between posts (seconds)
        fail_fast: Stop on first failure if True

    """

    def __init__(
        self,
        publisher: BasePublisher,
        outputs_dir: Path | str = DEFAULT_OUTPUTS_DIR,
        platforms: list[Platform] | None = None,
        stagger_delay_min: int = 30,
        stagger_delay_max: int = 60,
        fail_fast: bool = False,
        retry_failed: bool = False,
    ):
        """Initialize batch publisher.

        Args:
        ----
            publisher: BasePublisher instance (already initialized)
            outputs_dir: Directory containing product outputs (default: "outputs")
            platforms: Target platforms (default: [YouTube, TikTok, Instagram])
            stagger_delay_min: Minimum delay between posts in seconds (default: 30)
            stagger_delay_max: Maximum delay between posts in seconds (default: 60)
            fail_fast: Stop processing on first failure (default: False)
            retry_failed: Only process items from retry queue (default: False)

        Example:
        -------
            >>> from src.publisher import create_publisher, PublisherProvider
            >>> from src.publisher.batch import BatchPublisher
            >>> publisher = create_publisher(
            ...     provider=PublisherProvider.LATE,
            ...     api_key="sk_live_...",
            ...     vercel_token="vercel_..."
            ... )
            >>> batch = BatchPublisher(
            ...     publisher=publisher,
            ...     platforms=[Platform.YOUTUBE, Platform.TIKTOK]
            ... )
            >>> summary = await batch.publish_batch()

        Example (retry failed items):
        -------
            >>> batch = BatchPublisher(publisher=publisher, retry_failed=True)
            >>> summary = await batch.publish_batch()  # Only processes failed items

        """
        self.publisher = publisher
        self.outputs_dir = (
            Path(outputs_dir) if isinstance(outputs_dir, str) else outputs_dir
        )
        self.platforms = platforms or list(DEFAULT_PLATFORMS)
        self.stagger_delay_min = stagger_delay_min
        self.stagger_delay_max = stagger_delay_max
        self.fail_fast = fail_fast
        self.retry_failed = retry_failed

        platforms_str = [p.value for p in self.platforms]
        mode = "RETRY MODE" if retry_failed else "normal"
        logger.info(
            "Initialized BatchPublisher: platforms=%s, stagger=%d-%ds, "
            "fail_fast=%s, mode=%s",
            platforms_str,
            stagger_delay_min,
            stagger_delay_max,
            fail_fast,
            mode,
        )

    async def publish_batch(self) -> BatchPublishSummary:
        """Execute batch publishing for all discovered videos.

        Workflow:
        1. Discover videos in outputs directory (or get from retry queue)
        2. For each video:
           a. Load platform-specific metadata
           b. Upload video
           c. Create posts for target platforms
           d. Apply staggered delay
           e. On failure, add to retry queue
           f. On success, remove from retry queue (if was retry)
        3. Generate summary report

        Returns:
        -------
            BatchPublishSummary with success/failure counts and detailed errors

        Example:
        -------
            >>> summary = await batch.publish_batch()
            >>> print(f"Published: {summary.successful}/{summary.total_videos}")
            >>> print(f"Failed: {summary.failed}")
            >>> for error in summary.errors:
            ...     print(f"  {error['video_id']}: {error['error']}")

        """
        batch_start = time.time()

        logger.info("=" * 80)
        if self.retry_failed:
            logger.info("BATCH PUBLISHING STARTED (RETRY MODE)")
        else:
            logger.info("BATCH PUBLISHING STARTED")
        logger.info("=" * 80)

        # Get videos to process
        if self.retry_failed:
            videos = self._get_retry_queue_videos()
            if not videos:
                logger.info("Retry queue is empty - no failed items to retry")
                return BatchPublishSummary(
                    total_videos=0,
                    successful=0,
                    failed=0,
                    skipped=0,
                    duration_seconds=time.time() - batch_start,
                )
            logger.info("Found %d failed video(s) to retry", len(videos))
        else:
            videos = self._discover_videos()
            if not videos:
                logger.warning("No videos found for publishing")
                return BatchPublishSummary(
                    total_videos=0,
                    successful=0,
                    failed=0,
                    skipped=0,
                    duration_seconds=time.time() - batch_start,
                )
            logger.info("Found %d video(s) to publish", len(videos))

        logger.info("Target platforms: %s", [p.value for p in self.platforms])

        # Track statistics - initialize with zeros, set total_videos at end
        successful = 0
        failed = 0
        skipped = 0
        total_video_count = len(videos)
        summary = BatchPublishSummary(
            total_videos=0,
            successful=0,
            failed=0,
            skipped=0,
        )

        # Fetch connected accounts once (reused for all videos/platforms)
        accounts = await self.publisher.get_accounts()

        # Process each video
        for idx, video_info in enumerate(videos, 1):
            video_path = video_info["path"]
            product_id = video_info["product_id"]
            # Get scheduled_time from retry queue item (if any) to preserve scheduling
            scheduled_time = video_info.get("scheduled_time")

            logger.info("-" * 80)
            logger.info("[%d/%d] Processing: %s", idx, len(videos), video_path.name)
            logger.info("Product ID: %s", product_id)
            if self.retry_failed:
                retry_count = video_info.get("retry_count", 1)
                logger.info("Retry attempt: %d", retry_count)

            try:
                # Publish video to all target platforms
                publish_result = await self._publish_single_video(
                    video_path, product_id, idx, len(videos), accounts
                )

                if publish_result["status"] == "success":
                    successful += 1
                    summary.successful += 1
                    # Track platform-specific results
                    for platform in self.platforms:
                        summary.add_platform_result(platform, success=True)
                    # Remove from retry queue on success (idempotent)
                    remove_from_retry_queue(product_id, self.outputs_dir)
                    # Add to published products registry
                    try:
                        from src.publisher.product_registry import add_to_registry

                        add_to_registry(product_id, self.outputs_dir)
                    except Exception as exc:
                        logger.warning("Failed to update product registry: %s", exc)

                elif publish_result["status"] == "skipped":
                    skipped += 1
                    summary.skipped += 1
                    error_msg = publish_result.get("error", "Unknown skip reason")
                    summary.add_error(product_id, error_msg)
                    # Add to retry queue for skipped items (missing metadata, etc.)
                    self._add_failed_to_retry_queue(
                        product_id, error_msg, scheduled_time
                    )

                else:
                    failed += 1
                    summary.failed += 1
                    error_msg = publish_result.get("error", "Unknown error")
                    summary.add_error(product_id, error_msg)
                    # Track platform-specific failures
                    for platform in self.platforms:
                        summary.add_platform_result(platform, success=False)
                    # Add to retry queue for later retry
                    self._add_failed_to_retry_queue(
                        product_id, error_msg, scheduled_time
                    )

                    if self.fail_fast:
                        logger.error("Fail-fast enabled, stopping batch processing")
                        break

            except Exception as e:
                failed += 1
                summary.failed += 1
                error_msg = f"Unexpected error: {e}"
                logger.error("[%d/%d] %s", idx, len(videos), error_msg)
                summary.add_error(product_id, error_msg)
                # Add to retry queue for later retry
                self._add_failed_to_retry_queue(product_id, error_msg, scheduled_time)

                if self.fail_fast:
                    logger.error("Fail-fast enabled, stopping batch processing")
                    break

            # Apply staggered delay (except for last video)
            if idx < len(videos):
                await self._apply_staggered_delay(idx, len(videos))

        # Finalize summary
        batch_duration = time.time() - batch_start
        summary.total_videos = total_video_count
        summary.duration_seconds = batch_duration

        # Log final summary
        self._log_summary(summary)

        return summary

    def _discover_videos(self) -> list[dict]:
        """Discover completed videos in outputs directory.

        Scans outputs/{product_id}/video_*.mp4 files and extracts product IDs.

        Returns
        -------
            List of video info dicts: [{"path": Path, "product_id": str}, ...]

        """
        logger.info("Scanning for videos in: %s", self.outputs_dir)

        if not self.outputs_dir.exists():
            logger.warning("Outputs directory not found: %s", self.outputs_dir)
            return []

        videos = []

        # Scan each product directory
        for product_dir in self.outputs_dir.iterdir():
            if not product_dir.is_dir():
                continue

            product_id = product_dir.name

            # Find video files (video_*.mp4)
            video_files = list(product_dir.glob("video_*.mp4"))

            for video_file in video_files:
                videos.append({"path": video_file, "product_id": product_id})

        logger.info("Discovered %d video(s)", len(videos))
        return videos

    def _get_retry_queue_videos(self) -> list[dict]:
        """Get videos from the retry queue.

        Retrieves failed items from the retry queue and resolves their video paths.
        Only includes items where the video file still exists.

        Returns
        -------
            List of video info dicts with path, product_id, scheduled_time, retry_count

        """
        logger.info("Checking retry queue in: %s", self.outputs_dir)

        retry_items = get_retry_queue(self.outputs_dir)
        if not retry_items:
            return []

        videos = []
        for item in retry_items:
            product_id = item["product_id"]
            product_dir = self.outputs_dir / product_id

            if not product_dir.exists():
                logger.warning(
                    "Product directory not found for retry item: %s", product_id
                )
                continue

            # Find video file
            video_files = list(product_dir.glob("video_*.mp4"))
            if not video_files:
                logger.warning("No video file found for retry item: %s", product_id)
                continue

            # Use first video file found
            video_path = video_files[0]
            videos.append(
                {
                    "path": video_path,
                    "product_id": product_id,
                    "scheduled_time": item.get("scheduled_time"),
                    "retry_count": item.get("retry_count", 1),
                    "original_error": item.get("error"),
                }
            )

        logger.info("Found %d video(s) in retry queue", len(videos))
        return videos

    def _add_failed_to_retry_queue(
        self,
        product_id: str,
        error: str,
        scheduled_time: str | None = None,
    ) -> None:
        """Add a failed product to the retry queue.

        Args:
        ----
            product_id: Product identifier
            error: Error message
            scheduled_time: Original scheduled time to preserve

        """
        platforms = [p.value for p in self.platforms]
        add_to_retry_queue(
            product_id=product_id,
            platforms=platforms,
            error=error,
            scheduled_time=scheduled_time,
            outputs_dir=self.outputs_dir,
        )

    async def _publish_single_video(
        self,
        video_path: Path,
        product_id: str,
        current_idx: int,
        total_count: int,
        accounts: list[dict],
    ) -> dict:
        """Publish a single video to target platforms.

        Args:
        ----
            video_path: Path to video file
            product_id: Product identifier
            current_idx: Current video index (1-based)
            total_count: Total number of videos
            accounts: Pre-fetched list of connected platform accounts

        Returns:
        -------
            Result dict: {"status": "success"|"failed"|"skipped", "error": str}

        """
        try:
            # Upload video once (reuse media_id for all platforms)
            logger.info("[%d/%d] Uploading video...", current_idx, total_count)
            media_id = await self.publisher.upload_media(video_path)
            logger.info(
                "[%d/%d] Upload complete: %s", current_idx, total_count, media_id
            )

            # Publish to each platform
            for platform in self.platforms:
                logger.info(
                    "[%d/%d] Publishing to %s...",
                    current_idx, total_count, platform.value,
                )

                # Load platform-specific metadata
                metadata = load_platform_metadata(
                    product_id, platform, self.outputs_dir
                )

                if not metadata:
                    logger.warning(
                        "[%d/%d] Skipping %s: metadata not found",
                        current_idx,
                        total_count,
                        platform.value,
                    )
                    return {
                        "status": "skipped",
                        "error": f"Missing metadata for {platform.value}",
                    }

                # Get account ID for this platform (from pre-fetched accounts)
                platform_account = next(
                    (
                        acc
                        for acc in accounts
                        if acc["platform"].lower() == platform.value
                    ),
                    None,
                )

                if not platform_account:
                    logger.warning(
                        "[%d/%d] Skipping %s: no connected account",
                        current_idx,
                        total_count,
                        platform.value,
                    )
                    continue

                # Format content
                content = metadata.format_content()

                # Create post
                try:
                    result = await self.publisher.publish(
                        media_id=media_id,
                        platforms=[
                            {
                                "platform": platform.value,
                                "account_id": platform_account["account_id"],
                            }
                        ],
                        content=content,
                        scheduled_time=None,  # Immediate publish
                    )

                    post_id = str(result["post_id"])
                    post_status = result["status"]

                    logger.info(
                        "[%d/%d] Published to %s: post_id=%s, status=%s",
                        current_idx,
                        total_count,
                        platform.value,
                        post_id,
                        post_status,
                    )

                    # Log published URLs if available
                    published_urls = result.get("published_urls")
                    if published_urls and isinstance(published_urls, list):
                        logger.info(
                            "[%d/%d] Published URLs for %s:",
                            current_idx,
                            total_count,
                            platform.value,
                        )
                        for url in published_urls:
                            logger.info("[%d/%d]   - %s", current_idx, total_count, url)

                    # Fetch and log post status after creation (non-blocking)
                    try:
                        status_info = await self.publisher.get_status(post_id)
                        if status_info["status"] != "unknown":
                            logger.debug(
                                "[%d/%d] Status for %s: %s",
                                current_idx,
                                total_count,
                                platform.value,
                                status_info["status"],
                            )
                            # If status check found additional URLs
                            status_urls = status_info.get("published_urls")
                            if (
                                status_urls
                                and isinstance(status_urls, list)
                                and not published_urls
                            ):
                                logger.info(
                                    "[%d/%d] URLs from status check:",
                                    current_idx,
                                    total_count,
                                )
                                for url in status_urls:
                                    logger.info(
                                        "[%d/%d]   - %s", current_idx, total_count, url
                                    )
                    except Exception as status_err:
                        # Status check failure is non-critical
                        logger.debug(
                            "[%d/%d] Status check failed: %s",
                            current_idx,
                            total_count,
                            status_err,
                        )

                except PublishError as e:
                    # Check for rate limit (429)
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        wait_time = LATE_DEFAULT_RETRY_AFTER_SEC
                        logger.warning(
                            "[%d/%d] Rate limit hit for %s, "
                            "waiting %ds before retry...",
                            current_idx,
                            total_count,
                            platform.value,
                            wait_time,
                        )
                        # Wait for retry-after period
                        await asyncio.sleep(LATE_DEFAULT_RETRY_AFTER_SEC)
                        # Retry once
                        result = await self.publisher.publish(
                            media_id=media_id,
                            platforms=[
                                {
                                    "platform": platform.value,
                                    "account_id": platform_account["account_id"],
                                }
                            ],
                            content=content,
                            scheduled_time=None,
                        )
                        logger.info(
                            "[%d/%d] Retry successful for %s",
                            current_idx,
                            total_count,
                            platform.value,
                        )
                    else:
                        raise

            return {"status": "success"}

        except Exception as e:
            error_msg = f"Publishing failed: {e}"
            logger.error("[%d/%d] %s", current_idx, total_count, error_msg)
            return {"status": "failed", "error": error_msg}

    async def _apply_staggered_delay(self, current_idx: int, total_count: int):
        """Apply random staggered delay between posts.

        Args:
        ----
            current_idx: Current video index (1-based)
            total_count: Total number of videos

        """
        delay = random.randint(self.stagger_delay_min, self.stagger_delay_max)  # noqa: S311
        logger.info(
            "[%d/%d] Waiting %ds before next video...", current_idx, total_count, delay
        )
        await asyncio.sleep(delay)

    def _log_summary(self, summary: BatchPublishSummary):
        """Log comprehensive batch publishing summary.

        Args:
        ----
            summary: BatchPublishSummary with detailed statistics

        """
        logger.info("=" * 80)
        logger.info("BATCH PUBLISHING COMPLETE")
        logger.info("=" * 80)

        # Overall statistics
        logger.info("Total videos attempted: %d", summary.total_videos)
        logger.info("Successful: %d", summary.successful)
        logger.info("Failed: %d", summary.failed)
        logger.info("Skipped: %d", summary.skipped)
        logger.info("Success rate: %.1f%%", summary.get_success_rate())

        # Duration formatting
        if summary.duration_seconds < 60:
            duration_str = f"{summary.duration_seconds:.1f}s"
        elif summary.duration_seconds < 3600:
            minutes = int(summary.duration_seconds // 60)
            seconds = int(summary.duration_seconds % 60)
            duration_str = f"{minutes}m {seconds}s"
        else:
            hours = int(summary.duration_seconds // 3600)
            minutes = int((summary.duration_seconds % 3600) // 60)
            duration_str = f"{hours}h {minutes}m"

        logger.info("Total duration: %s", duration_str)

        # Average time per video
        if summary.successful > 0:
            avg_time = summary.duration_seconds / summary.successful
            logger.info("Average time per successful video: %.1fs", avg_time)

        # Platform-specific results
        if summary.platform_results:
            logger.info("-" * 80)
            logger.info("Per-Platform Results:")
            header = f"{'Platform':<15} {'Successful':<12} {'Failed':<10} "
            header += f"{'Total':<10} {'Rate':<10}"
            logger.info(header)
            logger.info("-" * 80)

            for platform, counts in summary.platform_results.items():
                total_attempts = counts["successful"] + counts["failed"]
                success_rate = (
                    (counts["successful"] / total_attempts * 100)
                    if total_attempts > 0
                    else 0.0
                )
                logger.info(
                    "%s %d %d %d %6.1f%%",
                    f"{platform.value:<15}",
                    counts["successful"],
                    counts["failed"],
                    total_attempts,
                    success_rate,
                )

        # Errors (show first 10, summarize rest)
        if summary.errors:
            logger.info("-" * 80)
            logger.info("Errors (%d total):", len(summary.errors))

            # Group errors by type for better readability
            error_types: dict[str, list[str]] = {}
            for error in summary.errors:
                error_msg = error["error"]
                # Extract error type (first sentence or up to 50 chars)
                error_type = error_msg.split(".")[0][:50]
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error["video_id"])

            # Display grouped errors
            for error_type, video_ids in list(error_types.items())[:5]:
                logger.info("  %s:", error_type)
                for video_id in video_ids[:3]:
                    logger.info("    - %s", video_id)
                if len(video_ids) > 3:
                    logger.info("    ... and %d more videos", len(video_ids) - 3)

            if len(error_types) > 5:
                logger.info("  ... and %d more error types", len(error_types) - 5)

        # Success summary
        logger.info("=" * 80)
        if summary.failed == 0 and summary.skipped == 0:
            logger.info("All videos published successfully!")
        elif summary.successful > 0:
            logger.info(
                "Completed with %d successful, %d failed, %d skipped",
                summary.successful,
                summary.failed,
                summary.skipped,
            )
        else:
            logger.info("No videos were successfully published")
        logger.info("=" * 80)
