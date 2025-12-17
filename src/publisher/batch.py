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
from src.publisher.metadata import load_platform_metadata
from src.publisher.models import BatchPublishSummary, Platform, PublishStatus

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
        outputs_dir: Path | str = Path("outputs"),
        platforms: list[Platform] | None = None,
        stagger_delay_min: int = 30,
        stagger_delay_max: int = 60,
        fail_fast: bool = False,
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

        """
        self.publisher = publisher
        self.outputs_dir = (
            Path(outputs_dir) if isinstance(outputs_dir, str) else outputs_dir
        )
        self.platforms = platforms or [
            Platform.YOUTUBE,
            Platform.TIKTOK,
            Platform.INSTAGRAM,
        ]
        self.stagger_delay_min = stagger_delay_min
        self.stagger_delay_max = stagger_delay_max
        self.fail_fast = fail_fast

        logger.info(
            f"Initialized BatchPublisher: platforms={[p.value for p in self.platforms]}, "
            f"stagger_delay={stagger_delay_min}-{stagger_delay_max}s, "
            f"fail_fast={fail_fast}"
        )

    async def publish_batch(self) -> BatchPublishSummary:
        """Execute batch publishing for all discovered videos.

        Workflow:
        1. Discover videos in outputs directory
        2. For each video:
           a. Load platform-specific metadata
           b. Upload video
           c. Create posts for target platforms
           d. Apply staggered delay
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
        logger.info("BATCH PUBLISHING STARTED")
        logger.info("=" * 80)

        # Discover videos
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

        logger.info(f"Found {len(videos)} video(s) to publish")
        logger.info(f"Target platforms: {[p.value for p in self.platforms]}")

        # Track statistics
        successful = 0
        failed = 0
        skipped = 0
        summary = BatchPublishSummary(
            total_videos=len(videos),
            successful=0,
            failed=0,
            skipped=0,
        )

        # Process each video
        for idx, video_info in enumerate(videos, 1):
            video_path = video_info["path"]
            product_id = video_info["product_id"]

            logger.info("-" * 80)
            logger.info(f"[{idx}/{len(videos)}] Processing: {video_path.name}")
            logger.info(f"Product ID: {product_id}")

            try:
                # Publish video to all target platforms
                publish_result = await self._publish_single_video(
                    video_path, product_id, idx, len(videos)
                )

                if publish_result["status"] == "success":
                    successful += 1
                    summary.successful += 1
                    # Track platform-specific results
                    for platform in self.platforms:
                        summary.add_platform_result(platform, success=True)
                elif publish_result["status"] == "skipped":
                    skipped += 1
                    summary.skipped += 1
                    summary.add_error(
                        product_id, publish_result.get("error", "Unknown skip reason")
                    )
                else:
                    failed += 1
                    summary.failed += 1
                    summary.add_error(
                        product_id, publish_result.get("error", "Unknown error")
                    )
                    # Track platform-specific failures
                    for platform in self.platforms:
                        summary.add_platform_result(platform, success=False)

                    if self.fail_fast:
                        logger.error("Fail-fast enabled, stopping batch processing")
                        break

            except Exception as e:
                failed += 1
                summary.failed += 1
                error_msg = f"Unexpected error: {e}"
                logger.error(f"[{idx}/{len(videos)}] {error_msg}")
                summary.add_error(product_id, error_msg)

                if self.fail_fast:
                    logger.error("Fail-fast enabled, stopping batch processing")
                    break

            # Apply staggered delay (except for last video)
            if idx < len(videos):
                await self._apply_staggered_delay(idx, len(videos))

        # Finalize summary
        batch_duration = time.time() - batch_start
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
        logger.info(f"Scanning for videos in: {self.outputs_dir}")

        if not self.outputs_dir.exists():
            logger.warning(f"Outputs directory not found: {self.outputs_dir}")
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

        logger.info(f"Discovered {len(videos)} video(s)")
        return videos

    async def _publish_single_video(
        self,
        video_path: Path,
        product_id: str,
        current_idx: int,
        total_count: int,
    ) -> dict:
        """Publish a single video to target platforms.

        Args:
        ----
            video_path: Path to video file
            product_id: Product identifier
            current_idx: Current video index (1-based)
            total_count: Total number of videos

        Returns:
        -------
            Result dict: {"status": "success"|"failed"|"skipped", "error": str}

        """
        try:
            # Upload video once (reuse media_id for all platforms)
            logger.info(f"[{current_idx}/{total_count}] Uploading video...")
            media_id = await self.publisher.upload_media(video_path)
            logger.info(f"[{current_idx}/{total_count}] Upload complete: {media_id}")

            # Publish to each platform
            for platform in self.platforms:
                logger.info(
                    f"[{current_idx}/{total_count}] Publishing to {platform.value}..."
                )

                # Load platform-specific metadata
                metadata = load_platform_metadata(
                    product_id, platform, self.outputs_dir
                )

                if not metadata:
                    logger.warning(
                        f"[{current_idx}/{total_count}] Skipping {platform.value}: "
                        f"metadata not found"
                    )
                    return {
                        "status": "skipped",
                        "error": f"Missing metadata for {platform.value}",
                    }

                # Get account ID for this platform (from connected accounts)
                accounts = await self.publisher.get_accounts()
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
                        f"[{current_idx}/{total_count}] Skipping {platform.value}: "
                        f"no connected account"
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

                    post_id = result["post_id"]
                    post_status = result["status"]

                    logger.info(
                        f"[{current_idx}/{total_count}] Published to {platform.value}: "
                        f"post_id={post_id}, status={post_status}"
                    )

                    # Log published URLs if available
                    if result.get("published_urls"):
                        logger.info(
                            f"[{current_idx}/{total_count}] Published URLs for {platform.value}:"
                        )
                        for url in result["published_urls"]:
                            logger.info(f"[{current_idx}/{total_count}]   - {url}")

                    # Fetch and log post status after creation (non-blocking)
                    try:
                        status_info = await self.publisher.get_status(post_id)
                        if status_info["status"] != "unknown":
                            logger.debug(
                                f"[{current_idx}/{total_count}] Status verification for {platform.value}: "
                                f"{status_info['status']}"
                            )
                            # If status check found additional URLs not in publish response
                            if status_info["published_urls"] and not result.get(
                                "published_urls"
                            ):
                                logger.info(
                                    f"[{current_idx}/{total_count}] Additional URLs from status check:"
                                )
                                for url in status_info["published_urls"]:
                                    logger.info(
                                        f"[{current_idx}/{total_count}]   - {url}"
                                    )
                    except Exception as status_err:
                        # Status check failure is non-critical, just log and continue
                        logger.debug(
                            f"[{current_idx}/{total_count}] Status check failed for {platform.value}: "
                            f"{status_err}"
                        )

                except PublishError as e:
                    # Check for rate limit (429)
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        logger.warning(
                            f"[{current_idx}/{total_count}] Rate limit hit for "
                            f"{platform.value}, waiting before retry..."
                        )
                        # Wait for retry-after period (default: 60s)
                        await asyncio.sleep(60)
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
                            f"[{current_idx}/{total_count}] Retry successful for "
                            f"{platform.value}"
                        )
                    else:
                        raise

            return {"status": "success"}

        except Exception as e:
            error_msg = f"Publishing failed: {e}"
            logger.error(f"[{current_idx}/{total_count}] {error_msg}")
            return {"status": "failed", "error": error_msg}

    async def _apply_staggered_delay(self, current_idx: int, total_count: int):
        """Apply random staggered delay between posts.

        Args:
        ----
            current_idx: Current video index (1-based)
            total_count: Total number of videos

        """
        delay = random.randint(self.stagger_delay_min, self.stagger_delay_max)
        logger.info(
            f"[{current_idx}/{total_count}] Waiting {delay}s before next video..."
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
        logger.info(f"Total videos attempted: {summary.total_videos}")
        logger.info(f"Successful: {summary.successful}")
        logger.info(f"Failed: {summary.failed}")
        logger.info(f"Skipped: {summary.skipped}")
        logger.info(f"Success rate: {summary.get_success_rate():.1f}%")

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

        logger.info(f"Total duration: {duration_str}")

        # Average time per video
        if summary.successful > 0:
            avg_time = summary.duration_seconds / summary.successful
            logger.info(f"Average time per successful video: {avg_time:.1f}s")

        # Platform-specific results
        if summary.platform_results:
            logger.info("-" * 80)
            logger.info("Per-Platform Results:")
            logger.info(
                f"{'Platform':<15} {'Successful':<12} {'Failed':<10} {'Total':<10} {'Rate':<10}"
            )
            logger.info("-" * 80)

            for platform, counts in summary.platform_results.items():
                total_attempts = counts["successful"] + counts["failed"]
                success_rate = (
                    (counts["successful"] / total_attempts * 100)
                    if total_attempts > 0
                    else 0.0
                )
                logger.info(
                    f"{platform.value:<15} {counts['successful']:<12} "
                    f"{counts['failed']:<10} {total_attempts:<10} "
                    f"{success_rate:>6.1f}%"
                )

        # Errors (show first 10, summarize rest)
        if summary.errors:
            logger.info("-" * 80)
            logger.info(f"Errors ({len(summary.errors)} total):")

            # Group errors by type for better readability
            error_types = {}
            for error in summary.errors:
                error_msg = error["error"]
                # Extract error type (first sentence or up to 50 chars)
                error_type = error_msg.split(".")[0][:50]
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error["video_id"])

            # Display grouped errors
            for error_type, video_ids in list(error_types.items())[:5]:
                logger.info(f"  {error_type}:")
                for video_id in video_ids[:3]:
                    logger.info(f"    - {video_id}")
                if len(video_ids) > 3:
                    logger.info(f"    ... and {len(video_ids) - 3} more videos")

            if len(error_types) > 5:
                logger.info(f"  ... and {len(error_types) - 5} more error types")

        # Success summary
        logger.info("=" * 80)
        if summary.failed == 0 and summary.skipped == 0:
            logger.info("✓ All videos published successfully!")
        elif summary.successful > 0:
            logger.info(
                f"✓ Completed with {summary.successful} successful, "
                f"{summary.failed} failed, {summary.skipped} skipped"
            )
        else:
            logger.info("✗ No videos were successfully published")
        logger.info("=" * 80)
