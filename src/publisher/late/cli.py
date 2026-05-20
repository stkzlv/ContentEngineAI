r"""CLI interface for Late.dev publisher.

Provides command-line access to publish videos to social media platforms via Late.dev.

Usage:
    # Publish single product (auto-schedules to next slot)
    python -m src.publisher.late single B0ABC123 --debug

    # Schedule all videos to calendar slots
    python -m src.publisher.late schedule --debug

    # Publish all videos immediately
    python -m src.publisher.late schedule --immediate --debug
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from src.publisher import PublisherProvider, create_publisher
from src.publisher.batch import BatchPublisher
from src.publisher.cleanup import CleanupManager
from src.publisher.config import load_publisher_config
from src.publisher.models import DEFAULT_PLATFORMS, Platform
from src.publisher.product_registry import add_to_registry, rebuild_registry
from src.publisher.schedule import ScheduleManager
from src.publisher.tracking import is_already_published, record_publish
from src.utils.logging_setup import setup_debug_logging

logger = logging.getLogger(__name__)


def _create_publisher_from_config(config, session: aiohttp.ClientSession):
    """Create a publisher instance from loaded config.

    Args:
    ----
        config: PublisherConfig instance
        session: aiohttp ClientSession

    Returns:
    -------
        Configured publisher instance

    """
    return create_publisher(
        provider=PublisherProvider(config.provider),
        api_key=config.api_key,
        session=session,
        vercel_token=config.vercel_token,
        timeout=config.timeout,
        max_retries=config.max_retries,
        tiktok_settings=config.tiktok_settings,
        first_comment_config=config.first_comment_config,
    )


def parse_datetime(datetime_str: str) -> datetime:
    """Parse datetime string in ISO format or common formats.

    Args:
    ----
        datetime_str: Datetime string (e.g., "2025-01-20 14:00:00"
            or "2025-01-20T14:00:00")

    Returns:
    -------
        datetime object

    Raises:
    ------
        ValueError: If datetime string is invalid

    """
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(datetime_str, fmt)
            # Assume UTC if no timezone provided
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue

    raise ValueError(
        f"Invalid datetime format: {datetime_str}. "
        f"Use format like '2025-01-20 14:00:00' or '2025-01-20T14:00:00'"
    )


async def cmd_list_accounts(
    args: argparse.Namespace, config, session: aiohttp.ClientSession
):
    """Execute list-accounts command.

    Args:
    ----
        args: Parsed command-line arguments
        config: PublisherConfig instance
        session: aiohttp ClientSession

    """
    logger.info("Listing connected social media accounts...")

    publisher = _create_publisher_from_config(config, session)

    try:
        # Authenticate first
        is_authenticated = await publisher.authenticate()
        if not is_authenticated:
            logger.error("Authentication failed - check your API key")
            sys.exit(1)

        logger.info("Authentication successful")

        # Get accounts
        accounts = await publisher.get_accounts()

        if not accounts:
            logger.warning("No connected accounts found")
            return

        logger.info("Found %d connected account(s):", len(accounts))
        logger.info("-" * 80)

        for account in accounts:
            logger.info("Platform: %s", account["platform"])
            logger.info("Account ID: %s", account["account_id"])
            logger.info("Username: %s", account.get("username", "N/A"))
            logger.info("-" * 80)

    except Exception as e:
        logger.error("Failed to list accounts: %s", e, exc_info=args.debug)
        sys.exit(1)


async def cmd_single(args: argparse.Namespace, config, session: aiohttp.ClientSession):
    """Execute single video publish command.

    Args:
    ----
        args: Parsed command-line arguments
        config: PublisherConfig instance
        session: aiohttp ClientSession

    """
    product_id = args.product_id
    outputs_dir = Path("outputs")
    product_dir = outputs_dir / product_id

    if not product_dir.exists():
        logger.error("Product directory not found: %s", product_dir)
        sys.exit(1)

    # Default to all 3 platforms if none specified
    if not args.platforms:
        args.platforms = list(DEFAULT_PLATFORMS)
        logger.info("Using default platforms: youtube, tiktok, instagram")

    # Auto-discover video file. If profiles are configured, prefer the
    # render for the first platform in the list. The unified upload path
    # uses one file for all platforms; full per-platform uploads are a
    # follow-up after multi-profile renders ship.
    from src.publisher.video_selector import select_video_for_platform

    first_platform = (
        args.platforms[0].value
        if hasattr(args.platforms[0], "value")
        else str(args.platforms[0])
    )
    video_path = select_video_for_platform(
        product_dir, product_id, first_platform, getattr(config, "profiles", None)
    )
    if video_path is None:
        logger.error("No video files found in %s", product_dir)
        sys.exit(1)
    logger.info("Auto-discovered video: %s", video_path.name)

    logger.info("Publishing single video: %s", video_path.name)
    logger.info("Target platforms: %s", [p.value for p in args.platforms])

    publisher = _create_publisher_from_config(config, session)

    try:
        # Authenticate
        is_authenticated = await publisher.authenticate()
        if not is_authenticated:
            logger.error("Authentication failed - check your API key")
            sys.exit(1)

        # Get accounts for mapping
        accounts = await publisher.get_accounts()
        if not accounts:
            logger.error("No connected accounts found")
            sys.exit(1)

        # Auto-discover next slot if --schedule not provided (and not --immediate)
        schedule_time = args.schedule
        if not schedule_time and not args.immediate:
            logger.info("Auto-discovering next available schedule slot...")
            schedule_mgr = ScheduleManager(config=config.schedule_config)
            slots = config.schedule_config.slots
            if not slots:
                logger.error("No recurring slots configured for auto-discovery")
                sys.exit(1)

            # Build set of occupied slot times from ALL posts (scheduled + published)
            occupied_slot_times: set[datetime] = set()
            try:
                logger.debug("Fetching all posts from Late.dev...")
                api_posts = await publisher.list_posts()
                logger.debug("Found %d posts on Late.dev", len(api_posts))

                for api_post in api_posts:
                    scheduled_for = api_post.get("scheduledFor")
                    if not scheduled_for:
                        continue
                    # Parse datetime
                    if isinstance(scheduled_for, str):
                        scheduled_dt = datetime.fromisoformat(
                            scheduled_for.replace("+00:00", "+00:00")
                        )
                    else:
                        scheduled_dt = scheduled_for
                    # Ensure timezone-aware
                    if scheduled_dt.tzinfo is None:
                        scheduled_dt = scheduled_dt.replace(tzinfo=UTC)
                    # Normalize to minute precision for comparison
                    normalized = scheduled_dt.replace(second=0, microsecond=0)
                    occupied_slot_times.add(normalized)
                logger.debug("Occupied slots: %d times", len(occupied_slot_times))
            except Exception as e:
                logger.warning("Failed to fetch existing posts: %s", e)

            # Find first available slot (gap detection)
            search_time = datetime.now(UTC)
            max_attempts = 365  # Search up to a year ahead
            attempts = 0
            current_slot = 0

            while attempts < max_attempts:
                next_time, current_slot = schedule_mgr.get_next_slot(
                    slots, search_time, slot_index=current_slot
                )
                normalized = next_time.replace(second=0, microsecond=0)

                if normalized not in occupied_slot_times:
                    schedule_time = next_time
                    logger.info("Found available slot: %s", schedule_time.isoformat())
                    break

                # Slot is occupied, try next
                search_time = next_time
                attempts += 1
                logger.debug("Slot %s occupied, trying next...", normalized)

            if not schedule_time:
                logger.error("Could not find available slot within search range")
                sys.exit(1)

        if schedule_time:
            logger.info("Scheduled time: %s", schedule_time)
        else:
            logger.info("Publishing immediately")

        # Upload video
        logger.info("Uploading video...")
        media_url = await publisher.upload_media(video_path)
        logger.info("Upload complete: %s", media_url)

        # Build platforms list (filter duplicates and validate accounts)
        platforms_to_publish = []
        for platform in args.platforms:
            # Check for duplicates (unless --force)
            if not args.force and is_already_published(
                product_id, platform.value, outputs_dir
            ):
                logger.warning(
                    "Product %s already published to %s. Use --force to republish.",
                    product_id,
                    platform.value,
                )
                continue

            # Find account for this platform
            platform_account = next(
                (acc for acc in accounts if acc["platform"].lower() == platform.value),
                None,
            )

            if not platform_account:
                logger.warning("No connected account for %s, skipping", platform.value)
                continue

            platforms_to_publish.append(
                {
                    "platform": platform.value,
                    "account_id": platform_account["account_id"],
                }
            )

        if not platforms_to_publish:
            logger.warning("No platforms to publish to after filtering")
            return

        # Publish (unified or platform-specific mode)
        from src.publisher.publish_modes import publish_product

        platform_specific = (
            getattr(args, "platform_specific", False)
            or config.use_platform_specific_content
        )

        publish_results = await publish_product(
            publisher=publisher,
            media_id=media_url,
            product_id=product_id,
            platforms=platforms_to_publish,
            outputs_dir=outputs_dir,
            platform_specific=platform_specific,
            schedule_time=schedule_time,
        )

        # Record successful publish for each result
        for pub_result in publish_results:
            result_data = pub_result["result"]
            post_id = str(result_data.get("post_id", ""))
            logger.info(
                "Published: post_id=%s, status=%s",
                post_id,
                result_data.get("status"),
            )

            if pub_result["platform"] == "all":
                for p_info in platforms_to_publish:
                    record_publish(product_id, p_info["platform"], post_id, outputs_dir)
            else:
                record_publish(product_id, pub_result["platform"], post_id, outputs_dir)

        logger.info("Single video publishing complete")

        # Add to published products registry
        try:
            add_to_registry(product_id, outputs_dir)
        except Exception as exc:
            logger.warning("Failed to update product registry: %s", exc)

        # Link-in-bio update if enabled (CLI flags override config)
        link_in_bio_enabled = config.link_in_bio_config.enabled
        if getattr(args, "no_link_in_bio", False):
            link_in_bio_enabled = False
        elif getattr(args, "link_in_bio", None):
            link_in_bio_enabled = True
        if link_in_bio_enabled:
            try:
                from src.publisher.link_in_bio.manager import (
                    create_link_in_bio_manager,
                )

                link_bio_mgr = create_link_in_bio_manager(
                    provider_name=config.link_in_bio_config.provider,
                    max_links=config.link_in_bio_config.max_links,
                    max_title_length=config.link_in_bio_config.max_title_length,
                )
                bio_result = await link_bio_mgr.update(product_id, outputs_dir)
                if bio_result.get("success"):
                    logger.info("Link-in-bio updated for %s", product_id)
                else:
                    logger.warning(
                        "Link-in-bio skipped: %s", bio_result.get("reason", "unknown")
                    )
            except Exception as bio_error:
                logger.warning("Link-in-bio failed: %s", bio_error)

        # Automatic cleanup if enabled
        if config.cleanup_config.enabled and not args.no_cleanup:
            logger.info("Running automatic cleanup...")

            try:
                cleanup_mgr = CleanupManager(
                    outputs_dir=outputs_dir,
                    config=config.cleanup_config,
                    publisher=publisher,
                )

                cleanup_result = await cleanup_mgr.cleanup(
                    product_id=product_id,
                    platforms=args.platforms,
                    dry_run=False,
                )

                if cleanup_result["success"]:
                    logger.info("✓ Cleanup complete: %s", cleanup_result["message"])
                    disk_freed = cleanup_result["disk_freed"]
                    if isinstance(disk_freed, int) and disk_freed > 0:
                        logger.info("  Disk space freed: %s", format_bytes(disk_freed))
                else:
                    logger.warning("Cleanup skipped: %s", cleanup_result["message"])

            except Exception as cleanup_error:
                logger.warning(
                    "Cleanup failed but publish was successful: %s", cleanup_error
                )

        elif args.no_cleanup:
            logger.info("Cleanup disabled via --no-cleanup flag")
        else:
            logger.debug("Cleanup not configured in config file")

    except Exception as e:
        logger.error("Failed to publish video: %s", e, exc_info=args.debug)
        sys.exit(1)


async def cmd_calendar(
    args: argparse.Namespace, config, session: aiohttp.ClientSession
):
    """Execute calendar list command.

    Args:
    ----
        args: Parsed command-line arguments
        config: PublisherConfig instance
        session: aiohttp ClientSession

    """
    logger.info("Listing scheduled posts...")

    # Create schedule manager
    schedule_mgr = ScheduleManager()

    # Parse date filters if provided
    date_from = None
    date_to = None

    if args.date_from:
        try:
            date_from = datetime.fromisoformat(args.date_from)
            if date_from.tzinfo is None:
                date_from = date_from.replace(tzinfo=UTC)
        except ValueError as e:
            logger.error("Invalid date-from format: %s", e)
            sys.exit(1)

    if args.date_to:
        try:
            date_to = datetime.fromisoformat(args.date_to)
            if date_to.tzinfo is None:
                date_to = date_to.replace(tzinfo=UTC)
        except ValueError as e:
            logger.error("Invalid date-to format: %s", e)
            sys.exit(1)

    # List scheduled posts
    entries = schedule_mgr.list_scheduled(
        platform=args.platform,
        status=args.status,
        date_from=date_from,
        date_to=date_to,
    )

    if not entries:
        logger.info("No scheduled posts found")
        return

    logger.info("Found %d scheduled post(s):", len(entries))
    logger.info("=" * 80)

    for entry in entries:
        logger.info("Product: %s", entry.product_id)
        logger.info("Scheduled: %s (UTC)", entry.scheduled_time.isoformat())
        logger.info("Platforms: %s", ", ".join([p.value for p in entry.platforms]))
        logger.info("Status: %s", entry.status)
        if entry.post_id:
            logger.info("Post ID: %s", entry.post_id)
        if entry.slot_index is not None:
            logger.info("Slot Index: %d", entry.slot_index)
        logger.info("-" * 80)


async def cmd_schedule_auto(
    args: argparse.Namespace, config, session: aiohttp.ClientSession
):
    """Execute schedule command (calendar slots or immediate).

    Handles both scheduled and immediate publishing. When --immediate is set,
    delegates to BatchPublisher for direct publishing with stagger delays.
    Otherwise uses ScheduleManager to assign calendar slots.

    Args:
    ----
        args: Parsed command-line arguments
        config: PublisherConfig instance
        session: aiohttp ClientSession

    """
    immediate = getattr(args, "immediate", False)
    mode = "IMMEDIATE PUBLISH" if immediate else "AUTO-SCHEDULING"

    logger.info("=" * 80)
    logger.info("%s MODE", mode)
    logger.info("=" * 80)
    logger.info("Target platforms: %s", [p.value for p in args.platforms])
    logger.info("Outputs directory: %s", args.outputs_dir)
    if getattr(args, "dry_run", False) and not immediate:
        logger.info("DRY RUN MODE - No actual scheduling will occur")

    # Create publisher and authenticate
    publisher = _create_publisher_from_config(config, session)

    try:
        is_authenticated = await publisher.authenticate()
        if not is_authenticated:
            logger.error("Authentication failed - check your API key")
            sys.exit(1)

        logger.info("Authentication successful")

        # Immediate mode: delegate to BatchPublisher
        if immediate:
            await _run_immediate_batch(args, config, publisher)
            return

        # Scheduled mode: scan, filter, and assign calendar slots
        unpublished_videos = _scan_and_filter_videos(args)
        if not unpublished_videos:
            return

        schedule_mgr = ScheduleManager(config=config.schedule_config)
        cleanup_config = None if args.no_cleanup else config.cleanup_config

        summary = await schedule_mgr.auto_schedule(
            videos=unpublished_videos,
            platforms=args.platforms,
            publisher=publisher,
            start_slot=0,
            dry_run=args.dry_run,
            cleanup_config=cleanup_config,
            outputs_dir=args.outputs_dir,
            auto_resolve=getattr(args, "auto_resolve", False),
            force=getattr(args, "force", False),
        )

        logger.info("--- PUBLISHER SUMMARY ---")
        logger.info(
            "Products: %d attempted, %d scheduled, %d failed, %d skipped",
            len(unpublished_videos),
            summary["scheduled"],
            summary["failed"],
            summary["skipped"],
        )
        if summary.get("conflicts_resolved", 0) > 0:
            logger.info("Conflicts auto-resolved: %d", summary["conflicts_resolved"])
        logger.info("---")

        if args.dry_run:
            logger.info(
                "[DRY RUN] No actual scheduling occurred - "
                "run without --dry-run to schedule"
            )

        if summary["failed"] > 0:
            sys.exit(1)

    except Exception as e:
        logger.error("Publishing failed: %s", e, exc_info=args.debug)
        sys.exit(1)


def _scan_and_filter_videos(args: argparse.Namespace) -> list[Path]:
    """Scan outputs dir for videos, optionally filtering published ones.

    Args:
    ----
        args: Parsed CLI args (needs outputs_dir, force, platforms)

    Returns:
    -------
        List of video file paths to process

    """
    logger.info("Scanning %s for videos...", args.outputs_dir)
    video_paths = []

    for product_dir in args.outputs_dir.iterdir():
        if not product_dir.is_dir():
            continue
        for video_file in product_dir.glob("video_*.mp4"):
            video_paths.append(video_file)

    if not video_paths:
        logger.warning("No video files found in %s", args.outputs_dir)
        return []

    logger.info("Found %d video(s)", len(video_paths))

    if getattr(args, "force", False):
        logger.info("Force mode - including already published videos")
        return video_paths

    logger.info("Filtering already published videos...")
    unpublished = []

    for video_path in video_paths:
        product_id = video_path.parent.name

        already_published = all(
            is_already_published(product_id, platform.value, args.outputs_dir)
            for platform in args.platforms
        )

        if not already_published:
            unpublished.append(video_path)
        else:
            logger.debug(
                "Skipping %s - already published to all target platforms",
                product_id,
            )

    if not unpublished:
        logger.info("No unpublished videos to process")
        return []

    logger.info("Found %d unpublished video(s) ready to process", len(unpublished))
    return unpublished


async def _run_immediate_batch(
    args: argparse.Namespace,
    config,
    publisher,
) -> None:
    """Run immediate batch publishing via BatchPublisher.

    Args:
    ----
        args: Parsed CLI args
        config: PublisherConfig instance
        publisher: Authenticated publisher instance

    """
    batch_publisher = BatchPublisher(
        publisher=publisher,
        outputs_dir=args.outputs_dir,
        platforms=args.platforms,
        stagger_delay_min=config.stagger_delay_min,
        stagger_delay_max=config.stagger_delay_max,
        fail_fast=getattr(args, "fail_fast", False),
        retry_failed=getattr(args, "retry_failed", False),
    )

    summary = await batch_publisher.publish_batch()

    # Automatic cleanup if enabled
    if config.cleanup_config.enabled and not args.no_cleanup and summary.successful > 0:
        logger.info("=" * 80)
        logger.info("Running automatic cleanup for successfully published products...")

        try:
            cleanup_mgr = CleanupManager(
                outputs_dir=args.outputs_dir,
                config=config.cleanup_config,
                publisher=publisher,
            )

            cleanup_summary = await cleanup_mgr.cleanup_all(
                platforms=args.platforms,
                dry_run=False,
            )

            logger.info("Cleanup complete")
            logger.info("  Products cleaned: %d", cleanup_summary["cleaned"])
            logger.info("  Products skipped: %d", cleanup_summary["skipped"])
            logger.info(
                "  Total disk space freed: %s",
                format_bytes(cleanup_summary["disk_freed"]),
            )

        except Exception as cleanup_error:
            logger.warning(
                "Cleanup failed but publishing was successful: %s",
                cleanup_error,
            )

    elif args.no_cleanup:
        logger.info("Cleanup disabled via --no-cleanup flag")
    elif summary.successful == 0:
        logger.debug("No successful publishes - skipping cleanup")
    else:
        logger.debug("Cleanup not configured in config file")

    if summary.failed > 0:
        sys.exit(1)


def format_bytes(bytes_value: int | float) -> str:
    """Format bytes to human-readable string (KB, MB, GB).

    Args:
    ----
        bytes_value: Size in bytes

    Returns:
    -------
        Formatted string (e.g., "1.5 GB")

    """
    value = float(bytes_value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


async def cmd_cleanup(args: argparse.Namespace, config, session: aiohttp.ClientSession):
    """Execute cleanup command.

    Args:
    ----
        args: Parsed command-line arguments
        config: PublisherConfig instance
        session: aiohttp ClientSession

    """
    logger.info("=" * 80)
    logger.info("CLEANUP MODE")
    logger.info("=" * 80)

    # Validate --all requires --confirm (unless dry-run)
    if args.all and not args.confirm and not args.dry_run:
        logger.error(
            "ERROR: --all mode requires --confirm flag to prevent accidental deletion"
        )
        logger.info("Use: --all --confirm (or add --dry-run to preview)")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN MODE - No actual deletion will occur")

    # Determine platforms
    if args.platforms:
        platforms = [Platform[p.upper()] for p in args.platforms]
    else:
        # Use default platforms from config
        platforms = config.default_platforms or [
            Platform.YOUTUBE,
            Platform.TIKTOK,
            Platform.INSTAGRAM,
        ]

    logger.info("Target platforms: %s", [p.value for p in platforms])
    logger.info("Outputs directory: %s", args.outputs_dir)

    # Create publisher
    publisher = _create_publisher_from_config(config, session)

    try:
        # Authenticate
        is_authenticated = await publisher.authenticate()
        if not is_authenticated:
            logger.error("Authentication failed - check your API key")
            sys.exit(1)

        logger.info("Authentication successful")

        # Create cleanup manager
        cleanup_mgr = CleanupManager(
            outputs_dir=args.outputs_dir,
            config=config.cleanup_config,
            publisher=publisher,
        )

        if args.product_id:
            # Single product cleanup
            logger.info("Cleaning up product: %s", args.product_id)
            logger.info("-" * 80)

            result = await cleanup_mgr.cleanup(
                product_id=args.product_id,
                platforms=platforms,
                dry_run=args.dry_run,
            )

            if result["success"]:
                logger.info("✓ %s", result["message"])
                disk_freed = result["disk_freed"]
                if isinstance(disk_freed, int) and disk_freed > 0:
                    logger.info("  Disk space freed: %s", format_bytes(disk_freed))
            else:
                logger.warning("✗ %s", result["message"])
                sys.exit(1)

        elif args.all:
            # Batch cleanup
            logger.info("Cleaning up all successfully published products...")
            logger.info("-" * 80)

            summary = await cleanup_mgr.cleanup_all(
                platforms=platforms,
                dry_run=args.dry_run,
            )

            # Display summary
            logger.info("--- CLEANUP SUMMARY ---")
            logger.info(
                "Products: %d cleaned, %d skipped, %s freed",
                summary["cleaned"],
                summary["skipped"],
                format_bytes(summary["disk_freed"]),
            )
            logger.info("---")

            if args.dry_run:
                logger.info(
                    "[DRY RUN] No actual deletion occurred - "
                    "run without --dry-run to cleanup"
                )

    except Exception as e:
        logger.error("Cleanup failed: %s", e, exc_info=args.debug)
        sys.exit(1)


async def cmd_delete(args: argparse.Namespace, config, session: aiohttp.ClientSession):
    """Delete a post from Late.dev."""
    publisher = _create_publisher_from_config(config, session)

    try:
        # Authenticate
        is_authenticated = await publisher.authenticate()
        if not is_authenticated:
            logger.error("Authentication failed - check your API key")
            sys.exit(1)

        logger.info("Deleting post: %s", args.post_id)
        success = await publisher.delete_post(args.post_id)

        if success:
            logger.info("Successfully deleted post: %s", args.post_id)
        else:
            logger.error("Failed to delete post: %s", args.post_id)
            sys.exit(1)

    except Exception as e:
        logger.error("Delete failed: %s", e, exc_info=args.debug)
        sys.exit(1)


def cmd_registry(args: argparse.Namespace) -> None:
    """Manage published products registry."""
    outputs_dir = args.outputs_dir

    if args.rebuild:
        scan_dir = getattr(args, "scan_dir", None)
        count = rebuild_registry(outputs_dir, scan_dir=scan_dir)
        logger.info("Registry rebuilt: %d products in %s", count, outputs_dir)
    else:
        logger.error("No action specified. Use --rebuild to rebuild the registry.")
        sys.exit(1)


async def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Late.dev Publisher - Publish videos to social media platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Publish single product (auto-schedules to next available slot)
  python -m src.publisher.late single B0ABC123 --debug

  # Publish to specific platforms
  python -m src.publisher.late single B0ABC123 --platform youtube --platform tiktok

  # Schedule all unpublished videos to calendar slots
  python -m src.publisher.late schedule --dry-run --debug

  # Force-schedule already published products
  python -m src.publisher.late schedule --force --debug

  # Publish all videos immediately (no scheduling)
  python -m src.publisher.late schedule --immediate --debug

  # Retry failed items only
  python -m src.publisher.late schedule --immediate --retry-failed --debug

  # Use specific account (multi-account mode)
  python -m src.publisher.late single B0ABC123 --account secondary
        """,
    )

    # Global argument for multi-account support
    parser.add_argument(
        "--account",
        type=str,
        metavar="NAME",
        help="Account name to use (from config/publisher.yaml accounts section)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # list-accounts command
    list_parser = subparsers.add_parser(
        "list-accounts",
        help="List connected social media accounts",
    )
    list_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # single command
    single_parser = subparsers.add_parser(
        "single",
        help="Publish a single product (auto-discovers video, schedules to next slot)",
    )
    single_parser.add_argument(
        "product_id",
        type=str,
        help="Product ID (e.g., B00TF9E6XE) - video is auto-discovered from outputs/",
    )
    single_parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        choices=["youtube", "tiktok", "instagram", "facebook", "twitter", "linkedin"],
        help="Target platform(s) (default: youtube, tiktok, instagram)",
    )
    single_parser.add_argument(
        "--schedule",
        type=str,
        metavar="DATETIME",
        help=(
            "Schedule for specific time (format: 'YYYY-MM-DD HH:MM:SS'). "
            "Without this, uses next available calendar slot"
        ),
    )
    single_parser.add_argument(
        "--immediate",
        action="store_true",
        help="Publish immediately (requires explicit flag to prevent accidents)",
    )
    single_parser.add_argument(
        "--force",
        action="store_true",
        help="Force republish even if already published to platform",
    )
    single_parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable automatic cleanup after publish (default: cleanup enabled)",
    )
    single_parser.add_argument(
        "--link-in-bio",
        action="store_true",
        default=None,
        help="Enable link-in-bio update after publish (overrides config)",
    )
    single_parser.add_argument(
        "--no-link-in-bio",
        action="store_true",
        help="Disable link-in-bio update after publish (overrides config)",
    )
    single_parser.add_argument(
        "--platform-specific",
        action="store_true",
        help=(
            "Create separate posts per platform with optimized metadata. "
            "Default: single post for all platforms."
        ),
    )
    single_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # calendar command
    calendar_parser = subparsers.add_parser(
        "calendar",
        help="View scheduled posts calendar",
    )
    calendar_parser.add_argument(
        "action",
        nargs="?",
        default="list",
        choices=["list"],
        help="Calendar action (default: list)",
    )
    calendar_parser.add_argument(
        "--platform",
        help="Filter by platform (e.g., youtube, tiktok)",
    )
    calendar_parser.add_argument(
        "--status",
        choices=["pending", "scheduled", "published", "failed", "partial"],
        help="Filter by status",
    )
    calendar_parser.add_argument(
        "--date-from",
        help="Start date filter (YYYY-MM-DD)",
    )
    calendar_parser.add_argument(
        "--date-to",
        help="End date filter (YYYY-MM-DD)",
    )
    calendar_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # schedule command
    schedule_parser = subparsers.add_parser(
        "schedule",
        help="Publish all videos (scheduled to calendar slots or immediately)",
    )
    schedule_parser.add_argument(
        "action",
        nargs="?",
        default="auto",
        choices=["auto"],
        help="Schedule action (default: auto)",
    )
    schedule_parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        choices=["youtube", "tiktok", "instagram", "facebook", "twitter", "linkedin"],
        help="Target platform(s) (default: youtube, tiktok, instagram)",
    )
    schedule_parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to scan for videos (default: outputs/)",
    )
    schedule_parser.add_argument(
        "--immediate",
        action="store_true",
        help="Publish immediately instead of scheduling to calendar slots",
    )
    schedule_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview scheduling without making changes",
    )
    schedule_parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable automatic cleanup after publishing (default: cleanup enabled)",
    )
    schedule_parser.add_argument(
        "--auto-resolve",
        action="store_true",
        help="Automatically resolve slot conflicts by using first available",
    )
    schedule_parser.add_argument(
        "--force",
        action="store_true",
        help="Include already published products",
    )
    schedule_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure",
    )
    schedule_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only retry previously failed items from the retry queue",
    )
    schedule_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # cleanup command
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Cleanup published product directories",
    )
    cleanup_group = cleanup_parser.add_mutually_exclusive_group(required=True)
    cleanup_group.add_argument(
        "--product-id",
        help="Clean up specific product by ID",
    )
    cleanup_group.add_argument(
        "--all",
        action="store_true",
        help="Clean up all successfully published products",
    )
    cleanup_parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        choices=["youtube", "tiktok", "instagram", "facebook", "twitter", "linkedin"],
        help="Target platform(s) (default: all platforms)",
    )
    cleanup_parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to scan for products (default: outputs/)",
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview cleanup without deleting anything",
    )
    cleanup_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required for --all mode to prevent accidents",
    )
    cleanup_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # delete command
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a post from Late.dev",
    )
    delete_parser.add_argument(
        "post_id",
        help="The ID of the post to delete",
    )
    delete_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # registry command
    registry_parser = subparsers.add_parser(
        "registry",
        help="Manage published products registry (JSON + CSV)",
    )
    registry_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild registry from all data.json files in outputs directory",
    )
    registry_parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save registry files (default: outputs)",
    )
    registry_parser.add_argument(
        "--scan-dir",
        type=Path,
        default=None,
        help="Directory to scan for product data (default: same as --outputs-dir)",
    )
    registry_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.command == "calendar":
        # No special validation needed for calendar command
        pass

    elif args.command == "schedule":
        # Convert platform strings to Platform enums, default to all 3
        if args.platforms:
            args.platforms = [Platform[p.upper()] for p in args.platforms]
        else:
            args.platforms = list(DEFAULT_PLATFORMS)

    elif args.command == "cleanup":
        # Platform conversion handled in cmd_cleanup for better defaults
        pass

    elif args.command == "single":
        if args.schedule:
            try:
                args.schedule = parse_datetime(args.schedule)
            except ValueError as e:
                parser.error(str(e))
        # Note: If neither --schedule nor --immediate, auto-discover next slot

        # Convert platform strings to Platform enums (if provided)
        if args.platforms:
            args.platforms = [Platform[p.upper()] for p in args.platforms]
        # Else: cmd_single will default to all 3 platforms

        # Initialize force if not set
        if not hasattr(args, "force"):
            args.force = False

    # Load .env
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    load_dotenv(project_root / ".env")

    # Setup logging
    setup_debug_logging(
        log_file=project_root / "outputs" / "logs" / "publisher.log",
        debug_mode=args.debug,
        verbose=args.debug,
        component_name="Publisher",
    )

    if args.debug:
        logger.info("Debug mode enabled")

    # Load configuration
    try:
        # Build CLI overrides
        cli_overrides = {}
        if hasattr(args, "platforms") and args.platforms:
            cli_overrides["default_platforms"] = args.platforms
        if hasattr(args, "immediate") and args.immediate:
            cli_overrides["immediate_publish"] = True
        if hasattr(args, "fail_fast") and args.fail_fast:
            cli_overrides["fail_fast"] = args.fail_fast
        if hasattr(args, "account") and args.account:
            cli_overrides["account"] = args.account

        config = load_publisher_config(
            config_path=project_root / "config" / "publisher.yaml",
            cli_overrides=cli_overrides,
        )

        account_info = (
            f", account={config.active_account}" if config.active_account else ""
        )
        logger.info(
            "Configuration loaded: provider=%s%s",
            config.provider,
            account_info,
        )

    except Exception as e:
        logger.error("Configuration loading failed: %s", e, exc_info=args.debug)
        sys.exit(1)

    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Execute command
        if args.command == "list-accounts":
            await cmd_list_accounts(args, config, session)
        elif args.command == "single":
            await cmd_single(args, config, session)
        elif args.command == "calendar":
            await cmd_calendar(args, config, session)
        elif args.command == "schedule":
            await cmd_schedule_auto(args, config, session)
        elif args.command == "cleanup":
            await cmd_cleanup(args, config, session)
        elif args.command == "delete":
            await cmd_delete(args, config, session)
        elif args.command == "registry":
            cmd_registry(args)


if __name__ == "__main__":
    asyncio.run(main())
