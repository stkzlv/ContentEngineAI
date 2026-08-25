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
import json
import logging
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

from src.publisher import PublisherProvider, create_publisher
from src.publisher.analytics import (
    load_metrics,
    publish_time,
    rank_by_durability,
    save_metrics,
    summarize_post,
    timeline_resource,
)
from src.publisher.batch import BatchPublisher
from src.publisher.blob_retention import run_blob_retention
from src.publisher.cleanup import CleanupManager
from src.publisher.comment_verify import verify_post_first_comments
from src.publisher.config import load_publisher_config
from src.publisher.link_in_bio.manager import update_link_in_bio_safe
from src.publisher.models import DEFAULT_PLATFORMS, Platform
from src.publisher.partial_post_sweep import sweep_partial_posts
from src.publisher.product_registry import (
    add_to_registry,
    load_registry,
    rebuild_registry,
    summarize_by_content_format,
)
from src.publisher.schedule import ScheduleManager
from src.publisher.tracking import is_already_published, record_publish
from src.publisher.video_selector import sole_render_for_product
from src.utils.logging_setup import setup_debug_logging

logger = logging.getLogger(__name__)


def _record_publish_results(
    product_id: str,
    publish_results: list[dict],
    platforms_to_publish: list[dict],
    outputs_dir: Path,
) -> int:
    """Write each publish result to publish_history.json.

    Each call is wrapped so a tracking write that fails for one platform
    doesn't drop the others. Returns the number of platforms recorded
    successfully.
    """
    recorded = 0
    for pub_result in publish_results:
        result_data = pub_result["result"]
        post_id = str(result_data.get("post_id", ""))
        logger.info(
            "Published: post_id=%s, status=%s",
            post_id,
            result_data.get("status"),
        )

        targets = (
            [p["platform"] for p in platforms_to_publish]
            if pub_result["platform"] == "all"
            else [pub_result["platform"]]
        )
        for plat in targets:
            try:
                record_publish(product_id, plat, post_id, outputs_dir)
                recorded += 1
            except OSError as e:
                logger.error(
                    "Failed to record publish %s:%s to history: %s",
                    product_id,
                    plat,
                    e,
                )
    return recorded


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
        synthetic_media_disclosure=config.synthetic_media_disclosure,
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


def _load_product_map(outputs_dir: Path) -> dict[str, str]:
    """Map Zernio post_id to product_id from publish_history.json (best effort)."""
    try:
        posts = json.loads((outputs_dir / "publish_history.json").read_text())["posts"]
    except (OSError, KeyError, ValueError):
        return {}
    out: dict[str, str] = {}
    for value in posts.values():
        post_id = value.get("post_id")
        if post_id:
            out[post_id] = value.get("product_id", post_id)
    return out


async def cmd_analytics(
    args: argparse.Namespace, config, session: aiohttp.ClientSession
):
    """Capture day-N views and a durability ratio for published posts.

    Ranking by total views answers a different question from ranking by
    durability, and for a format comparison it answers the wrong one: a post
    that spiked and stopped outranks one still earning months later. The day-7
    figure has the same problem, since at day 7 both look like their launch
    curve.
    """
    outputs_dir = args.outputs_dir
    if args.rank_only:
        # No client and no authentication: the flag promises no network call.
        # Config still loads before dispatch, so a key must be configured even
        # though this path never uses one.
        metrics = load_metrics(outputs_dir)
        if not metrics:
            logger.warning(
                "No stored metrics in %s; run without --rank-only", outputs_dir
            )
            return
    else:
        publisher = _create_publisher_from_config(config, session)
        if not await publisher.authenticate():
            logger.error("Authentication failed - check your API key")
            sys.exit(1)
        posts = await publisher.list_posts(status="published")
        resource = timeline_resource(publisher.client)
        metrics = []
        for post in posts[: args.limit]:
            post_id = post.get("id") or post.get("_id")
            if not post_id:
                continue
            try:
                raw = resource.get_post_timeline(post_id=post_id)
                # The SDK returns the parsed JSON body, a plain dict, not a
                # model. Reading it with getattr yields None for every key and
                # stores an empty record while reporting success.
                if isinstance(raw, dict):
                    rows = raw.get("timeline") or raw.get("data") or []
                else:
                    rows = (
                        getattr(raw, "timeline", None)
                        or getattr(raw, "data", None)
                        or []
                    )
                rows = [r if isinstance(r, dict) else r.model_dump() for r in rows]
            except Exception as exc:
                # One post's analytics failing must not lose the rest of the
                # sweep; a partial reading is still usable.
                logger.warning("No timeline for %s: %s", post_id, exc)
                continue
            metrics.append(summarize_post(post_id, publish_time(post), rows))
        save_metrics(metrics, outputs_dir)
        logger.info("Captured metrics for %d post(s) in %s", len(metrics), outputs_dir)

    logger.info("%-26s %8s %8s %8s %10s", "post", "day2", "day7", "total", "durability")
    for m in rank_by_durability(metrics):
        ratio = "-" if m.durability_ratio is None else f"{m.durability_ratio:.2f}"
        logger.info(
            "%-26s %8s %8s %8s %10s",
            m.post_id[:26],
            m.views_day_2 if m.views_day_2 is not None else "-",
            m.views_day_7 if m.views_day_7 is not None else "-",
            m.views_total if m.views_total is not None else "-",
            ratio,
        )


async def cmd_verify_comments(
    args: argparse.Namespace, config, session: aiohttp.ClientSession
):
    """Check recent published posts for a missing first comment.

    Sweeps the most recent published posts and WARNs when a YouTube or
    Instagram post is missing its owner first comment. Run after posts go live;
    the comment is the affiliate-link surface and can fail silently.
    """
    publisher = _create_publisher_from_config(config, session)
    if not await publisher.authenticate():
        logger.error("Authentication failed - check your API key")
        sys.exit(1)

    posts = (await publisher.list_posts(status="published"))[: args.limit]
    product_map = _load_product_map(args.outputs_dir)
    logger.info("Checking first comments on %d recent published post(s)", len(posts))

    checked = 0
    missing = 0
    for post in posts:
        post_id = post.get("id")
        if not post_id:
            continue
        product = product_map.get(post_id, post_id)
        try:
            checks = await verify_post_first_comments(publisher, post_id)
        except Exception as exc:
            logger.warning("Could not check post %s (%s): %s", product, post_id, exc)
            continue
        for check in checks:
            checked += 1
            if not check.present:
                missing += 1
                logger.warning(
                    "First comment MISSING: %s on %s (post %s)",
                    product,
                    check.platform,
                    post_id,
                )
    logger.info(
        "First-comment check: %d missing of %d checked across %d post(s)",
        missing,
        checked,
        len(posts),
    )


async def cmd_verify_delivery(
    args: argparse.Namespace, config, session: aiohttp.ClientSession
):
    """Sweep recent posts for silently-failed platform legs.

    Zernio leaves a post ``partial`` when one platform fails at publish time
    and flags it nowhere. This WARNs on every partial/failed recent post with
    its failing platform and error. Fix a flagged post with
    ``posts.retry(post_id)`` (re-publishes only the failed leg, no re-render).
    """
    publisher = _create_publisher_from_config(config, session)
    if not await publisher.authenticate():
        logger.error("Authentication failed - check your API key")
        sys.exit(1)

    results = await sweep_partial_posts(publisher, args.limit)
    product_map = _load_product_map(args.outputs_dir)
    logger.info("Swept %d recent post(s) for incomplete delivery", args.limit)

    for post in results:
        product = product_map.get(post.post_id, post.post_id)
        legs = (
            ", ".join(
                f"{leg.platform} ({leg.error_category or leg.status})"
                for leg in post.failed_legs
            )
            if post.failed_legs
            else "no per-leg detail"
        )
        logger.warning(
            "Delivery incomplete: %s post %s status=%s failing: %s",
            product,
            post.post_id,
            post.top_status,
            legs,
        )
        for leg in post.failed_legs:
            if leg.error_message:
                logger.warning("  %s error: %s", leg.platform, leg.error_message)
    logger.info("Delivery sweep: %d post(s) with failed/partial delivery", len(results))
    if results:
        logger.info(
            "Retry a failed leg with posts.retry(<post_id>): re-publishes only "
            "the failed platform from Zernio's CDN, no re-render"
        )


async def cmd_single(args: argparse.Namespace, config, session: aiohttp.ClientSession):
    """Execute single video publish command.

    Args:
    ----
        args: Parsed command-line arguments
        config: PublisherConfig instance
        session: aiohttp ClientSession

    """
    product_id = args.product_id
    outputs_dir = Path("outputs").resolve()
    product_dir = outputs_dir / product_id

    if not product_dir.exists():
        logger.error("Product directory not found: %s", product_dir)
        sys.exit(1)

    # Default to all 3 platforms if none specified
    if not args.platforms:
        args.platforms = list(DEFAULT_PLATFORMS)
        logger.info("Using default platforms: youtube, tiktok, instagram")

    # Link-in-bio enablement (CLI flags override config)
    link_in_bio_enabled = config.link_in_bio_config.enabled
    if getattr(args, "no_link_in_bio", False):
        link_in_bio_enabled = False
    elif getattr(args, "link_in_bio", None):
        link_in_bio_enabled = True

    # A fully-published product needs no video or upload: keep its bio link
    # fresh and exit. This also works after cleanup removed the rendered
    # video (only data.json is needed for the link).
    if not args.force and all(
        is_already_published(product_id, p.value, outputs_dir) for p in args.platforms
    ):
        logger.warning(
            "Product %s already published to all requested platforms. "
            "Use --force to republish.",
            product_id,
        )
        if link_in_bio_enabled:
            await update_link_in_bio_safe(
                product_id,
                outputs_dir,
                replace(config.link_in_bio_config, enabled=True),
            )
        return

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
        product_dir, product_id, first_platform, config.profiles
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
        skipped_already_published = False
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
                skipped_already_published = True
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
            # The product is already live; make sure its bio link exists so
            # `single <id>` on a published product isn't a link-in-bio no-op.
            if skipped_already_published and link_in_bio_enabled:
                await update_link_in_bio_safe(
                    product_id,
                    outputs_dir,
                    replace(config.link_in_bio_config, enabled=True),
                )
            return

        # Publish (unified or platform-specific mode)
        from src.publisher.publish_modes import publish_product

        platform_specific = (
            getattr(args, "platform_specific", False)
            or config.use_platform_specific_content
        )

        disc_cfg = config.affiliate_disclosure_config
        disclosure_phrase = disc_cfg.phrase if disc_cfg.enabled else None
        publish_results = await publish_product(
            publisher=publisher,
            media_id=media_url,
            product_id=product_id,
            platforms=platforms_to_publish,
            outputs_dir=outputs_dir,
            platform_specific=platform_specific,
            schedule_time=schedule_time,
            disclosure_phrase=disclosure_phrase,
        )

        # Record successful publish for each result
        _record_publish_results(
            product_id, publish_results, platforms_to_publish, outputs_dir
        )

        logger.info("Single video publishing complete")

        # Add to published products registry
        try:
            add_to_registry(product_id, outputs_dir)
        except Exception as exc:
            logger.warning("Failed to update product registry: %s", exc)

        # Link-in-bio update if enabled (flags resolved above)
        if link_in_bio_enabled:
            await update_link_in_bio_safe(
                product_id,
                outputs_dir,
                replace(config.link_in_bio_config, enabled=True),
            )

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
                    logger.info("Cleanup complete: %s", cleanup_result["message"])
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

        # Trim the Vercel Blob upload store (non-blocking)
        await run_blob_retention(publisher, config.blob_retention_config)

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
        unpublished_videos = _scan_and_filter_videos(args, config)
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
            link_in_bio_config=config.link_in_bio_config,
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


def _scan_and_filter_videos(
    args: argparse.Namespace, config: Any | None = None
) -> list[Path]:
    """Scan outputs dir for videos, optionally filtering published ones.

    Args:
    ----
        args: Parsed CLI args (needs outputs_dir, force, platforms)
        config: Loaded publisher config, for the per-platform render profile

    Returns:
    -------
        List of video file paths to process

    """
    logger.info("Scanning %s for videos...", args.outputs_dir)
    video_paths = []

    # One video per product, not one per file. A product rendered under a
    # second profile keeps both files, and taking each of them scheduled the
    # same product twice -- two posts on different days, each carrying a
    # different render, burning two slots. `single` has always resolved one
    # video per product; this makes the two paths agree.
    # The configured per-platform profile decides which render goes out, so
    # the scanner has to consult it too. Reading only the alphabetically
    # first file sent a different cut than `single` would have chosen.
    profiles = getattr(config, "profiles", None)
    first_platform = ""
    platform_args = getattr(args, "platforms", None)
    if platform_args:
        first = platform_args[0]
        first_platform = getattr(first, "value", str(first))

    for product_dir in sorted(args.outputs_dir.iterdir()):
        if not product_dir.is_dir():
            continue
        renders = sorted(product_dir.glob("video_*.mp4"))
        if not renders:
            continue
        chosen = sole_render_for_product(product_dir, profiles, first_platform)
        if chosen is None:
            continue
        video_paths.append(chosen)
        if len(renders) > 1:
            ignored = [r.name for r in renders if r != chosen]
            logger.info(
                "%s has %d renders; scheduling %s and ignoring %s",
                product_dir.name,
                len(renders),
                chosen.name,
                ", ".join(ignored),
            )

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
        profiles=getattr(config, "profiles", None),
        stagger_delay_min=config.stagger_delay_min,
        stagger_delay_max=config.stagger_delay_max,
        fail_fast=getattr(args, "fail_fast", False),
        retry_failed=getattr(args, "retry_failed", False),
        link_in_bio_config=config.link_in_bio_config,
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

    # Trim the Vercel Blob upload store (non-blocking)
    if summary.successful > 0:
        await run_blob_retention(publisher, config.blob_retention_config)

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
                logger.info("%s", result["message"])
                disk_freed = result["disk_freed"]
                if isinstance(disk_freed, int) and disk_freed > 0:
                    logger.info("  Disk space freed: %s", format_bytes(disk_freed))
            else:
                logger.warning("%s", result["message"])
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
        if count < 0:
            # The rebuild refused rather than emptying a registry it could not
            # read. Reporting success here would put an INFO line saying it
            # happened directly under the ERROR saying it did not.
            sys.exit(1)
        logger.info("Registry rebuilt: %d products in %s", count, outputs_dir)
    elif getattr(args, "summary", False):
        entries = load_registry(outputs_dir)
        counts = summarize_by_content_format(entries)
        logger.info("Registry: %d entries in %s", len(entries), outputs_dir)
        for arm, count in sorted(counts.items()):
            logger.info("  %-12s %d", arm, count)
    else:
        logger.error(
            "No action specified. Use --rebuild or --summary.",
        )
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

  # Republish a product already posted (bypass the duplicate guard; default is off)
  python -m src.publisher.late single B0ABC123 --force --debug

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
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Republish even if already published (default: off; --force to republish)",
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
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include already published products (default: off; --force to include)",
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
        "--summary",
        action="store_true",
        help="Count published products per content-format arm (one row per product)",
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

    analytics_parser = subparsers.add_parser(
        "analytics",
        help="Capture day-N views and a durability ratio per published post",
    )
    analytics_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="How many recent published posts to measure (default: 50)",
    )
    analytics_parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Rank stored metrics without fetching (no network)",
    )
    analytics_parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Where post_metrics.json lives (default: outputs)",
    )
    analytics_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    verify_parser = subparsers.add_parser(
        "verify-comments",
        help="Check that first comments landed on recent published posts",
    )
    verify_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of recent published posts to check (default: 25)",
    )
    verify_parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory holding publish_history.json for product names",
    )
    verify_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    verify_delivery_parser = subparsers.add_parser(
        "verify-delivery",
        help="Sweep recent posts for silently-failed platform legs",
    )
    verify_delivery_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of recent posts to sweep (default: 25)",
    )
    verify_delivery_parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory holding publish_history.json for product names",
    )
    verify_delivery_parser.add_argument(
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

        # Initialize force if not set (default off; safe duplicate guard)
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
        elif args.command == "analytics":
            await cmd_analytics(args, config, session)
        elif args.command == "verify-comments":
            await cmd_verify_comments(args, config, session)
        elif args.command == "verify-delivery":
            await cmd_verify_delivery(args, config, session)


if __name__ == "__main__":
    asyncio.run(main())
