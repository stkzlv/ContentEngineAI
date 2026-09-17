"""The batch's publishing phase: one post per rendered video, on a schedule.

Moved out of `global_batch.py` (#450), and unlike the other two phases not
verbatim: this is the phase with the drift history, so what the publisher
package already does is called rather than restated. The publisher comes
from `create_publisher_from_config`, the occupied slots and the next free
one from `ScheduleManager`, the history writes from
`record_publish_results`, the account pairing from `accounts_for_platforms`
and the cleanup from `remove_published_product_dir`. What stays here is the
batch's own loop: the stagger between posts, fail-fast, and the per-platform
tallies the summary reports.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path

from src.pipeline.config import GlobalBatchConfig, PublishingPhaseSummary
from src.publisher.models import Platform
from src.utils.outputs_paths import durable_state_path

logger = logging.getLogger(__name__)


async def run_publishing_phase(
    batch_config: GlobalBatchConfig, produced_videos: list[tuple[Path, str]]
) -> PublishingPhaseSummary:
    """Publish each produced video to the configured platforms.

    Authenticates once, resolves the scheduling strategy, and publishes each
    video with staggered delays and per-video error tracking.

    Args:
    ----
        batch_config: The batch configuration: platform and schedule
            overrides, outputs directory, fail-fast.
        produced_videos: List of (video_path, product_id) tuples from
            production.

    Returns:
    -------
        PublishingPhaseSummary with per-platform publishing statistics

    """
    from src.pipeline.global_batch import _publisher_settings
    from src.publisher.late.client import LatePublisher
    from src.publisher.registry import create_publisher_from_config
    from src.publisher.schedule import ScheduleManager

    phase_start = time.time()

    # One typed load. Every section below used to be re-parsed from a raw
    # dict here, which is how `tiktok_settings` went missing from this path
    # for several releases while `single` had it.
    published = _publisher_settings()

    platforms_to_publish = batch_config.platforms or [
        p.value for p in published.default_platforms
    ]
    platforms = [Platform(p.lower()) for p in platforms_to_publish]

    stagger_min = published.stagger_delay_min
    stagger_max = published.stagger_delay_max

    # Track statistics
    total_attempted = len(produced_videos)
    successful = 0
    failed = 0
    skipped = 0
    failed_videos: list[str] = []
    skipped_videos: list[str] = []
    platform_results: dict[str, dict[str, int]] = {
        p.value: {"successful": 0, "failed": 0} for p in platforms
    }
    errors: list[dict[str, str]] = []

    # Initialize publisher: the same factory the CLI uses, so every setting
    # the config names reaches the provider on this path too. The key comes
    # from the environment here, not from the settings object, which may
    # carry the placeholder `_publisher_settings` substitutes for an absent
    # credential at settings-read time.
    try:
        api_key = os.getenv("LATE_API_KEY")
        if not api_key:
            raise ValueError("LATE_API_KEY environment variable not set")

        vercel_token = os.getenv("LATE_VERCEL_TOKEN")
        logger.debug(
            "Publisher init: api_key=%s, vercel_token=%s",
            "set",
            "set" if vercel_token else "NOT SET",
        )
        publisher = create_publisher_from_config(
            published, api_key=api_key, vercel_token=vercel_token
        )

        logger.info("Authenticating with publisher...")
        await publisher.authenticate()
        logger.info("Authentication successful")

        accounts = await publisher.get_accounts()
        logger.info("Found %s connected account(s)", len(accounts))

    except Exception as e:
        logger.exception("Failed to initialize publisher: %s", e)
        # Return early with all videos marked as failed
        return PublishingPhaseSummary(
            total_attempted=total_attempted,
            successful=0,
            failed=total_attempted,
            skipped=0,
            failed_videos=[product_id for _, product_id in produced_videos],
            skipped_videos=[],
            platform_results=platform_results,
            errors=[
                {
                    "product_id": product_id,
                    "error": f"Publisher initialization failed: {e}",
                }
                for _, product_id in produced_videos
            ],
            duration_sec=time.time() - phase_start,
        )

    # Determine scheduling strategy with 3-tier precedence:
    # 1. Explicit CLI/YAML schedule_time (highest priority)
    # 2. Auto-schedule via recurring_schedule if immediate_publish=false
    # 3. Publish immediately (scheduled_time=None)
    #
    # For auto-schedule with multiple products, each product gets its own
    # slot, found per product in the loop against the occupancy read here.
    schedule_time: datetime | None = None
    schedule_manager: ScheduleManager | None = None
    occupied_slot_times: set[datetime] | None = None

    schedule_time_str = batch_config.schedule_time or published.schedule_time
    if schedule_time_str:
        schedule_time = datetime.fromisoformat(schedule_time_str.replace("Z", "+00:00"))
        logger.info("Using explicit schedule time: %s", schedule_time)
    else:
        immediate_publish = published.immediate_publish
        recurring_config = published.schedule_config
        recurring_enabled = recurring_config.enabled

        logger.debug(
            "Scheduling config: immediate_publish=%s, recurring_enabled=%s",
            immediate_publish,
            recurring_enabled,
        )

        if not immediate_publish and recurring_enabled:
            if not recurring_config.slots:
                logger.warning(
                    "recurring_schedule.enabled=true but no slots defined. "
                    "Publishing immediately."
                )
            else:
                logger.info("Auto-scheduling: preparing slot context...")
                try:
                    schedule_manager = ScheduleManager(
                        schedule_path=durable_state_path(
                            batch_config.outputs_dir, "schedule.json"
                        ),
                        config=recurring_config,
                    )
                    # The provider's posts and the local schedule, as one set;
                    # the same read `single` and `schedule` make.
                    occupied_slot_times = await schedule_manager.build_occupancy(
                        publisher, datetime.now(UTC)
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to auto-schedule: %s. Publishing immediately.", e
                    )
                    schedule_manager = None
                    occupied_slot_times = None
        elif immediate_publish:
            logger.info("immediate_publish=true: Publishing immediately")
        else:
            logger.info("recurring_schedule.enabled=false: Publishing immediately")

    # Publish each video
    for idx, (video_path, product_id) in enumerate(produced_videos, 1):
        logger.info("[%s/%s] Publishing video for %s", idx, total_attempted, product_id)

        video_successful = True
        video_errors: list[str] = []

        try:
            # Upload video once
            logger.info("[%s/%s] Uploading video...", idx, total_attempted)
            media_id = await publisher.upload_media(video_path)
            logger.info("[%s/%s] Upload complete: %s", idx, total_attempted, media_id)

            # Build platforms list (validate accounts upfront)
            from src.publisher.publish_modes import accounts_for_platforms

            pub_platforms, missing = accounts_for_platforms(platforms, accounts)
            for platform in missing:
                logger.warning(
                    "[%d/%d] No account for %s, skipping",
                    idx,
                    total_attempted,
                    platform.value,
                )
                platform_results[platform.value]["failed"] += 1

            if not pub_platforms:
                raise ValueError("No valid platform accounts found")

            # Find per-product schedule slot if auto-scheduling
            product_schedule_time = schedule_time
            product_slot_index: int | None = None
            if schedule_manager is not None and occupied_slot_times is not None:
                try:
                    product_schedule_time, product_slot_index = (
                        schedule_manager.next_free_slot(
                            product_id, datetime.now(UTC), 0, occupied_slot_times
                        )
                    )
                except ValueError:
                    logger.warning(
                        "All slots occupied for %s. Publishing immediately.",
                        product_id,
                    )
                    product_schedule_time = None
                else:
                    occupied_slot_times.add(
                        product_schedule_time.replace(second=0, microsecond=0)
                    )
                    logger.info(
                        "Auto-scheduled %s to slot #%s: %s",
                        product_id,
                        product_slot_index,
                        product_schedule_time.strftime("%A, %Y-%m-%d %H:%M:%S %Z"),
                    )

            # Publish (unified or platform-specific mode)
            from src.publisher.publish_modes import publish_product

            platform_specific = (
                batch_config.platform_specific_content
                or published.use_platform_specific_content
            )

            affiliate_cfg = published.affiliate_disclosure_config
            disclosure_phrase = affiliate_cfg.phrase if affiliate_cfg.enabled else None
            publish_results = await publish_product(
                publisher=publisher,
                media_id=media_id,
                product_id=product_id,
                platforms=pub_platforms,
                outputs_dir=batch_config.outputs_dir,
                platform_specific=platform_specific,
                schedule_time=product_schedule_time,
                disclosure_phrase=disclosure_phrase,
            )

            # Tally per platform, then record the way `single` records
            for pub_result in publish_results:
                published_to = (
                    [p["platform"] for p in pub_platforms]
                    if pub_result["platform"] == "all"
                    else [pub_result["platform"]]
                )
                for p_name in published_to:
                    platform_results[p_name]["successful"] += 1

            from src.publisher.tracking import record_publish_results

            record_publish_results(
                product_id, publish_results, pub_platforms, batch_config.outputs_dir
            )

            # A scheduled post also goes into the local schedule, so
            # `calendar` sees it. Same write as `single` (#485).
            if product_schedule_time is not None:
                from src.publisher.schedule import record_scheduled_posts

                schedule_writer = schedule_manager or ScheduleManager(
                    schedule_path=durable_state_path(
                        batch_config.outputs_dir, "schedule.json"
                    ),
                    config=published.schedule_config,
                )
                record_scheduled_posts(
                    product_id,
                    publish_results,
                    pub_platforms,
                    product_schedule_time,
                    product_slot_index,
                    schedule_writer,
                )

            # Add to product registry
            try:
                from src.publisher.product_registry import add_to_registry

                add_to_registry(product_id, batch_config.outputs_dir)
            except Exception as reg_exc:
                logger.warning("Failed to update product registry: %s", reg_exc)

            # Check if all platforms were published
            if len(pub_platforms) < len(platforms):
                video_successful = False
                video_errors.append("Some platforms skipped (no account)")

            # Check fail-fast after publish
            if not video_successful and batch_config.fail_fast_publish:
                logger.error("Fail-fast enabled, stopping publishing phase")
                failed += 1
                failed_videos.append(product_id)
                errors.append(
                    {"product_id": product_id, "error": "; ".join(video_errors)}
                )
                break

            # Track video-level success/failure
            if video_successful:
                successful += 1
                logger.info(
                    "[%s/%s] Successfully published %s to all platforms",
                    idx,
                    total_attempted,
                    product_id,
                )

                # Link-in-bio (non-blocking, before cleanup, default ON to
                # match the LinkInBioConfig dataclass and the other paths)
                from src.publisher.link_in_bio.manager import update_link_in_bio_safe

                await update_link_in_bio_safe(
                    product_id,
                    batch_config.outputs_dir,
                    published.link_in_bio_config,
                )

                # Cleanup product directory if configured
                from src.publisher.cleanup import remove_published_product_dir

                remove_published_product_dir(
                    batch_config.outputs_dir, product_id, published.cleanup_config
                )
            else:
                failed += 1
                failed_videos.append(product_id)
                errors.append(
                    {"product_id": product_id, "error": "; ".join(video_errors)}
                )
                logger.warning(
                    "[%s/%s] Partially failed for %s",
                    idx,
                    total_attempted,
                    product_id,
                )

        except Exception as e:
            failed += 1
            failed_videos.append(product_id)
            errors.append({"product_id": product_id, "error": str(e)})
            logger.exception(
                "[%s/%s] Failed to process %s: %s",
                idx,
                total_attempted,
                product_id,
                e,
            )

            if batch_config.fail_fast_publish:
                logger.error("Fail-fast enabled, stopping publishing phase")
                break

        # Apply staggered delay (except after last video)
        if idx < total_attempted:
            # Non-cryptographic random is acceptable for stagger delay
            delay = random.randint(stagger_min, stagger_max)  # noqa: S311
            logger.info(
                "[%s/%s] Waiting %ss before next publish...",
                idx,
                total_attempted,
                delay,
            )
            await asyncio.sleep(delay)

    # Trim the Vercel Blob upload store (non-blocking)
    if successful > 0:
        from src.publisher.blob_retention import run_blob_retention

        if isinstance(publisher, LatePublisher):
            await run_blob_retention(publisher, published.blob_retention_config)

    # Sweep a trailing window for silently-failed legs (non-blocking).
    # Not gated on `successful`: the value is in previous runs' posts.
    # No dry-run guard: main() exits on dry_run before any phase runs.
    from src.publisher.partial_post_sweep import run_delivery_sweep

    if isinstance(publisher, LatePublisher):
        await run_delivery_sweep(publisher, published.delivery_sweep_config)

    # Generate summary
    duration = time.time() - phase_start
    logger.info(
        "Publishing phase complete: %s successful, %s failed, %s skipped in %.1fs",
        successful,
        failed,
        skipped,
        duration,
    )

    return PublishingPhaseSummary(
        total_attempted=total_attempted,
        successful=successful,
        failed=failed,
        skipped=skipped,
        failed_videos=failed_videos,
        skipped_videos=skipped_videos,
        platform_results=platform_results,
        errors=errors,
        duration_sec=duration,
    )
