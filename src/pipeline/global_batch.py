"""Global batch pipeline orchestrator.

This module orchestrates the complete end-to-end workflow from scraping
products to generating and publishing promotional videos in four sequential phases:

1. Scraping Phase: Process product IDs/keywords through Amazon scraper
2. Handoff Phase: Discover products ready for video production
3. Video Production Phase: Generate videos for ready products
4. Publishing Phase: Publish videos to social media platforms (optional)

The orchestrator treats scraper, producer, and publisher as black boxes,
coordinating their execution without modifying their internals.

Usage:
    from src.pipeline.global_batch import GlobalPipelineOrchestrator
    from src.pipeline.config import GlobalBatchConfig

    config = GlobalBatchConfig(...)
    orchestrator = GlobalPipelineOrchestrator(config)
    summary = await orchestrator.run_pipeline()
"""

import logging
import time
from pathlib import Path
from typing import Any

from src.pipeline.config import (
    GlobalBatchConfig,
    PipelineSummary,
    ProductionPhaseSummary,
    PublishingPhaseSummary,
    ScrapingPhaseSummary,
)
from src.scraper.amazon.models import ProductData
from src.video.config_adapter import load_video_config_modular

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Create argument parser for global batch pipeline CLI.

    Returns
    -------
        argparse.ArgumentParser configured with all pipeline arguments

    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Global Batch Pipeline - "
            "End-to-end Amazon product scraping and video production"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape product and create video with fixed profile
  python -m src.pipeline --product-ids B0ABC123 --profile slideshow_images1

  # Scrape keywords and create videos with random profile selection
  python -m src.pipeline --keywords "wireless earbuds" --random-profile \\
      --profile-pool slideshow_images1 video_sequential

  # Batch with filters and fail-fast
  python -m src.pipeline --product-ids B0ABC123 B0DEF456 \\
      --profile slideshow_images1 --fail-fast --debug
        """,
    )

    # Input arguments
    input_group = parser.add_argument_group("Input Configuration")
    input_group.add_argument(
        "--product-ids",
        nargs="+",
        metavar="ASIN",
        help=(
            "Product IDs (ASINs) to scrape and produce videos for "
            "(e.g., B0ABC123 B0DEF456)"
        ),
    )
    input_group.add_argument(
        "--keywords",
        nargs="+",
        metavar="KEYWORD",
        help="Keywords to search for products (e.g., 'wireless earbuds' 'smart watch')",
    )
    input_group.add_argument(
        "--max-products",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of products to scrape per keyword (default: 10)",
    )

    # Scraper filter arguments
    filter_group = parser.add_argument_group("Scraper Filters")
    filter_group.add_argument(
        "--min-price",
        type=float,
        metavar="PRICE",
        help="Minimum price filter (e.g., 10.99)",
    )
    filter_group.add_argument(
        "--max-price",
        type=float,
        metavar="PRICE",
        help="Maximum price filter (e.g., 99.99)",
    )
    filter_group.add_argument(
        "--min-rating",
        type=float,
        metavar="RATING",
        help="Minimum rating filter (1-5 stars, e.g., 4.0)",
    )
    filter_group.add_argument(
        "--prime-only",
        action="store_true",
        help="Filter for Prime eligible items only",
    )

    # Producer arguments
    producer_group = parser.add_argument_group("Video Production Configuration")
    producer_group.add_argument(
        "--profile",
        type=str,
        metavar="NAME",
        help=(
            "Video profile to use for all products "
            "(mutually exclusive with --random-profile)"
        ),
    )
    producer_group.add_argument(
        "--random-profile",
        action="store_true",
        help=(
            "Enable random profile selection per product "
            "(deterministic by product ID). "
            "Mutually exclusive with --profile. "
            "Requires --profile-pool or uses all available profiles."
        ),
    )
    producer_group.add_argument(
        "--profile-pool",
        nargs="+",
        type=str,
        metavar="PROFILE",
        help=(
            "List of profile names for random selection (used with --random-profile). "
            "Example: --profile-pool slideshow_images1 video_sequential"
        ),
    )

    # Common arguments
    common_group = parser.add_argument_group("Common Options")
    common_group.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop pipeline on first failure (default: continue processing)",
    )
    common_group.add_argument(
        "--process-all-products",
        action="store_true",
        help=(
            "Process all products in outputs directory "
            "(default: only products from current scraping run)"
        ),
    )
    common_group.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        metavar="PATH",
        help="Directory for scraper output and producer input (default: outputs)",
    )
    common_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging",
    )

    # Publishing arguments
    publisher_group = parser.add_argument_group("Publishing Configuration")
    publisher_group.add_argument(
        "--skip-publish",
        action="store_true",
        help="Skip publishing phase (default: publish videos to social media)",
    )
    publisher_group.add_argument(
        "--platforms",
        nargs="+",
        choices=["youtube", "tiktok", "instagram"],
        metavar="PLATFORM",
        help=(
            "Platforms to publish to (default: use publisher.yaml default_platforms). "
            "Example: --platforms youtube tiktok"
        ),
    )
    publisher_group.add_argument(
        "--schedule-time",
        type=str,
        metavar="ISO8601",
        help=(
            "Schedule videos for specific time (ISO 8601 format). "
            "Example: --schedule-time '2025-01-20T10:00:00+00:00'"
        ),
    )
    publisher_group.add_argument(
        "--fail-fast-publish",
        action="store_true",
        help="Stop publishing on first failure (default: continue publishing)",
    )

    return parser


class GlobalPipelineOrchestrator:
    """Orchestrates scraping, video production, and publishing phases sequentially.

    Coordinates the complete pipeline from scraping to publishing,
    treating scraper, producer, and publisher as black boxes and managing
    the handoff between phases.

    Attributes
    ----------
        config: Unified pipeline configuration

    """

    def __init__(self, config: GlobalBatchConfig):
        """Initialize orchestrator with unified configuration.

        Args:
        ----
            config: Global batch configuration with scraper and producer settings

        """
        self.config = config

    async def run_pipeline(self) -> PipelineSummary:
        """Execute complete pipeline: scrape → handoff → produce → publish.

        Orchestrates four sequential phases:
        1. Scraping Phase: Scrape products using configured inputs
        2. Handoff Phase: Discover products ready for video production
        3. Video Production Phase: Generate videos for ready products
        4. Publishing Phase: Publish produced videos to social media platforms

        Returns
        -------
            PipelineSummary with aggregated statistics from all phases

        """
        pipeline_start = time.time()

        # Phase 1: Scraping
        logger.info("=" * 80)
        logger.info("SCRAPING PHASE")
        logger.info("=" * 80)
        scraping_summary = await self._execute_scraping_phase()

        # Phase 2: Handoff
        ready_products = self._execute_handoff_phase(
            scraping_summary.successful_products
        )

        # Check if any products are ready
        if not ready_products:
            logger.warning("No products with sufficient media for video production")
            # Return early with empty production summary
            production_summary = ProductionPhaseSummary(
                total_attempted=0,
                successful=0,
                failed=0,
                skipped=0,
                failed_products=[],
                skipped_products=[],
                profile_distribution=None,
                duration_sec=0.0,
            )
            produced_videos: list[tuple[Path, str]] = []
        else:
            # Phase 3: Video Production
            logger.info("=" * 80)
            logger.info("VIDEO PRODUCTION PHASE")
            logger.info("=" * 80)
            production_summary, produced_videos = await self._execute_production_phase(
                ready_products
            )

        # Phase 4: Publishing (conditional)
        publishing_summary = None
        if not self.config.skip_publish and produced_videos:
            logger.info("=" * 80)
            logger.info("PUBLISHING PHASE")
            logger.info("=" * 80)
            publishing_summary = await self._execute_publishing_phase(produced_videos)
        elif self.config.skip_publish:
            logger.info("→ Skipping publishing phase (--skip-publish)")
        else:
            logger.info("→ No videos to publish")

        # Generate final summary
        pipeline_duration = time.time() - pipeline_start
        final_summary = self._generate_final_summary(
            scraping_summary, production_summary, publishing_summary, pipeline_duration
        )

        return final_summary

    async def _execute_scraping_phase(self) -> ScrapingPhaseSummary:
        """Execute scraping phase and return summary.

        Invokes Amazon scraper with configured product IDs and keywords,
        tracks statistics, and generates phase summary.

        Returns
        -------
            ScrapingPhaseSummary with scraping statistics

        Raises
        ------
            Exception: If fail_fast is enabled and scraping fails

        """
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        phase_start = time.time()

        # Combine product IDs and keywords into single input list
        all_inputs = []
        if self.config.product_ids:
            all_inputs.extend(self.config.product_ids)
        if self.config.keywords:
            all_inputs.extend(self.config.keywords)

        total_inputs = len(all_inputs)
        logger.info(f"Scraping {total_inputs} product(s): {', '.join(all_inputs)}")

        # Initialize scraper
        scraper = BotasaurusAmazonScraper(
            debug_override=self.config.debug,
        )

        # Override max_products in scraper config if specified
        if self.config.max_products is not None:
            scraper.amazon_config["max_products"] = self.config.max_products

        # Track statistics
        successful = 0
        failed = 0
        successful_products: list[str] = []
        failed_products: list[str] = []
        total_images = 0
        total_videos = 0

        # Process each input
        for idx, input_item in enumerate(all_inputs, 1):
            logger.info(f"[{idx}/{total_inputs}] Scraping: {input_item}")

            try:
                # Call scraper with single input
                products = scraper.scrape_products(
                    keywords=[input_item], search_params=self.config.scraper_filters
                )

                if products:
                    successful += 1
                    # Track successful product IDs (ASINs)
                    for product in products:
                        if hasattr(product, "asin") and product.asin:
                            successful_products.append(product.asin)
                        # Count media for this product
                        if hasattr(product, "images") and product.images:
                            total_images += len(product.images)
                        if hasattr(product, "videos") and product.videos:
                            total_videos += len(product.videos)
                    logger.info(
                        f"✓ [{idx}/{total_inputs}] Successfully scraped {input_item}"
                    )
                else:
                    failed += 1
                    failed_products.append(input_item)
                    logger.warning(f"✗ [{idx}/{total_inputs}] No data for {input_item}")

                    if self.config.fail_fast:
                        logger.error("Fail-fast enabled, stopping scraping phase")
                        break

            except Exception as e:
                failed += 1
                failed_products.append(input_item)
                logger.error(
                    f"✗ [{idx}/{total_inputs}] Failed to scrape {input_item}: {e}"
                )

                if self.config.fail_fast:
                    logger.error("Fail-fast enabled, stopping scraping phase")
                    raise

        # Generate summary
        duration = time.time() - phase_start
        media_stats = {"total_images": total_images, "total_videos": total_videos}

        logger.info(
            f"Scraping phase complete: {successful} successful, "
            f"{failed} failed in {duration:.1f}s"
        )

        return ScrapingPhaseSummary(
            total_attempted=total_inputs,
            successful=successful,
            failed=failed,
            successful_products=successful_products,
            failed_products=failed_products,
            media_stats=media_stats,
            duration_sec=duration,
        )

    def _execute_handoff_phase(
        self, scraped_product_ids: list[str]
    ) -> list[tuple[Path, ProductData]]:
        """Discover products ready for video production.

        Scans outputs directory for products with data.json and filters
        by scraped product IDs unless process_all_products is enabled.

        Args:
        ----
            scraped_product_ids: List of ASINs scraped in current run

        Returns:
        -------
            List of (product_dir, ProductData) tuples for ready products

        """
        from src.video.producer.cli import discover_products_for_batch

        logger.info("Discovering products ready for video production...")

        # Use existing discover_products_for_batch function
        all_products = discover_products_for_batch(self.config.outputs_dir)

        logger.info(
            f"Found {len(all_products)} product(s) with data.json in "
            f"{self.config.outputs_dir}"
        )

        # Filter by scraped products unless process_all_products is enabled
        if self.config.process_all_products:
            ready_products = all_products
            logger.info("→ Processing all products in outputs directory")
        else:
            # Only process products scraped in current run
            scraped_set = set(scraped_product_ids)
            ready_products = [
                (path, data)
                for path, data in all_products
                if hasattr(data, "asin") and data.asin in scraped_set
            ]
            logger.info(
                f"→ Processing {len(ready_products)} product(s) "
                f"from current scraping run"
            )

        # Log transition
        if ready_products:
            logger.info(
                f"→ {len(ready_products)} product(s) ready for video production"
            )
        else:
            logger.warning("→ No products ready for video production")

        return ready_products

    async def _execute_production_phase(
        self, products: list[tuple[Path, ProductData]]
    ) -> tuple[ProductionPhaseSummary, list[tuple[Path, str]]]:
        """Execute video production phase and return summary with produced videos.

        Processes each product through video pipeline with configured profile,
        supports both fixed and random profile modes, tracks statistics.

        Args:
        ----
            products: List of (product_dir, ProductData) tuples to process

        Returns:
        -------
            Tuple of (ProductionPhaseSummary, list of (video_path, product_id) tuples)

        """
        import asyncio
        import os

        import aiohttp

        from src.video.config import load_video_config
        from src.video.producer.orchestration import create_video_for_product
        from src.video.producer.utils import (
            ProfileUsageTracker,
            select_profile_for_product,
        )

        phase_start = time.time()

        # Load video configuration
        config = load_video_config_modular()

        # Build secrets dict from environment variables
        secrets = {
            name: os.getenv(name)
            for name in [
                config.llm_settings.api_key_env_var,
                config.stock_media_settings.pexels_api_key_env_var,
                config.audio_settings.freesound_api_key_env_var,
                "GOOGLE_APPLICATION_CREDENTIALS",
                config.audio_settings.freesound_client_id_env_var,
                config.audio_settings.freesound_client_secret_env_var,
                config.audio_settings.freesound_refresh_token_env_var,
            ]
            if name and os.getenv(name)
        }

        # Initialize profile tracking if random mode
        profile_tracker: ProfileUsageTracker | None = None
        if self.config.random_profile:
            profile_tracker = ProfileUsageTracker()

        # Track statistics
        successful = 0
        failed = 0
        skipped = 0
        failed_products: list[str] = []
        skipped_products: list[str] = []
        produced_videos: list[tuple[Path, str]] = []

        total_products = len(products)
        logger.info(f"Processing {total_products} product(s) for video production")

        # Create HTTP session for API calls
        async with aiohttp.ClientSession() as session:
            for idx, (_product_dir, product) in enumerate(products, 1):
                product_id = product.asin or product.title or f"product_{idx}"

                # Select profile for this product
                if self.config.random_profile:
                    # Random profile selection (deterministic by product ID)
                    assert self.config.profile_pool is not None
                    assert profile_tracker is not None
                    current_profile = select_profile_for_product(
                        product_id=product_id,
                        profile_pool=self.config.profile_pool,
                        config=config,
                    )
                    profile_tracker.record_usage(current_profile)
                    logger.info(
                        f"[{idx}/{total_products}] Processing {product_id} "
                        f"with profile '{current_profile}'"
                    )
                else:
                    # Fixed profile mode
                    assert self.config.profile is not None
                    current_profile = self.config.profile
                    logger.info(
                        f"[{idx}/{total_products}] Processing product: {product_id}"
                    )

                try:
                    # Call video producer with timeout
                    result_path = await asyncio.wait_for(
                        create_video_for_product(
                            config=config,
                            product=product,
                            profile_name=current_profile,
                            secrets=secrets,
                            session=session,
                            debug_mode=self.config.debug,
                            clean_run=False,
                            debug_step_target=None,
                            cli_overrides=None,
                        ),
                        timeout=config.pipeline_timeout_sec,
                    )

                    if result_path:
                        successful += 1
                        produced_videos.append((result_path, product_id))
                        logger.info(
                            f"✓ [{idx}/{total_products}] Successfully created video "
                            f"for {product_id}"
                        )
                    else:
                        # Producer returned None - treat as skipped
                        skipped += 1
                        skipped_products.append(product_id)
                        logger.warning(
                            f"⊘ [{idx}/{total_products}] Skipped {product_id} "
                            f"(insufficient media)"
                        )

                except TimeoutError:
                    failed += 1
                    failed_products.append(product_id)
                    logger.error(
                        f"✗ [{idx}/{total_products}] Pipeline timed out "
                        f"after {config.pipeline_timeout_sec}s for {product_id}"
                    )

                    if self.config.fail_fast:
                        logger.error("Fail-fast enabled, stopping production phase")
                        break

                except Exception as e:
                    failed += 1
                    failed_products.append(product_id)
                    logger.error(
                        f"✗ [{idx}/{total_products}] "
                        f"Failed to process {product_id}: {e}",
                        exc_info=True,
                    )

                    if self.config.fail_fast:
                        logger.error("Fail-fast enabled, stopping production phase")
                        raise

        # Generate summary
        duration = time.time() - phase_start
        profile_distribution = profile_tracker.get_counts() if profile_tracker else None

        logger.info(
            f"Production phase complete: {successful} successful, "
            f"{failed} failed, {skipped} skipped in {duration:.1f}s"
        )

        summary = ProductionPhaseSummary(
            total_attempted=total_products,
            successful=successful,
            failed=failed,
            skipped=skipped,
            failed_products=failed_products,
            skipped_products=skipped_products,
            profile_distribution=profile_distribution,
            duration_sec=duration,
        )

        return summary, produced_videos

    async def _execute_publishing_phase(
        self, produced_videos: list[tuple[Path, str]]
    ) -> PublishingPhaseSummary:
        """Execute publishing phase for produced videos.

        Loads publisher configuration, authenticates, and publishes each video
        to configured platforms with staggered delays and comprehensive error tracking.

        Args:
        ----
            produced_videos: List of (video_path, product_id) tuples from production

        Returns:
        -------
            PublishingPhaseSummary with per-platform publishing statistics

        """
        import asyncio
        import os
        import random
        from datetime import datetime

        import yaml

        from src.publisher import PublisherProvider, create_publisher
        from src.publisher.metadata import load_platform_metadata
        from src.publisher.models import Platform

        phase_start = time.time()

        # Load publisher configuration
        config_path = Path("config/publisher.yaml")
        publisher_config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                publisher_config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded publisher config from {config_path}")
        else:
            logger.warning(f"Publisher config not found: {config_path}")

        # Apply CLI overrides to configuration
        platforms_to_publish = self.config.platforms or publisher_config.get(
            "default_platforms", ["youtube", "tiktok", "instagram"]
        )
        platforms = [Platform(p.lower()) for p in platforms_to_publish]

        # Determine scheduling strategy with 3-tier precedence:
        # 1. Explicit CLI/YAML schedule_time (highest priority)
        # 2. Auto-schedule via recurring_schedule if immediate_publish=false
        # 3. Publish immediately (scheduled_time=None)

        schedule_time = None

        # Priority 1: Explicit schedule time from CLI or YAML
        schedule_time_str = self.config.schedule_time or publisher_config.get(
            "schedule_time"
        )
        if schedule_time_str:
            schedule_time = datetime.fromisoformat(
                schedule_time_str.replace("Z", "+00:00")
            )
            logger.info(f"Using explicit schedule time: {schedule_time}")
        else:
            # Priority 2: Auto-schedule if configured
            immediate_publish = publisher_config.get("immediate_publish", True)
            recurring_config = publisher_config.get("recurring_schedule", {})
            recurring_enabled = recurring_config.get("enabled", False)

            logger.debug(
                f"Scheduling config: immediate_publish={immediate_publish}, "
                f"recurring_enabled={recurring_enabled}"
            )

            if not immediate_publish and recurring_enabled:
                # Use recurring schedule to find next available slot
                from src.publisher.models import RecurringSlot
                from src.publisher.schedule import ScheduleManager

                logger.info("Auto-scheduling: Finding next available slot...")

                # Parse recurring slots from config
                slots_config = recurring_config.get("slots", [])
                timezone_str = recurring_config.get("timezone", "UTC")

                if not slots_config:
                    logger.warning(
                        "recurring_schedule.enabled=true but no slots defined. "
                        "Publishing immediately."
                    )
                else:
                    try:
                        # Convert config slots to RecurringSlot objects
                        slots = [
                            RecurringSlot(
                                day_of_week=s["day_of_week"],
                                time=s["time"],
                                timezone=timezone_str,
                            )
                            for s in slots_config
                        ]

                        # Initialize schedule manager
                        schedule_manager = ScheduleManager(
                            schedule_path=self.config.outputs_dir / "schedule.json"
                        )

                        # Find next available UNOCCUPIED slot
                        from datetime import UTC

                        now = datetime.now(UTC)

                        # Fetch existing posts to check occupied slots
                        logger.debug("Checking occupied slots via API...")
                        occupied_slot_times: set[datetime] = set()

                        # Initialize publisher to query existing posts
                        api_key = os.getenv("LATE_API_KEY")
                        vercel_token = os.getenv("LATE_VERCEL_TOKEN")
                        if api_key:
                            logger.debug(
                                f"Temp publisher init: vercel_token="
                                f"{'set' if vercel_token else 'NOT SET'}"
                            )
                            temp_publisher = create_publisher(
                                provider=PublisherProvider.LATE,
                                api_key=api_key,
                                vercel_token=vercel_token,
                            )

                            try:
                                # Authenticate to access API
                                await temp_publisher.authenticate()

                                # Fetch all posts (scheduled + published)
                                api_posts = await temp_publisher.list_posts()
                                logger.debug(
                                    f"Found {len(api_posts)} existing posts on API"
                                )

                                # Build set of occupied slot times
                                for api_post in api_posts:
                                    scheduled_time = api_post.get("scheduledFor")
                                    if not scheduled_time:
                                        continue

                                    # Parse scheduled time
                                    if isinstance(scheduled_time, str):
                                        time_str = scheduled_time.replace("+00:00", "")
                                        scheduled_dt = datetime.fromisoformat(time_str)
                                        if scheduled_dt.tzinfo is None:
                                            scheduled_dt = scheduled_dt.replace(
                                                tzinfo=UTC
                                            )
                                    else:
                                        scheduled_dt = scheduled_time

                                    # Normalize to slot precision (minute)
                                    slot_time = scheduled_dt.replace(
                                        second=0, microsecond=0
                                    )
                                    occupied_slot_times.add(slot_time)
                                    logger.debug(
                                        f"Occupied slot: "
                                        f"{slot_time.strftime('%Y-%m-%d %H:%M %Z')}"
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to fetch occupied slots: {e}")

                        # Find next unoccupied slot
                        max_attempts = len(slots) * 8  # Check up to 8 weeks ahead
                        slot_index = 0

                        for _attempt in range(max_attempts):
                            next_slot_time, slot_index = schedule_manager.get_next_slot(
                                slots, after=now, slot_index=slot_index
                            )

                            # Normalize to slot precision for comparison
                            normalized_slot = next_slot_time.replace(
                                second=0, microsecond=0
                            )

                            if normalized_slot not in occupied_slot_times:
                                # Found unoccupied slot
                                schedule_time = next_slot_time
                                logger.info(
                                    f"Auto-scheduled to slot #{slot_index}: "
                                    f"{schedule_time.strftime(
                                        '%A, %Y-%m-%d %H:%M:%S %Z'
                                    )}"
                                )
                                break
                            else:
                                logger.debug(
                                    f"Slot #{slot_index} occupied, trying next slot..."
                                )
                                # Move to next slot for next iteration
                                now = next_slot_time
                                slot_index = (slot_index + 1) % len(slots)
                        else:
                            logger.warning(
                                "All slots occupied for next 8 weeks. "
                                "Publishing immediately."
                            )

                    except Exception as e:
                        logger.warning(
                            f"Failed to auto-schedule: {e}. Publishing immediately."
                        )
            elif immediate_publish:
                logger.info("immediate_publish=true: Publishing immediately")
            else:
                logger.info("recurring_schedule.enabled=false: Publishing immediately")

        stagger_min = publisher_config.get("stagger_delay_min", 30)
        stagger_max = publisher_config.get("stagger_delay_max", 60)

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

        # Initialize publisher
        try:
            api_key = os.getenv("LATE_API_KEY")
            if not api_key:
                raise ValueError("LATE_API_KEY environment variable not set")

            vercel_token = os.getenv("LATE_VERCEL_TOKEN")
            logger.debug(
                f"Publisher init: api_key={'set' if api_key else 'NOT SET'}, "
                f"vercel_token={'set' if vercel_token else 'NOT SET'}"
            )

            publisher = create_publisher(
                provider=PublisherProvider.LATE,
                api_key=api_key,
                vercel_token=vercel_token,
            )

            # Authenticate
            logger.info("Authenticating with publisher...")
            await publisher.authenticate()
            logger.info("✓ Authentication successful")

            # Get connected accounts
            accounts = await publisher.get_accounts()
            logger.info(f"Found {len(accounts)} connected account(s)")

        except Exception as e:
            logger.error(f"Failed to initialize publisher: {e}", exc_info=True)
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

        # Publish each video
        for idx, (video_path, product_id) in enumerate(produced_videos, 1):
            logger.info(f"[{idx}/{total_attempted}] Publishing video for {product_id}")

            video_successful = True
            video_errors = []

            try:
                # Upload video once (reuse media_id for all platforms)
                logger.info(f"[{idx}/{total_attempted}] Uploading video...")
                media_id = await publisher.upload_media(video_path)
                logger.info(f"[{idx}/{total_attempted}] Upload complete: {media_id}")

                # Publish to each platform
                for platform in platforms:
                    try:
                        logger.info(
                            f"[{idx}/{total_attempted}] "
                            f"Publishing to {platform.value}..."
                        )

                        # Load platform-specific metadata
                        metadata = load_platform_metadata(
                            product_id, platform, self.config.outputs_dir
                        )

                        if not metadata:
                            logger.warning(
                                f"[{idx}/{total_attempted}] Skipping {platform.value}: "
                                f"metadata not found"
                            )
                            video_errors.append(
                                f"Missing metadata for {platform.value}"
                            )
                            platform_results[platform.value]["failed"] += 1
                            video_successful = False
                            continue

                        # Get account ID for this platform
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
                                f"[{idx}/{total_attempted}] Skipping {platform.value}: "
                                f"no connected account"
                            )
                            video_errors.append(
                                f"No connected account for {platform.value}"
                            )
                            platform_results[platform.value]["failed"] += 1
                            video_successful = False
                            continue

                        # Format content
                        content = metadata.format_content()

                        # Publish
                        await publisher.publish(
                            media_id=media_id,
                            platforms=[
                                {
                                    "platform": platform.value,
                                    "account_id": platform_account["account_id"],
                                }
                            ],
                            content=content,
                            scheduled_time=schedule_time,
                        )

                        platform_results[platform.value]["successful"] += 1
                        logger.info(
                            f"✓ [{idx}/{total_attempted}] Published to {platform.value}"
                        )

                    except Exception as e:
                        platform_results[platform.value]["failed"] += 1
                        video_successful = False
                        error_msg = f"{platform.value}: {e}"
                        video_errors.append(error_msg)
                        logger.error(
                            f"✗ [{idx}/{total_attempted}] Failed to publish to "
                            f"{platform.value}: {e}"
                        )

                        if self.config.fail_fast_publish:
                            logger.error("Fail-fast enabled, stopping publishing phase")
                            raise

                # Track video-level success/failure
                if video_successful:
                    successful += 1
                    logger.info(
                        f"✓ [{idx}/{total_attempted}] "
                        f"Successfully published {product_id} to all platforms"
                    )

                    # Cleanup product directory if configured
                    cleanup_config = publisher_config.get("cleanup", {})
                    cleanup_enabled = cleanup_config.get("enabled", False)

                    if cleanup_enabled:
                        require_all_platforms = cleanup_config.get(
                            "require_all_platforms", True
                        )

                        # Only cleanup if published to ALL platforms
                        # (already verified above)
                        if require_all_platforms:
                            # Find product directory
                            product_dir = self.config.outputs_dir / product_id

                            if product_dir.exists():
                                try:
                                    import shutil

                                    logger.info(
                                        f"Cleaning up product directory: {product_dir}"
                                    )
                                    shutil.rmtree(product_dir)
                                    logger.info(f"✓ Removed {product_dir}")
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to cleanup {product_dir}: {e}"
                                    )
                else:
                    failed += 1
                    failed_videos.append(product_id)
                    errors.append(
                        {"product_id": product_id, "error": "; ".join(video_errors)}
                    )
                    logger.warning(
                        f"⚠ [{idx}/{total_attempted}] Partially failed for {product_id}"
                    )

            except Exception as e:
                failed += 1
                failed_videos.append(product_id)
                errors.append({"product_id": product_id, "error": str(e)})
                logger.error(
                    f"✗ [{idx}/{total_attempted}] Failed to process {product_id}: {e}",
                    exc_info=True,
                )

                if self.config.fail_fast_publish:
                    logger.error("Fail-fast enabled, stopping publishing phase")
                    break

            # Apply staggered delay (except after last video)
            if idx < total_attempted:
                # Non-cryptographic random is acceptable for stagger delay
                delay = random.randint(stagger_min, stagger_max)  # noqa: S311
                logger.info(
                    f"[{idx}/{total_attempted}] Waiting {delay}s before next publish..."
                )
                await asyncio.sleep(delay)

        # Generate summary
        duration = time.time() - phase_start
        logger.info(
            f"Publishing phase complete: {successful} successful, "
            f"{failed} failed, {skipped} skipped in {duration:.1f}s"
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

    def _generate_final_summary(
        self,
        scraping: ScrapingPhaseSummary,
        production: ProductionPhaseSummary,
        publishing: PublishingPhaseSummary | None,
        total_duration: float,
    ) -> PipelineSummary:
        """Generate end-to-end pipeline summary.

        Calculates derived statistics from phase summaries:
        - End-to-end success: Products scraped, produced, and published (if enabled)
        - Partial success: Products scraped but not fully produced/published
        - Total failures: Products that failed in any phase

        Logs formatted summary with all phase statistics and
        end-to-end metrics.

        Args:
        ----
            scraping: Scraping phase summary
            production: Video production phase summary
            publishing: Publishing phase summary (None if --skip-publish)
            total_duration: Total pipeline duration in seconds

        Returns:
        -------
            PipelineSummary with aggregated end-to-end statistics

        """
        # Calculate end-to-end metrics
        if publishing:
            # Full pipeline: scraped, produced, AND published successfully
            end_to_end_success = publishing.successful
            partial_success = scraping.successful - publishing.total_attempted
            total_failures = scraping.failed + production.failed + publishing.failed
        else:
            # Publishing skipped: scraped AND produced successfully
            end_to_end_success = production.successful
            partial_success = scraping.successful - production.total_attempted
            total_failures = scraping.failed + production.failed

        summary = PipelineSummary(
            scraping=scraping,
            production=production,
            publishing=publishing,
            end_to_end_success=end_to_end_success,
            partial_success=partial_success,
            total_failures=total_failures,
            total_duration_sec=total_duration,
        )

        # Log formatted summary
        logger.info(summary.format())

        return summary


async def main():
    """Main CLI entry point for global batch pipeline.

    Parses arguments, loads configuration, validates settings,
    executes pipeline, and handles errors gracefully.
    """
    import asyncio
    import sys

    from dotenv import load_dotenv

    from src.pipeline.config import (
        load_global_batch_config,
        validate_global_batch_config,
    )
    from src.utils.logging_setup import setup_debug_logging
    from src.video.config import load_video_config

    # Load environment variables from .env file
    load_dotenv()

    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set up logging early
    log_file = Path("outputs/logs/global_pipeline.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_debug_logging(
        log_file=log_file,
        debug_mode=args.debug,
        verbose=args.debug,
        component_name="GlobalPipeline",
    )

    logger.info("=" * 80)
    logger.info("GLOBAL BATCH PIPELINE STARTING")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")

    try:
        # Load configuration with CLI > YAML > defaults precedence
        logger.info("Loading configuration...")
        config = load_global_batch_config(args)

        # Load video configuration for validation
        video_config = load_video_config_modular()

        # Validate configuration
        logger.info("Validating configuration...")
        validate_global_batch_config(config, video_config)

        logger.info("Configuration validated successfully")
        logger.info(
            f"Inputs: {len(config.product_ids or [])} product IDs, "
            f"{len(config.keywords or [])} keywords"
        )

        if config.profile:
            logger.info(f"Profile: {config.profile} (fixed)")
        elif config.random_profile:
            pool_info = (
                ", ".join(config.profile_pool)
                if config.profile_pool
                else "all available"
            )
            logger.info(f"Profile: random selection from [{pool_info}]")

        logger.info(f"Outputs directory: {config.outputs_dir}")
        logger.info(f"Fail-fast: {config.fail_fast}")

        # Execute pipeline
        orchestrator = GlobalPipelineOrchestrator(config)
        await orchestrator.run_pipeline()

        # Success
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Complete log saved to: {log_file}")

        # Exit with success code
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 80)
        logger.warning("PIPELINE INTERRUPTED BY USER")
        logger.warning("=" * 80)
        logger.warning(f"Partial log saved to: {log_file}")
        sys.exit(130)  # Standard exit code for SIGINT

    except ValueError as e:
        # Configuration or validation errors
        logger.error("=" * 80)
        logger.error("CONFIGURATION ERROR")
        logger.error("=" * 80)
        logger.error(str(e))
        logger.error(f"Complete log saved to: {log_file}")
        sys.exit(1)

    except Exception as e:
        # Unexpected errors
        logger.critical("=" * 80)
        logger.critical("PIPELINE FAILED WITH ERROR")
        logger.critical("=" * 80)
        logger.critical(f"Error: {e}", exc_info=True)
        logger.critical(f"Complete log saved to: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
