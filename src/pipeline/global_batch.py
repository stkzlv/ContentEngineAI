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
    PipelinePhase,
    PipelineState,
    PipelineSummary,
    ProductionPhaseSummary,
    PublishingPhaseSummary,
    ScrapingPhaseSummary,
    clear_pipeline_state,
    load_pipeline_state,
    save_pipeline_state,
)
from src.pipeline.webhooks import WebhookConfig, WebhookNotifier
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
        default=None,
        metavar="N",
        help="Maximum number of products to scrape per keyword (default: from config)",
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
    common_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume interrupted pipeline from last checkpoint. "
            "Skips already-completed products and phases."
        ),
    )
    common_group.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate configuration and show planned actions without executing. "
            "Displays products to scrape, profiles to use, and platforms to publish."
        ),
    )
    common_group.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        metavar="FORMAT",
        help=(
            "Output format for pipeline summary: 'text' (default) for human-readable, "
            "'json' for machine-readable with all statistics and timestamps."
        ),
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

    Supports resume capability through state persistence, allowing interrupted
    pipelines to continue from the last successful checkpoint.

    Attributes
    ----------
        config: Unified pipeline configuration
        state: Pipeline state for tracking progress and enabling resume
        webhook_notifier: Optional webhook notifier for event notifications

    """

    def __init__(
        self,
        config: GlobalBatchConfig,
        state: PipelineState | None = None,
        webhook_notifier: WebhookNotifier | None = None,
    ):
        """Initialize orchestrator with unified configuration.

        Args:
        ----
            config: Global batch configuration with scraper and producer settings
            state: Optional pipeline state for resume capability
            webhook_notifier: Optional webhook notifier for pipeline events

        """
        self.config = config
        self.state = state or PipelineState.create_new(config)
        self.webhook_notifier = webhook_notifier

    def _save_state(self) -> None:
        """Save current pipeline state to disk."""
        save_pipeline_state(self.state, self.config.outputs_dir)

    async def _notify_webhook(
        self,
        event: str,
        data: dict[str, Any],
    ) -> None:
        """Send webhook notification (non-blocking).

        Args:
        ----
            event: Event type (e.g., "phase.complete")
            data: Event data payload

        """
        if self.webhook_notifier and self.webhook_notifier.is_ready():
            try:
                await self.webhook_notifier.notify(event, data)
            except Exception as e:
                # Never let webhook failures affect the pipeline
                logger.warning(f"Webhook notification failed: {e}")

    def display_execution_plan(self, video_config: Any) -> None:
        """Display planned execution without running the pipeline.

        Shows configuration validation results and planned actions for each phase.

        Args:
        ----
            video_config: Video configuration for profile information

        """
        import os

        import yaml

        separator = "=" * 80
        section = "-" * 40

        print(f"\n{separator}")
        print("DRY RUN - EXECUTION PLAN")
        print(f"{separator}\n")

        # Phase 1: Scraping Plan
        print(f"{section}")
        print("PHASE 1: SCRAPING")
        print(f"{section}")

        if self.config.product_ids:
            print(f"  Product IDs to scrape: {len(self.config.product_ids)}")
            for pid in self.config.product_ids[:10]:  # Show first 10
                print(f"    - {pid}")
            if len(self.config.product_ids) > 10:
                print(f"    ... and {len(self.config.product_ids) - 10} more")

        if self.config.keywords:
            print(f"  Keywords to search: {len(self.config.keywords)}")
            for kw in self.config.keywords[:5]:  # Show first 5
                print(f'    - "{kw}" (max {self.config.max_products} products)')
            if len(self.config.keywords) > 5:
                print(f"    ... and {len(self.config.keywords) - 5} more")

        # Show filters
        filters = self.config.scraper_filters
        active_filters = []
        if filters.min_price is not None:
            active_filters.append(f"min_price=${filters.min_price}")
        if filters.max_price is not None:
            active_filters.append(f"max_price=${filters.max_price}")
        if filters.min_rating is not None:
            active_filters.append(f"min_rating={filters.min_rating}★")
        if filters.prime_only:
            active_filters.append("prime_only=true")

        if active_filters:
            print(f"  Filters: {', '.join(active_filters)}")
        else:
            print("  Filters: none")

        print()

        # Phase 2: Handoff (informational)
        print(f"{section}")
        print("PHASE 2: HANDOFF")
        print(f"{section}")
        print("  Action: Discover scraped products with sufficient media")
        print("  Validation: Check data.json exists and has images/videos")
        print()

        # Phase 3: Video Production Plan
        print(f"{section}")
        print("PHASE 3: VIDEO PRODUCTION")
        print(f"{section}")

        if self.config.profile:
            print("  Profile mode: Fixed")
            print(f"  Profile: {self.config.profile}")

            # Show profile details if available
            if self.config.profile in video_config.video_profiles:
                profile = video_config.video_profiles[self.config.profile]
                print(f"    - Strategy: {profile.strategy}")
                print(f"    - Resolution: {profile.resolution}")
        elif self.config.random_profile:
            print("  Profile mode: Random selection")
            pool = self.config.profile_pool or list(video_config.video_profiles.keys())
            print(f"  Profile pool ({len(pool)} profiles):")
            for p in pool[:5]:
                print(f"    - {p}")
            if len(pool) > 5:
                print(f"    ... and {len(pool) - 5} more")
        else:
            print("  Profile mode: Not configured")
            print("  WARNING: No profile specified - will fail at runtime")

        print()

        # Phase 4: Publishing Plan
        print(f"{section}")
        print("PHASE 4: PUBLISHING")
        print(f"{section}")

        if self.config.skip_publish:
            print("  Status: SKIPPED (--skip-publish)")
        else:
            # Load publisher config to show platforms
            config_path = Path("config/publisher.yaml")
            publisher_config: dict[str, Any] = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    publisher_config = yaml.safe_load(f) or {}

            platforms = self.config.platforms or publisher_config.get(
                "default_platforms", ["youtube", "tiktok", "instagram"]
            )
            print(f"  Platforms: {', '.join(platforms)}")

            # Check API key
            api_key = os.getenv("LATE_API_KEY")
            if api_key:
                print("  API Key: ✓ LATE_API_KEY is set")
            else:
                print("  API Key: ✗ LATE_API_KEY NOT SET (will fail at runtime)")

            # Show scheduling mode
            if self.config.schedule_time:
                print(f"  Scheduling: Explicit time ({self.config.schedule_time})")
            else:
                immediate = publisher_config.get("immediate_publish", True)
                recurring = publisher_config.get("recurring_schedule", {}).get(
                    "enabled", False
                )
                if not immediate and recurring:
                    print("  Scheduling: Auto-schedule (find next available slot)")
                else:
                    print("  Scheduling: Immediate publish")

        print()

        # Common Options
        print(f"{section}")
        print("COMMON OPTIONS")
        print(f"{section}")
        print(f"  Outputs directory: {self.config.outputs_dir}")
        print(f"  Fail-fast: {self.config.fail_fast}")
        print(f"  Debug mode: {self.config.debug}")

        print()
        print(f"{separator}")
        print("DRY RUN COMPLETE - No actions were executed")
        print(f"{separator}\n")

    async def run_pipeline(self) -> PipelineSummary:
        """Execute complete pipeline: scrape → handoff → produce → publish.

        Orchestrates four sequential phases:
        1. Scraping Phase: Scrape products using configured inputs
        2. Handoff Phase: Discover products ready for video production
        3. Video Production Phase: Generate videos for ready products
        4. Publishing Phase: Publish produced videos to social media platforms

        Supports resume capability: if --resume flag is set and state file exists,
        skips already-completed phases and products.

        Returns
        -------
            PipelineSummary with aggregated statistics from all phases

        """
        from dataclasses import asdict

        pipeline_start = time.time()

        # Log resume status
        if self.config.resume:
            logger.info(f"Resuming pipeline run: {self.state.run_id}")
            logger.info(f"  Started: {self.state.started_at}")
            completed = ", ".join(self.state.completed_phases) or "none"
            logger.info(f"  Completed phases: {completed}")

        # Save initial state
        self._save_state()

        # Phase 1: Scraping
        if self.state.is_phase_completed(PipelinePhase.SCRAPING):
            logger.info("=" * 80)
            logger.info("SCRAPING PHASE (SKIPPED - Already completed)")
            logger.info("=" * 80)
            # Reconstruct summary from state
            scraping_summary = ScrapingPhaseSummary(
                **self.state.scraping_summary  # type: ignore[arg-type]
            )
            logger.info(
                f"→ Using cached results: {scraping_summary.successful} successful, "
                f"{scraping_summary.failed} failed"
            )
        else:
            logger.info("=" * 80)
            logger.info("SCRAPING PHASE")
            logger.info("=" * 80)
            self.state.advance_phase(PipelinePhase.SCRAPING)
            self._save_state()

            scraping_summary = await self._execute_scraping_phase()

            # Update state with scraping results
            self.state.scraping_completed_products = (
                scraping_summary.successful_products
            )
            self.state.scraping_failed_products = scraping_summary.failed_products
            self.state.scraping_summary = asdict(scraping_summary)
            self.state.mark_phase_complete(PipelinePhase.SCRAPING)
            self._save_state()

            # Notify webhook of scraping phase completion
            await self._notify_webhook(
                "phase.complete",
                {"phase": "scraping", "summary": asdict(scraping_summary)},
            )

        # Phase 2: Handoff
        self.state.advance_phase(PipelinePhase.HANDOFF)
        self._save_state()

        ready_products = self._execute_handoff_phase(
            scraping_summary.successful_products
        )

        self.state.mark_phase_complete(PipelinePhase.HANDOFF)
        self._save_state()

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
            if self.state.is_phase_completed(PipelinePhase.PRODUCTION):
                logger.info("=" * 80)
                logger.info("VIDEO PRODUCTION PHASE (SKIPPED - Already completed)")
                logger.info("=" * 80)
                # Reconstruct summary from state
                production_summary = ProductionPhaseSummary(
                    **self.state.production_summary  # type: ignore[arg-type]
                )
                msg = (
                    f"→ Using cached results: "
                    f"{production_summary.successful} successful, "
                    f"{production_summary.failed} failed"
                )
                logger.info(msg)
                # Reconstruct produced_videos from state
                produced_videos = [
                    (self.config.outputs_dir / pid / "video.mp4", pid)
                    for pid in self.state.production_completed_products
                ]
            else:
                logger.info("=" * 80)
                logger.info("VIDEO PRODUCTION PHASE")
                logger.info("=" * 80)
                self.state.advance_phase(PipelinePhase.PRODUCTION)
                self._save_state()

                (
                    production_summary,
                    produced_videos,
                ) = await self._execute_production_phase(ready_products)

                # Update state with production results
                self.state.production_completed_products = [
                    pid for _, pid in produced_videos
                ]
                self.state.production_failed_products = (
                    production_summary.failed_products
                )
                self.state.production_skipped_products = (
                    production_summary.skipped_products
                )
                self.state.production_summary = asdict(production_summary)
                self.state.mark_phase_complete(PipelinePhase.PRODUCTION)
                self._save_state()

                # Notify webhook of production phase completion
                await self._notify_webhook(
                    "phase.complete",
                    {"phase": "production", "summary": asdict(production_summary)},
                )

        # Phase 4: Publishing (conditional)
        publishing_summary = None
        if not self.config.skip_publish and produced_videos:
            if self.state.is_phase_completed(PipelinePhase.PUBLISHING):
                logger.info("=" * 80)
                logger.info("PUBLISHING PHASE (SKIPPED - Already completed)")
                logger.info("=" * 80)
                # Reconstruct summary from state
                publishing_summary = PublishingPhaseSummary(
                    **self.state.publishing_summary  # type: ignore[arg-type]
                )
                msg = (
                    f"→ Using cached results: "
                    f"{publishing_summary.successful} successful, "
                    f"{publishing_summary.failed} failed"
                )
                logger.info(msg)
            else:
                logger.info("=" * 80)
                logger.info("PUBLISHING PHASE")
                logger.info("=" * 80)
                self.state.advance_phase(PipelinePhase.PUBLISHING)
                self._save_state()

                publishing_summary = await self._execute_publishing_phase(
                    produced_videos
                )

                # Update state with publishing results
                self.state.publishing_completed_products = [
                    pid
                    for pid in self.state.production_completed_products
                    if pid not in publishing_summary.failed_videos
                ]
                self.state.publishing_failed_products = publishing_summary.failed_videos
                self.state.publishing_summary = asdict(publishing_summary)
                self.state.mark_phase_complete(PipelinePhase.PUBLISHING)
                self._save_state()

                # Notify webhook of publishing phase completion
                await self._notify_webhook(
                    "phase.complete",
                    {"phase": "publishing", "summary": asdict(publishing_summary)},
                )
        elif self.config.skip_publish:
            logger.info("→ Skipping publishing phase (--skip-publish)")
        else:
            logger.info("→ No videos to publish")

        # Mark pipeline as completed
        self.state.advance_phase(PipelinePhase.COMPLETED)
        self._save_state()

        # Generate final summary
        pipeline_duration = time.time() - pipeline_start
        final_summary = self._generate_final_summary(
            scraping_summary, production_summary, publishing_summary, pipeline_duration
        )

        # Notify webhook of pipeline completion
        await self._notify_webhook(
            "pipeline.complete",
            {"summary": final_summary.to_dict()},
        )

        # Clear state file on successful completion
        clear_pipeline_state(self.config.outputs_dir)
        logger.info("Pipeline completed successfully - state file cleared")

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

        # Helper function to publish to a single platform (for parallel execution)
        async def publish_to_platform(
            platform: Platform,
            media_id: str,
            product_id: str,
            idx: int,
        ) -> tuple[Platform, bool, str | None]:
            """Publish to a single platform with error isolation.

            Returns
            -------
                Tuple of (platform, success, error_message)

            """
            try:
                # Load platform-specific metadata
                metadata = load_platform_metadata(
                    product_id, platform, self.config.outputs_dir
                )

                if not metadata:
                    return (platform, False, f"Missing metadata for {platform.value}")

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
                    return (
                        platform,
                        False,
                        f"No connected account for {platform.value}",
                    )

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

                logger.info(
                    f"✓ [{idx}/{total_attempted}] Published to {platform.value}"
                )
                return (platform, True, None)

            except Exception as e:
                logger.error(
                    f"✗ [{idx}/{total_attempted}] Failed to publish to "
                    f"{platform.value}: {e}"
                )
                return (platform, False, f"{platform.value}: {e}")

        # Publish each video
        for idx, (video_path, product_id) in enumerate(produced_videos, 1):
            logger.info(f"[{idx}/{total_attempted}] Publishing video for {product_id}")

            video_successful = True
            video_errors: list[str] = []

            try:
                # Upload video once (reuse media_id for all platforms)
                logger.info(f"[{idx}/{total_attempted}] Uploading video...")
                media_id = await publisher.upload_media(video_path)
                logger.info(f"[{idx}/{total_attempted}] Upload complete: {media_id}")

                # Publish to all platforms concurrently
                logger.info(
                    f"[{idx}/{total_attempted}] Publishing to "
                    f"{len(platforms)} platform(s) in parallel..."
                )

                # Create tasks for parallel platform publishing
                publish_tasks = [
                    publish_to_platform(platform, media_id, product_id, idx)
                    for platform in platforms
                ]

                # Execute all platform publishes concurrently with error isolation
                results = await asyncio.gather(*publish_tasks, return_exceptions=True)

                # Process results and update statistics
                for result in results:
                    if isinstance(result, Exception):
                        # Unexpected exception during task execution
                        video_successful = False
                        error_msg = f"Unexpected error: {result}"
                        video_errors.append(error_msg)
                        logger.error(f"[{idx}/{total_attempted}] {error_msg}")
                    else:
                        platform, success, error = result  # type: ignore[misc]
                        if success:
                            platform_results[platform.value]["successful"] += 1
                        else:
                            platform_results[platform.value]["failed"] += 1
                            video_successful = False
                            if error:
                                video_errors.append(error)
                                logger.warning(
                                    f"[{idx}/{total_attempted}] {platform.value}: "
                                    f"{error}"
                                )

                # Check fail-fast after all platforms processed
                if not video_successful and self.config.fail_fast_publish:
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
        logger.info(f"Resume mode: {config.resume}")
        logger.info(f"Dry-run mode: {config.dry_run}")

        # Handle dry-run mode
        if config.dry_run:
            orchestrator = GlobalPipelineOrchestrator(config)
            orchestrator.display_execution_plan(video_config)
            logger.info("Dry-run completed - exiting without execution")
            sys.exit(0)

        # Handle resume mode
        state = None
        if config.resume:
            state = load_pipeline_state(config.outputs_dir)
            if state:
                logger.info(f"Resuming pipeline run: {state.run_id}")
                logger.info(f"  Current phase: {state.current_phase.value}")
                logger.info(
                    f"  Completed phases: "
                    f"{', '.join(state.completed_phases) or 'none'}"
                )
            else:
                logger.warning("No state file found - starting fresh pipeline")

        # Track start time for JSON output
        from datetime import UTC, datetime

        pipeline_started_at = datetime.now(UTC).isoformat()

        # Load webhook configuration
        import yaml

        from src.pipeline.webhooks import load_webhook_config

        webhook_notifier = None
        try:
            with open("config/pipeline.yaml") as f:
                yaml_config = yaml.safe_load(f) or {}
            global_batch_yaml = yaml_config.get("global_batch", {})
            webhook_config = load_webhook_config(global_batch_yaml)
            if webhook_config.is_configured():
                webhook_notifier = WebhookNotifier(webhook_config)
                if webhook_notifier.is_ready():
                    logger.info(f"Webhook notifications enabled: {webhook_config.url}")
                else:
                    logger.warning("Webhook URL configured but invalid")
        except FileNotFoundError:
            logger.debug("No pipeline.yaml found - webhooks disabled")
        except Exception as e:
            logger.warning(f"Failed to load webhook config: {e}")

        # Execute pipeline
        orchestrator = GlobalPipelineOrchestrator(
            config, state=state, webhook_notifier=webhook_notifier
        )
        summary = await orchestrator.run_pipeline()

        # Output summary in requested format
        if config.output_format == "json":
            # JSON output to stdout for machine parsing
            print(summary.to_json(started_at=pipeline_started_at))
        else:
            # Text output (already logged by _generate_final_summary)
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
        logger.warning("To resume from last checkpoint, run with --resume flag")
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
        logger.critical("To resume from last checkpoint, run with --resume flag")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
