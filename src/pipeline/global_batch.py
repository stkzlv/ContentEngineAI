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
        help=(
            "Maximum total products to collect across all keywords "
            "(default: from config)"
        ),
    )
    input_group.add_argument(
        "--products-per-keyword",
        type=int,
        default=None,
        metavar="N",
        help="Maximum products to scrape per individual keyword (default: from config)",
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
    producer_group.add_argument(
        "--voice-profile",
        type=str,
        metavar="NAME",
        help="Override voice profile selection for all products.",
    )
    producer_group.add_argument(
        "--script-template",
        type=str,
        metavar="NAME",
        help="Override script template for all products (name without .md).",
    )
    producer_group.add_argument(
        "--pillar",
        type=str,
        metavar="NAME",
        help=(
            "Content pillar for the run (e.g. value, novelty, utility). "
            "Filters template pool and prepends the pillar preamble to the "
            "LLM prompt. Without this flag, all templates are eligible."
        ),
    )
    producer_group.add_argument(
        "--subtitle-engine",
        choices=["ffmpeg", "pycaps"],
        help=(
            "Subtitle rendering engine. 'ffmpeg' (default) = SRT/ASS via "
            "libass. 'pycaps' = animated captions burned post-assembly. "
            "Install the optional group first: "
            "`poetry install --with pycaps`."
        ),
    )
    producer_group.add_argument(
        "--pycaps-template",
        type=str,
        metavar="NAME",
        help=(
            "Pycaps template name (e.g. word-focus, hype, minimalist). "
            "Forces this template for every product by clearing the template "
            "pool. To use a custom multi-entry pool, pass "
            "--pycaps-template-pool instead."
        ),
    )
    producer_group.add_argument(
        "--pycaps-template-pool",
        nargs="+",
        type=str,
        metavar="NAME",
        help=(
            "Pool of pycaps templates for deterministic per-product selection. "
            "Example: --pycaps-template-pool word-focus hype vibrant"
        ),
    )
    producer_group.add_argument(
        "--pycaps-renderer",
        choices=["css", "pictex"],
        help=(
            "Pycaps renderer backend. 'css' = Playwright+Chromium (default). "
            "'pictex' = browserless Skia path (lighter, no Chromium dep)."
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
        "--clean",
        action="store_true",
        help=(
            "Remove product directories from outputs before running. "
            "With --product-ids, removes only those products."
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
    publisher_group.add_argument(
        "--platform-specific",
        action="store_true",
        help=(
            "Create separate posts per platform with optimized metadata. "
            "Default: single post for all platforms."
        ),
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
        video_config: Any = None,
    ):
        """Initialize orchestrator with unified configuration.

        Args:
        ----
            config: Global batch configuration with scraper and producer settings
            state: Optional pipeline state for resume capability
            webhook_notifier: Optional webhook notifier for pipeline events
            video_config: Video configuration for profile-aware scraper validation

        """
        self.config = config
        self.state = state or PipelineState.create_new(config)
        self.webhook_notifier = webhook_notifier
        self.video_config = video_config

    def _save_state(self) -> None:
        """Save current pipeline state to disk."""
        save_pipeline_state(self.state, self.config.outputs_dir)

    def _build_cli_overrides(self) -> dict[str, Any] | None:
        """Build CLI overrides dict from pipeline config.

        Keys here must match the dotted override keys consumed by
        ``VideoConfig.get_profile_merged_settings`` — keep this in sync with
        ``src.video.producer.cli._build_cli_overrides`` per the
        Module/Batch Alignment Rule in CLAUDE.md.
        """
        overrides: dict[str, Any] = {}
        if self.config.voice_profile:
            overrides["voice_profile"] = self.config.voice_profile
        if self.config.script_template:
            overrides["script_template"] = self.config.script_template
        if self.config.pillar:
            overrides["pillar"] = self.config.pillar
        if self.config.subtitle_engine:
            overrides["subtitle_settings.subtitle_engine"] = self.config.subtitle_engine
        if self.config.pycaps_template:
            overrides["subtitle_settings.pycaps.template_name"] = (
                self.config.pycaps_template
            )
            # Clear the pool so the deterministic selector falls through to
            # template_name. Without this, a multi-entry pool would still win
            # via md5 hash and silently ignore --pycaps-template.
            overrides["subtitle_settings.pycaps.template_pool"] = []
        if self.config.pycaps_template_pool:
            # Explicit --pycaps-template-pool wins over the implicit clear
            # above when both flags are passed.
            overrides["subtitle_settings.pycaps.template_pool"] = (
                self.config.pycaps_template_pool
            )
        if self.config.pycaps_renderer:
            overrides["subtitle_settings.pycaps.renderer"] = self.config.pycaps_renderer
        return overrides or None

    def _resolve_profile_uses_videos(self) -> bool | None:
        """Check if the target profile(s) use scraped videos.

        Returns False if any profile in the selection doesn't use videos
        (strictest requirement wins). Returns None when no profile info
        is available.
        """
        if not self.video_config:
            return None

        if self.config.profile:
            profile = self.video_config.video_profiles.get(self.config.profile)
            if profile:
                return bool(profile.use_scraped_videos)
            return None

        if self.config.random_profile:
            from src.video.producer.utils import EXCLUDED_RANDOM_PROFILES

            pool = self.config.profile_pool or [
                p
                for p in self.video_config.video_profiles
                if p not in EXCLUDED_RANDOM_PROFILES
            ]
            for name in pool:
                profile = self.video_config.video_profiles.get(name)
                if profile and not profile.use_scraped_videos:
                    return False
            return True

        return None

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
                kw_limit = self.config.products_per_keyword
                print(f'    - "{kw}" (max {kw_limit} per keyword)')
            if len(self.config.keywords) > 5:
                print(f"    ... and {len(self.config.keywords) - 5} more")
            print(f"  Global limit: {self.config.max_products} products total")

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
            from src.video.producer.utils import EXCLUDED_RANDOM_PROFILES

            print("  Profile mode: Random selection")
            pool = self.config.profile_pool or [
                p
                for p in video_config.video_profiles
                if p not in EXCLUDED_RANDOM_PROFILES
            ]
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

        Uses a two-phase approach to avoid launching a separate Chrome process
        per keyword:
          1. Batch browser phase: one Chrome session scrapes ALL inputs
          2. Per-keyword post-processing: download media, validate, apply limits

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
        logger.info(f"Scraping {total_inputs} input(s): {', '.join(all_inputs)}")
        logger.info(
            f"Limits: {self.config.products_per_keyword} per keyword, "
            f"{self.config.max_products} total"
        )

        # Initialize scraper with profile-aware validation
        profile_uses_videos = self._resolve_profile_uses_videos()
        if profile_uses_videos is not None:
            logger.info(
                "Scraper validation aligned with profile: videos %s",
                "enabled" if profile_uses_videos else "disabled (image-only)",
            )
        scraper = BotasaurusAmazonScraper(
            debug_override=self.config.debug,
            profile_uses_videos=profile_uses_videos,
        )

        scraper.amazon_config["max_products"] = self.config.products_per_keyword

        # Track statistics
        inputs_processed = 0
        inputs_failed = 0
        successful_products: list[str] = []
        failed_inputs: list[str] = []
        total_images = 0
        total_videos = 0

        # --- Phase 1: batch browser scrape (one Chrome for all inputs) ---
        logger.info("Phase 1: batch browser scrape (%d inputs)", total_inputs)
        try:
            batch_results = scraper.scrape_batch_browser(
                all_inputs, search_params=self.config.scraper_filters
            )
        except Exception as e:
            logger.error("Batch browser scrape failed: %s", e)
            if self.config.fail_fast:
                raise
            batch_results = []

        # Build a lookup so we can iterate in original order
        results_by_input: dict[str, list[dict]] = {}
        for entry in batch_results:
            results_by_input[entry["input"]] = entry.get("products", [])

        # --- Phase 2: per-keyword post-processing ---
        logger.info("Phase 2: media download and validation")
        for idx, input_item in enumerate(all_inputs, 1):
            if len(successful_products) >= self.config.max_products:
                logger.info(
                    f"Reached max_products limit ({self.config.max_products}). "
                    f"Stopping with {len(all_inputs) - idx + 1} inputs remaining."
                )
                break

            remaining = self.config.max_products - len(successful_products)
            per_input_limit = min(self.config.products_per_keyword, remaining)
            scraper.amazon_config["max_products"] = per_input_limit

            collected = f"{len(successful_products)}/{self.config.max_products}"
            logger.info(
                f"[{idx}/{total_inputs}] Processing: {input_item} "
                f"(limit: {per_input_limit}, collected: {collected})"
            )

            raw_products = results_by_input.get(input_item, [])
            if not raw_products:
                inputs_failed += 1
                failed_inputs.append(input_item)
                logger.warning(f"✗ [{idx}/{total_inputs}] No data for {input_item}")
                if self.config.fail_fast:
                    logger.error("Fail-fast enabled, stopping scraping phase")
                    break
                continue

            try:
                products = scraper.process_raw_products(
                    raw_products,
                    target_download_count=per_input_limit,
                )

                # Retry with additional search pages if not enough
                # validated products (keywords only, not ASINs/URLs)
                is_keyword = not scraper._is_asin(input_item) and not scraper._is_url(
                    input_item
                )
                if is_keyword and len(products) < per_input_limit:
                    from src.scraper.amazon.config import CONFIG as SCRAPER_CONFIG

                    batch_cfg = SCRAPER_CONFIG.get(
                        "global_settings",
                        {},
                    ).get("batch_processing", {})
                    max_retry_pages = batch_cfg.get("max_retry_pages", 5)
                    page = 2
                    while len(products) < per_input_limit and page <= max_retry_pages:
                        remaining = per_input_limit - len(products)
                        logger.info(
                            "Retrying %s page %d (%d/%d validated)",
                            input_item,
                            page,
                            len(products),
                            per_input_limit,
                        )
                        extra_results = scraper.scrape_batch_browser(
                            [input_item],
                            search_params=self.config.scraper_filters,
                            start_page=page,
                        )
                        extra_raw = []
                        for entry in extra_results:
                            extra_raw.extend(entry.get("products", []))
                        if not extra_raw:
                            logger.info(
                                "No more results for %s on page %d",
                                input_item,
                                page,
                            )
                            break
                        extra_products = scraper.process_raw_products(
                            extra_raw,
                            target_download_count=remaining,
                        )
                        products.extend(extra_products)
                        page += 1

                if products:
                    inputs_processed += 1
                    kw_pillar = self.config.keyword_pillar_map.get(input_item)
                    for product in products:
                        if kw_pillar:
                            product.pillar = kw_pillar
                        if hasattr(product, "asin") and product.asin:
                            successful_products.append(product.asin)
                        if hasattr(product, "images") and product.images:
                            total_images += len(product.images)
                        if hasattr(product, "videos") and product.videos:
                            total_videos += len(product.videos)
                    logger.info(
                        f"✓ [{idx}/{total_inputs}] Found {len(products)} "
                        f"product(s) for {input_item}"
                    )
                else:
                    inputs_failed += 1
                    failed_inputs.append(input_item)
                    logger.warning(
                        f"✗ [{idx}/{total_inputs}] No valid products for {input_item}"
                    )
                    if self.config.fail_fast:
                        logger.error("Fail-fast enabled, stopping scraping phase")
                        break

            except Exception as e:
                inputs_failed += 1
                failed_inputs.append(input_item)
                logger.error(
                    f"✗ [{idx}/{total_inputs}] Failed to process {input_item}: {e}"
                )
                if self.config.fail_fast:
                    logger.error("Fail-fast enabled, stopping scraping phase")
                    raise

        # Generate summary
        duration = time.time() - phase_start
        media_stats = {"total_images": total_images, "total_videos": total_videos}

        logger.info(
            f"Scraping phase complete: {len(successful_products)} products from "
            f"{inputs_processed} inputs ({inputs_failed} failed) in {duration:.1f}s"
        )

        return ScrapingPhaseSummary(
            total_attempted=total_inputs,
            successful=inputs_processed,
            failed=inputs_failed,
            successful_products=successful_products,
            failed_products=failed_inputs,
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
        from src.video.producer.orchestration import (
            create_video_for_product,
            failed_step_from_result,
        )
        from src.video.producer.utils import (
            ProfileUsageTracker,
            select_profile_for_product,
        )

        phase_start = time.time()

        # Load video configuration
        config = load_video_config_modular()

        # Build secrets dict from environment variables
        secret_names = [
            config.llm_settings.api_key_env_var,
            config.stock_media_settings.pexels_api_key_env_var,
            config.audio_settings.freesound_api_key_env_var,
            "GOOGLE_APPLICATION_CREDENTIALS",
            config.audio_settings.freesound_client_id_env_var,
            config.audio_settings.freesound_client_secret_env_var,
            config.audio_settings.freesound_refresh_token_env_var,
        ]
        # Add env vars from audio provider configs
        for ap in config.audio_settings.audio_providers:
            for key in ("client_id_env_var", "api_key_env_var"):
                env_var = ap.settings.get(key)
                if env_var and env_var not in secret_names:
                    secret_names.append(env_var)
        if config.llm_settings.fallback_provider:
            secret_names.append(config.llm_settings.fallback_provider.api_key_env_var)
        secrets = {
            name: os.getenv(name) for name in secret_names if name and os.getenv(name)
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
                    cli_overrides = self._build_cli_overrides()
                    product_pillar = getattr(product, "pillar", None)
                    if product_pillar and not self.config.pillar:
                        if cli_overrides is None:
                            cli_overrides = {}
                        cli_overrides.setdefault("pillar", product_pillar)

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
                            cli_overrides=cli_overrides,
                        ),
                        timeout=config.pipeline_timeout_sec,
                    )

                    failed_step = failed_step_from_result(result_path)
                    if result_path == "SKIPPED":
                        skipped += 1
                        skipped_products.append(product_id)
                        logger.warning(
                            "[%d/%d] Skipped %s (insufficient media)",
                            idx,
                            total_products,
                            product_id,
                        )
                    elif failed_step is not None:
                        failed += 1
                        failed_products.append(product_id)
                        logger.error(
                            "[%d/%d] Failed to produce %s: "
                            "pipeline step '%s' failed",
                            idx,
                            total_products,
                            product_id,
                            failed_step,
                        )
                        if self.config.fail_fast:
                            logger.error("Fail-fast enabled, stopping production phase")
                            break
                    elif result_path:
                        successful += 1
                        produced_videos.append((result_path, product_id))
                        logger.info(
                            f"✓ [{idx}/{total_products}] Successfully created video "
                            f"for {product_id}"
                        )
                    else:
                        # The producer never returns None; a None here means
                        # the result contract was broken. Count as failed so
                        # the run doesn't underreport.
                        failed += 1
                        failed_products.append(product_id)
                        logger.error(
                            "[%d/%d] Failed to produce %s: "
                            "producer returned no result",
                            idx,
                            total_products,
                            product_id,
                        )
                        if self.config.fail_fast:
                            logger.error("Fail-fast enabled, stopping production phase")
                            break

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
        #
        # For auto-schedule with multiple products, each product gets its
        # own slot. The scheduling context is prepared here, and
        # _find_next_schedule_slot() is called per-product in the loop.

        schedule_time = None
        auto_schedule_ctx: dict | None = None

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

                logger.info("Auto-scheduling: preparing slot context...")

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

                        from datetime import UTC

                        # Fetch existing posts to check occupied slots
                        logger.debug("Checking occupied slots via API...")
                        occupied_slot_times: set[datetime] = set()

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
                                await temp_publisher.authenticate()

                                api_posts = await temp_publisher.list_posts()
                                logger.debug(
                                    f"Found {len(api_posts)} existing posts on API"
                                )

                                for api_post in api_posts:
                                    scheduled_time = api_post.get("scheduledFor")
                                    if not scheduled_time:
                                        continue

                                    if isinstance(scheduled_time, str):
                                        time_str = scheduled_time.replace("+00:00", "")
                                        scheduled_dt = datetime.fromisoformat(time_str)
                                        if scheduled_dt.tzinfo is None:
                                            scheduled_dt = scheduled_dt.replace(
                                                tzinfo=UTC
                                            )
                                    else:
                                        scheduled_dt = scheduled_time

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

                        # Store context for per-product slot finding
                        auto_schedule_ctx = {
                            "slots": slots,
                            "schedule_manager": schedule_manager,
                            "occupied_slot_times": occupied_slot_times,
                        }

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

            # Parse first_comment config from YAML
            from src.publisher.models import FirstCommentConfig

            fc_section = publisher_config.get("first_comment", {})
            try:
                first_comment_config = (
                    FirstCommentConfig(**fc_section) if fc_section else None
                )
            except (ValueError, TypeError):
                first_comment_config = None

            publisher = create_publisher(
                provider=PublisherProvider.LATE,
                api_key=api_key,
                vercel_token=vercel_token,
                first_comment_config=first_comment_config,
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
            video_errors: list[str] = []

            try:
                # Upload video once
                logger.info(f"[{idx}/{total_attempted}] Uploading video...")
                media_id = await publisher.upload_media(video_path)
                logger.info(f"[{idx}/{total_attempted}] Upload complete: {media_id}")

                # Build platforms list (validate accounts upfront)
                pub_platforms: list[dict[str, str]] = []
                for platform in platforms:
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
                            "[%d/%d] No account for %s, skipping",
                            idx,
                            total_attempted,
                            platform.value,
                        )
                        platform_results[platform.value]["failed"] += 1
                        continue
                    pub_platforms.append(
                        {
                            "platform": platform.value,
                            "account_id": platform_account["account_id"],
                        }
                    )

                if not pub_platforms:
                    raise ValueError("No valid platform accounts found")

                # Find per-product schedule slot if auto-scheduling
                product_schedule_time = schedule_time
                if auto_schedule_ctx is not None:
                    from datetime import UTC

                    ctx_slots = auto_schedule_ctx["slots"]
                    ctx_mgr = auto_schedule_ctx["schedule_manager"]
                    ctx_occupied = auto_schedule_ctx["occupied_slot_times"]

                    now = datetime.now(UTC)
                    max_attempts = len(ctx_slots) * 8
                    slot_index = 0

                    for _attempt in range(max_attempts):
                        next_slot_time, slot_index = ctx_mgr.get_next_slot(
                            ctx_slots, after=now, slot_index=slot_index
                        )
                        normalized_slot = next_slot_time.replace(
                            second=0, microsecond=0
                        )
                        if normalized_slot not in ctx_occupied:
                            product_schedule_time = next_slot_time
                            ctx_occupied.add(normalized_slot)
                            logger.info(
                                f"Auto-scheduled {product_id} to slot "
                                f"#{slot_index}: "
                                f"{next_slot_time.strftime(
                                    '%A, %Y-%m-%d %H:%M:%S %Z'
                                )}"
                            )
                            break
                        else:
                            logger.debug(f"Slot #{slot_index} occupied, trying next...")
                            now = next_slot_time
                            slot_index = (slot_index + 1) % len(ctx_slots)
                    else:
                        logger.warning(
                            f"All slots occupied for {product_id}. "
                            "Publishing immediately."
                        )
                        product_schedule_time = None

                # Publish (unified or platform-specific mode)
                from src.publisher.publish_modes import publish_product

                platform_specific = (
                    self.config.platform_specific_content
                    or publisher_config.get("use_platform_specific_content", False)
                )

                disc_raw = publisher_config.get("affiliate_disclosure", {})
                disclosure_phrase = (
                    disc_raw.get("phrase") if disc_raw.get("enabled", True) else None
                )
                publish_results = await publish_product(
                    publisher=publisher,
                    media_id=media_id,
                    product_id=product_id,
                    platforms=pub_platforms,
                    outputs_dir=self.config.outputs_dir,
                    platform_specific=platform_specific,
                    schedule_time=product_schedule_time,
                    disclosure_phrase=disclosure_phrase,
                )

                # Process results and record publish
                from src.publisher.tracking import record_publish

                for pub_result in publish_results:
                    result_data = pub_result["result"]
                    post_id = str(result_data.get("post_id", ""))
                    logger.info(
                        "✓ [%d/%d] Published: post_id=%s, status=%s",
                        idx,
                        total_attempted,
                        post_id,
                        result_data.get("status"),
                    )

                    if pub_result["platform"] == "all":
                        # Unified mode: one post_id for all platforms
                        for p_info in pub_platforms:
                            platform_results[p_info["platform"]]["successful"] += 1
                            record_publish(
                                product_id,
                                p_info["platform"],
                                post_id,
                                self.config.outputs_dir,
                            )
                    else:
                        # Platform-specific mode: per-platform post_id
                        p_name = pub_result["platform"]
                        platform_results[p_name]["successful"] += 1
                        record_publish(
                            product_id,
                            p_name,
                            post_id,
                            self.config.outputs_dir,
                        )

                # Add to product registry
                try:
                    from src.publisher.product_registry import add_to_registry

                    add_to_registry(product_id, self.config.outputs_dir)
                except Exception as reg_exc:
                    logger.warning("Failed to update product registry: %s", reg_exc)

                # Check if all platforms were published
                if len(pub_platforms) < len(platforms):
                    video_successful = False
                    video_errors.append("Some platforms skipped (no account)")

                # Check fail-fast after publish
                if not video_successful and self.config.fail_fast_publish:
                    logger.error("Fail-fast enabled, stopping publishing phase")
                    failed += 1
                    failed_videos.append(product_id)
                    errors.append(
                        {
                            "product_id": product_id,
                            "error": "; ".join(video_errors),
                        }
                    )
                    break

                # Track video-level success/failure
                if video_successful:
                    successful += 1
                    logger.info(
                        f"✓ [{idx}/{total_attempted}] "
                        f"Successfully published {product_id} to all platforms"
                    )

                    # Link-in-bio (non-blocking, before cleanup, default ON to
                    # match the LinkInBioConfig dataclass and the other paths)
                    from src.publisher.link_in_bio.manager import (
                        update_link_in_bio_safe,
                    )
                    from src.publisher.models import LinkInBioConfig

                    link_in_bio_cfg = publisher_config.get("link_in_bio", {})
                    await update_link_in_bio_safe(
                        product_id,
                        self.config.outputs_dir,
                        LinkInBioConfig(
                            enabled=link_in_bio_cfg.get("enabled", True),
                            provider=link_in_bio_cfg.get("provider", "lnkbio"),
                            max_links=link_in_bio_cfg.get("max_links", 0),
                            max_title_length=link_in_bio_cfg.get(
                                "max_title_length", 80
                            ),
                        ),
                    )

                    # Cleanup product directory if configured
                    cleanup_config = publisher_config.get("cleanup", {})
                    cleanup_enabled = cleanup_config.get("enabled", False)

                    if cleanup_enabled:
                        require_all_platforms = cleanup_config.get(
                            "require_all_platforms", True
                        )

                        # Only cleanup if published to ALL platforms
                        if require_all_platforms:
                            product_dir = self.config.outputs_dir / product_id

                            if product_dir.exists():
                                try:
                                    import shutil

                                    logger.info(
                                        "Cleaning up product directory: "
                                        f"{product_dir}"
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
                        {
                            "product_id": product_id,
                            "error": "; ".join(video_errors),
                        }
                    )
                    logger.warning(
                        f"⚠ [{idx}/{total_attempted}] "
                        f"Partially failed for {product_id}"
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

        # Trim the Vercel Blob upload store (non-blocking)
        if successful > 0:
            from src.publisher.blob_retention import run_blob_retention
            from src.publisher.late.client import LatePublisher
            from src.publisher.models import BlobRetentionConfig

            br_section = publisher_config.get("blob_retention", {})
            try:
                retention_policy = (
                    BlobRetentionConfig(**br_section) if br_section else None
                )
            except (ValueError, TypeError):
                retention_policy = None
            if isinstance(publisher, LatePublisher):
                await run_blob_retention(publisher, retention_policy)

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

        # Handle clean mode
        if config.clean:
            import re
            import shutil

            asin_pattern = re.compile(r"^([A-Z0-9]{10}|TEST[A-Z0-9]+)$")
            outputs = config.outputs_dir

            if outputs.exists():
                if config.product_ids:
                    for pid in config.product_ids:
                        prod_dir = outputs / pid
                        if prod_dir.is_dir():
                            shutil.rmtree(prod_dir)
                            logger.info("Cleaned product directory: %s", prod_dir)
                else:
                    for item in outputs.iterdir():
                        if item.is_dir() and asin_pattern.match(item.name):
                            shutil.rmtree(item)
                            logger.info("Cleaned product directory: %s", item)

        # Handle dry-run mode
        if config.dry_run:
            orchestrator = GlobalPipelineOrchestrator(config, video_config=video_config)
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
            config,
            state=state,
            webhook_notifier=webhook_notifier,
            video_config=video_config,
        )
        summary = await orchestrator.run_pipeline()

        # Exit code reflects whether the run did what was asked: non-zero when
        # no product completed end-to-end, so CI, cron, and wrappers checking
        # $? see the failure instead of a false success.
        exit_code = summary.exit_code()

        # Output summary in requested format
        if config.output_format == "json":
            # JSON output to stdout for machine parsing
            print(summary.to_json(started_at=pipeline_started_at))
        else:
            # Text output (already logged by _generate_final_summary)
            logger.info("=" * 80)
            if exit_code:
                logger.error(
                    "PIPELINE FAILED: no products completed end-to-end "
                    "(%d failed, %d skipped)",
                    summary.total_failures,
                    summary.production.skipped,
                )
            elif summary.total_failures > 0:
                logger.warning(
                    "PIPELINE COMPLETED WITH FAILURES: " "%d succeeded, %d failed",
                    summary.end_to_end_success,
                    summary.total_failures,
                )
            else:
                logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Complete log saved to: {log_file}")

        sys.exit(exit_code)

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

    finally:
        # Clean up HTTP connection pool to avoid "Unclosed connector" warnings
        from src.utils.connection_pool import close_global_pool

        await close_global_pool()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
