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
import re
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
from src.scraper.base.keyword_pillars import pillar_for as keyword_pillar_for
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
    # Same names and semantics as the producer CLI, per the Module/Batch
    # Alignment Rule. A topic run skips the scraping phase: there is no listing
    # behind it, so the input is the record rather than a search for one.
    input_group.add_argument(
        "--topic",
        metavar="TITLE",
        help=(
            "Render a video about a topic instead of a scraped product. "
            "Skips scraping; the record is built from the title."
        ),
    )
    input_group.add_argument(
        "--topic-description",
        metavar="TEXT",
        help="Source material the script is written from, for --topic.",
    )
    input_group.add_argument(
        "--topic-keywords",
        metavar="TERMS",
        help=(
            "Comma-separated stock media search terms for this topic, e.g. "
            "'wifi router, home network'."
        ),
    )
    input_group.add_argument(
        "--topics-file",
        type=Path,
        metavar="FILE",
        help=(
            "YAML list of topics to render, each with title, optional "
            "description and optional keywords."
        ),
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
        "--subtitle-format",
        choices=["srt", "ass"],
        help=(
            "Subtitle format: srt or ass (with animations). The pycaps engine "
            "ignores it, and the bundled YAML default is pycaps, so pair this "
            "with --subtitle-engine ffmpeg to have it apply."
        ),
    )
    producer_group.add_argument(
        "--subtitle-engine",
        choices=["ffmpeg", "pycaps"],
        help=(
            "Subtitle rendering engine. The bundled YAML selects pycaps. "
            "'ffmpeg' = SRT/ASS via "
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
            "Pycaps renderer backend. 'css' = Playwright+Chromium (default, "
            "the only production-safe option). 'pictex' = browserless Skia "
            "path; PREVIEW ONLY, it renders words with no gaps between them."
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
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero when any product was lost, to a failure or a "
            "skip, not only when none succeeded (default: a partial loss "
            "exits 0)"
        ),
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
        "--force",
        action="store_true",
        help=(
            "Render and publish products already recorded as published. "
            "By default the batch skips them before the render, not after."
        ),
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


# A run directory is a scraped product (ASIN-shaped, or a TEST fixture) or a
# topic. Topics were absent, so `--clean` walked past them and the dry-run plan
# under-reported what the run would remove.
_RUN_DIR_PATTERN = re.compile(r"^([A-Z0-9]{10}|TEST[A-Z0-9]+|topic-[a-z0-9-]+)$")


def resumed_record_kinds(config: "GlobalBatchConfig") -> tuple[bool, bool]:
    """What kinds of record a `--resume` is picking up: (topics, products).

    Topics are not persisted: the identifier carries a one-way digest of the
    title, so the specs cannot be read back out of the state. The ids can, and
    the prefix tells the two kinds apart.

    Both halves are needed, not just the first. A run can carry topics and
    scraped products together, and the config's own `keywords` cannot settle it
    -- a resume inherits the configured keywords whatever it is resuming, and
    the completed scraping phase already ignored them. Reading the state is the
    only way to know whether this resume has products in it.

    Separated from `main` so it can be tested. Left inline it was the one
    piece of this feature no test touched, and deleting it left the suite
    green while every resumed topic rendered under a product profile.
    """
    if not config.resume or config.topics:
        return (False, False)
    from src.video.producer.topic_input import TOPIC_ID_PREFIX

    saved = load_pipeline_state(config.outputs_dir)
    if saved is None:
        return (False, False)

    ids = saved.scraping_completed_products or []
    topics = any(pid.startswith(TOPIC_ID_PREFIX) for pid in ids)
    products = any(not pid.startswith(TOPIC_ID_PREFIX) for pid in ids)
    return (topics, products)


def apply_resume_record_kinds(config: "GlobalBatchConfig") -> None:
    """Stamp what a `--resume` is picking up onto the config.

    Both flags, not just the first. `topics_resume` narrows the profile pool
    to stock-sourced profiles, and `resume_has_products` is what stops that
    narrowing on a resume that also carries scraped products -- which would
    otherwise render them from generic footage, ignoring the photography
    scraped for them.

    A function rather than two lines in `main` so the values can be tested.
    Inline, the only reachable guard was an AST check that an assignment
    existed, which passes just as well when the assignment is a constant.
    """
    resuming_topics, resuming_products = resumed_record_kinds(config)
    if not resuming_topics:
        return

    logger.info("Resuming a topics run (recognised from saved state)")
    config.topics_resume = True
    config.resume_has_products = resuming_products


def _named_run_ids(config: "GlobalBatchConfig") -> list[str]:
    """The run directories this invocation named, if it named any.

    Topics are a named input exactly like product ids, and were falling into
    the unnamed branch below, which removes every run directory in `outputs/`.
    A `--topic X --clean` run would delete every scraped product the machine
    held, along with any rendered-but-unpublished video in them.

    Keywords name nothing: which products they produce is not known until the
    search runs, so a run carrying them is a sweep and the answer is the empty
    list. That has to be checked first, and both other kinds have to be
    unioned. Returning the first non-empty kind meant a run with a topic and
    keywords -- which the bundled config now produces with no flags at all --
    named only the topic, and `--clean` silently spared every product
    directory the operator asked it to remove.
    """
    from src.video.producer.topic_input import topic_product_id

    if config.keywords:
        return []

    named = list(config.product_ids)
    named += [topic_product_id(spec.title) for spec in config.topics]
    return named


def _clean_targets(outputs_dir: Path, product_ids: list[str] | None) -> list[Path]:
    """The product directories `--clean` would remove.

    Shared with the dry-run plan so the preview cannot describe something
    other than what the run does.
    """
    if not outputs_dir.exists():
        return []
    if product_ids:
        # Deduplicated: the caller may name a product twice, and hoisting
        # the selection ahead of the deletion means the second occurrence
        # would try to remove a directory the first one already took.
        return [
            outputs_dir / pid
            for pid in dict.fromkeys(product_ids)
            if (outputs_dir / pid).is_dir()
        ]
    return sorted(
        item
        for item in outputs_dir.iterdir()
        if item.is_dir() and _RUN_DIR_PATTERN.match(item.name)
    )


def _merge_scraping_summaries(
    *summaries: ScrapingPhaseSummary | None,
) -> ScrapingPhaseSummary:
    """Fold the topic and scrape phases into the one summary the state holds.

    Both write into the same directory shape and both feed the same handoff, so
    downstream reads one list of prepared ids. Reported as a single phase
    because that is what resume, the saved state and the phase summaries
    already understand.

    Duration is summed rather than maxed: the phases run one after the other.
    """
    present = [s for s in summaries if s is not None]
    if len(present) == 1:
        return present[0]

    media: dict[str, int] = {}
    for summary in present:
        for key, value in summary.media_stats.items():
            media[key] = media.get(key, 0) + value

    return ScrapingPhaseSummary(
        total_attempted=sum(s.total_attempted for s in present),
        successful=sum(s.successful for s in present),
        failed=sum(s.failed for s in present),
        successful_products=[p for s in present for p in s.successful_products],
        failed_products=[p for s in present for p in s.failed_products],
        media_stats=media,
        duration_sec=sum(s.duration_sec for s in present),
    )


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
        if self.config.subtitle_format:
            overrides["subtitle_settings.subtitle_format"] = self.config.subtitle_format
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
                logger.warning("Webhook notification failed: %s", e)

    def _resumed_topic_ids(self) -> list[str]:
        """Topic ids from the saved state, for a `--resume` plan.

        `config.topics` is empty on a resume, so the plan has nothing to name
        without reading them back.
        """
        if not self.config.topics_resume:
            return []
        from src.video.producer.topic_input import TOPIC_ID_PREFIX

        saved = load_pipeline_state(self.config.outputs_dir)
        if saved is None:
            return []
        return [
            pid
            for pid in (saved.scraping_completed_products or [])
            if pid.startswith(TOPIC_ID_PREFIX)
        ]

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

        # What --clean would remove. The plan exists to answer that before
        # the directories are gone, and it is the one companion flag whose
        # effect cannot be undone.
        if self.config.clean:
            print(f"{section}")
            print("CLEAN")
            print(f"{section}")
            targets = _clean_targets(
                self.config.outputs_dir, _named_run_ids(self.config)
            )
            if targets:
                print(
                    f"  Would remove {len(targets)} product director"
                    f"{'y' if len(targets) == 1 else 'ies'}:"
                )
                for target in targets[:10]:
                    print(f"    - {target.name}")
                if len(targets) > 10:
                    print(f"    ... and {len(targets) - 10} more")
            else:
                print("  Nothing to remove")
            print()

        # Phase 1: Scraping Plan
        print(f"{section}")
        print("PHASE 1: SCRAPING")
        print(f"{section}")

        # A resumed topics run scrapes nothing either, and its topics are not
        # on the config -- reading only `topics` printed a full keyword plan
        # for a run that would render the saved topic and search for nothing.
        resumed_ids = self._resumed_topic_ids()
        has_topics = bool(self.config.topics) or bool(resumed_ids)
        # A mixed run does both, so only a run with nothing to scrape may
        # suppress the scraping half. Suppressing it on a mixed run hides work
        # the run will do -- the same defect as printing work it would
        # discard, in the other direction.
        topics_only = has_topics and not (
            self.config.keywords or self.config.product_ids
        )
        if has_topics:
            # Named rather than omitted: a plan that simply prints nothing
            # under SCRAPING reads as a misconfigured run. Worded as "prepared"
            # rather than "skipped" because the topic IS produced -- only the
            # scraping is skipped, and on a mixed run "skipped" reads as if the
            # topic will not be rendered at all.
            named = [spec.title for spec in self.config.topics] or resumed_ids
            print(f"  Prepared without scraping: {len(named)} topic(s)")
            for title in named[:10]:
                print(f"    - {title}")
            if len(named) > 10:
                print(f"    ... and {len(named) - 10} more")

        # Everything below describes scraping, which a topics-only run does
        # not do. Printing it anyway promised work the run would discard,
        # which is the one thing the plan exists to rule out.
        if self.config.product_ids and not topics_only:
            print(f"  Product IDs to scrape: {len(self.config.product_ids)}")
            for pid in self.config.product_ids[:10]:  # Show first 10
                print(f"    - {pid}")
            if len(self.config.product_ids) > 10:
                print(f"    ... and {len(self.config.product_ids) - 10} more")

        if self.config.keywords and not topics_only:
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

        if not topics_only:
            # Scraper filters, so meaningless on a run that scrapes nothing.
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
                if profile.description:
                    print(f"    - {profile.description}")
                sources = []
                if profile.use_scraped_images:
                    sources.append("scraped images")
                if profile.use_scraped_videos:
                    sources.append("scraped videos")
                if profile.use_stock_images:
                    sources.append(f"{profile.stock_image_count} stock images")
                if profile.use_stock_videos:
                    sources.append(f"{profile.stock_video_count} stock videos")
                print(f"    - Visuals: {', '.join(sources) or 'none configured'}")
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

            # A topic draws from its own pool, so a mixed run has two. Printing
            # only the product one leaves the plan silent about which profile
            # the topics in it will actually use.
            topic_pool = self.config.topic_profile_pool
            if topic_pool and topic_pool != pool:
                print(f"  Topic profile pool ({len(topic_pool)} profiles):")
                for p in topic_pool[:5]:
                    print(f"    - {p}")
                if len(topic_pool) > 5:
                    print(f"    ... and {len(topic_pool) - 5} more")
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
            logger.info("Resuming pipeline run: %s", self.state.run_id)
            logger.info("  Started: %s", self.state.started_at)
            completed = ", ".join(self.state.completed_phases) or "none"
            logger.info("  Completed phases: %s", completed)

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
                "Using cached results: %s successful, %s failed",
                scraping_summary.successful,
                scraping_summary.failed,
            )
        else:
            logger.info("=" * 80)
            logger.info("SCRAPING PHASE")
            logger.info("=" * 80)
            self.state.advance_phase(PipelinePhase.SCRAPING)
            self._save_state()

            # A topic has no listing to scrape. Its records are built here
            # instead, into the same directory shape the scraper writes, so the
            # handoff and everything after it are unchanged. A run can carry
            # both kinds of input -- that is what a configured `topics:`
            # section alongside `keywords:` produces -- so this is two phases
            # that both may run, not a choice between them.
            topic_summary = (
                self._materialise_topics_phase() if self.config.topics else None
            )
            scrape_summary = (
                await self._execute_scraping_phase()
                if (self.config.keywords or self.config.product_ids)
                else None
            )
            scraping_summary = _merge_scraping_summaries(topic_summary, scrape_summary)

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
            logger.info("Skipping publishing phase (--skip-publish)")
        else:
            logger.info("No videos to publish")

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
        logger.info("Scraping %d input(s): %s", total_inputs, ", ".join(all_inputs))
        logger.info(
            "Limits: %s per keyword, %s total",
            self.config.products_per_keyword,
            self.config.max_products,
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
                    "Reached max_products limit (%s). "
                    "Stopping with %s inputs remaining.",
                    self.config.max_products,
                    len(all_inputs) - idx + 1,
                )
                break

            remaining = self.config.max_products - len(successful_products)
            per_input_limit = min(self.config.products_per_keyword, remaining)
            scraper.amazon_config["max_products"] = per_input_limit

            collected = f"{len(successful_products)}/{self.config.max_products}"
            logger.info(
                "[%s/%s] Processing: %s (limit: %s, collected: %s)",
                idx,
                total_inputs,
                input_item,
                per_input_limit,
                collected,
            )

            raw_products = results_by_input.get(input_item, [])
            if not raw_products:
                inputs_failed += 1
                failed_inputs.append(input_item)
                logger.warning("[%s/%s] No data for %s", idx, total_inputs, input_item)
                if self.config.fail_fast:
                    logger.error("Fail-fast enabled, stopping scraping phase")
                    break
                continue

            input_pillar = keyword_pillar_for(
                input_item, self.config.keyword_pillar_map
            )

            try:
                products = scraper.process_raw_products(
                    raw_products,
                    target_download_count=per_input_limit,
                    pillar=input_pillar,
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
                            pillar=input_pillar,
                        )
                        products.extend(extra_products)
                        page += 1

                if products:
                    inputs_processed += 1
                    # The pillar was applied before the write, above. Setting
                    # it here as well would be the bug this replaced: these
                    # objects are discarded and the directory re-read.
                    for product in products:
                        if hasattr(product, "asin") and product.asin:
                            successful_products.append(product.asin)
                        if hasattr(product, "images") and product.images:
                            total_images += len(product.images)
                        if hasattr(product, "videos") and product.videos:
                            total_videos += len(product.videos)
                    logger.info(
                        "[%s/%s] Found %s product(s) for %s",
                        idx,
                        total_inputs,
                        len(products),
                        input_item,
                    )
                else:
                    inputs_failed += 1
                    failed_inputs.append(input_item)
                    logger.warning(
                        "[%s/%s] No valid products for %s",
                        idx,
                        total_inputs,
                        input_item,
                    )
                    if self.config.fail_fast:
                        logger.error("Fail-fast enabled, stopping scraping phase")
                        break

            except Exception as e:
                inputs_failed += 1
                failed_inputs.append(input_item)
                logger.error(
                    "[%s/%s] Failed to process %s: %s",
                    idx,
                    total_inputs,
                    input_item,
                    e,
                )
                if self.config.fail_fast:
                    logger.error("Fail-fast enabled, stopping scraping phase")
                    raise

        # Generate summary
        duration = time.time() - phase_start
        media_stats = {"total_images": total_images, "total_videos": total_videos}

        logger.info(
            "Scraping phase complete: %s products from %s inputs "
            "(%s failed) in %.1fs",
            len(successful_products),
            inputs_processed,
            inputs_failed,
            duration,
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

    def _materialise_topics_phase(self) -> ScrapingPhaseSummary:
        """Stand in for the scraping phase when the inputs are topics.

        Reported as the scraping phase rather than as a new one so resume,
        state and the phase summaries keep working unchanged; what differs is
        where the records come from, not what the pipeline does with them.

        Returns a summary whose `successful_products` are the topic
        identifiers, which is what the handoff phase filters on.

        `--fail-fast` is deliberately not honoured here, unlike the scraping
        phase it stands in for. That flag exists to stop a run before it pays
        for more scrapes after one has failed; writing a record costs nothing,
        and the only way this fails is an unwritable outputs directory, which
        is not a reason to discard the topics that did write.
        """
        from src.video.producer.topic_input import materialise_topics

        phase_start = time.time()
        logger.info("Preparing %s topic(s) (no scraping)", len(self.config.topics))

        config = load_video_config_modular()
        # `materialise_topics` needs a profile only to resolve the run paths,
        # and the data.json it writes is profile-independent. A random-profile
        # run has no single profile yet, so any valid one resolves the same
        # directory.
        profile = self.config.profile or next(iter(config.video_profiles))

        prepared: list[str] = []
        failed: list[str] = []
        for spec in self.config.topics:
            try:
                ((topic_dir, product),) = materialise_topics(
                    [spec], config, profile, outputs_dir=self.config.outputs_dir
                )
            except OSError as e:
                # One unwritable directory must not lose the rest of the batch.
                logger.error("Could not prepare topic %r: %s", spec.title, e)
                failed.append(spec.title)
                continue
            logger.info("Prepared topic %r in %s", spec.title, topic_dir)
            prepared.append(str(product.asin))

        return ScrapingPhaseSummary(
            total_attempted=len(self.config.topics),
            successful=len(prepared),
            failed=len(failed),
            successful_products=prepared,
            failed_products=failed,
            media_stats={"total_images": 0, "total_videos": 0},
            duration_sec=time.time() - phase_start,
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
        # A topic run named its inputs, so its directories are what it asked
        # for. Discovery skips them by default, which would drop every topic
        # between the phase that wrote them and the phase that renders them.
        # Derived from the ids being handed off, not from the config: a
        # `--resume` carries no input flags, so `config.topics` is empty while
        # the saved state's ids are topics. Reading the config there returned
        # nothing and the resumed run reported PIPELINE FAILED.
        from src.video.producer.topic_input import TOPIC_ID_PREFIX

        # `topics_resume` is set in `main` before validation, which is what
        # narrows the profile pool. The id check stays as the fallback for a
        # caller that builds the orchestrator directly.
        include_topics = (
            bool(self.config.topics)
            or self.config.topics_resume
            or any(pid.startswith(TOPIC_ID_PREFIX) for pid in scraped_product_ids)
        )
        all_products = discover_products_for_batch(
            self.config.outputs_dir, include_topics=include_topics
        )

        logger.info(
            "Found %s product(s) with data.json in %s",
            len(all_products),
            self.config.outputs_dir,
        )

        # Filter by scraped products unless process_all_products is enabled
        if self.config.process_all_products:
            ready_products = all_products
            logger.info("Processing all products in outputs directory")
        else:
            # Only process products scraped in current run
            scraped_set = set(scraped_product_ids)
            ready_products = [
                (path, data)
                for path, data in all_products
                if hasattr(data, "asin") and data.asin in scraped_set
            ]
            logger.info(
                "Processing %s product(s) from current scraping run",
                len(ready_products),
            )

        ready_products = self._drop_already_published(ready_products)

        # Log transition
        if ready_products:
            logger.info("%s product(s) ready for video production", len(ready_products))
        else:
            logger.warning("No products ready for video production")

        return ready_products

    def _drop_already_published(
        self, products: list[tuple[Path, Any]]
    ) -> list[tuple[Path, Any]]:
        """Drop products already recorded as published on every platform.

        The batch had no duplicate guard at all. `single` and `schedule` skip
        an already-published product; the batch's publish phase did not, so a
        re-scraped product was rendered and then published a second time --
        a duplicate Zernio post, with the tracking row overwritten by the new
        `post_id` while the older post stayed live. So this stops the
        duplicate as well as the render, which is why it is on by default and
        why `--force` exists to get the old behaviour back.

        `publish_history.json` is the file that backs the guard, keyed by
        `<asin>:<platform>`; `published_products.json` records what was
        produced rather than what went live and cannot answer this. A product
        published to some platforms but not all is kept, since the run still
        has somewhere to send it.

        Skipped entirely when `--force` is set, which is the flag that already
        means "publish it again" on the single and schedule paths, and when
        `--skip-publish` is set, where there is no duplicate to prevent and
        re-rendering a published product is the point of the run.

        Topics are never dropped. A topic's id is a pure function of its
        title, so once published it would be skipped on every later run and
        the tutorial arm would stop producing silently -- the bundled config
        ships two topics at one per run, so that lands on day three.
        """
        from src.publisher.tracking import is_already_published
        from src.video.producer.topic_input import TOPIC_ID_PREFIX

        if (
            getattr(self.config, "force", False)
            or getattr(self.config, "skip_publish", False)
            or not products
        ):
            return products

        # The list the publish phase will actually target, read the same way
        # it reads it. A hardcoded triple demanded tiktok of an install whose
        # `default_platforms` omits it, so a product complete for that install
        # was re-rendered and re-published (#126 tracks folding the three
        # inline reads in this module into the loaded config).
        platforms = self.config.platforms or self._default_platforms()
        kept, skipped = [], []
        for path, data in products:
            asin = getattr(data, "asin", None)
            if (
                asin
                and not asin.startswith(TOPIC_ID_PREFIX)
                and all(
                    is_already_published(asin, platform, self.config.outputs_dir)
                    for platform in platforms
                )
            ):
                skipped.append(asin)
            else:
                kept.append((path, data))

        if skipped:
            logger.info(
                "Skipping %s already-published product(s) before render: %s",
                len(skipped),
                ", ".join(skipped),
            )
        self._skipped_as_published = skipped
        return kept

    def _default_platforms(self) -> list[str]:
        """The platforms a publish would target, absent a CLI override.

        Reads `publisher.yaml` first and falls back to the literal, which is
        what `print_plan` and `_execute_publishing_phase` already do. The
        duplicate guard has to ask the same question they do, or it decides
        completeness against platforms this install never publishes to.
        """
        import yaml

        config_path = Path("config/publisher.yaml")
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as handle:
                    publisher_config = yaml.safe_load(handle) or {}
            except (OSError, yaml.YAMLError):
                publisher_config = {}
            configured = publisher_config.get("default_platforms")
            if configured:
                return list(configured)
        return ["youtube", "tiktok", "instagram"]

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
        logger.info("Processing %s product(s) for video production", total_products)

        # Create HTTP session for API calls
        async with aiohttp.ClientSession() as session:
            for idx, (_product_dir, product) in enumerate(products, 1):
                product_id = product.asin or product.title or f"product_{idx}"

                # Select profile for this product. A topic draws from its own
                # pool: it has no product photography, so a profile that
                # sources only scraped media gathers nothing and the render
                # fails outright. On a topics-only run the two pools are the
                # same list; on a mixed run they are close to complements.
                is_topic = bool(getattr(product, "topic", None))
                pool = (
                    self.config.topic_profile_pool
                    if is_topic and self.config.topic_profile_pool
                    else self.config.profile_pool
                )
                if self.config.random_profile:
                    # Random profile selection (deterministic by product ID)
                    assert pool is not None
                    assert profile_tracker is not None
                    current_profile = select_profile_for_product(
                        product_id=product_id,
                        profile_pool=pool,
                        config=config,
                    )
                    profile_tracker.record_usage(current_profile)
                    logger.info(
                        "[%s/%s] Processing %s with profile '%s'",
                        idx,
                        total_products,
                        product_id,
                        current_profile,
                    )
                else:
                    # Fixed profile mode
                    assert self.config.profile is not None
                    current_profile = self.config.profile
                    logger.info(
                        "[%s/%s] Processing product: %s",
                        idx,
                        total_products,
                        product_id,
                    )

                try:
                    # The product's own pillar is NOT promoted into
                    # `cli_overrides` here. The producer reads it as the last
                    # term of its own resolution, and putting it in the CLI
                    # slot would rank it above a pillar a previous run
                    # recorded -- so a resumed batch would file the row under
                    # the scraped arm while reusing a script written for the
                    # overridden one.
                    cli_overrides = self._build_cli_overrides()

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
                            "[%s/%s] Successfully created video for %s",
                            idx,
                            total_products,
                            product_id,
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
                        "[%s/%s] Pipeline timed out after %ss for %s",
                        idx,
                        total_products,
                        config.pipeline_timeout_sec,
                        product_id,
                    )

                    if self.config.fail_fast:
                        logger.error("Fail-fast enabled, stopping production phase")
                        break

                except Exception as e:
                    failed += 1
                    failed_products.append(product_id)
                    logger.error(
                        "[%s/%s] Failed to process %s: %s",
                        idx,
                        total_products,
                        product_id,
                        e,
                        exc_info=True,
                    )

                    if self.config.fail_fast:
                        logger.error("Fail-fast enabled, stopping production phase")
                        raise

        # Generate summary
        duration = time.time() - phase_start
        profile_distribution = profile_tracker.get_counts() if profile_tracker else None

        logger.info(
            "Production phase complete: %s successful, %s failed, "
            "%s skipped in %.1fs",
            successful,
            failed,
            skipped,
            duration,
        )

        # Carried from the handoff drop so the summary says why a run that
        # rendered nothing rendered nothing. Without it the verdict is
        # "PIPELINE FAILED ... 0 failed, 0 skipped", which contradicts itself
        # and exits 1 on a correct result.
        already_published = getattr(self, "_skipped_as_published", [])
        summary = ProductionPhaseSummary(
            total_attempted=total_products,
            successful=successful,
            failed=failed,
            skipped=skipped,
            failed_products=failed_products,
            skipped_products=skipped_products,
            profile_distribution=profile_distribution,
            duration_sec=duration,
            already_published=len(already_published),
            already_published_products=list(already_published),
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
        from src.publisher.models import AffiliateDisclosureConfig, Platform

        phase_start = time.time()

        # Load publisher configuration
        config_path = Path("config/publisher.yaml")
        publisher_config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                publisher_config = yaml.safe_load(f) or {}
            logger.debug("Loaded publisher config from %s", config_path)
        else:
            logger.warning("Publisher config not found: %s", config_path)

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
            logger.info("Using explicit schedule time: %s", schedule_time)
        else:
            # Priority 2: Auto-schedule if configured
            immediate_publish = publisher_config.get("immediate_publish", True)
            recurring_config = publisher_config.get("recurring_schedule", {})
            recurring_enabled = recurring_config.get("enabled", False)

            logger.debug(
                "Scheduling config: immediate_publish=%s, recurring_enabled=%s",
                immediate_publish,
                recurring_enabled,
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
                                "Temp publisher init: vercel_token=%s",
                                "set" if vercel_token else "NOT SET",
                            )
                            # Reads slot occupancy and never publishes, so it
                            # needs none of the payload settings the publisher
                            # below is given.
                            temp_publisher = create_publisher(
                                provider=PublisherProvider.LATE,
                                api_key=api_key,
                                vercel_token=vercel_token,
                            )

                            try:
                                await temp_publisher.authenticate()

                                api_posts = await temp_publisher.list_posts()
                                logger.debug(
                                    "Found %s existing posts on API", len(api_posts)
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
                                        "Occupied slot: %s",
                                        slot_time.strftime("%Y-%m-%d %H:%M %Z"),
                                    )
                            except Exception as e:
                                logger.warning("Failed to fetch occupied slots: %s", e)

                        # Store context for per-product slot finding
                        auto_schedule_ctx = {
                            "slots": slots,
                            "schedule_manager": schedule_manager,
                            "occupied_slot_times": occupied_slot_times,
                        }

                    except Exception as e:
                        logger.warning(
                            "Failed to auto-schedule: %s. Publishing immediately.", e
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
                "Publisher init: api_key=%s, vercel_token=%s",
                "set" if api_key else "NOT SET",
                "set" if vercel_token else "NOT SET",
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

            from src.publisher.config import parse_tiktok_settings

            publisher = create_publisher(
                provider=PublisherProvider.LATE,
                api_key=api_key,
                vercel_token=vercel_token,
                first_comment_config=first_comment_config,
                # The batch builds its own publisher rather than reusing the
                # CLI's, so every setting has to be passed here too or the
                # same config produces different payloads on the two paths.
                synthetic_media_disclosure=bool(
                    publisher_config.get("synthetic_media_disclosure", False)
                ),
                tiktok_settings=parse_tiktok_settings(
                    publisher_config.get("tiktok_settings")
                ),
            )

            # Authenticate
            logger.info("Authenticating with publisher...")
            await publisher.authenticate()
            logger.info("Authentication successful")

            # Get connected accounts
            accounts = await publisher.get_accounts()
            logger.info("Found %s connected account(s)", len(accounts))

        except Exception as e:
            logger.error("Failed to initialize publisher: %s", e, exc_info=True)
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
            logger.info(
                "[%s/%s] Publishing video for %s", idx, total_attempted, product_id
            )

            video_successful = True
            video_errors: list[str] = []

            try:
                # Upload video once
                logger.info("[%s/%s] Uploading video...", idx, total_attempted)
                media_id = await publisher.upload_media(video_path)
                logger.info(
                    "[%s/%s] Upload complete: %s", idx, total_attempted, media_id
                )

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
                                "Auto-scheduled %s to slot #%s: %s",
                                product_id,
                                slot_index,
                                next_slot_time.strftime("%A, %Y-%m-%d %H:%M:%S %Z"),
                            )
                            break
                        else:
                            logger.debug(
                                "Slot #%s occupied, trying next...", slot_index
                            )
                            now = next_slot_time
                            slot_index = (slot_index + 1) % len(ctx_slots)
                    else:
                        logger.warning(
                            "All slots occupied for %s. Publishing immediately.",
                            product_id,
                        )
                        product_schedule_time = None

                # Publish (unified or platform-specific mode)
                from src.publisher.publish_modes import publish_product

                platform_specific = (
                    self.config.platform_specific_content
                    or publisher_config.get("use_platform_specific_content", False)
                )

                disc_raw = publisher_config.get("affiliate_disclosure", {}) or {}
                affiliate_cfg = (
                    AffiliateDisclosureConfig(**disc_raw)
                    if disc_raw
                    else AffiliateDisclosureConfig()
                )
                disclosure_phrase = (
                    affiliate_cfg.phrase if affiliate_cfg.enabled else None
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
                        "[%d/%d] Published: post_id=%s, status=%s",
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
                        "[%s/%s] Successfully published %s to all platforms",
                        idx,
                        total_attempted,
                        product_id,
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
                                        "Cleaning up product directory: %s", product_dir
                                    )
                                    shutil.rmtree(product_dir)
                                    logger.info("Removed %s", product_dir)
                                except Exception as e:
                                    logger.warning(
                                        "Failed to cleanup %s: %s", product_dir, e
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
                        "[%s/%s] Partially failed for %s",
                        idx,
                        total_attempted,
                        product_id,
                    )

            except Exception as e:
                failed += 1
                failed_videos.append(product_id)
                errors.append({"product_id": product_id, "error": str(e)})
                logger.error(
                    "[%s/%s] Failed to process %s: %s",
                    idx,
                    total_attempted,
                    product_id,
                    e,
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
                    "[%s/%s] Waiting %ss before next publish...",
                    idx,
                    total_attempted,
                    delay,
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
            "Publishing phase complete: %s successful, %s failed, "
            "%s skipped in %.1fs",
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
    logger.info("Log file: %s", log_file)

    try:
        # Load configuration with CLI > YAML > defaults precedence
        logger.info("Loading configuration...")
        config = load_global_batch_config(args)

        # Load video configuration for validation
        video_config = load_video_config_modular()

        # A `--resume` carries no input flags, so a topics run looks like a
        # product run to everything below unless the saved state is consulted
        # first. Reading it here rather than in the handoff phase keeps one
        # copy of the topic rules and lets the stock-key pre-flight see the
        # pool a topics run will actually draw from.
        apply_resume_record_kinds(config)

        # Validate configuration
        logger.info("Validating configuration...")
        validate_global_batch_config(config, video_config)

        # Same pre-flight the producer runs: a profile this batch may select
        # drawing every visual from the stock provider, with no key set, is a
        # whole run failing per product on a message that names neither.
        #
        # Guarded by the dry-run flag rather than by position. Sitting below
        # the `--clean` block would let a keyless run delete the product
        # directories and then abort, costing the scraped data as well as the
        # render.
        if not config.dry_run:
            from src.video.config_validator import check_stock_media_key

            if config.random_profile:
                # `validate_global_batch_config` fills an empty pool with the
                # selectable profiles, so this is the real draw set by now.
                candidate_profiles = list(config.profile_pool or [])
            else:
                candidate_profiles = [config.profile] if config.profile else []
            stock_key_error = check_stock_media_key(video_config, candidate_profiles)
            if stock_key_error:
                logger.critical(stock_key_error)
                sys.exit(1)

        logger.info("Configuration validated successfully")
        logger.info(
            "Inputs: %s product IDs, %s keywords, %s topics",
            len(config.product_ids or []),
            len(config.keywords or []),
            len(config.topics or []),
        )

        if config.profile:
            logger.info("Profile: %s (fixed)", config.profile)
        elif config.random_profile:
            pool_info = (
                ", ".join(config.profile_pool)
                if config.profile_pool
                else "all available"
            )
            logger.info("Profile: random selection from [%s]", pool_info)

        logger.info("Outputs directory: %s", config.outputs_dir)
        logger.info("Fail-fast: %s", config.fail_fast)
        logger.info("Resume mode: %s", config.resume)
        logger.info("Dry-run mode: %s", config.dry_run)

        # Dry-run first: it reports what a run would do, so nothing
        # destructive may precede it. `--clean` used to, which meant
        # `--dry-run --clean` removed the product directories and then
        # printed a plan for producing them.
        # Handle dry-run mode
        if config.dry_run:
            orchestrator = GlobalPipelineOrchestrator(config, video_config=video_config)
            orchestrator.display_execution_plan(video_config)
            logger.info("Dry-run completed - exiting without execution")
            sys.exit(0)

        # Handle clean mode
        if config.clean:
            import shutil

            for target in _clean_targets(config.outputs_dir, _named_run_ids(config)):
                shutil.rmtree(target)
                logger.info("Cleaned product directory: %s", target)

        # Handle resume mode
        state = None
        if config.resume:
            state = load_pipeline_state(config.outputs_dir)
            if state:
                logger.info("Resuming pipeline run: %s", state.run_id)
                logger.info("  Current phase: %s", state.current_phase.value)
                logger.info(
                    "  Completed phases: %s",
                    ", ".join(state.completed_phases) or "none",
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
                    logger.info("Webhook notifications enabled: %s", webhook_config.url)
                else:
                    logger.warning("Webhook URL configured but invalid")
        except FileNotFoundError:
            logger.debug("No pipeline.yaml found - webhooks disabled")
        except Exception as e:
            logger.warning("Failed to load webhook config: %s", e)

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
        exit_code = summary.exit_code(strict=args.strict)

        # Output summary in requested format
        if config.output_format == "json":
            # JSON output to stdout for machine parsing
            print(summary.to_json(started_at=pipeline_started_at))
        else:
            # Text output (already logged by _generate_final_summary)
            logger.info("=" * 80)
            # Keyed on what happened, not on the exit code: under --strict
            # a partial loss also exits non-zero, and calling that "no
            # products completed end-to-end" would be false. A loss counts
            # here for the same reason it counts there -- a product lost to
            # a skip is still a video that was asked for and does not
            # exist, and reporting success for it would contradict the
            # exit code of the very same run.
            skipped = summary.total_skipped()
            outcome = summary.outcome()
            if outcome == "failed":
                logger.error(
                    "PIPELINE FAILED: no products completed end-to-end "
                    "(%d failed, %d skipped)",
                    summary.total_failures,
                    skipped,
                )
            elif outcome == "lost":
                logger.warning(
                    "PIPELINE COMPLETED WITH LOSSES: "
                    "%d succeeded, %d failed, %d skipped",
                    summary.end_to_end_success,
                    summary.total_failures,
                    skipped,
                )
            else:
                logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info("Complete log saved to: %s", log_file)

        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 80)
        logger.warning("PIPELINE INTERRUPTED BY USER")
        logger.warning("=" * 80)
        logger.warning("Partial log saved to: %s", log_file)
        logger.warning("To resume from last checkpoint, run with --resume flag")
        sys.exit(130)  # Standard exit code for SIGINT

    except ValueError as e:
        # Configuration or validation errors
        logger.error("=" * 80)
        logger.error("CONFIGURATION ERROR")
        logger.error("=" * 80)
        logger.error(str(e))
        logger.error("Complete log saved to: %s", log_file)
        sys.exit(1)

    except Exception as e:
        # Unexpected errors
        logger.critical("=" * 80)
        logger.critical("PIPELINE FAILED WITH ERROR")
        logger.critical("=" * 80)
        logger.critical("Error: %s", e, exc_info=True)
        logger.critical("Complete log saved to: %s", log_file)
        logger.critical("To resume from last checkpoint, run with --resume flag")
        sys.exit(1)

    finally:
        # Clean up HTTP connection pool to avoid "Unclosed connector" warnings
        from src.utils.connection_pool import close_global_pool

        await close_global_pool()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
