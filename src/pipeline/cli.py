"""Command line for the global batch pipeline.

Moved out of `global_batch.py` (#450): the parser and `main` were a
third of a 2,800-line module whose other two thirds are the orchestrator.
`python -m src.pipeline.global_batch` still runs this `main`.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.pipeline.config import load_pipeline_state
from src.pipeline.global_batch import (
    GlobalPipelineOrchestrator,
    _clean_targets,
    _named_run_ids,
    _publisher_settings,
    apply_resume_record_kinds,
)
from src.pipeline.webhooks import WebhookNotifier
from src.video.config_adapter import load_video_config_modular
from src.video.producer.shared_cli import add_shared_render_args

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Create argument parser for global batch pipeline CLI.

    Returns
    -------
        argparse.ArgumentParser configured with all pipeline arguments

    """
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
    add_shared_render_args(producer_group)

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
        help=(
            "Directory for scraper output and producer input "
            "(default: the repo outputs/)"
        ),
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


async def main():
    """Main CLI entry point for global batch pipeline.

    Parses arguments, loads configuration, validates settings,
    executes pipeline, and handles errors gracefully.
    """
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
    from src.utils.outputs_paths import get_logs_directory

    log_file = get_logs_directory() / "global_pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_debug_logging(
        log_file=log_file,
        debug_mode=args.debug,
        verbose=args.debug,
        component_name="GlobalPipeline",
    )

    logger.info("GLOBAL BATCH PIPELINE STARTING")
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

        # Read the publisher config here rather than at first use, so an
        # unreadable one stops the run before anything is paid for. Resolution
        # is lazy and the first reader depends on the flags: without
        # `--platforms` it is the handoff filter, after the scrape; with it,
        # or with `--force`, nothing touches the file until the publishing
        # phase and every render is already spent.
        if not config.skip_publish:
            _publisher_settings()

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

        pipeline_started_at = datetime.now(UTC).isoformat()

        # Webhook configuration, from the already-loaded config rather than a
        # second relative-path read of the same file.
        from src.pipeline.webhooks import load_webhook_config

        webhook_notifier = None
        try:
            webhook_config = load_webhook_config(config.webhook_yaml)
            if webhook_config.is_configured():
                webhook_notifier = WebhookNotifier(webhook_config)
                if webhook_notifier.is_ready():
                    logger.info("Webhook notifications enabled: %s", webhook_config.url)
                else:
                    logger.warning("Webhook URL configured but invalid")
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
            logger.info("Complete log saved to: %s", log_file)

        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.warning("PIPELINE INTERRUPTED BY USER")
        logger.warning("Partial log saved to: %s", log_file)
        logger.warning("To resume from last checkpoint, run with --resume flag")
        sys.exit(130)  # Standard exit code for SIGINT

    except ValueError as e:
        # Configuration or validation errors
        logger.error("CONFIGURATION ERROR")
        logger.error(str(e))
        logger.error("Complete log saved to: %s", log_file)
        sys.exit(1)

    except Exception as e:
        # Unexpected errors
        logger.critical("PIPELINE FAILED WITH ERROR")
        logger.critical("Error: %s", e, exc_info=True)
        logger.critical("Complete log saved to: %s", log_file)
        logger.critical("To resume from last checkpoint, run with --resume flag")
        sys.exit(1)

    finally:
        # Clean up HTTP connection pool to avoid "Unclosed connector" warnings
        from src.utils.connection_pool import close_global_pool

        await close_global_pool()
