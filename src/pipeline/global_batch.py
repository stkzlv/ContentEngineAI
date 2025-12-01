"""Global batch pipeline orchestrator.

This module orchestrates the complete end-to-end workflow from scraping
products to generating promotional videos in three sequential phases:

1. Scraping Phase: Process product IDs/keywords through Amazon scraper
2. Handoff Phase: Discover products ready for video production
3. Video Production Phase: Generate videos for ready products

The orchestrator treats both scraper and producer as black boxes,
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

from src.pipeline.config import (
    GlobalBatchConfig,
    PipelineSummary,
    ProductionPhaseSummary,
    ScrapingPhaseSummary,
)
from src.scraper.amazon.models import ProductData

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Create argument parser for global batch pipeline CLI.

    Returns
    -------
        argparse.ArgumentParser configured with all pipeline arguments

    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Global Batch Pipeline - End-to-end Amazon product scraping and video production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape product and create video with fixed profile
  python -m src.pipeline --product-ids B0ABC123 --profile slideshow_images1

  # Scrape keywords and create videos with random profile selection
  python -m src.pipeline --keywords "wireless earbuds" --random-profile --profile-pool slideshow_images1 video_sequential

  # Batch with filters and fail-fast
  python -m src.pipeline --product-ids B0ABC123 B0DEF456 --profile slideshow_images1 --fail-fast --debug
        """,
    )

    # Input arguments
    input_group = parser.add_argument_group("Input Configuration")
    input_group.add_argument(
        "--product-ids",
        nargs="+",
        metavar="ASIN",
        help="Product IDs (ASINs) to scrape and produce videos for (e.g., B0ABC123 B0DEF456)",
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
        help="Video profile to use for all products (mutually exclusive with --random-profile)",
    )
    producer_group.add_argument(
        "--random-profile",
        action="store_true",
        help=(
            "Enable random profile selection per product (deterministic by product ID). "
            "Mutually exclusive with --profile. Requires --profile-pool or uses all available profiles."
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

    return parser


class GlobalPipelineOrchestrator:
    """Orchestrates scraping and video production phases sequentially.

    Coordinates the complete pipeline from scraping to video production,
    treating scraper and producer as black boxes and managing the handoff
    between phases.

    Attributes:
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
        """Execute complete pipeline: scrape → handoff → produce.

        Orchestrates three sequential phases:
        1. Scraping Phase: Scrape products using configured inputs
        2. Handoff Phase: Discover products ready for video production
        3. Video Production Phase: Generate videos for ready products

        Returns:
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
        ready_products = self._execute_handoff_phase()

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
        else:
            # Phase 3: Video Production
            logger.info("=" * 80)
            logger.info("VIDEO PRODUCTION PHASE")
            logger.info("=" * 80)
            production_summary = await self._execute_production_phase(ready_products)

        # Generate final summary
        pipeline_duration = time.time() - pipeline_start
        final_summary = self._generate_final_summary(
            scraping_summary, production_summary, pipeline_duration
        )

        return final_summary

    async def _execute_scraping_phase(self) -> ScrapingPhaseSummary:
        """Execute scraping phase and return summary.

        Invokes Amazon scraper with configured product IDs and keywords,
        tracks statistics, and generates phase summary.

        Returns:
        -------
            ScrapingPhaseSummary with scraping statistics

        Raises:
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
            output_dir=str(self.config.outputs_dir), debug_mode=self.config.debug
        )

        # Track statistics
        successful = 0
        failed = 0
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
                    # Count media for this product
                    for product in products:
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
                logger.error(f"✗ [{idx}/{total_inputs}] Failed to scrape {input_item}: {e}")

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
            failed_products=failed_products,
            media_stats=media_stats,
            duration_sec=duration,
        )

    def _execute_handoff_phase(self) -> list[tuple[Path, ProductData]]:
        """Discover products ready for video production.

        Scans outputs directory for products with data.json and filters
        by media availability based on profile requirements.

        Returns:
        -------
            List of (product_dir, ProductData) tuples for ready products

        """
        from src.video.producer.cli import discover_products_for_batch

        logger.info("Discovering products ready for video production...")

        # Use existing discover_products_for_batch function
        all_products = discover_products_for_batch(self.config.outputs_dir)

        logger.info(
            f"Found {len(all_products)} product(s) with data.json in {self.config.outputs_dir}"
        )

        # Filter products by media availability
        # Note: In a full implementation, we would check profile requirements
        # For now, we accept all products that have data.json as they were
        # successfully scraped and have the necessary structure
        ready_products = all_products

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
    ) -> ProductionPhaseSummary:
        """Execute video production phase and return summary.

        Processes each product through video pipeline with configured profile,
        supports both fixed and random profile modes, tracks statistics.

        Args:
        ----
            products: List of (product_dir, ProductData) tuples to process

        Returns:
        -------
            ProductionPhaseSummary with video production statistics

        """
        import asyncio

        import aiohttp

        import os

        from src.video.config import load_video_config
        from src.video.producer.orchestration import create_video_for_product
        from src.video.producer.utils import (
            ProfileUsageTracker,
            select_profile_for_product,
        )

        phase_start = time.time()

        # Load video configuration
        config = load_video_config()

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

        total_products = len(products)
        logger.info(f"Processing {total_products} product(s) for video production")

        # Create HTTP session for API calls
        async with aiohttp.ClientSession() as session:
            for idx, (product_dir, product) in enumerate(products, 1):
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
                    current_profile = self.config.profile
                    assert current_profile is not None
                    logger.info(f"[{idx}/{total_products}] Processing product: {product_id}")

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
                        f"✗ [{idx}/{total_products}] Failed to process {product_id}: {e}",
                        exc_info=True,
                    )

                    if self.config.fail_fast:
                        logger.error("Fail-fast enabled, stopping production phase")
                        raise

        # Generate summary
        duration = time.time() - phase_start
        profile_distribution = (
            profile_tracker.get_counts() if profile_tracker else None
        )

        logger.info(
            f"Production phase complete: {successful} successful, "
            f"{failed} failed, {skipped} skipped in {duration:.1f}s"
        )

        return ProductionPhaseSummary(
            total_attempted=total_products,
            successful=successful,
            failed=failed,
            skipped=skipped,
            failed_products=failed_products,
            skipped_products=skipped_products,
            profile_distribution=profile_distribution,
            duration_sec=duration,
        )

    def _generate_final_summary(
        self,
        scraping: ScrapingPhaseSummary,
        production: ProductionPhaseSummary,
        total_duration: float,
    ) -> PipelineSummary:
        """Generate end-to-end pipeline summary.

        Calculates derived statistics from phase summaries:
        - End-to-end success: Products scraped AND produced
        - Partial success: Products scraped but not produced
        - Total failures: Products that failed in either phase

        Logs formatted summary with all phase statistics and
        end-to-end metrics.

        Args:
        ----
            scraping: Scraping phase summary
            production: Video production phase summary
            total_duration: Total pipeline duration in seconds

        Returns:
        -------
            PipelineSummary with aggregated end-to-end statistics

        """
        # Calculate end-to-end metrics
        end_to_end_success = production.successful
        partial_success = scraping.successful - production.total_attempted
        total_failures = scraping.failed + production.failed

        summary = PipelineSummary(
            scraping=scraping,
            production=production,
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

    from src.pipeline.config import (
        load_global_batch_config,
        validate_global_batch_config,
    )
    from src.utils.logging_setup import setup_debug_logging
    from src.video.config import load_video_config

    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set up logging early
    log_file = Path("logs/global_pipeline.log")
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
        video_config = load_video_config()

        # Validate configuration
        logger.info("Validating configuration...")
        validate_global_batch_config(config, video_config)

        logger.info("Configuration validated successfully")
        logger.info(f"Inputs: {len(config.product_ids or [])} product IDs, {len(config.keywords or [])} keywords")

        if config.profile:
            logger.info(f"Profile: {config.profile} (fixed)")
        elif config.random_profile:
            pool_info = ", ".join(config.profile_pool) if config.profile_pool else "all available"
            logger.info(f"Profile: random selection from [{pool_info}]")

        logger.info(f"Outputs directory: {config.outputs_dir}")
        logger.info(f"Fail-fast: {config.fail_fast}")

        # Execute pipeline
        orchestrator = GlobalPipelineOrchestrator(config)
        summary = await orchestrator.run_pipeline()

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
