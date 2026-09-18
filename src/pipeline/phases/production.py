"""The batch's production phase: one render per ready product.

Moved out of `global_batch.py` (#450). The producer orchestration and
`aiohttp` stay function-local imports so this module's closure is as
light as the orchestrator's was.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.pipeline.config import GlobalBatchConfig, ProductionPhaseSummary
from src.scraper.amazon.models import ProductData
from src.utils.logging_setup import log_context
from src.utils.pipeline_deadline import set_pipeline_deadline
from src.video.config_adapter import load_video_config_modular
from src.video.producer.utils import (
    ProfileUsageTracker,
    collect_producer_secrets,
    select_profile_for_product,
)

logger = logging.getLogger(__name__)


async def run_production_phase(
    batch_config: GlobalBatchConfig,
    products: list[tuple[Path, ProductData]],
    build_cli_overrides: Callable[[], dict[str, Any] | None],
    already_published: list[str],
) -> tuple[ProductionPhaseSummary, list[tuple[Path, str]]]:
    """Execute video production phase and return summary with produced videos.

    Processes each product through video pipeline with configured profile,
    supports both fixed and random profile modes, tracks statistics.

    Args:
    ----
        batch_config: The batch configuration: profile mode, pools,
            fail-fast, debug.
        products: List of (product_dir, ProductData) tuples to process
        build_cli_overrides: Builds the producer's CLI override dict from
            the batch configuration; called once per product.
        already_published: Product ids the handoff dropped as already
            published, carried into the summary so a run that rendered
            nothing says why.

    Returns:
    -------
        Tuple of (ProductionPhaseSummary, list of (video_path, product_id) tuples)

    """
    import aiohttp

    from src.video.config import load_video_config
    from src.video.producer.orchestration import (
        create_video_for_product,
        failed_step_from_result,
    )

    phase_start = time.time()

    # Load video configuration
    config = load_video_config_modular()

    # Build secrets dict from environment variables (shared definition)
    secrets = collect_producer_secrets(config)

    # Initialize profile tracking if random mode
    profile_tracker: ProfileUsageTracker | None = None
    if batch_config.random_profile:
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
            with log_context(product_id=product_id):
                # Select profile for this product. A topic draws from its own
                # pool: it has no product photography, so a profile that
                # sources only scraped media gathers nothing and the render
                # fails outright. On a topics-only run the two pools are the
                # same list; on a mixed run they are close to complements.
                is_topic = bool(getattr(product, "topic", None))
                pool = (
                    batch_config.topic_profile_pool
                    if is_topic and batch_config.topic_profile_pool
                    else batch_config.profile_pool
                )
                if batch_config.random_profile:
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
                    assert batch_config.profile is not None
                    current_profile = batch_config.profile
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
                    cli_overrides = build_cli_overrides()

                    # Call video producer with timeout
                    # See the producer CLI: an inner limit must not exceed
                    # the budget this `wait_for` enforces (#398).
                    set_pipeline_deadline(config.pipeline_timeout_sec)
                    result_path = await asyncio.wait_for(
                        create_video_for_product(
                            config=config,
                            product=product,
                            profile_name=current_profile,
                            secrets=secrets,
                            session=session,
                            debug_mode=batch_config.debug,
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
                        if batch_config.fail_fast:
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
                        if batch_config.fail_fast:
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

                    if batch_config.fail_fast:
                        logger.error("Fail-fast enabled, stopping production phase")
                        break

                except Exception as e:
                    failed += 1
                    failed_products.append(product_id)
                    logger.exception(
                        "[%s/%s] Failed to process %s: %s",
                        idx,
                        total_products,
                        product_id,
                        e,
                    )

                    if batch_config.fail_fast:
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
