"""The batch's scraping phase: one browser session across every input.

Moved out of `global_batch.py` (#450).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from src.pipeline.config import GlobalBatchConfig, ScrapingPhaseSummary
from src.scraper.base.keyword_pillars import pillar_for as keyword_pillar_for

logger = logging.getLogger(__name__)


async def run_scraping_phase(
    config: GlobalBatchConfig,
    resolve_profile_uses_videos: Callable[[], bool | None],
) -> ScrapingPhaseSummary:
    """Execute scraping phase and return summary.

    Uses a two-phase approach to avoid launching a separate Chrome process
    per keyword:
      1. Batch browser phase: one Chrome session scrapes ALL inputs
      2. Per-keyword post-processing: download media, validate, apply limits

    Args:
    ----
        config: The batch configuration: inputs, limits, filters, fail-fast.
        resolve_profile_uses_videos: Reads whether the target profile(s)
            use scraped videos, or None when that cannot be known up
            front; called once, before the scraper is built.

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
    if config.product_ids:
        all_inputs.extend(config.product_ids)
    if config.keywords:
        all_inputs.extend(config.keywords)

    total_inputs = len(all_inputs)
    logger.info("Scraping %d input(s): %s", total_inputs, ", ".join(all_inputs))
    logger.info(
        "Limits: %s per keyword, %s total",
        config.products_per_keyword,
        config.max_products,
    )

    # Initialize scraper with profile-aware validation
    profile_uses_videos = resolve_profile_uses_videos()
    if profile_uses_videos is not None:
        logger.info(
            "Scraper validation aligned with profile: videos %s",
            "enabled" if profile_uses_videos else "disabled (image-only)",
        )
    scraper = BotasaurusAmazonScraper(
        debug_override=config.debug,
        profile_uses_videos=profile_uses_videos,
    )

    scraper.run_max_products = config.products_per_keyword

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
            all_inputs, search_params=config.scraper_filters
        )
    except Exception as e:
        logger.error("Batch browser scrape failed: %s", e)
        if config.fail_fast:
            raise
        batch_results = []

    # Build a lookup so we can iterate in original order
    results_by_input: dict[str, list[dict]] = {}
    for entry in batch_results:
        results_by_input[entry["input"]] = entry.get("products", [])

    # --- Phase 2: per-keyword post-processing ---
    logger.info("Phase 2: media download and validation")
    for idx, input_item in enumerate(all_inputs, 1):
        if len(successful_products) >= config.max_products:
            logger.info(
                "Reached max_products limit (%s). "
                "Stopping with %s inputs remaining.",
                config.max_products,
                len(all_inputs) - idx + 1,
            )
            break

        remaining = config.max_products - len(successful_products)
        per_input_limit = min(config.products_per_keyword, remaining)
        scraper.run_max_products = per_input_limit

        collected = f"{len(successful_products)}/{config.max_products}"
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
            if config.fail_fast:
                logger.error("Fail-fast enabled, stopping scraping phase")
                break
            continue

        input_pillar = keyword_pillar_for(input_item, config.keyword_pillar_map)

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
                        search_params=config.scraper_filters,
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
                    # Files downloaded and validated, not URLs found on the
                    # page: the summary said 15 images where 10 were on disk.
                    total_images += len(getattr(product, "downloaded_images", []))
                    total_videos += len(getattr(product, "downloaded_videos", []))
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
                if config.fail_fast:
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
            if config.fail_fast:
                logger.error("Fail-fast enabled, stopping scraping phase")
                raise

    # Generate summary
    duration = time.time() - phase_start
    media_stats = {"total_images": total_images, "total_videos": total_videos}

    logger.info(
        "Scraping phase complete: %s products from %s inputs " "(%s failed) in %.1fs",
        len(successful_products),
        inputs_processed,
        inputs_failed,
        duration,
    )

    for line in scraper.throttle.summary_lines():
        logger.warning("%s", line)

    return ScrapingPhaseSummary(
        total_attempted=total_inputs,
        successful=inputs_processed,
        failed=inputs_failed,
        successful_products=successful_products,
        failed_products=failed_inputs,
        dead_queries=scraper.throttle.dead_queries,
        throttled_inputs=scraper.throttle.throttled_inputs,
        media_stats=media_stats,
        duration_sec=duration,
    )
