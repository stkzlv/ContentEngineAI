"""The dry-run plan for a batch: what it would scrape, render and publish.

Moved out of `global_batch.py` (#450). Read-only: it prints what the
orchestrator is about to do and touches nothing.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from src.pipeline.config import GlobalBatchConfig
from src.video.producer.utils import eligible_random_profiles


def _batch_helpers():
    """The three orchestrator-module helpers the plan reads.

    Imported at call time rather than at module scope: `global_batch`
    imports this module for its delegating method, so a module-scope
    import here would be a cycle, and nine tests patch these names on
    `src.pipeline.global_batch`, which a call-time import still honours.
    """
    from src.pipeline.global_batch import (
        _clean_targets,
        _named_run_ids,
        _publisher_settings,
    )

    return _clean_targets, _named_run_ids, _publisher_settings


def display_execution_plan(
    config: GlobalBatchConfig,
    video_config: Any,
    resumed_topic_ids: Callable[[], list[str]],
) -> None:
    """Display planned execution without running the pipeline.

    Shows configuration validation results and planned actions for each phase.

    Args:
    ----
        config: The batch configuration the plan describes.
        video_config: The loaded video config, for profile lookups.
        resumed_topic_ids: Reads the saved state's topic ids on a
            `--resume`, when `config.topics` is empty.

    """
    _clean_targets, _named_run_ids, _publisher_settings = _batch_helpers()
    separator = "=" * 80
    section = "-" * 40

    print(f"\n{separator}")
    print("DRY RUN - EXECUTION PLAN")
    print(f"{separator}\n")

    # What --clean would remove. The plan exists to answer that before
    # the directories are gone, and it is the one companion flag whose
    # effect cannot be undone.
    if config.clean:
        print(f"{section}")
        print("CLEAN")
        print(f"{section}")
        targets = _clean_targets(config.outputs_dir, _named_run_ids(config))
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

    # An alternating run holds one pool empty by design; naming the drawn
    # side keeps that from reading as a misconfigured run.
    if config.alternated_format:
        print(
            f"  Alternating formats: today draws the "
            f"{config.alternated_format} side"
        )

    # A resumed topics run scrapes nothing either, and its topics are not
    # on the config -- reading only `topics` printed a full keyword plan
    # for a run that would render the saved topic and search for nothing.
    resumed_ids = resumed_topic_ids()
    has_topics = bool(config.topics) or bool(resumed_ids)
    # A mixed run does both, so only a run with nothing to scrape may
    # suppress the scraping half. Suppressing it on a mixed run hides work
    # the run will do -- the same defect as printing work it would
    # discard, in the other direction.
    topics_only = has_topics and not (config.keywords or config.product_ids)
    if has_topics:
        # Named rather than omitted: a plan that simply prints nothing
        # under SCRAPING reads as a misconfigured run. Worded as "prepared"
        # rather than "skipped" because the topic IS produced -- only the
        # scraping is skipped, and on a mixed run "skipped" reads as if the
        # topic will not be rendered at all.
        named = [spec.title for spec in config.topics] or resumed_ids
        print(f"  Prepared without scraping: {len(named)} topic(s)")
        for title in named[:10]:
            print(f"    - {title}")
        if len(named) > 10:
            print(f"    ... and {len(named) - 10} more")

    # Everything below describes scraping, which a topics-only run does
    # not do. Printing it anyway promised work the run would discard,
    # which is the one thing the plan exists to rule out.
    if config.product_ids and not topics_only:
        print(f"  Product IDs to scrape: {len(config.product_ids)}")
        for pid in config.product_ids[:10]:  # Show first 10
            print(f"    - {pid}")
        if len(config.product_ids) > 10:
            print(f"    ... and {len(config.product_ids) - 10} more")

    if config.keywords and not topics_only:
        print(f"  Keywords to search: {len(config.keywords)}")
        for kw in config.keywords[:5]:  # Show first 5
            kw_limit = config.products_per_keyword
            print(f'    - "{kw}" (max {kw_limit} per keyword)')
        if len(config.keywords) > 5:
            print(f"    ... and {len(config.keywords) - 5} more")
        print(f"  Global limit: {config.max_products} products total")

    # Show filters
    filters = config.scraper_filters
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
    if not (config.skip_publish or getattr(config, "force", False)):
        print(
            "  Filter: Skip products already published on every target "
            "platform (topics are exempt)"
        )
    print("  Validation: Check data.json exists and has images/videos")
    print()

    # Phase 3: Video Production Plan
    print(f"{section}")
    print("PHASE 3: VIDEO PRODUCTION")
    print(f"{section}")

    if config.profile:
        print("  Profile mode: Fixed")
        print(f"  Profile: {config.profile}")

        # Show profile details if available
        if config.profile in video_config.video_profiles:
            profile = video_config.video_profiles[config.profile]
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
    elif config.random_profile:
        print("  Profile mode: Random selection")
        pool = config.profile_pool or eligible_random_profiles(video_config)
        print(f"  Profile pool ({len(pool)} profiles):")
        for p in pool[:5]:
            print(f"    - {p}")
        if len(pool) > 5:
            print(f"    ... and {len(pool) - 5} more")

        # A topic draws from its own pool, so a mixed run has two. Printing
        # only the product one leaves the plan silent about which profile
        # the topics in it will actually use.
        topic_pool = config.topic_profile_pool
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

    if config.skip_publish:
        print("  Status: SKIPPED (--skip-publish)")
    else:
        published = _publisher_settings()
        platforms = config.platforms or [p.value for p in published.default_platforms]
        print(f"  Platforms: {', '.join(platforms)}")

        # Check API key
        api_key = os.getenv("LATE_API_KEY")
        if api_key:
            print("  API Key: ✓ LATE_API_KEY is set")
        else:
            print("  API Key: ✗ LATE_API_KEY NOT SET (will fail at runtime)")

        # Show scheduling mode
        if config.schedule_time:
            print(f"  Scheduling: Explicit time ({config.schedule_time})")
        else:
            immediate = published.immediate_publish
            recurring = published.schedule_config.enabled
            if not immediate and recurring:
                print("  Scheduling: Auto-schedule (find next available slot)")
            else:
                print("  Scheduling: Immediate publish")

    print()

    # Common Options
    print(f"{section}")
    print("COMMON OPTIONS")
    print(f"{section}")
    print(f"  Outputs directory: {config.outputs_dir}")
    print(f"  Fail-fast: {config.fail_fast}")
    print(f"  Debug mode: {config.debug}")

    print()
    print(f"{separator}")
    print("DRY RUN COMPLETE - No actions were executed")
    print(f"{separator}\n")
