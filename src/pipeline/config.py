"""Configuration data models for global batch pipeline.

This module defines unified configuration and summary structures
for orchestrating scraping and video production phases.

Data Models:
    - GlobalBatchConfig: Unified pipeline configuration
    - ScrapingPhaseSummary: Scraping phase statistics
    - ProductionPhaseSummary: Video production phase statistics
    - PublishingPhaseSummary: Publishing phase statistics
    - PipelineSummary: End-to-end pipeline summary
    - PipelineState: Pipeline state for resume capability

Configuration Functions:
    - load_global_batch_config: Load configuration with CLI > YAML > defaults precedence
    - validate_global_batch_config: Validate configuration before pipeline execution

State Persistence Functions:
    - save_pipeline_state: Save pipeline state to JSON file
    - load_pipeline_state: Load pipeline state from JSON file

Configuration Precedence:
    CLI arguments > YAML configuration > Default values
"""

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from src.scraper.amazon.models import SearchParameters
from src.scraper.base.keyword_pillars import read_keyword_pillars
from src.video.config import VideoConfig
from src.video.producer.topic_input import (
    TopicSpec,
    specs_from_args,
    specs_from_mappings,
)

logger = logging.getLogger(__name__)


class PipelinePhase(str, Enum):
    """Pipeline execution phases for state tracking.

    Phases execute in order: SCRAPING → HANDOFF → PRODUCTION → PUBLISHING → COMPLETED
    """

    SCRAPING = "scraping"
    HANDOFF = "handoff"
    PRODUCTION = "production"
    PUBLISHING = "publishing"
    COMPLETED = "completed"


@dataclass
class PipelineState:
    """Pipeline state for resume capability.

    Tracks pipeline progress to enable resuming interrupted pipelines
    without reprocessing already-completed products.

    Attributes
    ----------
        run_id: Unique identifier for this pipeline run
        started_at: ISO timestamp when pipeline started
        updated_at: ISO timestamp when state was last updated
        current_phase: Current phase being executed
        config_snapshot: Original configuration (preserved for resume)
        completed_phases: List of phases that completed successfully
        scraping_completed_products: ASINs successfully scraped
        scraping_failed_products: ASINs that failed scraping
        production_completed_products: ASINs with videos successfully produced
        production_failed_products: ASINs that failed video production
        production_skipped_products: ASINs skipped (insufficient media)
        publishing_completed_products: ASINs successfully published
        publishing_failed_products: ASINs that failed publishing
        scraping_summary: Serialized ScrapingPhaseSummary (if phase completed)
        production_summary: Serialized ProductionPhaseSummary (if phase completed)
        publishing_summary: Serialized PublishingPhaseSummary (if phase completed)

    """

    run_id: str
    started_at: str
    updated_at: str
    current_phase: PipelinePhase
    config_snapshot: dict[str, Any]
    completed_phases: list[str] = field(default_factory=list)
    scraping_completed_products: list[str] = field(default_factory=list)
    scraping_failed_products: list[str] = field(default_factory=list)
    production_completed_products: list[str] = field(default_factory=list)
    production_failed_products: list[str] = field(default_factory=list)
    production_skipped_products: list[str] = field(default_factory=list)
    publishing_completed_products: list[str] = field(default_factory=list)
    publishing_failed_products: list[str] = field(default_factory=list)
    scraping_summary: dict[str, Any] | None = None
    production_summary: dict[str, Any] | None = None
    publishing_summary: dict[str, Any] | None = None

    @classmethod
    def create_new(cls, config: "GlobalBatchConfig") -> "PipelineState":
        """Create new pipeline state from configuration.

        Args:
        ----
            config: Global batch configuration

        Returns:
        -------
            New PipelineState initialized for fresh pipeline run

        """
        import uuid

        now = datetime.now(UTC).isoformat()
        run_id = (
            f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        # Serialize config to dict (excluding non-serializable fields)
        config_dict = {
            "product_ids": config.product_ids,
            "keywords": config.keywords,
            "max_products": config.max_products,
            "products_per_keyword": config.products_per_keyword,
            "scraper_filters": {
                "min_price": config.scraper_filters.min_price,
                "max_price": config.scraper_filters.max_price,
                "min_rating": config.scraper_filters.min_rating,
                "prime_only": config.scraper_filters.prime_only,
            },
            "profile": config.profile,
            "random_profile": config.random_profile,
            "profile_pool": config.profile_pool,
            "fail_fast": config.fail_fast,
            "process_all_products": config.process_all_products,
            "outputs_dir": str(config.outputs_dir),
            "debug": config.debug,
            "skip_publish": config.skip_publish,
            "platforms": config.platforms,
            "schedule_time": config.schedule_time,
            "fail_fast_publish": config.fail_fast_publish,
        }

        return cls(
            run_id=run_id,
            started_at=now,
            updated_at=now,
            current_phase=PipelinePhase.SCRAPING,
            config_snapshot=config_dict,
        )

    def mark_phase_complete(self, phase: PipelinePhase) -> None:
        """Mark a phase as completed.

        Args:
        ----
            phase: Phase that completed

        """
        if phase.value not in self.completed_phases:
            self.completed_phases.append(phase.value)
        self.updated_at = datetime.now(UTC).isoformat()

    def advance_phase(self, next_phase: PipelinePhase) -> None:
        """Advance to the next phase.

        Args:
        ----
            next_phase: Phase to advance to

        """
        self.current_phase = next_phase
        self.updated_at = datetime.now(UTC).isoformat()

    def is_phase_completed(self, phase: PipelinePhase) -> bool:
        """Check if a phase has been completed.

        Args:
        ----
            phase: Phase to check

        Returns:
        -------
            True if phase is in completed_phases

        """
        return phase.value in self.completed_phases


def get_state_file_path(outputs_dir: Path) -> Path:
    """Get the path to the pipeline state file.

    Args:
    ----
        outputs_dir: Base outputs directory

    Returns:
    -------
        Path to .pipeline_state.json file

    """
    return outputs_dir / ".pipeline_state.json"


def save_pipeline_state(state: PipelineState, outputs_dir: Path) -> None:
    """Save pipeline state to JSON file.

    Writes state atomically by writing to temp file first, then renaming.

    Args:
    ----
        state: Pipeline state to save
        outputs_dir: Directory to save state file

    """
    state_path = get_state_file_path(outputs_dir)
    temp_path = state_path.with_suffix(".tmp")

    # Ensure outputs directory exists
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Update timestamp
    state.updated_at = datetime.now(UTC).isoformat()

    # Convert to dict for serialization
    state_dict = asdict(state)
    # Convert PipelinePhase enum to string
    state_dict["current_phase"] = state.current_phase.value

    # Write atomically
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=2)

    # Atomic rename
    temp_path.rename(state_path)
    logger.debug(f"Saved pipeline state to {state_path}")


def load_pipeline_state(outputs_dir: Path) -> PipelineState | None:
    """Load pipeline state from JSON file.

    Handles corrupted state files gracefully by returning None.

    Args:
    ----
        outputs_dir: Directory containing state file

    Returns:
    -------
        PipelineState if valid state file exists, None otherwise

    """
    state_path = get_state_file_path(outputs_dir)

    if not state_path.exists():
        logger.debug(f"No pipeline state file found at {state_path}")
        return None

    try:
        with open(state_path, encoding="utf-8") as f:
            state_dict = json.load(f)

        # Convert current_phase string back to enum
        state_dict["current_phase"] = PipelinePhase(state_dict["current_phase"])

        state = PipelineState(**state_dict)
        logger.info(f"Loaded pipeline state from {state_path}")
        logger.info(f"  Run ID: {state.run_id}")
        logger.info(f"  Current phase: {state.current_phase.value}")
        logger.info(
            f"  Completed phases: {', '.join(state.completed_phases) or 'none'}"
        )

        return state

    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted state file at {state_path}: {e}")
        logger.warning("State file will be ignored. Starting fresh pipeline.")
        return None

    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"Invalid state file format at {state_path}: {e}")
        logger.warning("State file will be ignored. Starting fresh pipeline.")
        return None


def clear_pipeline_state(outputs_dir: Path) -> None:
    """Remove pipeline state file.

    Args:
    ----
        outputs_dir: Directory containing state file

    """
    state_path = get_state_file_path(outputs_dir)
    if state_path.exists():
        state_path.unlink()
        logger.debug(f"Cleared pipeline state at {state_path}")


@dataclass
class GlobalBatchConfig:
    """Unified configuration for global batch pipeline.

    Combines scraper and producer settings for end-to-end automation.

    Attributes
    ----------
        product_ids: List of ASINs to scrape directly
        keywords: List of keywords to search for products
        topics: Topics to render without scraping
        topics_per_run: How many configured topics a no-flag run includes
        max_products: Maximum total products to collect across all keywords (global cap)
        products_per_keyword: Maximum products to scrape per individual keyword
        scraper_filters: SearchParameters for filtering products
        profile: Fixed video profile name (mutually exclusive with random_profile)
        random_profile: Enable random profile selection per product
        profile_pool: List of profiles for random selection
        fail_fast: Stop pipeline on first failure
        process_all_products: Process all products in outputs dir
                              (default: only current run)
        outputs_dir: Directory for scraper output and producer input
        debug: Enable debug logging

    """

    # Scraper configuration
    product_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    keyword_pillar_map: dict[str, str] = field(default_factory=dict)
    # Topics render without a scraper run, so they are an input source in their
    # own right rather than a filter on one.
    topics: list[TopicSpec] = field(default_factory=list)
    # How many of the configured topics a run with no CLI inputs includes,
    # alongside the configured keywords. Topics named on the command line are
    # taken in full and ignore this.
    topics_per_run: int = 1
    max_products: int = 10
    products_per_keyword: int = 1
    scraper_filters: SearchParameters = field(default_factory=SearchParameters)

    # Producer configuration
    profile: str | None = None
    random_profile: bool = False
    profile_pool: list[str] = field(default_factory=list)
    # Whether the pool was named for THIS run or inherited from YAML. A topics
    # run cannot use a product-run default, but must still refuse a pool the
    # operator named on the command line rather than silently replacing it.
    profile_pool_from_cli: bool = False
    # Profiles a topic record may draw. Separate from `profile_pool` because a
    # run can carry both kinds of record, and the two need opposite profiles:
    # a topic has no product photography and a product profile gathers nothing
    # for it. Populated by validation, not by the loader.
    topic_profile_pool: list[str] = field(default_factory=list)
    # Set when a `--resume` is picking up a topics run. Topics themselves are
    # not persisted -- the identifier carries a one-way digest of the title --
    # so the run is recognised from the ids in the saved state instead. Every
    # topic rule keys off this as well as `topics`, or a resume reaches
    # combinations a fresh run refuses.
    topics_resume: bool = False
    # Whether that resume also carries scraped products. Read from the saved
    # state, not from `keywords`: a resume inherits the configured keywords
    # whatever it is resuming, and the completed scraping phase already
    # ignored them.
    resume_has_products: bool = False

    # Common configuration
    fail_fast: bool = False
    process_all_products: bool = False
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))
    debug: bool = False

    # Publishing configuration
    skip_publish: bool = False
    # Render a product `publish_history.json` already records as published on
    # every target platform. Off by default: the batch used to pay the whole
    # render for such a product and let the publish phase drop it.
    force: bool = False
    platforms: list[str] | None = None
    schedule_time: str | None = None
    fail_fast_publish: bool = False
    platform_specific_content: bool = False

    # Voice profile override
    voice_profile: str | None = None

    # Script template override
    script_template: str | None = None

    # Content pillar override (filters template pool and adds runtime preamble)
    pillar: str | None = None

    # Pycaps subtitle engine overrides (all optional; None = inherit from YAML)
    subtitle_format: str | None = None
    subtitle_engine: str | None = None
    pycaps_template: str | None = None
    pycaps_template_pool: list[str] | None = None
    pycaps_renderer: str | None = None

    # Resume configuration
    resume: bool = False

    # Dry-run configuration
    dry_run: bool = False

    # Clean outputs before run
    clean: bool = False

    # Output format configuration
    output_format: str = "text"  # "text" or "json"


@dataclass
class ScrapingPhaseSummary:
    """Scraping phase statistics.

    Tracks scraping phase execution and results.

    Attributes
    ----------
        total_attempted: Total number of products attempted to scrape
        successful: Number of products scraped successfully
        failed: Number of products that failed to scrape
        successful_products: List of product IDs scraped successfully
        failed_products: List of product IDs that failed
        media_stats: Media statistics (e.g., total_images, total_videos)
        duration_sec: Phase duration in seconds

    """

    total_attempted: int
    successful: int
    failed: int
    successful_products: list[str]
    failed_products: list[str]
    media_stats: dict[str, int]
    duration_sec: float


@dataclass
class ProductionPhaseSummary:
    """Video production phase statistics.

    Tracks video production phase execution and results.

    Attributes
    ----------
        total_attempted: Total number of products attempted to process
        successful: Number of videos produced successfully
        failed: Number of products that failed video production
        skipped: Number of products skipped (insufficient media)
        failed_products: List of product IDs that failed
        skipped_products: List of product IDs that were skipped
        profile_distribution: Profile usage counts (only if randomization enabled)
        duration_sec: Phase duration in seconds

    """

    total_attempted: int
    successful: int
    failed: int
    skipped: int
    failed_products: list[str]
    skipped_products: list[str]
    profile_distribution: dict[str, int] | None
    duration_sec: float


@dataclass
class PublishingPhaseSummary:
    """Publishing phase statistics.

    Tracks publishing phase execution and per-platform results.

    Attributes
    ----------
        total_attempted: Total number of videos attempted to publish
        successful: Number of videos published successfully to ALL platforms
        failed: Number of videos that failed on ANY platform
        skipped: Number of videos skipped (no metadata/pre-publish errors)
        failed_videos: List of product IDs that failed
        skipped_videos: List of product IDs that were skipped
        platform_results: Per-platform success/failure counts
        errors: Detailed error information per video
        duration_sec: Phase duration in seconds

    """

    total_attempted: int
    successful: int
    failed: int
    skipped: int
    failed_videos: list[str]
    skipped_videos: list[str]
    platform_results: dict[str, dict[str, int]]
    errors: list[dict[str, str]]
    duration_sec: float


@dataclass
class PipelineSummary:
    """End-to-end pipeline statistics.

    Aggregates statistics across all phases and calculates
    derived end-to-end metrics.

    Attributes
    ----------
        scraping: Scraping phase summary
        production: Video production phase summary
        publishing: Publishing phase summary (None if --skip-publish)
        end_to_end_success: Products scraped AND produced successfully
        partial_success: Products scraped successfully but not produced
        total_failures: Products that failed in either phase
        total_duration_sec: Total pipeline duration in seconds

    """

    scraping: ScrapingPhaseSummary
    production: ProductionPhaseSummary
    publishing: PublishingPhaseSummary | None
    end_to_end_success: int
    partial_success: int
    total_failures: int
    total_duration_sec: float

    def total_skipped(self) -> int:
        """Products that did not get through without a step having broken.

        Counted apart from failures everywhere else, because a skip names a
        different cause. For the exit code they are the same thing: a video
        that was asked for and does not exist.
        """
        skipped = self.production.skipped
        if self.publishing is not None:
            skipped += self.publishing.skipped
        return skipped

    def outcome(self) -> str:
        """What the run did: ``"failed"``, ``"lost"`` or ``"succeeded"``.

        Derived here so the end-of-run verdict and the exit code cannot
        disagree. They were computed separately once, and a run that lost
        products only to skips exited non-zero while logging that it had
        completed successfully.
        """
        if self.end_to_end_success == 0:
            return "failed"
        if self.total_failures > 0 or self.total_skipped() > 0:
            return "lost"
        return "succeeded"

    def exit_code(self, strict: bool = False) -> int:
        """Process exit code for the run.

        By default 0 if any product completed end-to-end, else 1, so CI,
        cron, and wrappers checking ``$?`` see a run that produced nothing
        as a failure. A partial loss still exits 0: a batch that loses one
        product of twenty has done most of what was asked, and failing the
        whole run for it would stop a schedule over a single bad ASIN.

        ``strict`` makes any lost product non-zero, whether it was lost to
        a failure or to a skip. A profile misconfigured so that every
        product is rejected for insufficient media loses the whole run
        while reporting no failures at all, which is precisely the silence
        the flag exists to break.
        """
        if self.end_to_end_success == 0:
            return 1
        if strict and self.outcome() == "lost":
            return 1
        return 0

    def format(self) -> str:
        """Format pipeline summary as human-readable string.

        Returns
        -------
            Formatted multi-line summary string

        """
        separator = "=" * 80
        lines = [
            "",
            separator,
            "GLOBAL PIPELINE SUMMARY",
            separator,
            "",
            "SCRAPING PHASE:",
            f"  Total Attempted: {self.scraping.total_attempted}",
            f"  Successful: {self.scraping.successful}",
            f"  Failed: {self.scraping.failed}",
        ]

        if self.scraping.failed_products:
            lines.append(
                f"  Failed Products: {', '.join(self.scraping.failed_products)}"
            )

        lines.extend(
            [
                "",
                "  Media Statistics:",
                f"    - Total Images: "
                f"{self.scraping.media_stats.get('total_images', 0)}",
                f"    - Total Videos: "
                f"{self.scraping.media_stats.get('total_videos', 0)}",
                f"  Duration: {self.scraping.duration_sec:.1f}s",
                "",
                "VIDEO PRODUCTION PHASE:",
                f"  Total Attempted: {self.production.total_attempted}",
                f"  Successful: {self.production.successful}",
                f"  Failed: {self.production.failed}",
                f"  Skipped: {self.production.skipped}",
            ]
        )

        if self.production.skipped_products:
            lines.append(
                f"  Skipped Products (insufficient media): "
                f"{', '.join(self.production.skipped_products)}"
            )

        if self.production.failed_products:
            lines.append(
                f"  Failed Products: {', '.join(self.production.failed_products)}"
            )

        # Profile distribution (only if random profile mode)
        if self.production.profile_distribution:
            lines.append("")
            lines.append("  Profile Distribution:")
            total_uses = sum(self.production.profile_distribution.values())
            for profile_name, count in sorted(
                self.production.profile_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                percentage = (count / total_uses) * 100
                lines.append(f"    - {profile_name}: {count} ({percentage:.1f}%)")

        lines.append(f"  Duration: {self.production.duration_sec:.1f}s")

        # Publishing phase statistics (only if publishing was enabled)
        if self.publishing:
            lines.extend(
                [
                    "",
                    "PUBLISHING PHASE:",
                    f"  Total Attempted: {self.publishing.total_attempted}",
                    f"  Successful: {self.publishing.successful}",
                    f"  Failed: {self.publishing.failed}",
                    f"  Skipped: {self.publishing.skipped}",
                ]
            )

            if self.publishing.skipped_videos:
                lines.append(
                    f"  Skipped Videos (no metadata): "
                    f"{', '.join(self.publishing.skipped_videos)}"
                )

            if self.publishing.failed_videos:
                lines.append(
                    f"  Failed Videos: {', '.join(self.publishing.failed_videos)}"
                )

            # Per-platform success rates
            if self.publishing.platform_results:
                lines.append("")
                lines.append("  Platform Results:")
                for platform, stats in sorted(self.publishing.platform_results.items()):
                    successful = stats.get("successful", 0)
                    failed = stats.get("failed", 0)
                    total = successful + failed
                    if total > 0:
                        success_rate = (successful / total) * 100
                        lines.append(
                            f"    - {platform.title()}: "
                            f"{successful}/{total} ({success_rate:.1f}%)"
                        )

            lines.append(f"  Duration: {self.publishing.duration_sec:.1f}s")

        lines.extend(
            [
                "",
                "END-TO-END RESULTS:",
                f"  Complete Success (scraped + produced): {self.end_to_end_success}",
                f"  Partial Success (scraped only): {self.partial_success}",
                f"  Total Failures: {self.total_failures}",
                "",
                f"Total Pipeline Duration: {self.total_duration_sec:.1f}s",
                separator,
            ]
        )

        return "\n".join(lines)

    def to_dict(self, started_at: str | None = None) -> dict[str, Any]:
        """Convert pipeline summary to JSON-serializable dictionary.

        Args:
        ----
            started_at: ISO timestamp when pipeline started (optional)

        Returns:
        -------
            Dictionary with all summary fields for JSON serialization

        """
        completed_at = datetime.now(UTC).isoformat()

        result: dict[str, Any] = {
            "pipeline": {
                "started_at": started_at,
                "completed_at": completed_at,
                "total_duration_sec": round(self.total_duration_sec, 2),
            },
            "scraping": {
                "total_attempted": self.scraping.total_attempted,
                "successful": self.scraping.successful,
                "failed": self.scraping.failed,
                "successful_products": self.scraping.successful_products,
                "failed_products": self.scraping.failed_products,
                "media_stats": self.scraping.media_stats,
                "duration_sec": round(self.scraping.duration_sec, 2),
            },
            "production": {
                "total_attempted": self.production.total_attempted,
                "successful": self.production.successful,
                "failed": self.production.failed,
                "skipped": self.production.skipped,
                "failed_products": self.production.failed_products,
                "skipped_products": self.production.skipped_products,
                "profile_distribution": self.production.profile_distribution,
                "duration_sec": round(self.production.duration_sec, 2),
            },
            "publishing": None,
            "end_to_end": {
                "complete_success": self.end_to_end_success,
                "partial_success": self.partial_success,
                "total_failures": self.total_failures,
            },
        }

        # Add publishing summary if available
        if self.publishing:
            result["publishing"] = {
                "total_attempted": self.publishing.total_attempted,
                "successful": self.publishing.successful,
                "failed": self.publishing.failed,
                "skipped": self.publishing.skipped,
                "failed_videos": self.publishing.failed_videos,
                "skipped_videos": self.publishing.skipped_videos,
                "platform_results": self.publishing.platform_results,
                "errors": self.publishing.errors,
                "duration_sec": round(self.publishing.duration_sec, 2),
            }

        return result

    def to_json(self, started_at: str | None = None, indent: int = 2) -> str:
        """Convert pipeline summary to JSON string.

        Args:
        ----
            started_at: ISO timestamp when pipeline started (optional)
            indent: JSON indentation level (default: 2)

        Returns:
        -------
            JSON string representation of summary

        """
        return json.dumps(self.to_dict(started_at), indent=indent)


def topics_for_run(
    configured: list[TopicSpec], count: int, day_ordinal: int | None = None
) -> list[TopicSpec]:
    """Pick which configured topics this run renders.

    Taken in rotation from the day of the month rather than from the top of the
    list, so a daily cadence works through the list instead of re-rendering the
    first entry every morning. Interleaving matters beyond variety: comparing
    the two content formats fairly needs them mixed through the week rather
    than run in blocks, since a block comparison cannot separate the format
    from whatever else changed that week.

    Stateless on purpose. A cursor file would have to be written by every run,
    survive `--clean`, and be reconciled after a failed batch; the date already
    advances once a day on its own.
    """
    if count <= 0 or not configured:
        return []

    if day_ordinal is None:
        day_ordinal = date.today().toordinal()

    # Capped at the list length rather than wrapping. Wrapping returned the
    # same spec twice, and two entries with one title render into one
    # directory -- so the run wrote one video and counted two, which the
    # summary then reported as a product scraped but never produced.
    count = min(count, len(configured))
    start = day_ordinal % len(configured)
    return [configured[(start + i) % len(configured)] for i in range(count)]


def _scraper_keyword_pool(
    scraper_config_path: str | Path = "config/scraper.yaml",
) -> tuple[list[str], dict[str, str]]:
    """The scraper's `batch.keywords`, read through the shared reader.

    Not an import of the scraper's config loader: that builds a whole
    `BatchConfig` and pulls in the scraper package for what is one key. The
    reader is the same one both loaders already use, so the two files cannot
    disagree about the dict-of-pillars shape.

    Callers pass the sibling of the pipeline config being loaded rather than
    letting this default fire, so pointing the batch at another config
    directory reads that directory's scraper file. Reading the repo's copy
    regardless would mean a test or a fork loading its own pipeline.yaml
    silently picked up keywords from somewhere else.

    A missing or unreadable file yields an empty pool, which leaves the batch
    exactly where it was before this fallback existed.
    """
    path = Path(scraper_config_path)
    if not path.exists():
        return [], {}
    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not read %s for keywords: %s", scraper_config_path, exc)
        return [], {}
    return read_keyword_pillars((raw.get("batch") or {}).get("keywords", []) or [])


def keywords_for_run(
    configured: list[str], count: int, day_ordinal: int | None = None
) -> list[str]:
    """Pick which configured keywords this run searches.

    The batch stops at `max_products`, so a run only ever reaches the first
    few keywords of the list it is given. Taking them from the top every time
    made the effective catalogue as wide as the cap rather than as wide as the
    pool: two runs an hour apart returned the same products, several of them
    already published.

    Rotated by date, like `topics_for_run`, and stateless for the same reason
    -- a cursor would have to survive `--clean` and be reconciled after a
    failed batch, where the date advances on its own.

    The stride is `count`, not 1. `topics_for_run` takes one of a handful, so
    stepping by one already hands back something new; a run taking ten of
    fifty-four that stepped by one would repeat nine of yesterday's ten.
    Stepping by the slice width makes consecutive days disjoint until the pool
    wraps.
    """
    if count <= 0 or not configured:
        return []

    if day_ordinal is None:
        day_ordinal = date.today().toordinal()

    count = min(count, len(configured))
    start = (day_ordinal * count) % len(configured)
    return [configured[(start + i) % len(configured)] for i in range(count)]


def load_global_batch_config(
    cli_args: argparse.Namespace, config_path: str = "config/pipeline.yaml"
) -> GlobalBatchConfig:
    """Load global batch configuration with 3-tier precedence.

    Precedence order: CLI arguments > YAML configuration > Default values

    Args:
    ----
        cli_args: Parsed CLI arguments from argparse
        config_path: Path to YAML configuration file (default: config/pipeline.yaml)

    Returns:
    -------
        GlobalBatchConfig with merged configuration from all sources

    Raises:
    ------
        FileNotFoundError: If YAML file specified but not found
        yaml.YAMLError: If YAML file is malformed

    """
    # Load YAML configuration if file exists
    yaml_config: dict[str, Any] = {}
    yaml_path = Path(config_path)

    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and "global_batch" in loaded:
                yaml_config = loaded["global_batch"]

    # Apply 3-tier precedence: CLI > YAML > defaults
    # When ANY CLI input is provided (product_ids or keywords), treat CLI as
    # the complete input set. Don't mix CLI product_ids with YAML keywords
    # or vice versa. This prevents e.g. --product-ids B0ABC from also
    # picking up 28 YAML keywords.
    cli_product_ids = getattr(cli_args, "product_ids", None)
    cli_keywords = getattr(cli_args, "keywords", None)
    # Topics count as a CLI input set. Leaving them out meant `--topic` fell
    # through to the YAML branch and the run scraped every configured keyword
    # alongside the topic nobody asked it to pair them with.
    cli_topics = specs_from_args(
        topic=getattr(cli_args, "topic", None),
        topic_description=getattr(cli_args, "topic_description", None),
        topic_keywords=getattr(cli_args, "topic_keywords", None),
        topics_file=getattr(cli_args, "topics_file", None),
    )
    cli_has_inputs = cli_product_ids or cli_keywords or cli_topics

    # Build the pillar map from YAML whichever source supplies the keyword
    # list. It describes which pillar a configured keyword belongs to, which is
    # true regardless of how the keyword reached this run, so a CLI keyword that
    # matches a configured one still carries its pillar. Mirrors the standalone
    # scraper (src/scraper/amazon/config.py), which already works this way.
    yaml_keywords, keyword_pillar_map = read_keyword_pillars(
        yaml_config.get("keywords", []) or []
    )

    # An empty `global_batch.keywords` falls back to the scraper's pool rather
    # than meaning "no keywords". The two files each carried their own list,
    # and the batch's was a six-entry subset of the scraper's fifty-four, so
    # the batch searched six however many were configured next door -- and at
    # `products_per_keyword: 1` a six-product run exhausted it exactly, which
    # is why two runs an hour apart returned the same already-published
    # products. Reading one pool is also what makes the rotation below worth
    # anything: rotating a list the cap already consumes whole changes
    # nothing.
    if not yaml_keywords:
        yaml_keywords, scraper_pillars = _scraper_keyword_pool(
            Path(config_path).parent / "scraper.yaml"
        )
        # The batch's own map wins where both name a keyword, so a pillar set
        # here still overrides; everything else comes from the scraper file.
        keyword_pillar_map = {**scraper_pillars, **keyword_pillar_map}

    # How many configured topics a no-flag run includes. CLI topics ignore it:
    # a topic named on the command line was asked for explicitly.
    topics_per_run = yaml_config.get("topics_per_run", 1)
    if not isinstance(topics_per_run, int) or isinstance(topics_per_run, bool):
        raise ValueError(
            f"global_batch.topics_per_run must be an integer, got "
            f"{topics_per_run!r}"
        )

    # Only the configured pool rotates. Keywords typed on the command line
    # were asked for by name, so a run must search exactly those.
    rotate_keywords = False
    if cli_has_inputs:
        product_ids = cli_product_ids or []
        keywords = cli_keywords or []
        topics = cli_topics
    else:
        product_ids = yaml_config.get("product_ids", []) or []
        keywords = yaml_keywords
        rotate_keywords = True
        # Without this the tutorial arm could only enter a run by being typed
        # on that day's command line, so the repeatable path -- the one a
        # scheduled run uses -- produced product renders and nothing else.
        configured_topics = (
            specs_from_mappings(yaml_config["topics"], config_path)
            if yaml_config.get("topics")
            else []
        )
        topics = topics_for_run(configured_topics, topics_per_run)

    # Max products (global cap across all keywords)
    max_products = (
        getattr(cli_args, "max_products", None) or yaml_config.get("max_products") or 10
    )

    # Products per keyword
    products_per_keyword = (
        getattr(cli_args, "products_per_keyword", None)
        or yaml_config.get("products_per_keyword")
        or 1
    )

    # Rotate here rather than where the list is read, because the slice width
    # is what the run will actually consume: the batch stops at
    # `max_products`, so at one product per keyword it reaches that many
    # keywords and no more. Rotating by that width makes consecutive days
    # disjoint; rotating by the whole pool would be no rotation at all, since
    # a start offset of a multiple of the length is zero.
    if rotate_keywords and keywords:
        per_run = yaml_config.get("keywords_per_run") or math.ceil(
            max_products / max(1, products_per_keyword)
        )
        keywords = keywords_for_run(keywords, per_run)

    # Scraper filters (SearchParameters)
    yaml_filters = yaml_config.get("scraper_filters", {})
    scraper_filters = SearchParameters(
        min_price=getattr(cli_args, "min_price", None) or yaml_filters.get("min_price"),
        max_price=getattr(cli_args, "max_price", None) or yaml_filters.get("max_price"),
        min_rating=getattr(cli_args, "min_rating", None)
        or yaml_filters.get("min_rating"),
        prime_only=getattr(cli_args, "prime_only", False)
        or yaml_filters.get("prime_only", False),
    )

    # Profile configuration
    profile = getattr(cli_args, "profile", None) or yaml_config.get("profile")

    random_profile = getattr(cli_args, "random_profile", False) or yaml_config.get(
        "random_profile", False
    )

    # Default to random profile when no explicit profile is set
    if not profile and not random_profile:
        random_profile = True

    cli_profile_pool = getattr(cli_args, "profile_pool", None)
    profile_pool = cli_profile_pool or yaml_config.get("profile_pool", []) or []

    # Common configuration
    fail_fast = getattr(cli_args, "fail_fast", False) or yaml_config.get(
        "fail_fast", False
    )

    process_all_products = getattr(
        cli_args, "process_all_products", False
    ) or yaml_config.get("process_all_products", False)

    outputs_dir_str = getattr(cli_args, "outputs_dir", None) or yaml_config.get(
        "outputs_dir", "outputs"
    )
    outputs_dir = Path(outputs_dir_str)

    debug = getattr(cli_args, "debug", False) or yaml_config.get("debug", False)

    # Publishing configuration
    skip_publish = getattr(cli_args, "skip_publish", False) or yaml_config.get(
        "skip_publish", False
    )

    platforms = getattr(cli_args, "platforms", None) or yaml_config.get("platforms")

    schedule_time = getattr(cli_args, "schedule_time", None) or yaml_config.get(
        "schedule_time"
    )

    fail_fast_publish = getattr(
        cli_args, "fail_fast_publish", False
    ) or yaml_config.get("fail_fast_publish", False)

    platform_specific_content = getattr(
        cli_args, "platform_specific", False
    ) or yaml_config.get("platform_specific_content", False)

    # Voice profile override
    voice_profile = getattr(cli_args, "voice_profile", None) or yaml_config.get(
        "voice_profile"
    )

    # Script template override
    script_template = getattr(cli_args, "script_template", None) or yaml_config.get(
        "script_template"
    )

    # Content pillar override
    pillar = getattr(cli_args, "pillar", None) or yaml_config.get("pillar")

    # Pycaps subtitle engine overrides
    subtitle_format = getattr(cli_args, "subtitle_format", None) or yaml_config.get(
        "subtitle_format"
    )
    subtitle_engine = getattr(cli_args, "subtitle_engine", None) or yaml_config.get(
        "subtitle_engine"
    )
    pycaps_template = getattr(cli_args, "pycaps_template", None) or yaml_config.get(
        "pycaps_template"
    )
    pycaps_template_pool = getattr(
        cli_args, "pycaps_template_pool", None
    ) or yaml_config.get("pycaps_template_pool")
    pycaps_renderer = getattr(cli_args, "pycaps_renderer", None) or yaml_config.get(
        "pycaps_renderer"
    )

    # Resume configuration
    resume = getattr(cli_args, "resume", False)

    # Dry-run configuration
    dry_run = getattr(cli_args, "dry_run", False)

    # Clean outputs before run
    clean = getattr(cli_args, "clean", False)

    # Output format configuration
    output_format = getattr(cli_args, "output_format", None) or "text"

    return GlobalBatchConfig(
        product_ids=product_ids,
        keywords=keywords,
        keyword_pillar_map=keyword_pillar_map,
        topics=topics,
        topics_per_run=topics_per_run,
        max_products=max_products,
        products_per_keyword=products_per_keyword,
        scraper_filters=scraper_filters,
        profile=profile,
        random_profile=random_profile,
        profile_pool=profile_pool,
        profile_pool_from_cli=bool(cli_profile_pool),
        fail_fast=fail_fast,
        process_all_products=process_all_products,
        outputs_dir=outputs_dir,
        debug=debug,
        skip_publish=skip_publish,
        force=getattr(cli_args, "force", False),
        platforms=platforms,
        schedule_time=schedule_time,
        fail_fast_publish=fail_fast_publish,
        platform_specific_content=platform_specific_content,
        voice_profile=voice_profile,
        script_template=script_template,
        pillar=pillar,
        subtitle_format=subtitle_format,
        subtitle_engine=subtitle_engine,
        pycaps_template=pycaps_template,
        pycaps_template_pool=pycaps_template_pool,
        pycaps_renderer=pycaps_renderer,
        resume=resume,
        dry_run=dry_run,
        clean=clean,
        output_format=output_format,
    )


def topic_capable_profiles(video_config: VideoConfig) -> list[str]:
    """Profiles that can render a topic: ones that source stock media.

    `slideshow_stock` is in `EXCLUDED_RANDOM_PROFILES` precisely because a
    *product* batch must not draw it, so the topic pool cannot be the product
    pool minus exclusions -- it is close to the complement.
    """
    from src.video.producer.utils import (
        draws_visuals_from_script,
        profile_needs_stock_media,
    )

    # Both halves, matching `config_validator.check_stock_media_key`: asking
    # for stock is not enough on its own. A hybrid profile that also draws
    # scraped images would gather only its stock share on a topic, which below
    # `min_images_if_no_video` reports the run SKIPPED rather than rendering.
    return sorted(
        name
        for name, profile in video_config.video_profiles.items()
        if name != "base"
        and profile_needs_stock_media(profile)
        and draws_visuals_from_script(profile)
    )


def _run_has_product_records(config: GlobalBatchConfig) -> bool:
    """Whether this run renders scraped products as well as topics.

    On a resume the answer comes from the saved state rather than from the
    inputs: `keywords` is inherited from the config whatever the run is
    resuming, and the completed scraping phase already ignored it, so reading
    it here would call every resumed topics run "mixed" and stop narrowing its
    profile pool.
    """
    if config.topics_resume:
        return config.resume_has_products
    return bool(config.product_ids or config.keywords)


def _validate_topic_profiles(
    config: GlobalBatchConfig, video_config: VideoConfig
) -> None:
    """Refuse a topics run that would draw a product-only profile."""
    capable = topic_capable_profiles(video_config)

    if not capable:
        raise ValueError(
            "No configured profile can render a topic: none source stock "
            "media. Add one, or pass --profile explicitly."
        )

    # A run carrying both kinds of record cannot narrow the shared pool: the
    # products in it need the product profiles. The two pools coexist instead,
    # and the production loop picks by record. A fixed profile has no such
    # escape -- one name cannot serve both -- so it is refused rather than
    # quietly applied to the products and ignored for the topics.
    if _run_has_product_records(config):
        if config.profile and config.profile not in capable:
            raise ValueError(
                f"Profile '{config.profile}' draws no stock media, so it "
                "cannot render the topics in this run, and one fixed profile "
                "cannot serve both kinds of record.\n"
                "Drop --profile to let each record pick its own, or set "
                "topics_per_run: 0 to run products alone.\n"
                f"Profiles that can render a topic: {', '.join(capable)}"
            )
        config.topic_profile_pool = capable
        # The products still need a pool of their own. Left unset with no
        # fixed profile there is nothing to select from, so say what a run
        # carrying two kinds of record means: each picks from its own pool.
        # The generic block below fills `profile_pool` with the product
        # profiles.
        if not config.profile:
            config.random_profile = True
        return

    config.topic_profile_pool = capable

    if config.profile:
        if config.profile not in capable:
            raise ValueError(
                f"Profile '{config.profile}' draws no stock media, so a topic "
                "render under it gathers no visuals and the run fails.\n"
                f"Profiles that can render a topic: {', '.join(capable) or 'none'}"
            )
        return

    # A pool inherited from YAML describes the default product run, not this
    # one, so it is replaced rather than refused; only a pool named on the
    # command line for this run is an instruction worth contradicting.
    if config.profile_pool and config.profile_pool_from_cli:
        unusable = [p for p in config.profile_pool if p not in capable]
        if unusable:
            raise ValueError(
                f"Profile pool contains {', '.join(sorted(unusable))}, which "
                "draw no stock media and cannot render a topic.\n"
                f"Profiles that can render a topic: {', '.join(capable) or 'none'}"
            )
        return

    # No profile named at all. The default pool is built from the product
    # profiles, every one of which fails on a topic, so fill it here instead
    # of letting the run pick one and die in `gather_visuals`.
    config.random_profile = True
    config.profile_pool = capable


def validate_global_batch_config(
    config: GlobalBatchConfig, video_config: VideoConfig
) -> None:
    """Validate global batch configuration before pipeline execution.

    Validates:
    - At least one input source (product_ids, keywords or topics) is provided
    - Profile configuration is valid (profile XOR random_profile)
    - Profile names exist in video configuration
    - Profile pool is not empty when random_profile is enabled
    - Publishing configuration (LATE_API_KEY, platforms, schedule_time) if enabled

    Args:
    ----
        config: Global batch configuration to validate
        video_config: Video configuration for profile validation

    Raises:
    ------
        ValueError: If validation fails with actionable error message

    """
    # Validate inputs exist
    # A resumed topics run has its inputs in the saved state, not on the
    # command line.
    if (
        not config.product_ids
        and not config.keywords
        and not config.topics
        and not config.topics_resume
    ):
        raise ValueError(
            "No inputs provided. Specify at least one of:\n"
            "  --product-ids ASIN1 ASIN2 ...\n"
            "  --keywords 'keyword1' 'keyword2' ...\n"
            "  --topic 'subject' / --topics-file topics.yaml"
        )

    # A topic run replaces the scraping phase outright, so anything to scrape
    # named alongside it is silently discarded. Refuse rather than drop an
    # input the operator asked for.
    is_topics_run = bool(config.topics) or config.topics_resume

    # Only on a topics-ONLY run. That is the one that narrows the shared pool;
    # a mixed run keeps the product pool for products, so sweeping in an old
    # product directory renders it the way it would have been rendered anyway.
    topics_only = is_topics_run and not _run_has_product_records(config)
    if topics_only and config.process_all_products:
        raise ValueError(
            "Cannot combine topics with --process-all-products. A topics-only "
            "run narrows its profile pool to stock-sourced profiles, which "
            "draw no product imagery, so every scraped product swept in would "
            "be rendered from generic stock footage and published."
        )

    # Topics and products used to be refused together, because a topic run
    # replaced the scraping phase outright and the scraped inputs were
    # silently discarded. Both phases now run, so the combination is the
    # supported mix rather than a contradiction -- and it has to be, since a
    # no-flag run reads both from the same config file.

    # Validate profile configuration
    if config.profile and config.random_profile:
        raise ValueError(
            "Cannot use both --profile and --random-profile. "
            "Choose one profile mode:\n"
            "  --profile PROFILE_NAME (fixed profile for all products)\n"
            "  --random-profile --profile-pool PROFILE1 PROFILE2 ... "
            "(random selection per product)"
        )

    # Validate fixed profile exists
    if config.profile and config.profile not in video_config.video_profiles:
        available = ", ".join(sorted(video_config.video_profiles.keys()))
        raise ValueError(
            f"Invalid profile: '{config.profile}'\n" f"Available profiles: {available}"
        )

    # A topic has no product imagery, so a profile that draws only scraped
    # media gathers nothing and the run fails outright -- it does not degrade
    # to a skip, because `step_gather_visuals` raises before the media check
    # that reports one. Checked here rather than left to the render, which
    # reports a configuration mistake as a render failure, once per product.
    if is_topics_run:
        _validate_topic_profiles(config, video_config)

    # Validate random profile configuration
    if config.random_profile:
        # If no profile pool specified, use all available render profiles
        # (excluding `base`, the inheritance template).
        if not config.profile_pool:
            from src.video.producer.utils import EXCLUDED_RANDOM_PROFILES

            config.profile_pool = [
                p
                for p in video_config.video_profiles
                if p not in EXCLUDED_RANDOM_PROFILES
            ]

        # Validate all profiles in pool exist
        invalid_profiles = [
            p for p in config.profile_pool if p not in video_config.video_profiles
        ]
        if invalid_profiles:
            available = ", ".join(sorted(video_config.video_profiles.keys()))
            invalid = ", ".join(invalid_profiles)
            raise ValueError(
                f"Invalid profiles in pool: {invalid}\n"
                f"Available profiles: {available}"
            )

    # Validate publishing configuration (only if publishing is enabled)
    if not config.skip_publish:
        # Validate LATE_API_KEY environment variable exists
        if not os.getenv("LATE_API_KEY"):
            raise ValueError(
                "Publishing enabled but LATE_API_KEY environment variable not set.\n"
                "Either set LATE_API_KEY or use --skip-publish to disable publishing."
            )

        # Validate platforms if specified
        if config.platforms:
            valid_platforms = {"youtube", "tiktok", "instagram"}
            invalid_platforms = [
                p for p in config.platforms if p.lower() not in valid_platforms
            ]
            if invalid_platforms:
                invalid = ", ".join(invalid_platforms)
                raise ValueError(
                    f"Invalid platforms: {invalid}\n"
                    f"Valid platforms: youtube, tiktok, instagram"
                )

        # Validate schedule_time format if specified
        if config.schedule_time:
            try:
                datetime.fromisoformat(config.schedule_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError) as e:
                raise ValueError(
                    f"Invalid schedule_time format: '{config.schedule_time}'\n"
                    f"Expected ISO 8601 format (e.g., '2025-01-20T10:00:00+00:00')\n"
                    f"Error: {e}"
                ) from e
