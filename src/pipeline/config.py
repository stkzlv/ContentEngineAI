"""Configuration data models for global batch pipeline.

This module defines unified configuration and summary structures
for orchestrating scraping and video production phases.

Data Models:
    - GlobalBatchConfig: Unified pipeline configuration
    - ScrapingPhaseSummary: Scraping phase statistics
    - ProductionPhaseSummary: Video production phase statistics
    - PipelineSummary: End-to-end pipeline summary

Configuration Functions:
    - load_global_batch_config: Load configuration with CLI > YAML > defaults precedence
    - validate_global_batch_config: Validate configuration before pipeline execution

Configuration Precedence:
    CLI arguments > YAML configuration > Default values
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.scraper.amazon.models import SearchParameters
from src.video.config import VideoConfig


@dataclass
class GlobalBatchConfig:
    """Unified configuration for global batch pipeline.

    Combines scraper and producer settings for end-to-end automation.

    Attributes
    ----------
        product_ids: List of ASINs to scrape directly
        keywords: List of keywords to search for products
        max_products: Maximum number of products to scrape per keyword
        scraper_filters: SearchParameters for filtering products
        profile: Fixed video profile name (mutually exclusive with random_profile)
        random_profile: Enable random profile selection per product
        profile_pool: List of profiles for random selection
        fail_fast: Stop pipeline on first failure
        outputs_dir: Directory for scraper output and producer input
        debug: Enable debug logging

    """

    # Scraper configuration
    product_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    max_products: int = 10
    scraper_filters: SearchParameters = field(default_factory=SearchParameters)

    # Producer configuration
    profile: str | None = None
    random_profile: bool = False
    profile_pool: list[str] = field(default_factory=list)

    # Common configuration
    fail_fast: bool = False
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))
    debug: bool = False


@dataclass
class ScrapingPhaseSummary:
    """Scraping phase statistics.

    Tracks scraping phase execution and results.

    Attributes
    ----------
        total_attempted: Total number of products attempted to scrape
        successful: Number of products scraped successfully
        failed: Number of products that failed to scrape
        failed_products: List of product IDs that failed
        media_stats: Media statistics (e.g., total_images, total_videos)
        duration_sec: Phase duration in seconds

    """

    total_attempted: int
    successful: int
    failed: int
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
class PipelineSummary:
    """End-to-end pipeline statistics.

    Aggregates statistics across all phases and calculates
    derived end-to-end metrics.

    Attributes
    ----------
        scraping: Scraping phase summary
        production: Video production phase summary
        end_to_end_success: Products scraped AND produced successfully
        partial_success: Products scraped successfully but not produced
        total_failures: Products that failed in either phase
        total_duration_sec: Total pipeline duration in seconds

    """

    scraping: ScrapingPhaseSummary
    production: ProductionPhaseSummary
    end_to_end_success: int
    partial_success: int
    total_failures: int
    total_duration_sec: float

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

        lines.extend(
            [
                f"  Duration: {self.production.duration_sec:.1f}s",
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
    # Product IDs
    product_ids = (
        getattr(cli_args, "product_ids", None)
        or yaml_config.get("product_ids", [])
        or []
    )

    # Keywords
    keywords = (
        getattr(cli_args, "keywords", None) or yaml_config.get("keywords", []) or []
    )

    # Max products
    max_products = (
        getattr(cli_args, "max_products", None) or yaml_config.get("max_products") or 10
    )

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

    profile_pool = (
        getattr(cli_args, "profile_pool", None)
        or yaml_config.get("profile_pool", [])
        or []
    )

    # Common configuration
    fail_fast = getattr(cli_args, "fail_fast", False) or yaml_config.get(
        "fail_fast", False
    )

    outputs_dir_str = getattr(cli_args, "outputs_dir", None) or yaml_config.get(
        "outputs_dir", "outputs"
    )
    outputs_dir = Path(outputs_dir_str)

    debug = getattr(cli_args, "debug", False) or yaml_config.get("debug", False)

    return GlobalBatchConfig(
        product_ids=product_ids,
        keywords=keywords,
        max_products=max_products,
        scraper_filters=scraper_filters,
        profile=profile,
        random_profile=random_profile,
        profile_pool=profile_pool,
        fail_fast=fail_fast,
        outputs_dir=outputs_dir,
        debug=debug,
    )


def validate_global_batch_config(
    config: GlobalBatchConfig, video_config: VideoConfig
) -> None:
    """Validate global batch configuration before pipeline execution.

    Validates:
    - At least one input source (product_ids or keywords) is provided
    - Profile configuration is valid (profile XOR random_profile)
    - Profile names exist in video configuration
    - Profile pool is not empty when random_profile is enabled

    Args:
    ----
        config: Global batch configuration to validate
        video_config: Video configuration for profile validation

    Raises:
    ------
        ValueError: If validation fails with actionable error message

    """
    # Validate inputs exist
    if not config.product_ids and not config.keywords:
        raise ValueError(
            "No inputs provided. Specify at least one of:\n"
            "  --product-ids ASIN1 ASIN2 ...\n"
            "  --keywords 'keyword1' 'keyword2' ..."
        )

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

    # Validate random profile configuration
    if config.random_profile:
        if not config.profile_pool:
            raise ValueError(
                "Random profile mode requires --profile-pool with at least one profile"
            )

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
