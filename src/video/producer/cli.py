# src/video/producer/cli.py
"""CLI argument parsing and batch processing for video producer."""

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from src.scraper.amazon.scraper import ProductData
from src.utils import cleanup_temp_dirs
from src.utils.background_processing import cleanup_global_background_processor
from src.utils.connection_pool import get_http_session
from src.utils.logging_setup import setup_debug_logging
from src.utils.performance import PerformanceHistoryManager
from src.video.config import VideoConfig
from src.video.config_adapter import load_video_config_modular
from src.video.config_validator import (
    check_stock_media_key,
    validate_config_and_exit_on_error,
)
from src.video.producer.orchestration import (
    create_video_for_product,
    failed_step_from_result,
)
from src.video.producer.state import STEP_GATHER_VISUALS, VALID_STEPS
from src.video.producer.topic_input import (
    TOPIC_ID_PREFIX,
    TopicSpec,
    build_topic_product,
    load_topics_file,
    topic_product_id,
)
from src.video.producer.utils import (
    ProfileUsageTracker,
    load_profile_pool,
    select_profile_for_product,
    setup_logging,
    validate_profiles,
)

logger = logging.getLogger(__name__)


@dataclass
class ProductResult:
    """Result of processing a single product."""

    id: str
    status: str  # SUCCESS, FAILED, SKIPPED
    profile: str
    duration_sec: float = 0.0
    error: str | None = None
    output_path: str | None = None


@dataclass
class BatchSummary:
    """Summary of a batch processing run."""

    total_attempted: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    total_duration_sec: float = 0.0
    average_duration_sec: float = 0.0
    start_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    end_time: str | None = None
    profile_distribution: dict[str, int] = field(default_factory=dict)
    results: list[ProductResult] = field(default_factory=list)

    def exit_code(self, strict: bool = False) -> int:
        """Process exit code for the run.

        Mirrors ``PipelineSummary.exit_code``: nothing produced is a
        failure, a partial loss is not unless the caller asked. ``strict``
        counts a skipped product too — reported apart from a failure
        because the cause differs, but the same thing for an exit code.
        """
        if self.succeeded_count == 0:
            return 1
        if strict and (self.failed_count or self.skipped_count):
            return 1
        return 0

    def to_json(self) -> str:
        """Convert summary to JSON string."""
        return json.dumps(asdict(self), indent=2)


def discover_products_for_batch(outputs_dir: Path) -> list[tuple[Path, ProductData]]:
    """Discover products in the outputs directory for batch processing.

    Args:
    ----
        outputs_dir: Directory to scan for product subdirectories

    Returns:
    -------
        List of (product_dir_path, ProductData) tuples for valid products

    """
    products: list[tuple[Path, ProductData]] = []

    if not outputs_dir.exists():
        logger.warning(f"Outputs directory does not exist: {outputs_dir}")
        return products

    for product_dir in outputs_dir.iterdir():
        if not product_dir.is_dir():
            continue

        # Skip global directories (cache, logs, reports, etc.)
        if product_dir.name in {
            "cache",
            "logs",
            "reports",
            "coverage",
            "error_logs",
            "output",
            "outputs",
            "performance_history",
            "unknown_product",
        }:
            continue

        data_file = product_dir / "data.json"
        if not data_file.exists():
            logger.debug(f"Skipping {product_dir.name}: no data.json found")
            continue

        if product_dir.name.startswith(TOPIC_ID_PREFIX):
            # Topic renders need a stock-sourced profile. Batch discovery hands
            # products to product profiles, which would find no imagery here and
            # fail the run rather than skip it.
            logger.debug(f"Skipping {product_dir.name}: topic directory")
            continue

        try:
            product_data = json.loads(data_file.read_text(encoding="utf-8"))
            if isinstance(product_data, list):
                # Handle list format - take first product
                if product_data:
                    product = ProductData(**product_data[0])
                else:
                    logger.warning(f"Empty product list in {data_file}")
                    continue
            else:
                product = ProductData(**product_data)

            products.append((product_dir, product))
            logger.debug(f"Found valid product: {product_dir.name}")

        except Exception as e:
            logger.warning(f"Failed to load product data from {data_file}: {e}")
            continue

    logger.info(f"Discovered {len(products)} valid products for batch processing")
    return products


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build CLI overrides dictionary from parsed arguments.

    Args:
    ----
        args: Parsed command-line arguments

    Returns:
    -------
        Dictionary mapping config paths to override values

    """
    overrides: dict[str, Any] = {}

    # Subtitle format and effects (legacy args)
    if args.subtitle_format:
        overrides["subtitle_settings.subtitle_format"] = args.subtitle_format
    if args.ass_karaoke:
        overrides["subtitle_settings.ass_enable_karaoke"] = True
    if args.ass_fade:
        overrides["subtitle_settings.ass_enable_fade"] = True
    if args.preset:
        overrides["subtitle_settings.style_preset"] = args.preset

    # Pycaps subtitle engine overrides (highest precedence)
    if getattr(args, "subtitle_engine", None):
        overrides["subtitle_settings.subtitle_engine"] = args.subtitle_engine
    if getattr(args, "pycaps_template", None):
        overrides["subtitle_settings.pycaps.template_name"] = args.pycaps_template
        # Clear the pool so the deterministic selector falls through to
        # template_name. Without this, a multi-entry pool would still win
        # via md5 hash and silently ignore --pycaps-template.
        overrides["subtitle_settings.pycaps.template_pool"] = []
    if getattr(args, "pycaps_template_pool", None):
        # Explicit --pycaps-template-pool wins over the implicit clear above
        # when both flags are passed.
        overrides["subtitle_settings.pycaps.template_pool"] = args.pycaps_template_pool
    if getattr(args, "pycaps_renderer", None):
        overrides["subtitle_settings.pycaps.renderer"] = args.pycaps_renderer

    # Positioning
    if args.subtitle_anchor:
        overrides["subtitle_settings.anchor"] = args.subtitle_anchor
    if args.subtitle_margin is not None:
        overrides["subtitle_settings.margin"] = args.subtitle_margin
    if (
        hasattr(args, "subtitle_content_aware")
        and args.subtitle_content_aware is not None
    ):
        overrides["subtitle_settings.content_aware"] = args.subtitle_content_aware

    # Styling
    if args.font_size_scale is not None:
        overrides["subtitle_settings.font_size_scale"] = args.font_size_scale
    if args.max_subtitle_width_fraction is not None:
        overrides["subtitle_settings.max_subtitle_width_fraction"] = (
            args.max_subtitle_width_fraction
        )
    if args.subtitle_alignment:
        overrides["subtitle_settings.horizontal_alignment"] = args.subtitle_alignment

    # Text formatting
    if args.max_line_length is not None:
        overrides["subtitle_settings.max_line_length"] = args.max_line_length
    if args.max_words_per_line is not None:
        overrides["subtitle_settings.max_words_per_line"] = args.max_words_per_line
    if args.max_duration is not None:
        overrides["subtitle_settings.max_duration"] = args.max_duration
    if args.min_duration is not None:
        overrides["subtitle_settings.min_duration"] = args.min_duration

    # Randomization
    if (
        hasattr(args, "subtitle_randomize_fonts")
        and args.subtitle_randomize_fonts is not None
    ):
        overrides["subtitle_settings.randomize_fonts"] = args.subtitle_randomize_fonts
    if (
        hasattr(args, "subtitle_randomize_colors")
        and args.subtitle_randomize_colors is not None
    ):
        overrides["subtitle_settings.randomize_colors"] = args.subtitle_randomize_colors
    if (
        hasattr(args, "subtitle_randomize_effects")
        and args.subtitle_randomize_effects is not None
    ):
        overrides["subtitle_settings.randomize_effects"] = (
            args.subtitle_randomize_effects
        )

    # Image positioning
    if hasattr(args, "image_width_percent") and args.image_width_percent is not None:
        overrides["video_settings.image_width_percent"] = args.image_width_percent
    if (
        hasattr(args, "image_top_position_percent")
        and args.image_top_position_percent is not None
    ):
        overrides["video_settings.image_top_position_percent"] = (
            args.image_top_position_percent
        )

    # Platform targeting
    if hasattr(args, "target_platform") and args.target_platform is not None:
        overrides["description_settings.target_platform"] = args.target_platform

    # Metadata mode
    if hasattr(args, "metadata_mode") and args.metadata_mode is not None:
        overrides["description_settings.metadata_mode"] = args.metadata_mode

    # Voice profile override
    if hasattr(args, "voice_profile") and args.voice_profile is not None:
        overrides["voice_profile"] = args.voice_profile

    # Script template override
    if hasattr(args, "script_template") and args.script_template is not None:
        overrides["script_template"] = args.script_template

    # Content pillar override (drives template filter and runtime preamble)
    if hasattr(args, "pillar") and args.pillar is not None:
        overrides["pillar"] = args.pillar

    return overrides


def _step_exempts_stock_check(step: str | None) -> bool:
    """Whether `--step <step>` can skip the stock-provider key check.

    A single named step runs alone, so a run that will not reach the fetcher
    should not be refused for a key it never asks for. `gather_visuals` is the
    exception: it is the step that asks, and exempting it would restore the
    generic "No visual inputs were found" error this check exists to replace.
    """
    return step is not None and step != STEP_GATHER_VISUALS


def _profiles_this_run_may_use(args, config) -> list[str]:
    """Every profile the run could select, not just the one it names.

    With `--random-profile` any pool member can be drawn, so checking only the
    profile that happens to be picked would make a missing provider key an
    intermittent failure rather than a deterministic one.
    """
    if getattr(args, "random_profile", False):
        try:
            return load_profile_pool(
                getattr(args, "profile_pool", None),
                getattr(config, "profile_pool", None),
                config,
            )
        except ValueError:
            # An unusable pool is reported later, with a message naming the
            # bad profile. Raising here would replace that with a traceback.
            return []
    named = getattr(args, "batch_profile", None) or getattr(args, "profile", None)
    return [named] if named else []


def create_argument_parser() -> argparse.ArgumentParser:
    """Build the producer CLI parser.

    Extracted so the flags can be asserted without running a render;
    `main` is the only caller.
    """
    parser = argparse.ArgumentParser(
        description="Generate promotional videos for e-commerce products."
    )
    parser.add_argument(
        "products_file",
        type=Path,
        nargs="?",
        help="Path to JSON file with product data (not required with --batch).",
    )
    parser.add_argument(
        "profile",
        type=str,
        nargs="?",
        help="Video profile name from config (not required with --batch).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all products found in outputs directory.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help=(
            "Render a video about a topic instead of a scraped product. "
            "Replaces products_file; profile is still required."
        ),
    )
    parser.add_argument(
        "--topic-description",
        type=str,
        default="",
        help=(
            "Source material the script is written from. The script generator "
            "reads only the title and this description."
        ),
    )
    parser.add_argument(
        "--topic-keywords",
        type=str,
        help=(
            "Comma-separated stock media search terms for this topic, e.g. "
            "'wifi router, home network'. Comma-separated rather than repeated "
            "because a multi-value flag before a positional swallows it."
        ),
    )
    parser.add_argument(
        "--topics-file",
        type=Path,
        help=(
            "YAML list of topics to render, each with title, optional "
            "description and optional keywords."
        ),
    )
    parser.add_argument(
        "--batch-profile",
        type=str,
        help="Video profile to use for batch processing (required with --batch).",
    )
    parser.add_argument(
        "--random-profile",
        action="store_true",
        help=(
            "Enable random profile selection for batch processing. "
            "Each product gets a randomly selected profile "
            "(deterministic by product ID). "
            "Cannot be used with --batch-profile. Requires --batch."
        ),
    )
    parser.add_argument(
        "--profile-pool",
        nargs="+",
        type=str,
        help=(
            "List of profile names to randomly select from when "
            "--random-profile is enabled. "
            "If not specified, all available profiles will be used. "
            "Example: --profile-pool slideshow_images1 video_sequential"
        ),
    )
    parser.add_argument(
        "--product-ids",
        nargs="+",
        type=str,
        help=(
            "Filter batch processing to specific product IDs (ASINs). "
            "Only products matching these IDs will be processed. Requires --batch."
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to scan for products (default: outputs).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch processing on first failure.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero when any product was lost, to a failure or a "
            "skip, not only when none succeeded (default: a partial loss "
            "exits 0)."
        ),
    )
    parser.add_argument(
        "--product-index", type=int, help="0-based index of product in JSON list."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--step",
        type=str,
        choices=VALID_STEPS,
        help="Run a single, specific pipeline step.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Force a clean run by deleting the existing output directory "
            "before starting."
        ),
    )
    parser.add_argument(
        "--subtitle-format",
        choices=["srt", "ass"],
        help="Subtitle format: srt (default) or ass (with animations).",
    )
    parser.add_argument(
        "--subtitle-engine",
        choices=["ffmpeg", "pycaps"],
        help=(
            "Subtitle rendering engine. 'ffmpeg' (default) uses SRT/ASS "
            "burned via libass. 'pycaps' runs the pycaps library as a "
            "post-assembly step for animated TikTok-style captions. See "
            "docs/pycaps-subtitles.md for install."
        ),
    )
    parser.add_argument(
        "--pycaps-template",
        type=str,
        help=(
            "Pycaps template name (e.g. word-focus, hype, minimalist). "
            "Forces this template for every product by clearing the template "
            "pool. To use a custom multi-entry pool, pass "
            "--pycaps-template-pool instead."
        ),
    )
    parser.add_argument(
        "--pycaps-template-pool",
        nargs="+",
        type=str,
        help=(
            "Pool of pycaps templates for deterministic per-product selection. "
            "Example: --pycaps-template-pool word-focus hype vibrant"
        ),
    )
    parser.add_argument(
        "--pycaps-renderer",
        choices=["css", "pictex"],
        help=(
            "Pycaps renderer backend. 'css' = Playwright+Chromium (default, "
            "the only production-safe option). 'pictex' = browserless Skia "
            "path; PREVIEW ONLY, it renders words with no gaps between them."
        ),
    )
    parser.add_argument(
        "--ass-karaoke",
        action="store_true",
        help="Enable karaoke word highlighting (ASS format only).",
    )
    parser.add_argument(
        "--ass-fade",
        action="store_true",
        help="Enable fade-in/out effects (ASS format only).",
    )
    parser.add_argument(
        "--preset",
        choices=["minimal", "modern", "bold", "animated", "random"],
        help="Override subtitle style preset: minimal, modern, bold, animated, random.",
    )

    # Subtitle positioning arguments
    parser.add_argument(
        "--subtitle-anchor",
        choices=["top", "center", "bottom", "above_content", "below_content"],
        help="Subtitle anchor position.",
    )
    parser.add_argument(
        "--subtitle-margin",
        type=float,
        help="Subtitle margin as fraction of frame height (0.0-0.5).",
    )
    parser.add_argument(
        "--content-aware",
        action="store_true",
        dest="subtitle_content_aware",
        default=None,
        help="Enable content-aware subtitle positioning.",
    )
    parser.add_argument(
        "--no-content-aware",
        action="store_false",
        dest="subtitle_content_aware",
        default=None,
        help="Disable content-aware subtitle positioning.",
    )

    # Subtitle styling arguments
    parser.add_argument(
        "--font-size-scale",
        type=float,
        help="Font size scale factor (0.5-2.0).",
    )
    parser.add_argument(
        "--max-subtitle-width-fraction",
        type=float,
        help="Max subtitle width as fraction of frame width (0.0-1.0).",
    )
    parser.add_argument(
        "--subtitle-alignment",
        choices=["left", "center", "right"],
        help="Horizontal text alignment.",
    )

    # Subtitle text formatting arguments
    parser.add_argument(
        "--max-line-length",
        type=int,
        help="Maximum characters per subtitle line.",
    )
    parser.add_argument(
        "--max-words-per-line",
        type=int,
        help="Maximum words per subtitle line (0 to disable).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Maximum subtitle duration in seconds.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Minimum subtitle duration in seconds.",
    )

    # Randomization arguments
    parser.add_argument(
        "--randomize-fonts",
        action="store_true",
        dest="subtitle_randomize_fonts",
        default=None,
        help="Enable font randomization.",
    )
    parser.add_argument(
        "--no-randomize-fonts",
        action="store_false",
        dest="subtitle_randomize_fonts",
        default=None,
        help="Disable font randomization.",
    )
    parser.add_argument(
        "--randomize-colors",
        action="store_true",
        dest="subtitle_randomize_colors",
        default=None,
        help="Enable color randomization.",
    )
    parser.add_argument(
        "--no-randomize-colors",
        action="store_false",
        dest="subtitle_randomize_colors",
        default=None,
        help="Disable color randomization.",
    )
    parser.add_argument(
        "--randomize-effects",
        action="store_true",
        dest="subtitle_randomize_effects",
        default=None,
        help="Enable effect randomization.",
    )
    parser.add_argument(
        "--no-randomize-effects",
        action="store_false",
        dest="subtitle_randomize_effects",
        default=None,
        help="Disable effect randomization.",
    )

    # Image positioning arguments
    parser.add_argument(
        "--image-width-percent",
        type=float,
        help="Override image width as percentage of frame (0.0-1.0).",
    )
    parser.add_argument(
        "--image-top-position-percent",
        type=float,
        help="Override image top position as percentage from top (0.0-1.0).",
    )

    # Platform targeting argument
    parser.add_argument(
        "--target-platform",
        choices=["youtube", "tiktok", "instagram", "multi"],
        help=(
            "Override target platform for video metadata and captions. "
            "Choices: youtube (YouTube Shorts), tiktok (TikTok), "
            "instagram (Instagram Reels), multi (generate for all platforms). "
            "Example: --target-platform youtube"
        ),
    )

    # Metadata mode argument
    parser.add_argument(
        "--metadata-mode",
        choices=["unified", "optimized"],
        help=(
            "Metadata generation mode. "
            "unified: Single title/description/hashtags for all platforms (default). "
            "optimized: Platform-specific SEO-tailored metadata."
        ),
    )
    parser.add_argument(
        "--voice-profile",
        type=str,
        help="Override voice profile selection.",
    )
    parser.add_argument(
        "--script-template",
        type=str,
        help="Override script template (name without .md).",
    )
    parser.add_argument(
        "--pillar",
        type=str,
        help=(
            "Content pillar for this run (e.g. value, novelty, utility). "
            "Filters the script template pool to templates listed under the "
            "pillar in ai_services.yaml and prepends the pillar preamble to "
            "the LLM prompt. Without this flag, all templates are eligible."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Format for batch summary output (default: text).",
    )

    return parser


async def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    topic_mode = args.topic is not None or args.topics_file is not None
    if topic_mode and args.products_file and not args.profile:
        # Both positionals are optional, so argparse binds a lone profile name
        # to products_file. In topic mode there is no products_file, so the bare
        # word can only be the profile.
        args.profile = str(args.products_file)
        args.products_file = None

    # Validate argument combinations
    if args.batch:
        # Mutual exclusivity: --batch-profile and --random-profile
        if args.batch_profile and args.random_profile:
            parser.error(
                "Cannot use both --batch-profile and --random-profile. "
                "Use --batch-profile for a fixed profile or --random-profile "
                "for randomized selection."
            )
        # Require either --batch-profile or --random-profile
        if not args.batch_profile and not args.random_profile:
            parser.error(
                "--batch mode requires either --batch-profile (fixed profile) "
                "or --random-profile (randomized selection)"
            )
        if args.products_file or args.profile:
            parser.error(
                "products_file and profile arguments cannot be used with --batch"
            )
        if topic_mode:
            # Without this the batch branch wins silently and renders every
            # product directory instead, with nothing saying the topic was
            # dropped.
            parser.error("--topic/--topics-file cannot be used with --batch")
    elif topic_mode:
        # Topic mode: the record is built from the topic, so there is no
        # products_file to read. A profile is still required, and must be one
        # that sources its visuals from stock.
        if args.topic and args.topics_file:
            parser.error("--topic and --topics-file cannot be used together")
        if args.products_file:
            parser.error("products_file cannot be used with --topic/--topics-file")
        if not args.profile:
            parser.error("profile is required with --topic/--topics-file")
        if args.batch_profile or args.fail_fast or args.random_profile:
            parser.error(
                "--batch-profile, --fail-fast and --random-profile "
                "can only be used with --batch"
            )
        if args.pillar:
            # Every pillar preamble and audience hint is written about a
            # product ("the product fixes...", "practical buyers"), which
            # contradicts the topic templates' own instruction not to invent
            # one. Refusing beats emitting a prompt that argues with itself.
            parser.error(
                "--pillar cannot be used with --topic/--topics-file: the "
                "pillar preambles and audiences are written about a product"
            )
    else:
        # Non-batch mode validation
        if not args.products_file or not args.profile:
            parser.error(
                "products_file and profile are required when not using --batch, "
                "--topic or --topics-file"
            )
        if args.batch_profile or args.fail_fast:
            parser.error(
                "--batch-profile and --fail-fast can only be used with --batch"
            )
        # --random-profile requires --batch
        if args.random_profile:
            parser.error("--random-profile can only be used with --batch")
        # --profile-pool requires --random-profile and --batch
        if args.profile_pool:
            parser.error(
                "--profile-pool can only be used with --batch and --random-profile"
            )

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    load_dotenv(project_root / ".env")

    # Build CLI overrides dict from parsed arguments
    cli_overrides = _build_cli_overrides(args)

    # Load config first to get log directory path
    try:
        # Use modular config loading (automatically handles modular vs monolithic)
        config = load_video_config_modular(cli_overrides=cli_overrides)
    except Exception as e:
        # Fallback logging setup if config fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.critical(f"Config loading failed, using fallback logging: {e}")
        sys.exit(1)

    # Set up logging to both console and file
    log_file = setup_logging(config, args.debug)
    logger.info(f"Video producer started - Log file: {log_file}")

    # Validate configuration early to catch errors before processing
    logger.info("Validating configuration and runtime dependencies...")
    try:
        validate_config_and_exit_on_error(config)
    except SystemExit:
        logger.critical(f"Complete log saved to: {log_file}")
        raise

    # Fail now, not three steps into the render, when a profile this run may
    # select needs the stock provider and the key is absent. The message that
    # would otherwise appear names neither the provider nor the variable.
    #
    # Skipped for `--step`, which runs exactly one named step, so refusing it
    # would block debugging a run for a resource it is not going to ask for.
    # `--step gather_visuals` is the exception: it is the step that asks, and
    # skipping the check there restores the generic media error this exists to
    # replace.
    stock_key_error = (
        None
        if _step_exempts_stock_check(getattr(args, "step", None))
        else check_stock_media_key(config, _profiles_this_run_may_use(args, config))
    )
    if stock_key_error:
        logger.critical(stock_key_error)
        logger.critical(f"Complete log saved to: {log_file}")
        sys.exit(1)

    # Apply script template override to LLM settings
    if cli_overrides.get("script_template"):
        config.llm_settings.script_templates.fixed_template = cli_overrides[
            "script_template"
        ]

    # Log applied CLI overrides (already applied via config loader)
    if cli_overrides:
        logger.info(f"Applied {len(cli_overrides)} CLI override(s):")
        for key, value in cli_overrides.items():
            logger.info(f"  {key} = {value}")

    try:
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
    except Exception as e:
        logger.critical(f"Config/Secrets Error: {e}", exc_info=True)
        logger.critical(f"Complete log saved to: {log_file}")
        sys.exit(1)

    if not shutil.which(config.ffmpeg_settings.executable_path or "ffmpeg"):
        logger.error("FFmpeg not found in PATH or at specified executable_path.")
        logger.error(f"Complete log saved to: {log_file}")
        sys.exit(1)
    try:
        if args.batch:
            # Batch mode: discover products from outputs directory
            # Resolve outputs_dir relative to project root to handle working
            # directory changes
            if args.outputs_dir.is_absolute():
                outputs_path = args.outputs_dir
            else:
                outputs_path = project_root / args.outputs_dir
            discovered_products = discover_products_for_batch(outputs_path)
            if not discovered_products:
                logger.error(f"No valid products found in {outputs_path}")
                sys.exit(1)

            # Filter by product IDs if specified
            if args.product_ids:
                id_set = set(args.product_ids)
                discovered_products = [
                    (d, p) for d, p in discovered_products if d.name in id_set
                ]
                if not discovered_products:
                    logger.error(
                        "None of the specified product IDs found in %s: %s",
                        outputs_path,
                        args.product_ids,
                    )
                    sys.exit(1)
                logger.info(
                    "Filtered to %d product(s): %s",
                    len(discovered_products),
                    [d.name for d, _ in discovered_products],
                )

            # Create products list with directory info for batch processing
            products_list = list(discovered_products)
            profile_name = args.batch_profile
        elif topic_mode:
            # Topic mode: build the record instead of reading one the scraper
            # wrote. Everything downstream is unchanged; the run directory comes
            # from the record's identifier the same way a scraped product's does.
            if args.topics_file:
                specs = load_topics_file(args.topics_file)
            else:
                specs = [
                    TopicSpec(
                        title=args.topic,
                        description=args.topic_description or "",
                        keywords=[
                            k.strip()
                            for k in (args.topic_keywords or "").split(",")
                            if k.strip()
                        ],
                    )
                ]
            products_list = []
            for spec in specs:
                product = build_topic_product(spec)
                # Materialise data.json so the run is inspectable and resumable
                # in the same shape as a scraped product's directory.
                # `scraped_data` is the data.json path the producer reads, so
                # take the directory from it rather than naming a key that only
                # happens to exist today.
                data_path = config.get_product_paths(
                    topic_product_id(spec.title), args.profile
                )["scraped_data"]
                topic_dir = data_path.parent
                topic_dir.mkdir(parents=True, exist_ok=True)
                data_path.write_text(
                    json.dumps(product.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info("Prepared topic %r in %s", spec.title, topic_dir)
                products_list.append((topic_dir, product))
            profile_name = args.profile
        else:
            # Single product mode: load from file
            # Fix path resolution: resolve relative to project root, not current
            # working
            # directory
            # This handles cases where Botasaurus changes the working directory to
            # outputs/
            if args.products_file.is_absolute():
                products_file_path = args.products_file
            else:
                # Resolve relative paths against the original project root
                products_file_path = project_root / args.products_file
            product_data = json.loads(products_file_path.read_text(encoding="utf-8"))
            raw_products = [
                ProductData(**p)
                for p in (
                    product_data if isinstance(product_data, list) else [product_data]
                )
            ]
            # For single mode, we don't have directory info, so use a placeholder path
            placeholder_path = Path(".")  # Use current directory as placeholder
            products_list = [(placeholder_path, product) for product in raw_products]
            profile_name = args.profile
    except Exception as e:
        error_msg = f"Failed to load products: {e}"
        if not args.batch and not topic_mode:
            error_msg = (
                f"Failed to load or validate products from {products_file_path}: {e}"
            )
        logger.critical(error_msg, exc_info=True)
        logger.critical(f"Complete log saved to: {log_file}")
        sys.exit(1)

    # Handle product index for single product mode only
    if args.batch and args.product_index is not None:
        logger.error("--product-index cannot be used with --batch mode")
        sys.exit(1)

    indices = (
        [args.product_index]
        if args.product_index is not None
        and 0 <= args.product_index < len(products_list)
        else range(len(products_list))
    )
    if args.product_index is not None and not indices:
        logger.error(
            f"Product index {args.product_index} out of range for file with "
            f"{len(products_list)} products."
        )
        sys.exit(1)

    batch_summary = BatchSummary(total_attempted=len(indices))
    batch_start_time = datetime.now(UTC)
    session = await get_http_session()  # Use global connection pool

    # Initialize profile selection for batch mode
    profile_tracker = None
    profile_pool = None
    if args.batch and args.random_profile:
        # Load profile pool with CLI > YAML > all profiles precedence
        yaml_profile_pool = (
            config.batch.get("profile_pool") if hasattr(config, "batch") else None
        )
        try:
            profile_pool = load_profile_pool(
                cli_pool=args.profile_pool,
                yaml_pool=yaml_profile_pool,
                config=config,
            )
            logger.info(
                f"Profile randomization enabled with pool: {profile_pool} "
                f"({len(profile_pool)} profiles)"
            )
        except ValueError as e:
            logger.critical(f"Invalid profile pool configuration: {e}")
            sys.exit(1)

        # Initialize usage tracker
        profile_tracker = ProfileUsageTracker()
    else:
        # Validate fixed profile before starting
        if profile_name is None:
            logger.critical("Profile name is required in fixed profile mode")
            sys.exit(1)
        try:
            validate_profiles([profile_name], config)
        except ValueError as e:
            logger.critical(f"Invalid profile selection: {e}")
            sys.exit(1)

    # Enhanced progress reporting for batch mode
    total_products = len(indices)
    if args.batch:
        if args.random_profile:
            logger.info(
                f"Starting batch processing of {total_products} products with "
                f"random profile selection"
            )
        else:
            logger.info(
                f"Starting batch processing of {total_products} products with "
                f"profile '{profile_name}'"
            )

    for i, idx in enumerate(indices):
        product_dir, product = products_list[idx]
        product_id = product.asin or product.title or f"product_{idx}"

        # Select profile for this product
        if args.batch and args.random_profile:
            # Random profile selection (deterministic by product ID)
            # profile_pool and profile_tracker are guaranteed set in this branch
            current_profile = select_profile_for_product(
                product_id=product_id,
                profile_pool=cast(list[str], profile_pool),
                config=config,
            )
            cast(ProfileUsageTracker, profile_tracker).record_usage(current_profile)
            logger.info(
                f"[{i+1}/{total_products}] Processing {product_id} "
                f"with profile '{current_profile}'"
            )
        else:
            # Fixed profile mode (profile_name validated at startup)
            current_profile = cast(str, profile_name)
            if args.batch:
                logger.info(
                    f"[{i+1}/{total_products}] Processing product: {product_id}"
                )

        product_start_time = datetime.now(UTC)
        product_error = None
        try:
            result_path = await asyncio.wait_for(
                create_video_for_product(
                    config,
                    product,
                    current_profile,
                    secrets,
                    session,
                    args.debug,
                    args.clean,
                    args.step,
                    cli_overrides,
                ),
                timeout=config.pipeline_timeout_sec,
            )
        except TimeoutError:
            product_error = f"Pipeline timed out after {config.pipeline_timeout_sec}s"
            logger.error(f"{product_error} for product {product_id}")
            result_path = None
        except Exception as e:
            product_error = str(e)
            logger.error(
                f"Unexpected error processing product {product_id}: {e}", exc_info=True
            )
            result_path = None

        duration = (datetime.now(UTC) - product_start_time).total_seconds()

        failed_step = failed_step_from_result(result_path)
        if result_path == "SKIPPED":
            batch_summary.skipped_count += 1
            batch_summary.results.append(
                ProductResult(
                    id=product_id,
                    status="SKIPPED",
                    profile=current_profile,
                    duration_sec=duration,
                )
            )
            if args.batch:
                logger.info(
                    f"[{i+1}/{total_products}] Skipped {product_id} "
                    f"(insufficient media)"
                )
        elif failed_step is not None:
            batch_summary.failed_count += 1
            batch_summary.results.append(
                ProductResult(
                    id=product_id,
                    status="FAILED",
                    profile=current_profile,
                    duration_sec=duration,
                    error=f"pipeline step '{failed_step}' failed",
                )
            )
            if args.batch:
                logger.error(
                    "[%d/%d] Failed to process %s: step '%s' failed",
                    i + 1,
                    total_products,
                    product_id,
                    failed_step,
                )
                if args.fail_fast:
                    logger.error(
                        "Stopping batch processing due to --fail-fast "
                        "(failed on product %s)",
                        product_id,
                    )
                    break
        elif result_path:
            batch_summary.succeeded_count += 1
            batch_summary.results.append(
                ProductResult(
                    id=product_id,
                    status="SUCCESS",
                    profile=current_profile,
                    duration_sec=duration,
                    output_path=str(result_path),
                )
            )
            if args.batch:
                logger.info(
                    f"[{i+1}/{total_products}] Successfully completed {product_id}"
                )
        elif not args.step:
            batch_summary.failed_count += 1
            batch_summary.results.append(
                ProductResult(
                    id=product_id,
                    status="FAILED",
                    profile=current_profile,
                    duration_sec=duration,
                    error=product_error,
                )
            )
            if args.batch:
                logger.error(f"[{i+1}/{total_products}] Failed to process {product_id}")
                if args.fail_fast:
                    logger.error(
                        f"Stopping batch processing due to --fail-fast "
                        f"(failed on product {product_id})"
                    )
                    break

        if i < len(indices) - 1:
            delay = random.uniform(  # noqa: S311
                config.video_settings.inter_product_delay_min_sec,
                config.video_settings.inter_product_delay_max_sec,
            )
            await asyncio.sleep(delay)

    # Calculate final summary metrics
    batch_end_time = datetime.now(UTC)
    batch_summary.end_time = batch_end_time.isoformat()
    batch_summary.total_duration_sec = (
        batch_end_time - batch_start_time
    ).total_seconds()

    if batch_summary.results:
        batch_summary.average_duration_sec = sum(
            r.duration_sec for r in batch_summary.results
        ) / len(batch_summary.results)

    if profile_tracker:
        batch_summary.profile_distribution = profile_tracker.get_counts()

    # Output summary based on requested format
    if args.output_format == "json":
        # Write pure JSON to stdout for machine parsing
        print(batch_summary.to_json())
    else:
        # Standard text summary
        logger.info("--- PRODUCER SUMMARY ---")
        logger.info(
            "Products: %d attempted, %d successful, %d failed, %d skipped",
            batch_summary.total_attempted,
            batch_summary.succeeded_count,
            batch_summary.failed_count,
            batch_summary.skipped_count,
        )

        success_ids = [r.id for r in batch_summary.results if r.status == "SUCCESS"]
        if success_ids:
            logger.info("Successful: %s", ", ".join(success_ids))

        skipped_ids = [r.id for r in batch_summary.results if r.status == "SKIPPED"]
        if skipped_ids:
            logger.info("Skipped (insufficient media): %s", ", ".join(skipped_ids))

        failed_results = [r for r in batch_summary.results if r.status == "FAILED"]
        if failed_results:
            for r in failed_results:
                logger.info("Failed: %s (%s)", r.id, r.error)

        if args.batch and args.random_profile and profile_tracker:
            dist = profile_tracker.format_summary()
            logger.info("Profiles: %s", dist)

        logger.info(
            "Duration: %.1fs (avg %.1fs/product)",
            batch_summary.total_duration_sec,
            batch_summary.average_duration_sec,
        )
        logger.info("---")

    if args.step:
        logger.info(f"NOTE: Run was limited to debug step '{args.step}'.")
    if args.batch and args.fail_fast and batch_summary.failed_count > 0:
        logger.info("NOTE: Batch processing stopped early due to --fail-fast.")

    # Non-zero exit when nothing was produced, so CI, cron, and wrappers
    # checking $? see the failure.
    exit_code = batch_summary.exit_code(strict=args.strict)

    # Keyed on what happened, not on the exit code: under --strict a partial
    # failure also exits non-zero, and calling that "no videos produced"
    # would contradict the files on disk.
    if batch_summary.succeeded_count == 0:
        logger.error(
            "Video producer failed: no videos produced (%d failed, %d skipped)",
            batch_summary.failed_count,
            batch_summary.skipped_count,
        )
    elif batch_summary.failed_count:
        logger.warning(
            "Video producer completed with failures: %d succeeded, %d failed",
            batch_summary.succeeded_count,
            batch_summary.failed_count,
        )
    else:
        logger.info("Video producer completed successfully")
    logger.info(f"Complete log saved to: {log_file}")

    # Clean up HTTP connection pool
    from src.utils.connection_pool import close_global_pool

    await close_global_pool()

    # Ensure all log messages are flushed
    for handler in logging.getLogger().handlers:
        handler.flush()

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
