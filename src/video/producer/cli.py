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
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.scraper.amazon.scraper import ProductData
from src.utils import cleanup_temp_dirs
from src.utils.background_processing import cleanup_global_background_processor
from src.utils.connection_pool import get_http_session
from src.utils.logging_setup import setup_debug_logging
from src.utils.performance import PerformanceHistoryManager
from src.video.config import VideoConfig
from src.video.config_adapter import load_video_config_modular
from src.video.config_validator import validate_config_and_exit_on_error
from src.video.producer.orchestration import create_video_for_product
from src.video.producer.state import VALID_STEPS
from src.video.producer.utils import setup_logging

logger = logging.getLogger(__name__)


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

    # Advanced styling
    if args.subtitle_font:
        overrides["subtitle_settings.font_name"] = args.subtitle_font
    if args.subtitle_font_color:
        overrides["subtitle_settings.font_color"] = args.subtitle_font_color
    if args.subtitle_outline_color:
        overrides["subtitle_settings.outline_color"] = args.subtitle_outline_color
    if args.subtitle_background_color:
        overrides["subtitle_settings.background_color"] = args.subtitle_background_color

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

    return overrides


async def main():
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
        "--batch-profile",
        type=str,
        help="Video profile to use for batch processing (required with --batch).",
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

    # Advanced subtitle styling (colors and fonts)
    parser.add_argument(
        "--subtitle-font",
        help="Override subtitle font family.",
    )
    parser.add_argument(
        "--subtitle-font-color",
        help="Override subtitle text color (ASS format: &H00RRGGBB).",
    )
    parser.add_argument(
        "--subtitle-outline-color",
        help="Override subtitle outline color (ASS format: &H00RRGGBB).",
    )
    parser.add_argument(
        "--subtitle-background-color",
        help="Override subtitle background color (ASS format: &H00RRGGBB).",
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

    args = parser.parse_args()

    # Validate argument combinations
    if args.batch:
        if not args.batch_profile:
            parser.error("--batch-profile is required when using --batch")
        if args.products_file or args.profile:
            parser.error(
                "products_file and profile arguments cannot be used with --batch"
            )
    else:
        if not args.products_file or not args.profile:
            parser.error(
                "products_file and profile are required when not using --batch"
            )
        if args.batch_profile or args.fail_fast:
            parser.error(
                "--batch-profile and --fail-fast can only be used with --batch"
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

    # Log applied CLI overrides (already applied via config loader)
    if cli_overrides:
        logger.info(f"Applied {len(cli_overrides)} CLI override(s):")
        for key, value in cli_overrides.items():
            logger.info(f"  {key} = {value}")

    try:
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

            # Create products list with directory info for batch processing
            products_list = list(discovered_products)
            profile_name = args.batch_profile
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
        if not args.batch:
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

    succeeded, failed, skipped = 0, 0, 0
    skipped_products = []
    failed_products = []
    session = await get_http_session()  # Use global connection pool

    # Enhanced progress reporting for batch mode
    total_products = len(indices)
    if args.batch:
        logger.info(
            f"Starting batch processing of {total_products} products with "
            f"profile '{profile_name}'"
        )

    for i, idx in enumerate(indices):
        product_dir, product = products_list[idx]
        product_id = product.asin or product.title or f"product_{idx}"

        # Enhanced progress reporting
        if args.batch:
            logger.info(f"[{i+1}/{total_products}] Processing product: {product_id}")

        try:
            result_path = await asyncio.wait_for(
                create_video_for_product(
                    config,
                    product,
                    profile_name,
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
            logger.error(
                f"Pipeline timed out after {config.pipeline_timeout_sec} seconds "
                f"for product {product_id}"
            )
            result_path = None
        except Exception as e:
            logger.error(
                f"Unexpected error processing product {product_id}: {e}", exc_info=True
            )
            result_path = None

        if result_path == "SKIPPED":
            skipped += 1
            skipped_products.append(product_id)
            if args.batch:
                logger.info(
                    f"[{i+1}/{total_products}] Skipped {product_id} "
                    f"(insufficient media)"
                )
        elif result_path:
            succeeded += 1
            if args.batch:
                logger.info(
                    f"[{i+1}/{total_products}] Successfully completed {product_id}"
                )
        elif not args.step:
            failed += 1
            failed_products.append(product_id)
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

    logger.info("\n--- Run Summary ---")
    if args.batch:
        logger.info(f"Batch Processing Summary (Profile: {profile_name})")
    logger.info(f"Total Products Processed: {len(indices)}")
    logger.info(f"Succeeded: {succeeded}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    if skipped_products:
        logger.info(
            f"Skipped products (insufficient media): {', '.join(skipped_products)}"
        )
    if failed_products:
        logger.info(f"Failed products: {', '.join(failed_products)}")
    if args.step:
        logger.info(f"NOTE: Run was limited to debug step '{args.step}'.")
    if args.batch and args.fail_fast and failed > 0:
        logger.info("NOTE: Batch processing stopped early due to --fail-fast.")

    logger.info("Video producer completed successfully")
    logger.info(f"Complete log saved to: {log_file}")

    # Ensure all log messages are flushed
    for handler in logging.getLogger().handlers:
        handler.flush()


if __name__ == "__main__":
    asyncio.run(main())
