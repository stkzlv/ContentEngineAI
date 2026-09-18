"""Command-line entry point for the Amazon scraper.

Split out of `scraper.py`, which had grown a 580-line `main()` mixing
argument parsing, input resolution, debug wiring and the batch-versus-single
dispatch. The scraper class stays there; everything that belongs to running
it from a terminal is here.

`python -m src.scraper.amazon.scraper` still works: that module re-exports
`main`.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import traceback
from pathlib import Path
from typing import Any

import yaml

from src.scraper.base.keyword_pillars import (
    read_keyword_pillars,
    rotate_keyword_pool,
)
from src.scraper.config_models import ScraperConfig
from src.utils.logging_setup import dated_log_path, setup_debug_logging
from src.utils.outputs_paths import get_logs_directory, get_project_root

from . import scraper as scraper_module
from .config import get_default_search_parameters, get_output_path
from .models import SearchParameters
from .scraper import BotasaurusAmazonScraper, WebsocketFilter

logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """The scraper CLI parser.

    Extracted from `main` so a test can read what an omitted flag
    resolves to. `load_batch_config` resolves several arguments with an
    `is not None` sentinel, so a flag defaulting to a falsy value rather
    than to `None` makes the configured value unreachable -- and that is
    a property of the parser, not of any one run.
    """
    parser = argparse.ArgumentParser(
        description="Botasaurus Amazon Scraper for ContentEngineAI"
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        required=False,
        help=(
            "Keywords or ASINs to scrape - supports multiple values "
            "for batch mode (overrides config file)"
        ),
    )
    parser.add_argument(
        "--product-ids",
        nargs="+",
        required=False,
        help=(
            "Product IDs (ASINs) for batch scraping - supports multiple "
            "values (e.g., --product-ids B0ABC123 B0DEF456)"
        ),
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        metavar="N",
        help="Global cap on total products to collect across all keywords",
    )
    parser.add_argument(
        "--products-per-keyword",
        type=int,
        default=None,
        metavar="N",
        help="Maximum products to scrape per individual keyword",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero when any product failed, not only when none was "
            "scraped (default: a partial failure exits 0)"
        ),
    )
    parser.add_argument(
        "--fail-fast",
        # `BooleanOptionalAction` with `default=None`, not `store_true`. The
        # loader resolves this with `cli_fail_fast if cli_fail_fast is not
        # None`, so an omitted `store_true` flag arriving as False was
        # indistinguishable from one passed deliberately, and
        # `batch.fail_fast` in the YAML could never win. Same collision as the
        # chunked-keywords defect: a not-supplied sentinel meeting a supplied
        # value.
        #
        # The paired form matters once the default is None: `store_true` can
        # then only produce True or "unset", leaving a user who configured
        # `fail_fast: true` no way to ask for continue-on-error for one run.
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stop batch processing on first failure "
            "(default: batch.fail_fast in config/scraper.yaml, else continue)"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging and browser visibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (more detailed than debug)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean output directory before scraping"
    )
    parser.add_argument(
        "--pause-on-error",
        action="store_true",
        help="Pause execution when errors occur (debug mode only)",
    )
    parser.add_argument(
        "--save-screenshots",
        action="store_true",
        help="Save screenshots at key steps (debug mode only)",
    )
    parser.add_argument(
        "--save-page-source",
        action="store_true",
        help="Save HTML page source for analysis (debug mode only)",
    )
    parser.add_argument(
        "--analyze-images",
        action="store_true",
        help="Deep analysis of all images found on page (debug mode only)",
    )
    parser.add_argument(
        "--dump-image-urls",
        action="store_true",
        help="Save all discovered image URLs to file (debug mode only)",
    )
    _add_search_arguments(parser)
    _add_run_arguments(parser)
    return parser


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    """Filters that shape the Amazon search itself."""
    # Search parameter arguments
    parser.add_argument(
        "--min-price",
        type=float,
        metavar="PRICE",
        help="Minimum price filter (e.g., 10.99)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        metavar="PRICE",
        help="Maximum price filter (e.g., 99.99)",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        metavar="RATING",
        help="Minimum rating filter (1-5 stars, e.g., 4.0)",
    )
    parser.add_argument(
        "--prime-only", action="store_true", help="Filter for Prime eligible items only"
    )
    parser.add_argument(
        "--free-shipping",
        action="store_true",
        help="Filter for items with free shipping",
    )
    parser.add_argument(
        "--brands",
        nargs="+",
        metavar="BRAND",
        help="Filter by brand names (e.g., --brands Apple Samsung)",
    )
    parser.add_argument(
        "--sort",
        choices=[
            "relevance",
            "price-low",
            "price-high",
            "rating",
            "newest",
            "featured",
        ],
        default="relevance",
        help="Sort order for search results",
    )
    parser.add_argument(
        "--category", metavar="ID", help="Category ID for filtering (advanced usage)"
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Media-validation alignment, and where inputs and outputs live."""
    # Media validation alignment. The boolean rather than a profile name:
    # resolving a name means loading the video config, which is a different
    # package's five YAML files read for one field, and the batch -- where
    # profiles actually live -- computes this itself and passes it to the
    # scraper class directly.
    videos = parser.add_mutually_exclusive_group()
    videos.add_argument(
        "--profile-uses-videos",
        dest="profile_uses_videos",
        action="store_true",
        default=None,
        help=(
            "Keep the configured media requirements, which already count "
            "scraped videos. Same as passing neither flag; here so a script "
            "can say which rule it means."
        ),
    )
    videos.add_argument(
        "--no-profile-uses-videos",
        dest="profile_uses_videos",
        action="store_false",
        help=(
            "Validate media for an image-only profile: videos are ignored "
            "and enough images are required on their own."
        ),
    )

    # Batch input/output arguments
    parser.add_argument(
        "--input-file",
        metavar="FILE",
        help=(
            "Read product IDs or URLs from file (one per line), "
            "merged with --product-ids"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        metavar="N",
        help="Process products in batches of N (default: all at once)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help='Override output directory (default: "outputs" from config)',
    )


def _rotate_pool_for_today(
    pool: list[str],
    args: argparse.Namespace,
    config: dict[str, Any],
    batch_config: dict[str, Any],
) -> list[str]:
    """Reorder a configured keyword pool so runs do not re-tread its head.

    Only reached when the operator named no keywords. `--keywords` is what
    they typed and stays in the order given, since a date-dependent result is
    surprising in a tool used to reproduce a problem. The distinction has to
    be made here rather than in `load_batch_config`, because the caller
    assigns the pool to `args.keywords` before the loader sees it and the two
    become indistinguishable.

    The stride is what one run consumes, so consecutive days reach different
    keywords. It is a stride and not a slice width: the pool stays whole, and
    the entries past the stride remain as fallback for a barren search.
    """
    max_products = args.max_products or config["scrapers"]["amazon"]["max_products"]
    per_keyword = args.products_per_keyword or batch_config["products_per_keyword"]
    stride = max(1, -(-max_products // max(1, per_keyword)))
    rotated = rotate_keyword_pool(pool, stride)
    logger.info(
        "Rotated %d configured keywords for today; starting at %s",
        len(rotated),
        rotated[0] if rotated else "(none)",
    )
    return rotated


def _start_logging() -> Path:
    """Configure logging for an invoked run and return the log file.

    After argument parsing, so `--help` and an argparse error exit without
    touching the log at all. The marker records an invoked run, not a
    completed scrape.
    """
    log_file = dated_log_path(get_logs_directory() / "scraper.log")
    setup_debug_logging(
        log_file=log_file,
        debug_mode=False,
        verbose=False,
        component_name="AmazonScraper",
    )
    return log_file


def _merge_input_file(args: argparse.Namespace) -> bool:
    """Fold `--input-file` entries into `--product-ids`. False stops the run."""
    if not args.input_file:
        return True

    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = get_project_root() / input_path
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return False

    with open(input_path, encoding="utf-8") as f:
        file_ids = [line.strip() for line in f if line.strip()]
    # Deduplicate while preserving order
    existing = list(args.product_ids or [])
    seen = set(existing)
    for fid in file_ids:
        if fid not in seen:
            existing.append(fid)
            seen.add(fid)
    args.product_ids = existing
    logger.info(
        "Loaded %d entries from %s (%d unique total)",
        len(file_ids),
        args.input_file,
        len(args.product_ids),
    )
    return True


def _load_scraper_config() -> dict[str, Any] | None:
    """The scraper's own YAML, validated and defaults-filled, or None.

    The project root rather than the working directory, because Botasaurus
    changes it underneath the run.
    """
    config_path = get_project_root() / "config/scraper.yaml"
    if not config_path.exists():
        return None
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ScraperConfig.from_legacy_dict(raw).to_runtime_dict()


def _inputs_from_config(args: argparse.Namespace) -> bool:
    """Fill keywords/product IDs from the config. False stops the run."""
    if args.keywords or args.product_ids:
        return True

    try:
        config = _load_scraper_config()
        if config is None:
            logger.error("No keywords provided via CLI and config file not found")
            logger.debug("Use --keywords 'your keyword' to specify what to scrape")
            return False

        batch_config = config.get("batch", {})
        batch_product_ids = batch_config.get("product_ids", [])
        # Flattened: the config groups keywords by pillar, and iterating that
        # dict yields the pillar names. A run with no --keywords searched for
        # "value" and "utility" instead of any configured keyword.
        batch_keywords, _ = read_keyword_pillars(batch_config.get("keywords", []))
        if batch_keywords:
            batch_keywords = _rotate_pool_for_today(
                batch_keywords, args, config, batch_config
            )

        if batch_product_ids or batch_keywords:
            # A list or None; an empty list would read as "the CLI named some".
            args.product_ids = batch_product_ids or None
            args.keywords = batch_keywords or None
            if batch_product_ids and batch_keywords:
                logger.debug(
                    "Using batch mode from config: %d product IDs, %d keywords",
                    len(batch_product_ids),
                    len(batch_keywords),
                )
            elif batch_product_ids:
                logger.debug(
                    "Using batch product IDs from config: %s",
                    ", ".join(batch_product_ids),
                )
            else:
                logger.debug(
                    "Using batch keywords from config: %s", ", ".join(batch_keywords)
                )
            return True

        amazon_config = config.get("scrapers", {}).get("amazon", {})
        config_keywords = amazon_config.get("keywords", [])
        if config_keywords:
            args.keywords = _rotate_pool_for_today(
                config_keywords, args, config, batch_config
            )
            logger.debug(
                "Using keywords from config file: %s", ", ".join(config_keywords)
            )
            return True

        logger.error(
            "No keywords/product_ids provided via CLI and none found in config file"
        )
        logger.debug(
            "Either use --keywords/--product-ids or add to the batch section "
            "in config/scraper.yaml"
        )
        return False
    except Exception as e:
        logger.error("Error loading config file: %s", e)
        logger.debug("Use --keywords 'your keyword' to specify what to scrape")
        return False


def _resolve_debug(args: argparse.Namespace, log_file: Path) -> bool:
    """Settle debug mode from CLI and config, and reconfigure logging for it."""
    config_debug_mode = False
    if not args.debug and not args.verbose:
        try:
            config = _load_scraper_config() or {}
            config_debug_mode = config.get("global_settings", {}).get(
                "debug_mode", False
            )
        except Exception:
            config_debug_mode = False

    debug_enabled = args.debug or args.verbose or config_debug_mode
    if debug_enabled:
        # Set on the scraper module, which is where the browser code reads it.
        # `global DEBUG_MODE` here would bind a name in this module and leave
        # a config-driven debug run with an invisible browser.
        scraper_module.DEBUG_MODE = True
        setup_debug_logging(
            log_file=log_file,
            debug_mode=True,
            verbose=args.verbose,
            component_name="AmazonScraper",
            mark_run=False,  # already marked above; this is the same run
        )

    # Apply websocket filter to suppress cleanup messages
    websocket_filter = WebsocketFilter()
    logging.getLogger().addFilter(websocket_filter)
    logging.getLogger("websocket").addFilter(websocket_filter)

    if debug_enabled:
        _log_debug_options(args, config_debug_mode)
    return debug_enabled


def _log_debug_options(args: argparse.Namespace, config_debug_mode: bool) -> None:
    """Say which debug switches are on, once logging can show it."""
    if args.verbose:
        logger.debug("Verbose mode enabled - detailed logging active")
    elif config_debug_mode and not args.debug:
        logger.debug(
            "Debug mode enabled from config - browser visibility and "
            "detailed logging active"
        )
    else:
        logger.debug("Debug mode enabled - browser visibility and detailed logging")
    logger.debug("Debug mode set globally for browser visibility")

    switches = (
        (args.pause_on_error, "Pause-on-error enabled - execution pauses on errors"),
        (args.save_screenshots, "Screenshot saving enabled - key steps captured"),
        (args.save_page_source, "Page source saving enabled - HTML saved for analysis"),
        (args.analyze_images, "Deep image analysis enabled - all images analyzed"),
        (args.dump_image_urls, "Image URL dumping enabled - URLs saved to file"),
    )
    for enabled, message in switches:
        if enabled:
            logger.debug("%s", message)


def _clean_outputs() -> None:
    """Remove scraped product directories and scraper working files.

    The project root, not the working directory, because Botasaurus changes
    it; reports and logs are left alone.
    """
    base_output_path = get_project_root() / get_output_path("base")
    if not base_output_path.exists():
        return

    logger.info("Cleaning all scraper outputs in: %s", base_output_path)
    # 10-character ASINs, plus the TEST ids the suite writes.
    asin_pattern = re.compile(r"^([A-Z0-9]{10}|TEST[A-Z0-9]+)$")
    for item in base_output_path.iterdir():
        if item.is_dir():
            if asin_pattern.match(item.name):
                shutil.rmtree(item)
                logger.debug("Cleaned product directory: %s", item)
            elif item.name in ("cache", "temp", "screenshots"):
                shutil.rmtree(item)
                logger.debug("Cleaned scraper directory: %s", item)
        elif (
            item.is_file()
            and item.suffix in (".json", ".csv", ".xlsx", ".html")
            and not item.name.startswith("report")
        ):
            item.unlink()
            logger.debug("Cleaned scraper file: %s", item)
    logger.info("Cleanup completed - all scraper outputs removed")


_SORT_MAPPING = {
    "relevance": "relevanceblender",
    "price-low": "price-asc-rank",
    "price-high": "price-desc-rank",
    "rating": "review-rank",
    "newest": "date-desc-rank",
    "featured": "featured-rank",
}


def _build_search_params(
    args: argparse.Namespace,
) -> tuple[SearchParameters, dict[str, Any]] | None:
    """Config defaults with the CLI's overrides on top. None stops the run."""
    search_params = get_default_search_parameters()

    cli_overrides: dict[str, Any] = {}
    # Two groups, as before the split: a numeric filter of 0 is a real bound,
    # while an empty string or list means the flag was not really given.
    for name in ("min_price", "max_price", "min_rating"):
        value = getattr(args, name)
        if value is not None:
            cli_overrides[name] = value
    for name in ("prime_only", "free_shipping", "brands", "category"):
        value = getattr(args, name)
        if value:
            cli_overrides[name] = value
    if args.sort != "relevance":
        cli_overrides["sort_order"] = _SORT_MAPPING[args.sort]

    if cli_overrides:
        search_params = SearchParameters(
            **{
                name: cli_overrides.get(name, getattr(search_params, name))
                for name in (
                    "min_price",
                    "max_price",
                    "min_rating",
                    "prime_only",
                    "free_shipping",
                    "brands",
                    "sort_order",
                    "category",
                )
            }
        )

    validation_errors = search_params.validate()
    if validation_errors:
        logger.error("Invalid search parameters:")
        for error in validation_errors:
            logger.error("   %s", error)
        return None
    return search_params, cli_overrides


def _log_search_params(
    search_params: SearchParameters, cli_overrides: dict[str, Any]
) -> None:
    logger.debug("Search parameters configured:")
    if search_params.min_price or search_params.max_price:
        logger.debug(
            "   Price range: $%.2f-$%s",
            search_params.min_price or 0,
            search_params.max_price or "inf",
        )
    if search_params.min_rating:
        logger.debug("   Minimum rating: %s+ stars", search_params.min_rating)
    if search_params.prime_only:
        logger.debug("   Prime only: Yes")
    if search_params.free_shipping:
        logger.debug("   Free shipping: Yes")
    if search_params.brands:
        logger.debug("   Brands: %s", ", ".join(search_params.brands))
    if search_params.sort_order != "relevanceblender":
        logger.debug("   Sort: %s", search_params.sort_order)

    config_defaults = get_default_search_parameters()
    if cli_overrides:
        logger.debug("   CLI overrides applied: %s", list(cli_overrides))
    if (
        search_params.min_price != config_defaults.min_price
        or search_params.max_price != config_defaults.max_price
    ):
        logger.debug(
            "   Config defaults: $%.2f-$%s",
            config_defaults.min_price or 0,
            config_defaults.max_price or "inf",
        )


def cmd_batch(
    scraper: BotasaurusAmazonScraper,
    args: argparse.Namespace,
    search_params: SearchParameters,
) -> tuple[int, Any]:
    """Run the batch controller, chunking product IDs if asked.

    Returns the number of products scraped and the merged summary.
    """
    from .batch_controller import BatchController
    from .config import load_batch_config
    from .models import BatchSummary

    batch_size = getattr(args, "batch_size", None)
    all_product_ids = list(args.product_ids or [])
    all_keywords = list(args.keywords or [])

    if batch_size and all_product_ids:
        chunks = [
            all_product_ids[i : i + batch_size]
            for i in range(0, len(all_product_ids), batch_size)
        ]
        logger.info(
            "Splitting %d products into %d batches of %d",
            len(all_product_ids),
            len(chunks),
            batch_size,
        )
    else:
        chunks = [all_product_ids] if all_product_ids else [[]]

    total_summary: BatchSummary | None = None
    for chunk_idx, chunk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info(
                "Batch %d/%d (%d products)",
                chunk_idx + 1,
                len(chunks),
                len(chunk) if chunk else 0,
            )

        batch_config = load_batch_config(
            cli_product_ids=chunk,
            # `[]`, not `None`, for the later chunks. The loader reads `None`
            # as "the CLI named no keywords" and falls back to the configured
            # list, so a chunked `--product-ids` run searched every keyword in
            # `scraper.yaml` from the second chunk on -- silently, since the
            # log reads like a normal keyword run. The keywords belong to the
            # first chunk because they are searched once for the whole run.
            cli_keywords=all_keywords if chunk_idx == 0 else [],
            cli_fail_fast=args.fail_fast,
            cli_max_products=args.max_products,
            cli_products_per_keyword=args.products_per_keyword,
        )
        batch_config.search_params = search_params

        summary = BatchController(scraper, batch_config).run_batch()
        if total_summary is None:
            total_summary = summary
        else:
            total_summary.total_attempted += summary.total_attempted
            total_summary.product_ids_attempted += summary.product_ids_attempted
            total_summary.successful += summary.successful
            total_summary.failed += summary.failed
            total_summary.failed_products.extend(summary.failed_products)
            # Merged like the product-level failures: without this a keyword
            # lost in any chunk after the first is invisible to --strict,
            # which is the loss the field exists for.
            total_summary.failed_keywords.extend(summary.failed_keywords)
            total_summary.duration_sec += summary.duration_sec

    if total_summary is None:
        logger.warning("No batches were processed")
        raise SystemExit(1)

    logger.info("--- SCRAPER SUMMARY ---")
    logger.info(
        "Products: %d attempted, %d successful, %d failed",
        total_summary.total_attempted,
        total_summary.successful,
        total_summary.failed,
    )
    if total_summary.successful_products:
        logger.info("Successful: %s", ", ".join(total_summary.successful_products))
    if total_summary.failed_products:
        logger.info("Failed: %s", ", ".join(total_summary.failed_products))
    logger.info(
        "Images: %d, Videos: %d",
        total_summary.media_stats.get("total_images", 0),
        total_summary.media_stats.get("total_videos", 0),
    )
    logger.info("Duration: %.1fs", total_summary.duration_sec)
    logger.info("---")
    return total_summary.successful, total_summary


def cmd_single(
    scraper: BotasaurusAmazonScraper,
    args: argparse.Namespace,
    search_params: SearchParameters,
) -> tuple[int, None]:
    """Scrape one keyword through the cycling unified path."""
    products = scraper.scrape_products(args.keywords, search_params)

    logger.info("--- SCRAPER SUMMARY ---")
    if products:
        logger.info(
            "Products: %d scraped for keywords: %s",
            len(products),
            ", ".join(args.keywords),
        )
    else:
        logger.info("Products: 0 scraped")
    # The batch arm reports these through BatchSummary; this one builds no
    # summary object, so it reads the tracker directly. Without it the
    # single-keyword run -- which is what the runbook and the end-to-end cases
    # use -- printed no verdict at all, while the docs said both were
    # reported at the end of the run.
    for line in scraper.throttle.summary_lines():
        logger.warning("%s", line)
    logger.info("---")
    return len(products), None


def _exit_code_for(
    args: argparse.Namespace, products_scraped: int, summary: Any
) -> None:
    """Raise SystemExit when the run lost something the caller asked for."""
    if products_scraped == 0:
        logger.error("Scraper failed: 0 products scraped")
        raise SystemExit(1)

    # A partial failure exits 0 by default, matching the global batch: a run
    # that lost one product of twenty has done most of what was asked.
    # `--strict` is for a caller that would rather investigate than lose a
    # product silently. Both kinds of loss count: a product id that yielded
    # nothing, and a keyword whose search returned nothing or raised. The
    # keyword arm records no per-product result, so counting only `failed`
    # would make --strict a no-op on exactly the runs the docs use as
    # examples.
    failed = getattr(summary, "failed", 0)
    lost_keywords = list(getattr(summary, "failed_keywords", []) or [])
    if args.strict and (failed or lost_keywords):
        logger.error(
            "Scraper failed under --strict: %d scraped, %d products failed, "
            "%d keywords produced nothing%s",
            products_scraped,
            failed,
            len(lost_keywords),
            f" ({', '.join(lost_keywords)})" if lost_keywords else "",
        )
        raise SystemExit(1)


def main() -> None:
    """Command-line interface for the Botasaurus Amazon scraper."""
    # Load .env BEFORE anything reads env vars. Without this,
    # AMAZON_ASSOCIATE_TAG (and any other secret in .env) is invisible to
    # build_affiliate_url, which silently falls back to returning the input
    # URL unchanged: untagged affiliate links end up in data.json. The global
    # batch entry point loads .env the same way for the same reason.
    from dotenv import load_dotenv

    load_dotenv()

    args = build_argument_parser().parse_args()
    log_file = _start_logging()

    if not _merge_input_file(args):
        return
    if not _inputs_from_config(args):
        return

    _resolve_debug(args, log_file)

    if args.clean:
        _clean_outputs()

    if args.debug:
        from ...utils.outputs_paths import get_temp_directory

        logger.debug("Debug files will be saved to: %s", get_temp_directory())

    built = _build_search_params(args)
    if built is None:
        return
    search_params, cli_overrides = built
    if args.debug:
        _log_search_params(search_params, cli_overrides)

    if args.profile_uses_videos is not None:
        # The `--profile NAME` path this replaces logged which profile it had
        # aligned with; an operator reading scraper.log still needs to see
        # which rule a run used. Only the false side changes anything: the
        # one consumer tests `is False` (scraper.py, `effective_vid_count`),
        # so the true side restates the configured requirements, as naming a
        # video-using profile did before.
        logger.info(
            "Media validation: %s",
            "the configured requirements, which count videos"
            if args.profile_uses_videos
            else "videos ignored, images required on their own",
        )

    try:
        scraper = BotasaurusAmazonScraper(
            debug_override=args.debug if args.debug else None,
            debug_options={
                "save_screenshots": args.save_screenshots if args.debug else False,
                "save_page_source": args.save_page_source if args.debug else False,
                "analyze_images": args.analyze_images if args.debug else False,
                "dump_image_urls": args.dump_image_urls if args.debug else False,
                "pause_on_error": args.pause_on_error if args.debug else False,
            },
            output_dir=getattr(args, "output_dir", None),
            profile_uses_videos=args.profile_uses_videos,
        )

        # --product-ids, or more than one keyword, is what the batch
        # controller is for; a single keyword runs the cycling unified path.
        if bool(args.product_ids) or (args.keywords and len(args.keywords) > 1):
            products_scraped, summary = cmd_batch(scraper, args, search_params)
        else:
            products_scraped, summary = cmd_single(scraper, args, search_params)

        _exit_code_for(args, products_scraped, summary)
    except Exception as e:
        logger.error("Scraper failed: %s", e)
        if args.debug:
            logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
