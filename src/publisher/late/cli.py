"""CLI interface for Late.dev publisher.

Provides command-line access to publish videos to social media platforms via Late.dev.

Usage:
    # List connected accounts
    python -m src.publisher.late list-accounts --debug

    # Publish single video immediately to multiple platforms
    python -m src.publisher.late single --video outputs/B0ABC123/video_sequential.mp4 \\
        --platform youtube --platform tiktok --immediate

    # Schedule single video for later
    python -m src.publisher.late single --video outputs/B0ABC123/video_sequential.mp4 \\
        --platform youtube --schedule "2025-01-20 14:00:00"

    # Batch publish all videos in outputs directory
    python -m src.publisher.late batch --platform youtube --platform tiktok \\
        --platform instagram --immediate --debug
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from src.publisher import PublisherProvider, create_publisher
from src.publisher.batch import BatchPublisher
from src.publisher.config import load_publisher_config
from src.publisher.models import Platform
from src.publisher.tracking import is_already_published, record_publish
from src.utils.logging_setup import setup_debug_logging

logger = logging.getLogger(__name__)


def parse_datetime(datetime_str: str) -> datetime:
    """Parse datetime string in ISO format or common formats.

    Args:
    ----
        datetime_str: Datetime string (e.g., "2025-01-20 14:00:00"
            or "2025-01-20T14:00:00")

    Returns:
    -------
        datetime object

    Raises:
    ------
        ValueError: If datetime string is invalid

    """
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Invalid datetime format: {datetime_str}. "
        f"Use format like '2025-01-20 14:00:00' or '2025-01-20T14:00:00'"
    )


async def cmd_list_accounts(
    args: argparse.Namespace, config, session: aiohttp.ClientSession
):
    """Execute list-accounts command.

    Args:
    ----
        args: Parsed command-line arguments
        config: PublisherConfig instance
        session: aiohttp ClientSession

    """
    logger.info("Listing connected social media accounts...")

    publisher = create_publisher(
        provider=PublisherProvider(config.provider),
        api_key=config.api_key,
        session=session,
        vercel_token=config.vercel_token,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )

    try:
        # Authenticate first
        is_authenticated = await publisher.authenticate()
        if not is_authenticated:
            logger.error("Authentication failed - check your API key")
            sys.exit(1)

        logger.info("Authentication successful")

        # Get accounts
        accounts = await publisher.get_accounts()

        if not accounts:
            logger.warning("No connected accounts found")
            return

        logger.info(f"Found {len(accounts)} connected account(s):")
        logger.info("-" * 80)

        for account in accounts:
            logger.info(f"Platform: {account['platform']}")
            logger.info(f"Account ID: {account['account_id']}")
            logger.info(f"Username: {account.get('username', 'N/A')}")
            logger.info("-" * 80)

    except Exception as e:
        logger.error(f"Failed to list accounts: {e}", exc_info=args.debug)
        sys.exit(1)


async def cmd_single(args: argparse.Namespace, config, session: aiohttp.ClientSession):
    """Execute single video publish command.

    Args:
    ----
        args: Parsed command-line arguments
        config: PublisherConfig instance
        session: aiohttp ClientSession

    """
    video_path = Path(args.video)

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    logger.info(f"Publishing single video: {video_path.name}")
    logger.info(f"Target platforms: {[p.value for p in args.platforms]}")
    if args.schedule:
        logger.info(f"Scheduled time: {args.schedule}")
    else:
        logger.info("Publishing immediately")

    publisher = create_publisher(
        provider=PublisherProvider(config.provider),
        api_key=config.api_key,
        session=session,
        vercel_token=config.vercel_token,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )

    try:
        # Authenticate
        is_authenticated = await publisher.authenticate()
        if not is_authenticated:
            logger.error("Authentication failed - check your API key")
            sys.exit(1)

        # Get accounts for mapping
        accounts = await publisher.get_accounts()
        if not accounts:
            logger.error("No connected accounts found")
            sys.exit(1)

        # Upload video
        logger.info("Uploading video...")
        media_url = await publisher.upload_media(video_path)
        logger.info(f"Upload complete: {media_url}")

        # Extract product ID from path (outputs/B0ABC123/video_*.mp4)
        if video_path.parent.parent.name == "outputs":
            product_id = video_path.parent.name
        else:
            product_id = video_path.stem

        # Load metadata for content
        from src.publisher.metadata import load_platform_metadata

        # Determine outputs_dir for tracking
        outputs_dir = (
            video_path.parent.parent
            if video_path.parent.parent.name == "outputs"
            else Path("outputs")
        )

        # Publish to each platform
        for platform in args.platforms:
            logger.info(f"Publishing to {platform.value}...")

            # Check for duplicates (unless --force)
            if not args.force and is_already_published(
                product_id, platform.value, outputs_dir
            ):
                logger.warning(
                    f"Product {product_id} already published to {platform.value}. "
                    f"Use --force to republish."
                )
                continue

            # Find account for this platform
            platform_account = next(
                (acc for acc in accounts if acc["platform"].lower() == platform.value),
                None,
            )

            if not platform_account:
                logger.warning(f"No connected account for {platform.value}, skipping")
                continue

            # Load platform-specific metadata
            metadata = load_platform_metadata(product_id, platform, outputs_dir)

            if metadata:
                content = metadata.format_content()
            else:
                logger.warning(
                    f"No metadata found for {platform.value}, using basic content"
                )
                content = f"Check out this product: {product_id}"

            # Publish
            result = await publisher.publish(
                media_id=media_url,
                platforms=[
                    {
                        "platform": platform.value,
                        "account_id": platform_account["account_id"],
                    }
                ],
                content=content,
                scheduled_time=args.schedule,
            )

            logger.info(
                f"Published to {platform.value}: "
                f"post_id={result['post_id']}, status={result['status']}"
            )

            # Record successful publish to prevent duplicates
            record_publish(
                product_id, platform.value, str(result["post_id"]), outputs_dir
            )

        logger.info("Single video publishing complete")

    except Exception as e:
        logger.error(f"Failed to publish video: {e}", exc_info=args.debug)
        sys.exit(1)


async def cmd_batch(args: argparse.Namespace, config, session: aiohttp.ClientSession):
    """Execute batch publish command.

    Args:
    ----
        args: Parsed command-line arguments
        config: PublisherConfig instance
        session: aiohttp ClientSession

    """
    logger.info("=" * 80)
    logger.info("BATCH PUBLISHING MODE")
    logger.info("=" * 80)
    logger.info(f"Target platforms: {[p.value for p in args.platforms]}")
    logger.info(f"Outputs directory: {args.outputs_dir}")
    logger.info(
        f"Stagger delay: {config.stagger_delay_min}-{config.stagger_delay_max}s"
    )
    logger.info(f"Fail-fast: {args.fail_fast}")

    publisher = create_publisher(
        provider=PublisherProvider(config.provider),
        api_key=config.api_key,
        session=session,
        vercel_token=config.vercel_token,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )

    try:
        # Authenticate
        is_authenticated = await publisher.authenticate()
        if not is_authenticated:
            logger.error("Authentication failed - check your API key")
            sys.exit(1)

        # Create batch publisher
        batch_publisher = BatchPublisher(
            publisher=publisher,
            outputs_dir=args.outputs_dir,
            platforms=args.platforms,
            stagger_delay_min=config.stagger_delay_min,
            stagger_delay_max=config.stagger_delay_max,
            fail_fast=args.fail_fast,
        )

        # Execute batch
        summary = await batch_publisher.publish_batch()

        # Exit with error code if any failures
        if summary.failed > 0:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Batch publishing failed: {e}", exc_info=args.debug)
        sys.exit(1)


async def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Late.dev Publisher - Publish videos to social media platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List connected accounts
  python -m src.publisher.late list-accounts --debug

  # Publish single video immediately
  python -m src.publisher.late single --video outputs/B0ABC123/video.mp4 \\
      --platform youtube --platform tiktok --immediate

  # Schedule single video
  python -m src.publisher.late single --video outputs/B0ABC123/video.mp4 \\
      --platform youtube --schedule "2025-01-20 14:00:00"

  # Batch publish all videos
  python -m src.publisher.late batch --platform youtube --platform tiktok \\
      --immediate --debug
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # list-accounts command
    list_parser = subparsers.add_parser(
        "list-accounts",
        help="List connected social media accounts",
    )
    list_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # single command
    single_parser = subparsers.add_parser(
        "single",
        help="Publish a single video",
    )
    single_parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to video file to publish",
    )
    single_parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        choices=["youtube", "tiktok", "instagram", "facebook", "twitter", "linkedin"],
        required=True,
        help="Target platform(s) - can be specified multiple times",
    )
    single_parser.add_argument(
        "--schedule",
        type=str,
        metavar="DATETIME",
        help="Schedule for later (format: '2025-01-20 14:00:00') - RECOMMENDED",
    )
    single_parser.add_argument(
        "--immediate",
        action="store_true",
        help="Publish immediately (requires explicit flag to prevent accidents)",
    )
    single_parser.add_argument(
        "--force",
        action="store_true",
        help="Force republish even if already published to platform",
    )
    single_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch publish all videos in outputs directory",
    )
    batch_parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        choices=["youtube", "tiktok", "instagram", "facebook", "twitter", "linkedin"],
        required=True,
        help="Target platform(s) - can be specified multiple times",
    )
    batch_parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to scan for videos (default: outputs)",
    )
    batch_parser.add_argument(
        "--immediate",
        action="store_true",
        help="Publish immediately (only immediate publishing supported in batch mode)",
    )
    batch_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch processing on first failure",
    )
    batch_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.command == "single":
        if args.schedule:
            try:
                args.schedule = parse_datetime(args.schedule)
            except ValueError as e:
                parser.error(str(e))
        elif not args.immediate:
            # Require explicit --schedule or --immediate
            parser.error(
                "Must specify --schedule DATETIME or --immediate. "
                "Use --schedule to prevent accidental immediate posts."
            )

        # Initialize force if not set
        if not hasattr(args, "force"):
            args.force = False

        # Convert platform strings to Platform enums (use [] for name lookup)
        args.platforms = [Platform[p.upper()] for p in args.platforms]

    elif args.command == "batch":
        if not args.immediate:
            parser.error(
                "Batch mode requires --immediate "
                "(scheduled publishing not supported in batch mode)"
            )

        # Convert platform strings to Platform enums (use [] for name lookup)
        args.platforms = [Platform[p.upper()] for p in args.platforms]

    # Load .env
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    load_dotenv(project_root / ".env")

    # Setup logging
    setup_debug_logging(
        log_file=project_root / "outputs" / "logs" / "publisher.log",
        debug_mode=args.debug,
        verbose=args.debug,
        component_name="Publisher",
    )

    if args.debug:
        logger.info("Debug mode enabled")

    # Load configuration
    try:
        # Build CLI overrides
        cli_overrides = {}
        if hasattr(args, "platforms") and args.platforms:
            cli_overrides["default_platforms"] = args.platforms
        if hasattr(args, "immediate") and args.immediate:
            cli_overrides["immediate_publish"] = True
        if hasattr(args, "fail_fast") and args.fail_fast:
            cli_overrides["fail_fast"] = args.fail_fast

        config = load_publisher_config(
            config_path=project_root / "config" / "publisher.yaml",
            cli_overrides=cli_overrides,
        )

        logger.info(f"Configuration loaded: provider={config.provider}")

    except Exception as e:
        logger.error(f"Configuration loading failed: {e}", exc_info=args.debug)
        sys.exit(1)

    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Execute command
        if args.command == "list-accounts":
            await cmd_list_accounts(args, config, session)
        elif args.command == "single":
            await cmd_single(args, config, session)
        elif args.command == "batch":
            await cmd_batch(args, config, session)


if __name__ == "__main__":
    asyncio.run(main())
