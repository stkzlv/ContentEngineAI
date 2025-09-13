#!/usr/bin/env python3
"""Cleanup utility for ContentEngineAI outputs directory.

This script provides an easy way to clean up unexpected files and directories
in the outputs directory based on the configured directory structure.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video.video_config import load_video_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Clean up unexpected files in outputs directory"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "video_producer.yaml",
        help="Path to video producer config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned up without actually deleting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Perform cleanup even if disabled in config",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (only errors and summary)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output (show all actions)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_video_config(args.config)

        if not config.cleanup_settings.enabled and not args.force:
            print("Cleanup is disabled in configuration. Use --force to override.")
            return 1

        # Override dry_run if specified
        dry_run = args.dry_run if args.dry_run else None

        if not args.quiet:
            mode = (
                "DRY RUN"
                if (dry_run or config.cleanup_settings.dry_run)
                else "ACTUAL CLEANUP"
            )
            print(f"Starting outputs directory cleanup ({mode})")
            print(f"Config: {args.config}")
            print(f"Outputs: {config.global_output_root_path}")
            print()

        # Run cleanup
        result = config.cleanup_outputs_directory(dry_run=dry_run)

        if result.get("status") == "disabled":
            print("Cleanup is disabled in configuration")
            return 1

        # Show results
        stats = result["statistics"]
        if not args.quiet:
            print("Cleanup Results:")
            print(f"  Files removed: {stats['files_removed']}")
            print(f"  Directories removed: {stats['directories_removed']}")
            print(f"  Bytes freed: {stats['bytes_freed']:,}")
            print(f"  Errors: {stats['errors']}")
            print(f"  Total actions: {len(result['actions'])}")

        if args.verbose and result["actions"]:
            print("\nActions taken:")
            for action in result["actions"]:
                action_type = action["action"]
                path = action["path"]
                if "size" in action:
                    print(f"  {action_type}: {path} ({action['size']} bytes)")
                else:
                    print(f"  {action_type}: {path}")

        if stats["errors"] > 0:
            print(f"\nWarning: {stats['errors']} errors occurred during cleanup")
            return 1

        if (
            not args.quiet
            and config.cleanup_settings.create_report
            and not (dry_run or config.cleanup_settings.dry_run)
        ):
            report_path = (
                config.global_output_root_path / config.cleanup_settings.report_file
            )
            print(f"\nDetailed report saved to: {report_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
