"""Post-publication cleanup for product directories.

This module handles cleanup of product directories after successful publication,
with safety features including verification, archiving, and audit logging.
"""

import contextlib
import json
import logging
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.publisher.constants import DEFAULT_OUTPUTS_DIR, MAX_CONCURRENT_CLEANUPS
from src.publisher.models import CleanupConfig, Platform
from src.publisher.tracking import get_publish_record

logger = logging.getLogger(__name__)


def get_schedule_entry(
    product_id: str,
    platform: str,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> dict[str, Any] | None:
    """Get schedule entry for product/platform from schedule.json.

    Args:
    ----
        product_id: Product identifier
        platform: Platform name
        outputs_dir: Root directory containing schedule.json

    Returns:
    -------
        Schedule entry dict if found, None otherwise

    """
    schedule_path = outputs_dir / "schedule.json"
    if not schedule_path.exists():
        return None

    try:
        data = json.loads(schedule_path.read_text())
        entries = data.get("entries", [])

        # Find entry matching product_id and platform
        for entry in entries:
            if entry.get("product_id") == product_id:
                platforms = entry.get("platforms", [])
                if platform in platforms:
                    # Cast to dict to satisfy type checker
                    return dict(entry) if isinstance(entry, dict) else None

        return None
    except Exception as e:
        logger.warning("Failed to load schedule.json: %s", e)
        return None


class CleanupManager:
    """Manages post-publication cleanup of product directories.

    Provides safe cleanup with optional verification, archiving, and comprehensive
    audit logging. Ensures data is only removed after confirming successful
    publication to all required platforms.

    Attributes:
    ----------
        outputs_dir: Root directory containing product outputs
        config: Cleanup configuration with safety rules
        publisher: Publisher instance for verification
        audit_log_path: Path to cleanup_audit.json file

    Example:
    -------
        >>> config = CleanupConfig(enabled=True, verify_before_delete=True)
        >>> manager = CleanupManager(outputs_dir, config, publisher)
        >>> result = await manager.cleanup("B0TEST001", [Platform.YOUTUBE])
        >>> print(f"Cleaned up {result['disk_freed']} bytes")

    """

    def __init__(
        self,
        outputs_dir: Path,
        config: CleanupConfig,
        publisher: Any,
    ):
        """Initialize cleanup manager.

        Args:
        ----
            outputs_dir: Root directory containing product outputs
            config: Cleanup configuration
            publisher: Publisher instance for status verification

        """
        self.outputs_dir = Path(outputs_dir)
        self.config = config
        self.publisher = publisher
        self.audit_log_path = self.outputs_dir / "cleanup_audit.json"

        logger.debug(
            "CleanupManager initialized with outputs_dir=%s, enabled=%s",
            outputs_dir,
            config.enabled,
        )

    def _calculate_dir_size(self, directory: Path) -> int:
        """Calculate total size of directory recursively.

        Computes the total disk space used by all files and subdirectories
        within the specified directory.

        Args:
        ----
            directory: Path to directory to measure

        Returns:
        -------
            Total size in bytes, including all files and subdirectories

        Example:
        -------
            >>> size = manager._calculate_dir_size(Path("outputs/B0TEST001"))
            >>> print(f"Directory uses {size / 1024 / 1024:.2f} MB")

        """
        total_size = 0
        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
        except OSError as e:
            logger.warning("Error calculating size for %s: %s", directory, e)
        return total_size

    def _log_cleanup(
        self,
        product_id: str,
        platforms: list[Platform],
        post_urls: list[str],
        disk_freed_bytes: int,
        archive_path: Path | None = None,
    ) -> None:
        """Append cleanup record to audit log atomically.

        Uses atomic write pattern (temp file + rename) to safely append
        cleanup records without data corruption.

        Args:
        ----
            product_id: Product identifier that was cleaned
            platforms: List of platforms the product was published to
            post_urls: List of published post URLs
            disk_freed_bytes: Amount of disk space freed in bytes
            archive_path: Path to ZIP archive if created, None otherwise

        Raises:
        ------
            IOError: If audit log write fails

        Example:
        -------
            >>> manager._log_cleanup(
            ...     "B0TEST001",
            ...     [Platform.YOUTUBE, Platform.TIKTOK],
            ...     ["https://youtube.com/post123", "https://tiktok.com/post456"],
            ...     1024000,
            ...     Path("outputs/archive/B0TEST001_2025-01-20.zip")
            ... )

        """
        # Load existing audit log or create new one
        if self.audit_log_path.exists():
            try:
                data = json.loads(self.audit_log_path.read_text())
                if not isinstance(data, dict) or "cleanups" not in data:
                    data = {"cleanups": []}
            except Exception as e:
                logger.warning("Failed to load audit log, creating new: %s", e)
                data = {"cleanups": []}
        else:
            data = {"cleanups": []}

        # Create new cleanup record
        record = {
            "product_id": product_id,
            "platforms": [p.value for p in platforms],
            "post_urls": post_urls,
            "cleaned_at": datetime.now(UTC).isoformat(),
            "disk_freed_bytes": disk_freed_bytes,
            "archive_path": str(archive_path) if archive_path else None,
        }

        # Append record
        data["cleanups"].append(record)

        # Atomic write using temp file + rename pattern
        temp_path = self.audit_log_path.with_suffix(".tmp")
        try:
            temp_path.write_text(json.dumps(data, indent=2))
            temp_path.replace(self.audit_log_path)
            logger.info(
                "Logged cleanup for %s: %.2f MB freed",
                product_id,
                disk_freed_bytes / 1024 / 1024,
            )
        except OSError as e:
            logger.error("Failed to write audit log: %s", e)
            if temp_path.exists():
                temp_path.unlink()
            raise OSError(f"Failed to save cleanup audit log: {e}") from e

    async def verify_publication(
        self, product_id: str, platforms: list[Platform]
    ) -> tuple[bool, dict[str, str]]:
        """Verify all platforms successfully published via API.

        Checks publication status for each platform by querying the publisher's
        API. If require_all_platforms is True, verifies all platforms are published.

        Args:
        ----
            product_id: Product identifier to verify
            platforms: List of platforms to check

        Returns:
        -------
            Tuple of (all_published, platform_statuses)
                - all_published: True if all required platforms published
                - platform_statuses: Dict mapping platform name to status

        Example:
        -------
            >>> success, statuses = await manager.verify_publication(
            ...     "B0TEST001", [Platform.YOUTUBE, Platform.TIKTOK]
            ... )
            >>> if success:
            ...     print("Ready for cleanup")
            >>> else:
            ...     print(f"Not published: {statuses}")

        """
        platform_statuses: dict[str, str] = {}

        for platform in platforms:
            # Try loading publish record first (immediate publishing)
            record = get_publish_record(product_id, platform.value, self.outputs_dir)

            # If no publish record, check schedule entries (scheduled posts)
            if not record:
                schedule_entry = get_schedule_entry(
                    product_id, platform.value, self.outputs_dir
                )
                if schedule_entry:
                    record = {
                        "post_id": str(schedule_entry.get("post_id", "")),
                        "status": str(schedule_entry.get("status", "")),
                    }

            if not record:
                platform_statuses[platform.value] = "not_published"
                logger.warning(
                    "No publish or schedule record found for %s on %s",
                    product_id,
                    platform.value,
                )
                continue

            post_id = record.get("post_id")
            local_status = record.get("status", "")

            # For scheduled posts, trust local status (API returns 404)
            if local_status == "scheduled":
                platform_statuses[platform.value] = "scheduled"
                logger.debug(
                    "Platform %s status: scheduled (post_id: %s, from local record)",
                    platform.value,
                    post_id,
                )
                continue

            # For published/unknown status, query API
            if not post_id:
                platform_statuses[platform.value] = "missing_post_id"
                logger.warning(
                    "Record missing post_id for %s on %s", product_id, platform.value
                )
                continue

            try:
                status_info = await self.publisher.get_status(post_id)
                status = status_info.get("status", "unknown")
                platform_statuses[platform.value] = status

                logger.debug(
                    "Platform %s status: %s (post_id: %s, from API)",
                    platform.value,
                    status,
                    post_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to get status for %s (post_id: %s): %s",
                    platform.value,
                    post_id,
                    e,
                )
                platform_statuses[platform.value] = "api_error"

        # Determine if all platforms published/scheduled successfully
        # Accept both "published" and "scheduled" as valid statuses for cleanup
        valid_statuses = {"published", "scheduled"}
        if self.config.require_all_platforms:
            all_published = all(
                status in valid_statuses for status in platform_statuses.values()
            )
        else:
            # At least one platform published/scheduled
            all_published = any(
                status in valid_statuses for status in platform_statuses.values()
            )

        if all_published:
            logger.info(
                "Verification passed for %s (require_all=%s)",
                product_id,
                self.config.require_all_platforms,
            )
        else:
            failed = [
                f"{p}={s}" for p, s in platform_statuses.items() if s != "published"
            ]
            logger.warning(
                "Verification failed for %s: %s", product_id, ", ".join(failed)
            )

        return all_published, platform_statuses

    def archive_directory(self, product_dir: Path) -> Path:
        """Create ZIP archive of directory before deletion.

        Creates a compressed ZIP archive of the entire product directory
        with timestamp in the filename for uniqueness.

        Args:
        ----
            product_dir: Path to product directory to archive

        Returns:
        -------
            Path to created ZIP archive file

        Raises:
        ------
            ValueError: If product_dir doesn't exist or isn't a directory
            IOError: If archive creation fails

        Example:
        -------
            >>> archive = manager.archive_directory(
            ...     Path("outputs/B0TEST001")
            ... )
            >>> print(f"Archived to {archive}")
            outputs/archive/B0TEST001_20250120T143000.zip

        """
        if not product_dir.exists():
            raise ValueError(f"Product directory does not exist: {product_dir}")
        if not product_dir.is_dir():
            raise ValueError(f"Path is not a directory: {product_dir}")

        # Create archive directory if it doesn't exist
        archive_dir = self.config.archive_dir
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Generate archive filename with timestamp
        product_id = product_dir.name
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        archive_name = f"{product_id}_{timestamp}"
        archive_base = archive_dir / archive_name

        logger.info("Creating archive for %s...", product_id)

        try:
            # Create ZIP archive (shutil.make_archive adds .zip extension)
            archive_path = shutil.make_archive(
                str(archive_base), "zip", product_dir.parent, product_dir.name
            )
            archive_path_obj = Path(archive_path)

            archive_size = archive_path_obj.stat().st_size
            logger.info(
                "Archive created: %s (%.2f MB)",
                archive_path_obj.name,
                archive_size / 1024 / 1024,
            )

            return archive_path_obj
        except Exception as e:
            logger.error("Failed to create archive for %s: %s", product_dir, e)
            raise OSError(f"Archive creation failed: {e}") from e

    def _should_cleanup(self, product_id: str, published_at: datetime | None) -> bool:
        """Check if product should be cleaned up based on age.

        Applies keep_published_days rule to determine if enough time has
        passed since publication.

        Args:
        ----
            product_id: Product identifier
            published_at: When product was published (None = cleanup immediately)

        Returns:
        -------
            True if product should be cleaned up, False otherwise

        """
        if self.config.keep_published_days == 0:
            # Immediate cleanup
            return True

        if not published_at:
            logger.warning(
                "No published_at timestamp for %s, allowing cleanup", product_id
            )
            return True

        # Check if enough days have passed
        age = datetime.now(UTC) - published_at
        required_age = timedelta(days=self.config.keep_published_days)

        if age >= required_age:
            logger.debug(
                "%s is %d days old (>= %d), cleanup allowed",
                product_id,
                age.days,
                self.config.keep_published_days,
            )
            return True
        else:
            logger.info(
                "%s is only %d days old (< %d), skipping cleanup",
                product_id,
                age.days,
                self.config.keep_published_days,
            )
            return False

    async def cleanup(
        self, product_id: str, platforms: list[Platform], dry_run: bool = False
    ) -> dict[str, bool | str | int]:
        """Cleanup published product directory with safety checks.

        Executes cleanup workflow with optional verification and archiving
        based on configuration. Removes product directory and logs to audit.

        Args:
        ----
            product_id: Product identifier to cleanup
            platforms: List of platforms product was published to
            dry_run: If True, preview without actual deletion

        Returns:
        -------
            Dictionary with:
                - success: Whether cleanup succeeded
                - message: Human-readable result message
                - disk_freed: Bytes of disk space freed (0 if dry_run)

        Example:
        -------
            >>> result = await manager.cleanup(
            ...     "B0TEST001", [Platform.YOUTUBE], dry_run=True
            ... )
            >>> print(result['message'])
            [DRY RUN] Would cleanup B0TEST001 (150.5 MB)

        """
        if not self.config.enabled:
            return {
                "success": False,
                "message": f"Cleanup disabled in configuration for {product_id}",
                "disk_freed": 0,
            }

        product_dir = self.outputs_dir / product_id

        if not product_dir.exists():
            return {
                "success": False,
                "message": f"Product directory not found: {product_id}",
                "disk_freed": 0,
            }

        # Check age requirement
        # Try to get published_at from tracking
        if platforms:
            first_platform = platforms[0]
            record = get_publish_record(
                product_id, first_platform.value, self.outputs_dir
            )
            published_at_str = record.get("published_at") if record else None
            published_at = None
            if published_at_str:
                with contextlib.suppress(ValueError, TypeError):
                    published_at = datetime.fromisoformat(published_at_str)

            if not self._should_cleanup(product_id, published_at):
                return {
                    "success": False,
                    "message": (
                        f"Product {product_id} not old enough for cleanup "
                        f"(keep_published_days={self.config.keep_published_days})"
                    ),
                    "disk_freed": 0,
                }

        # Verify publication if configured
        if self.config.verify_before_delete:
            logger.info("Verifying publication status for %s...", product_id)
            all_published, statuses = await self.verify_publication(
                product_id, platforms
            )

            if not all_published:
                failed_platforms = [
                    f"{p}={s}" for p, s in statuses.items() if s != "published"
                ]
                return {
                    "success": False,
                    "message": (
                        f"Cannot cleanup {product_id}: not published to all "
                        f"platforms ({', '.join(failed_platforms)})"
                    ),
                    "disk_freed": 0,
                }

        # Calculate disk space
        disk_freed = self._calculate_dir_size(product_dir)

        # Dry run mode
        if dry_run:
            logger.info(
                "[DRY RUN] Would cleanup %s (%.2f MB)",
                product_id,
                disk_freed / 1024 / 1024,
            )
            return {
                "success": True,
                "message": (
                    f"[DRY RUN] Would cleanup {product_id} "
                    f"({disk_freed / 1024 / 1024:.2f} MB)"
                ),
                "disk_freed": 0,
            }

        # Archive if configured
        archive_path = None
        if self.config.archive_before_delete:
            try:
                archive_path = self.archive_directory(product_dir)
            except Exception as e:
                logger.error("Archive failed for %s: %s", product_id, e)
                return {
                    "success": False,
                    "message": f"Archive creation failed: {e}",
                    "disk_freed": 0,
                }

        # Get post URLs for audit log
        post_urls = []
        for platform in platforms:
            record = get_publish_record(product_id, platform.value, self.outputs_dir)
            if record and record.get("post_url"):
                post_urls.append(record["post_url"])

        # Remove directory
        try:
            shutil.rmtree(product_dir)
            logger.info(
                "Removed product directory: %s (%.2f MB freed)",
                product_id,
                disk_freed / 1024 / 1024,
            )
        except Exception as e:
            logger.error("Failed to remove directory %s: %s", product_dir, e)
            return {
                "success": False,
                "message": f"Directory removal failed: {e}",
                "disk_freed": 0,
            }

        # Log cleanup to audit
        try:
            self._log_cleanup(
                product_id, platforms, post_urls, disk_freed, archive_path
            )
        except Exception as e:
            logger.error("Failed to log cleanup for %s: %s", product_id, e)
            # Don't fail cleanup if logging fails

        return {
            "success": True,
            "message": (
                f"Successfully cleaned up {product_id} "
                f"({disk_freed / 1024 / 1024:.2f} MB freed)"
            ),
            "disk_freed": disk_freed,
        }

    async def cleanup_all(
        self, platforms: list[Platform], dry_run: bool = False
    ) -> dict[str, int]:
        """Cleanup all successfully published products in batch.

        Scans outputs directory for all product directories and cleans up
        those that have been published to all required platforms.

        Args:
        ----
            platforms: List of platforms to check for publication
            dry_run: If True, preview without actual deletion

        Returns:
        -------
            Dictionary with:
                - cleaned: Number of products cleaned
                - skipped: Number of products skipped
                - disk_freed: Total bytes of disk space freed

        Example:
        -------
            >>> summary = await manager.cleanup_all(
            ...     [Platform.YOUTUBE, Platform.TIKTOK], dry_run=False
            ... )
            >>> print(f"Cleaned {summary['cleaned']} products")
            >>> print(f"Freed {summary['disk_freed'] / 1024 / 1024:.2f} MB")

        """
        if not self.config.enabled:
            logger.info("Cleanup disabled in configuration")
            return {"cleaned": 0, "skipped": 0, "disk_freed": 0}

        # Scan for product directories
        product_dirs = [d for d in self.outputs_dir.iterdir() if d.is_dir()]

        # Filter out non-product directories
        product_dirs = [
            d
            for d in product_dirs
            if not d.name.startswith(".") and d.name not in ["archive", "__pycache__"]
        ]

        logger.info("Found %d product directories", len(product_dirs))

        cleaned = 0
        skipped = 0
        disk_freed = 0

        # Limit concurrent operations
        from asyncio import Semaphore, gather

        semaphore = Semaphore(MAX_CONCURRENT_CLEANUPS)

        async def cleanup_with_semaphore(product_id: str):
            async with semaphore:
                return await self.cleanup(product_id, platforms, dry_run)

        # Process each product
        results = await gather(
            *[cleanup_with_semaphore(d.name) for d in product_dirs],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("Cleanup failed: %s", result)
                skipped += 1
            elif isinstance(result, dict):
                if result.get("success"):
                    cleaned += 1
                    disk_freed += result.get("disk_freed", 0)
                else:
                    skipped += 1
            else:
                skipped += 1

        logger.info(
            "Cleanup complete: %d cleaned, %d skipped, %.2f MB freed",
            cleaned,
            skipped,
            disk_freed / 1024 / 1024,
        )

        return {"cleaned": cleaned, "skipped": skipped, "disk_freed": disk_freed}
