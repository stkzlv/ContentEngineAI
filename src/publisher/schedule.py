"""Schedule management for video publishing.

Provides calendar view, recurring schedule slots, and batch scheduling
capabilities for the publisher module.
"""

import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from src.publisher.models import Platform, RecurringSlot, ScheduleConfig, ScheduleEntry
from src.publisher.schedule_validator import ScheduleValidator

logger = logging.getLogger(__name__)


class ScheduleManager:
    """Manages calendar view and recurring schedule operations.

    Handles loading/saving schedule entries, calculating next available
    recurring slots, and filtering scheduled posts.

    Attributes
    ----------
        schedule_path: Path to schedule.json file
        config: Schedule configuration with validation rules
        entries: List of schedule entries loaded from disk

    """

    def __init__(
        self,
        schedule_path: Path | str = Path("outputs/schedule.json"),
        config: ScheduleConfig | None = None,
    ):
        """Initialize schedule manager.

        Args:
        ----
            schedule_path: Path to schedule.json file
            config: Schedule configuration (uses defaults if None)

        """
        self.schedule_path = (
            Path(schedule_path) if isinstance(schedule_path, str) else schedule_path
        )
        self.config = config or ScheduleConfig()
        self.entries: list[ScheduleEntry] = []
        self._load_schedule()

    def _load_schedule(self) -> None:
        """Load schedule entries from JSON file.

        Handles missing files gracefully by starting with empty schedule.
        Logs warnings for corrupted data but continues operation.
        """
        if not self.schedule_path.exists():
            logger.debug(
                f"Schedule file not found: {self.schedule_path}, starting empty"
            )
            self.entries = []
            return

        try:
            data = json.loads(self.schedule_path.read_text())
            if not isinstance(data, dict):
                logger.warning(
                    f"Invalid schedule format in {self.schedule_path}, starting empty"
                )
                self.entries = []
                return

            # Parse entries from JSON
            entries_data = data.get("entries", [])
            if not isinstance(entries_data, list):
                logger.warning("Invalid entries format, starting empty")
                self.entries = []
                return

            # Convert dict entries to ScheduleEntry objects
            entries = []
            for entry_dict in entries_data:
                try:
                    # Parse datetime fields
                    scheduled_time = datetime.fromisoformat(
                        entry_dict["scheduled_time"]
                    )
                    created_at = datetime.fromisoformat(entry_dict["created_at"])

                    # Parse platforms
                    from src.publisher.models import Platform

                    platforms = [Platform(p) for p in entry_dict["platforms"]]

                    # Create ScheduleEntry
                    entry = ScheduleEntry(
                        product_id=entry_dict["product_id"],
                        scheduled_time=scheduled_time,
                        platforms=platforms,
                        post_id=entry_dict.get("post_id"),
                        status=entry_dict.get("status", "pending"),
                        created_at=created_at,
                        slot_index=entry_dict.get("slot_index"),
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to parse entry {entry_dict}: {e}")
                    continue

            self.entries = entries
            logger.info(f"Loaded {len(self.entries)} schedule entries")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse schedule JSON: {e}, starting empty")
            self.entries = []
        except Exception as e:
            logger.error(f"Error loading schedule: {e}, starting empty")
            self.entries = []

    def _save_schedule(self) -> None:
        """Save schedule entries to JSON file atomically.

        Uses temp file + rename for atomic write operation to prevent
        corruption if process is interrupted.
        """
        # Ensure parent directory exists
        self.schedule_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for serialization
        data = {
            "entries": [entry.to_dict() for entry in self.entries],
            "last_updated": datetime.now(UTC).isoformat(),
        }

        # Atomic write: write to temp file, then rename
        try:
            # Create temp file in same directory as target
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.schedule_path.parent,
                prefix=".schedule_",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2, default=str)
                tmp_path = Path(tmp_file.name)

            # Atomic rename
            tmp_path.replace(self.schedule_path)
            logger.debug(f"Saved {len(self.entries)} entries to {self.schedule_path}")

        except Exception as e:
            logger.error(f"Failed to save schedule: {e}")
            # Clean up temp file if it exists
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def get_next_slot(
        self, slots: list[RecurringSlot], after: datetime, slot_index: int = 0
    ) -> tuple[datetime, int]:
        """Get next available recurring slot.

        Calculates the next occurrence of recurring slots starting from
        slot_index, wrapping around if necessary.

        Args:
        ----
            slots: List of recurring slots to check
            after: Reference datetime to calculate from (timezone-aware)
            slot_index: Starting slot index (default: 0)

        Returns:
        -------
            Tuple of (next_datetime, slot_index)

        Raises:
        ------
            ValueError: If slots list is empty or after is timezone-naive

        Example:
        -------
            >>> slots = [
            ...     RecurringSlot("monday", "10:00:00", "UTC"),
            ...     RecurringSlot("wednesday", "14:00:00", "UTC"),
            ... ]
            >>> after = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)  # Wednesday
            >>> next_time, next_idx = manager.get_next_slot(slots, after)
            >>> # Returns next Wednesday at 14:00 UTC, index 1

        """
        if not slots:
            raise ValueError("slots list cannot be empty")

        if after.tzinfo is None:
            raise ValueError("after datetime must be timezone-aware")

        # Validate slot_index
        if slot_index < 0 or slot_index >= len(slots):
            raise ValueError(
                f"slot_index must be between 0 and {len(slots) - 1}, got {slot_index}"
            )

        # Calculate next occurrence for each slot starting from slot_index
        # Find the earliest next occurrence
        min_time = None
        min_index = slot_index

        # Check slots starting from slot_index (wrap around)
        for i in range(len(slots)):
            idx = (slot_index + i) % len(slots)
            slot = slots[idx]

            try:
                next_time = slot.next_occurrence(after)

                # Track the earliest next occurrence
                if min_time is None or next_time < min_time:
                    min_time = next_time
                    min_index = idx

            except Exception as e:
                logger.warning(
                    f"Failed to calculate next occurrence for slot {idx}: {e}"
                )
                continue

        if min_time is None:
            raise ValueError("No valid next slot found")

        return min_time, min_index

    def list_scheduled(
        self,
        platform: str | None = None,
        status: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[ScheduleEntry]:
        """List scheduled posts with optional filtering.

        Filters entries by platform, status, and date range, then sorts
        by scheduled_time in ascending order.

        Args:
        ----
            platform: Filter by platform name (e.g., "youtube", "tiktok")
            status: Filter by status (pending, scheduled, published, failed, partial)
            date_from: Only include entries scheduled on or after this datetime
            date_to: Only include entries scheduled on or before this datetime

        Returns:
        -------
            List of ScheduleEntry objects matching filters, sorted by scheduled_time

        Example:
        -------
            >>> # Get all YouTube posts scheduled this week
            >>> entries = manager.list_scheduled(
            ...     platform="youtube",
            ...     date_from=datetime(2025, 1, 20, tzinfo=UTC),
            ...     date_to=datetime(2025, 1, 26, 23, 59, 59, tzinfo=UTC)
            ... )

        """
        from src.publisher.models import Platform

        # Start with all entries
        filtered = self.entries.copy()

        # Filter by platform
        if platform is not None:
            # Convert string to Platform enum for comparison
            try:
                platform_enum = Platform(platform.lower())
                filtered = [
                    entry
                    for entry in filtered
                    if platform_enum in entry.platforms
                ]
            except ValueError:
                logger.warning(f"Invalid platform '{platform}', returning empty list")
                return []

        # Filter by status
        if status is not None:
            filtered = [entry for entry in filtered if entry.status == status]

        # Filter by date_from (inclusive)
        if date_from is not None:
            # Ensure timezone-aware comparison
            if date_from.tzinfo is None:
                logger.warning("date_from is timezone-naive, treating as UTC")
                date_from = date_from.replace(tzinfo=UTC)

            filtered = [
                entry
                for entry in filtered
                if entry.scheduled_time >= date_from
            ]

        # Filter by date_to (inclusive)
        if date_to is not None:
            # Ensure timezone-aware comparison
            if date_to.tzinfo is None:
                logger.warning("date_to is timezone-naive, treating as UTC")
                date_to = date_to.replace(tzinfo=UTC)

            filtered = [
                entry
                for entry in filtered
                if entry.scheduled_time <= date_to
            ]

        # Sort by scheduled_time (ascending)
        filtered.sort(key=lambda e: e.scheduled_time)

        logger.debug(
            f"Filtered {len(self.entries)} entries to {len(filtered)} "
            f"(platform={platform}, status={status}, "
            f"date_from={date_from}, date_to={date_to})"
        )

        return filtered

    async def auto_schedule(
        self,
        videos: list[Path],
        platforms: list[Platform],
        publisher: "BasePublisher",  # type: ignore[name-defined]
        start_slot: int = 0,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Auto-assign videos to recurring slots.

        Batch schedules multiple videos to recurring time slots, respecting
        configuration rules and skipping already-published content.

        Args:
        ----
            videos: List of video file paths to schedule
            platforms: List of platforms to publish to
            publisher: Publisher instance for calling publish()
            start_slot: Starting slot index (default: 0)
            dry_run: Preview without publishing (default: False)

        Returns:
        -------
            Summary dictionary with keys: scheduled, skipped, failed

        Raises:
        ------
            ValueError: If recurring schedule not enabled or no slots configured

        Example:
        -------
            >>> from pathlib import Path
            >>> from src.publisher.models import Platform
            >>> videos = [Path("outputs/B0ABC123/video_B0ABC123.mp4")]
            >>> summary = await manager.auto_schedule(
            ...     videos=videos,
            ...     platforms=[Platform.YOUTUBE, Platform.TIKTOK],
            ...     publisher=publisher_instance,
            ...     start_slot=0,
            ...     dry_run=False
            ... )
            >>> print(f"Scheduled: {summary['scheduled']}")

        """
        # Validate recurring schedule is enabled and has slots
        if not self.config.enabled:
            raise ValueError(
                "Recurring schedule is not enabled. "
                "Set recurring_schedule.enabled: true in config"
            )

        if not self.config.slots:
            raise ValueError(
                "No recurring slots configured. "
                "Add slots to recurring_schedule section in config"
            )

        logger.info(
            f"Auto-scheduling {len(videos)} video(s) to "
            f"{len(self.config.slots)} recurring slot(s)"
        )
        logger.info(f"Platforms: {', '.join([p.value for p in platforms])}")
        logger.info(f"Start slot: {start_slot}, Dry run: {dry_run}")

        # Initialize counters
        scheduled_count = 0
        skipped_count = 0
        failed_count = 0

        # Current slot index (wraps around)
        current_slot = start_slot
        # Reference time for calculating next slot
        current_time = datetime.now(UTC)

        # Import tracking utilities
        from src.publisher.tracking import is_already_published

        for video in videos:
            try:
                # Extract product_id from video path
                # e.g., "outputs/B0ABC123/video_B0ABC123.mp4" -> "B0ABC123"
                product_id = video.parent.name
                logger.debug(f"Processing video: {product_id}")

                # Check if already published to ANY of the specified platforms
                already_published = []
                for platform in platforms:
                    if is_already_published(product_id, platform.value):
                        already_published.append(platform.value)

                if already_published:
                    logger.info(
                        f"Skipping {product_id}: already published to "
                        f"{', '.join(already_published)}"
                    )
                    skipped_count += 1
                    continue

                # Find next available slot
                try:
                    next_time, next_idx = self.get_next_slot(
                        slots=self.config.slots,
                        after=current_time,
                        slot_index=current_slot,
                    )
                    logger.debug(
                        f"Next slot for {product_id}: {next_time} (slot {next_idx})"
                    )
                except Exception as e:
                    logger.error(f"Failed to calculate next slot: {e}")
                    failed_count += 1
                    continue

                # Create ScheduleEntry
                entry = ScheduleEntry(
                    product_id=product_id,
                    scheduled_time=next_time,
                    platforms=platforms.copy(),
                    post_id=None,
                    status="pending",
                    created_at=datetime.now(UTC),
                    slot_index=next_idx,
                )

                # Validate entry before publishing
                validator = ScheduleValidator(self.config, self.entries)
                is_valid, error_message = validator.validate(entry)
                if not is_valid:
                    logger.warning(
                        f"Validation failed for {product_id}: {error_message}"
                    )
                    failed_count += 1
                    continue

                if dry_run:
                    # Dry run mode: just log without publishing
                    logger.info(
                        f"[DRY RUN] Would schedule {product_id} at "
                        f"{next_time} (slot {next_idx})"
                    )
                    scheduled_count += 1
                else:
                    # Call publisher.publish() with scheduled_time
                    try:
                        logger.info(
                            f"Scheduling {product_id} at {next_time} (slot {next_idx})"
                        )

                        # Get actual account IDs from publisher
                        accounts = await publisher.get_accounts()
                        account_map = {
                            acc["platform"]: acc["account_id"] for acc in accounts
                        }

                        # Prepare platforms for publisher
                        platform_dicts = []
                        for p in platforms:
                            account_id = account_map.get(p.value)
                            if not account_id:
                                logger.warning(f"No account for {p.value}")
                                continue
                            platform_dicts.append({
                                "platform": p.value,
                                "account_id": account_id,
                            })

                        if not platform_dicts:
                            raise ValueError("No valid accounts for platforms")

                        # Upload video first (publisher needs media_id)
                        media_id = await publisher.upload_media(video)

                        # Build per-platform content from metadata files
                        import json
                        platform_contents = {}
                        for p in platforms:
                            meta_file = f"metadata_{p.value}.json"
                            platform_meta = video.parent / meta_file
                            if platform_meta.exists():
                                meta = json.loads(platform_meta.read_text())
                                desc = meta.get("description", "")
                                hashtags = meta.get("hashtags", [])
                                if hashtags:
                                    desc = f"{desc}\n\n{' '.join(hashtags)}"
                                if p.value == "youtube" and meta.get("title"):
                                    platform_contents[p.value] = {
                                        "content": desc,
                                        "title": meta.get("title"),
                                    }
                                else:
                                    platform_contents[p.value] = {"content": desc}
                            else:
                                # Fallback to data.json
                                fallback_path = video.parent / "data.json"
                                if fallback_path.exists():
                                    fb = json.loads(fallback_path.read_text())
                                    if isinstance(fb, list) and fb:
                                        fb = fb[0]
                                    title = fb.get("title", "Product Video")
                                    desc = fb.get("description", "")
                                    platform_contents[p.value] = {
                                        "content": f"{title}\n\n{desc}"
                                    }
                                else:
                                    platform_contents[p.value] = {
                                        "content": f"Product video for {product_id}"
                                    }

                        # Publish with per-platform content
                        result = await publisher.publish(
                            media_id=media_id,
                            platforms=platform_dicts,
                            platform_contents=platform_contents,
                            scheduled_time=next_time,
                        )

                        # Update entry with post_id
                        entry.post_id = result.get("post_id")
                        entry.status = "scheduled"

                        # Add entry to schedule and save
                        self.entries.append(entry)
                        self._save_schedule()

                        logger.info(
                            f"Scheduled {product_id} (post: {entry.post_id})"
                        )
                        scheduled_count += 1

                    except Exception as e:
                        logger.error(f"Failed to schedule {product_id}: {e}")
                        entry.status = "failed"
                        # Still add to schedule to track failures
                        self.entries.append(entry)
                        self._save_schedule()
                        failed_count += 1
                        continue

                # Move to next slot for next video
                current_slot = (next_idx + 1) % len(self.config.slots)
                # Update current_time to after this scheduled post
                current_time = next_time

            except Exception as e:
                logger.error(f"Unexpected error processing {video}: {e}")
                failed_count += 1
                continue

        # Log summary
        logger.info(
            f"Auto-schedule complete: "
            f"scheduled={scheduled_count}, "
            f"skipped={skipped_count}, "
            f"failed={failed_count}"
        )

        return {
            "scheduled": scheduled_count,
            "skipped": skipped_count,
            "failed": failed_count,
        }

    def add_entry(self, entry: ScheduleEntry) -> None:
        """Add a schedule entry atomically.

        Validates the entry using ScheduleValidator and adds it to the schedule
        with atomic write operation to prevent data corruption.

        Note: Basic field validation (product_id, scheduled_time, platforms,
        status) is already handled by ScheduleEntry.__post_init__(). This
        method performs comprehensive validation including duplicates, spacing,
        and daily limits using ScheduleValidator.

        Args:
        ----
            entry: ScheduleEntry object to add (must pass __post_init__ validation)

        Raises:
        ------
            ValueError: If entry fails validation (duplicates, spacing, etc.)
            IOError: If schedule file write fails

        Example:
        -------
            >>> entry = ScheduleEntry(
            ...     product_id="B0TEST001",
            ...     scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
            ...     platforms=[Platform.YOUTUBE],
            ...     post_id=None,
            ...     status="pending",
            ...     created_at=datetime.now(UTC),
            ... )
            >>> manager.add_entry(entry)

        """
        # Validate entry using ScheduleValidator
        validator = ScheduleValidator(self.config, self.entries)
        is_valid, error_message = validator.validate(entry)

        if not is_valid:
            logger.warning(
                f"Validation failed for {entry.product_id}: {error_message}"
            )
            raise ValueError(f"Entry validation failed: {error_message}")

        # Add to entries list
        self.entries.append(entry)
        logger.debug(
            f"Adding entry for {entry.product_id} scheduled at {entry.scheduled_time}"
        )

        # Atomic write to disk
        try:
            self._save_schedule()
            logger.info(
                f"Successfully added entry for {entry.product_id} "
                f"(total entries: {len(self.entries)})"
            )
        except Exception as e:
            # Roll back on write failure
            self.entries.pop()
            logger.error(f"Failed to save schedule after adding entry: {e}")
            raise OSError(f"Failed to save schedule: {e}") from e
