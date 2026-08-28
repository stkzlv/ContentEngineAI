"""Schedule management for video publishing.

Provides calendar view, recurring schedule slots, and batch scheduling
capabilities for the publisher module.
"""

import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.publisher.base import PublishError
from src.publisher.first_comment import build_first_comment
from src.publisher.link_in_bio.manager import update_link_in_bio_safe
from src.publisher.models import (
    CleanupConfig,
    ConflictResolution,
    LinkInBioConfig,
    Platform,
    RecurringSlot,
    ScheduleConfig,
    ScheduleEntry,
    _trim_on_word_boundary,
    strip_disclosure_tokens,
)
from src.publisher.product_registry import add_to_registry
from src.publisher.schedule_validator import ScheduleValidator
from src.publisher.tracking import is_already_published, record_publish
from src.video.config.constants import (
    SCHEDULE_ALTERNATIVE_SEARCH_MULTIPLIER,
    SCHEDULE_MAX_SLOT_SEARCH_ATTEMPTS,
)

if TYPE_CHECKING:
    from src.publisher.base import BasePublisher

logger = logging.getLogger(__name__)


def caption_from_metadata(meta: dict, product_id: str | None) -> str:
    """Build the caption `schedule auto` publishes, from a metadata file.

    Extracted so it can be driven directly. `schedule auto` does not go
    through `PublishMetadata`, so the disclosure guard on that object does not
    protect this path -- and a test asserting the shared function is merely
    *called* here passes while the call sits behind a dead branch.
    """
    desc = str(meta.get("description", "") or "")
    hashtags = list(meta.get("hashtags", []))

    if not bool(meta.get("carries_affiliate_content", True)):
        # The caption prompts write `#ad` whatever the render carries, and
        # this path gets neither the object guard nor the loader's
        # trailing-hashtag rule.
        desc, hashtags = strip_disclosure_tokens(desc, hashtags)

    if product_id and product_id not in hashtags:
        hashtags.append(product_id)
    if hashtags:
        hashtag_str = " ".join(
            f"#{t}" if not t.startswith("#") else t for t in hashtags
        )
        desc = f"{desc}\n\n{hashtag_str}"
    return desc


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
                "Schedule file not found: %s, starting empty", self.schedule_path
            )
            self.entries = []
            return

        try:
            data = json.loads(self.schedule_path.read_text())
            if not isinstance(data, dict):
                logger.warning(
                    "Invalid schedule format in %s, starting empty", self.schedule_path
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
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning("Failed to parse entry %s: %s", entry_dict, e)
                    continue

            self.entries = entries
            logger.info("Loaded %d schedule entries", len(self.entries))

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse schedule JSON: %s, starting empty", e)
            self.entries = []
        except OSError as e:
            logger.error("Error loading schedule: %s, starting empty", e)
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
            logger.debug(
                "Saved %d entries to %s", len(self.entries), self.schedule_path
            )

        except OSError as e:
            logger.error("Failed to save schedule: %s", e)
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

            except (ValueError, KeyError) as e:
                logger.warning(
                    "Failed to calculate next occurrence for slot %d: %s", idx, e
                )
                continue

        if min_time is None:
            raise ValueError("No valid next slot found")

        return min_time, min_index

    def find_alternatives(
        self,
        preferred_time: datetime,
        platforms: list[Platform],
        occupied_slots: set[datetime],
        count: int | None = None,
    ) -> ConflictResolution:
        """Find alternative slots when preferred time has a conflict.

        Searches for the next N available slots starting from the preferred time,
        sorted by proximity to preserve user's time preference.

        Args:
        ----
            preferred_time: User's originally preferred schedule time
            platforms: Platforms to check for conflicts
            occupied_slots: Set of already-occupied slot times
            count: Number of alternatives to find (defaults to config value)

        Returns:
        -------
            ConflictResolution with alternatives sorted by time proximity

        Example:
        -------
            >>> resolution = manager.find_alternatives(
            ...     preferred_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
            ...     platforms=[Platform.YOUTUBE],
            ...     occupied_slots={datetime(2025, 1, 20, 10, 0, tzinfo=UTC)},
            ... )
            >>> for alt in resolution.alternatives:
            ...     print(f"Alternative: {alt}")

        """
        if count is None:
            count = self.config.conflict_alternatives_count

        if not self.config.slots:
            return ConflictResolution(
                original_time=preferred_time,
                conflict_reason="No recurring slots configured",
                alternatives=[],
            )

        # Normalize preferred time for comparison
        normalized_preferred = preferred_time.replace(second=0, microsecond=0)

        # Determine conflict reason
        conflict_reason = "Slot occupied"
        if normalized_preferred in occupied_slots:
            conflict_reason = f"Slot at {preferred_time.isoformat()} already occupied"
        else:
            # Check validation issues
            temp_entry = ScheduleEntry(
                product_id="__temp__",
                scheduled_time=preferred_time,
                platforms=platforms,
                status="pending",
                created_at=datetime.now(UTC),
            )
            validator = ScheduleValidator(self.config, self.entries)
            is_valid, error_msg = validator.validate(temp_entry)
            if not is_valid:
                conflict_reason = error_msg

        # Find alternatives
        alternatives: list[datetime] = []
        search_time = preferred_time
        max_attempts = count * SCHEDULE_ALTERNATIVE_SEARCH_MULTIPLIER
        attempts = 0
        current_slot = 0

        while len(alternatives) < count and attempts < max_attempts:
            try:
                next_time, next_idx = self.get_next_slot(
                    slots=self.config.slots,
                    after=search_time,
                    slot_index=current_slot,
                )

                # Normalize for comparison
                normalized = next_time.replace(second=0, microsecond=0)

                # Skip if occupied
                if normalized in occupied_slots:
                    search_time = next_time
                    current_slot = (next_idx + 1) % len(self.config.slots)
                    attempts += 1
                    continue

                # Validate the slot
                temp_entry = ScheduleEntry(
                    product_id="__temp__",
                    scheduled_time=next_time,
                    platforms=platforms,
                    status="pending",
                    created_at=datetime.now(UTC),
                )
                validator = ScheduleValidator(self.config, self.entries)
                is_valid, _ = validator.validate(temp_entry)

                if is_valid and next_time not in alternatives:
                    alternatives.append(next_time)
                    logger.debug("Found alternative slot: %s", next_time)

                # Move to next slot
                search_time = next_time
                current_slot = (next_idx + 1) % len(self.config.slots)
                attempts += 1

            except (ValueError, KeyError) as e:
                logger.warning("Error finding alternative: %s", e)
                attempts += 1
                break

        # Sort by proximity to preferred time
        alternatives.sort(key=lambda t: abs((t - preferred_time).total_seconds()))

        logger.info(
            "Found %d alternatives for conflict at %s",
            len(alternatives),
            preferred_time,
        )

        return ConflictResolution(
            original_time=preferred_time,
            conflict_reason=conflict_reason,
            alternatives=alternatives,
        )

    def resolve_conflict(
        self,
        preferred_time: datetime,
        platforms: list[Platform],
        occupied_slots: set[datetime],
        auto_resolve: bool = False,
    ) -> ConflictResolution:
        """Resolve a scheduling conflict with optional auto-resolution.

        Finds alternative slots and optionally auto-selects the first available one.

        Args:
        ----
            preferred_time: User's originally preferred schedule time
            platforms: Platforms to check for conflicts
            occupied_slots: Set of already-occupied slot times
            auto_resolve: If True, automatically use first available alternative

        Returns:
        -------
            ConflictResolution with auto_resolved=True and resolved_time set
            if auto_resolve was enabled and an alternative was found

        Example:
        -------
            >>> resolution = manager.resolve_conflict(
            ...     preferred_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
            ...     platforms=[Platform.YOUTUBE],
            ...     occupied_slots={datetime(2025, 1, 20, 10, 0, tzinfo=UTC)},
            ...     auto_resolve=True,
            ... )
            >>> if resolution.auto_resolved:
            ...     print(f"Auto-resolved to: {resolution.resolved_time}")

        """
        resolution = self.find_alternatives(
            preferred_time=preferred_time,
            platforms=platforms,
            occupied_slots=occupied_slots,
        )

        if auto_resolve and resolution.alternatives:
            resolved_time = resolution.alternatives[0]
            resolution.auto_resolved = True
            resolution.resolved_time = resolved_time
            logger.info(
                "Auto-resolved conflict: %s -> %s (reason: %s)",
                preferred_time,
                resolved_time,
                resolution.conflict_reason,
            )

        return resolution

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
        # Start with all entries
        filtered = self.entries.copy()

        # Filter by platform
        if platform is not None:
            # Convert string to Platform enum for comparison
            try:
                platform_enum = Platform(platform.lower())
                filtered = [
                    entry for entry in filtered if platform_enum in entry.platforms
                ]
            except ValueError:
                logger.warning("Invalid platform '%s', returning empty list", platform)
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
                entry for entry in filtered if entry.scheduled_time >= date_from
            ]

        # Filter by date_to (inclusive)
        if date_to is not None:
            # Ensure timezone-aware comparison
            if date_to.tzinfo is None:
                logger.warning("date_to is timezone-naive, treating as UTC")
                date_to = date_to.replace(tzinfo=UTC)

            filtered = [entry for entry in filtered if entry.scheduled_time <= date_to]

        # Sort by scheduled_time (ascending)
        filtered.sort(key=lambda e: e.scheduled_time)

        logger.debug(
            "Filtered %d entries to %d "
            "(platform=%s, status=%s, date_from=%s, date_to=%s)",
            len(self.entries),
            len(filtered),
            platform,
            status,
            date_from,
            date_to,
        )

        return filtered

    async def auto_schedule(
        self,
        videos: list[Path],
        platforms: list[Platform],
        publisher: "BasePublisher",
        start_slot: int = 0,
        dry_run: bool = False,
        cleanup_config: CleanupConfig | None = None,
        outputs_dir: Path | None = None,
        auto_resolve: bool = False,
        force: bool = False,
        link_in_bio_config: LinkInBioConfig | None = None,
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
            cleanup_config: Cleanup configuration (default: CleanupConfig() with
                enabled=True). Runs cleanup after successful scheduling.
            outputs_dir: Base outputs directory for cleanup (required if cleanup
                is enabled)
            auto_resolve: Automatically resolve conflicts using first alternative
            force: Skip already-published check and schedule regardless
            link_in_bio_config: Link-in-bio configuration (default: enabled).
                Bio link is added after each successful schedule, before cleanup

        Returns:
        -------
            Summary dictionary with keys: scheduled, skipped, failed, cleaned,
            conflicts_resolved

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
            ...     dry_run=False,
            ...     auto_resolve=True
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
            "Auto-scheduling %d video(s) to %d recurring slot(s)",
            len(videos),
            len(self.config.slots),
        )
        logger.info("Platforms: %s", ", ".join([p.value for p in platforms]))
        logger.info("Start slot: %d, Dry run: %s", start_slot, dry_run)

        # Initialize reference time for calculating next slot
        current_time = datetime.now(UTC)

        # Build set of occupied slots from API to find gaps
        # NOTE: We do NOT create local entries from API posts because the API
        # doesn't return the original product_id (Amazon ASIN). Creating entries
        # with placeholder product_ids (e.g., API_abc123) causes confusion and
        # breaks duplicate detection. Instead, we track occupied slot times.
        occupied_slot_times: set[datetime] = set()
        try:
            logger.debug("Fetching existing posts from API (all statuses)...")
            # Fetch ALL posts (scheduled + published) to avoid slot conflicts
            api_posts = await publisher.list_posts()
            logger.debug("Found %d posts on API", len(api_posts))

            # Build set of all occupied slot times from API posts
            for api_post in api_posts:
                scheduled_time = api_post.get("scheduledFor")
                if not scheduled_time:
                    continue

                # Parse scheduled time (handle both datetime and string)
                if isinstance(scheduled_time, str):
                    # Parse ISO format datetime string
                    time_str = scheduled_time.replace("+00:00", "")
                    scheduled_dt = datetime.fromisoformat(time_str)
                else:
                    scheduled_dt = scheduled_time

                # Ensure timezone-aware
                if scheduled_dt.tzinfo is None:
                    scheduled_dt = scheduled_dt.replace(tzinfo=UTC)

                # Normalize to slot time (remove seconds/microseconds)
                normalized = scheduled_dt.replace(second=0, microsecond=0)
                occupied_slot_times.add(normalized)

            logger.info("Found %d occupied slots from API", len(occupied_slot_times))

            # Also include local schedule entries in occupied slots
            # This prevents duplicates and enables proper gap-filling
            for entry in self.entries:
                normalized = entry.scheduled_time.replace(second=0, microsecond=0)
                occupied_slot_times.add(normalized)

            logger.info(
                "Total %d occupied slots (API + local)", len(occupied_slot_times)
            )

            # Log latest post time for debugging, but keep current_time at NOW
            # to enable gap-filling from current time forward
            if occupied_slot_times:
                api_latest_time = max(occupied_slot_times)
                logger.info("Latest post on API: %s", api_latest_time)
                logger.info(
                    "Searching for next available slot from now (%s), "
                    "skipping %d occupied slots",
                    current_time,
                    len(occupied_slot_times),
                )

        except (PublishError, OSError, TimeoutError) as e:
            logger.warning("Failed to check API schedule: %s", e)

        # Initialize counters
        scheduled_count = 0
        skipped_count = 0
        failed_count = 0
        cleaned_count = 0
        conflicts_resolved_count = 0

        # Initialize cleanup manager if cleanup enabled
        cleanup_manager = None
        if cleanup_config is None:
            cleanup_config = CleanupConfig()  # Default: enabled=True
        if cleanup_config.enabled and outputs_dir:
            from src.publisher.cleanup import CleanupManager

            cleanup_manager = CleanupManager(outputs_dir, cleanup_config, publisher)
            logger.info("Cleanup enabled - will cleanup after successful scheduling")

        # Current slot index (wraps around)
        current_slot = start_slot

        for video in videos:
            try:
                # Extract product_id from video path
                # e.g., "outputs/B0ABC123/video_B0ABC123.mp4" -> "B0ABC123"
                product_id = video.parent.name
                logger.debug("Processing video: %s", product_id)

                # Check if already published to ANY of the specified platforms
                if not force:
                    already_published = []
                    for platform in platforms:
                        if is_already_published(product_id, platform.value):
                            already_published.append(platform.value)

                    if already_published:
                        logger.info(
                            "Skipping %s: already published to %s",
                            product_id,
                            ", ".join(already_published),
                        )
                        skipped_count += 1
                        continue

                # Find next available slot (skip occupied slots from API)
                try:
                    search_time = current_time
                    max_attempts = SCHEDULE_MAX_SLOT_SEARCH_ATTEMPTS
                    attempts = 0

                    while attempts < max_attempts:
                        next_time, next_idx = self.get_next_slot(
                            slots=self.config.slots,
                            after=search_time,
                            slot_index=current_slot,
                        )

                        # Normalize slot time for comparison
                        normalized = next_time.replace(second=0, microsecond=0)

                        # Check if slot is occupied by API post
                        if normalized not in occupied_slot_times:
                            logger.debug(
                                "Next slot for %s: %s (slot %d)",
                                product_id,
                                next_time,
                                next_idx,
                            )
                            break

                        # Slot occupied, try next one
                        logger.debug(
                            "Slot %s occupied by API post, trying next slot", next_time
                        )
                        search_time = next_time
                        attempts += 1

                    if attempts >= max_attempts:
                        raise ValueError(
                            f"No available slot after {max_attempts} attempts"
                        )

                except (ValueError, KeyError) as e:
                    logger.error("Failed to calculate next slot: %s", e)
                    failed_count += 1
                    continue

                # Validate scheduling (check slot availability for this product)
                # Note: We'll create separate posts per platform, but validate once
                # for all
                temp_entry = ScheduleEntry(
                    product_id=product_id,
                    scheduled_time=next_time,
                    platforms=[platforms[0]],  # Validate with first platform only
                    post_id=None,
                    status="pending",
                    created_at=datetime.now(UTC),
                    slot_index=next_idx,
                )

                validator = ScheduleValidator(self.config, self.entries)
                is_valid, error_message = validator.validate(temp_entry)
                if not is_valid:
                    if auto_resolve:
                        # Try to resolve conflict by finding alternative
                        resolution = self.resolve_conflict(
                            preferred_time=next_time,
                            platforms=platforms,
                            occupied_slots=occupied_slot_times,
                            auto_resolve=True,
                        )
                        if resolution.auto_resolved and resolution.resolved_time:
                            logger.info(
                                "Conflict resolved for %s: %s -> %s (reason: %s)",
                                product_id,
                                next_time,
                                resolution.resolved_time,
                                resolution.conflict_reason,
                            )
                            next_time = resolution.resolved_time
                            conflicts_resolved_count += 1
                            # Mark the resolved time as occupied
                            occupied_slot_times.add(
                                next_time.replace(second=0, microsecond=0)
                            )
                        else:
                            logger.warning(
                                "Could not resolve conflict for %s: %s",
                                product_id,
                                error_message,
                            )
                            if resolution.alternatives:
                                alt_str = ", ".join(
                                    t.isoformat() for t in resolution.alternatives[:3]
                                )
                                logger.info("Available alternatives: %s", alt_str)
                            failed_count += 1
                            continue
                    else:
                        logger.warning(
                            "Validation failed for %s: %s", product_id, error_message
                        )
                        # Suggest alternatives even without auto-resolve
                        resolution = self.find_alternatives(
                            preferred_time=next_time,
                            platforms=platforms,
                            occupied_slots=occupied_slot_times,
                        )
                        if resolution.alternatives:
                            alt_str = ", ".join(
                                t.isoformat() for t in resolution.alternatives[:3]
                            )
                            logger.info(
                                "Suggested alternatives for %s: %s", product_id, alt_str
                            )
                            logger.info("Use --auto-resolve to automatically use first")
                        failed_count += 1
                        continue

                if dry_run:
                    # Dry run mode: just log without publishing
                    logger.info(
                        "[DRY RUN] Would schedule %s at %s (slot %d)",
                        product_id,
                        next_time,
                        next_idx,
                    )
                    scheduled_count += 1
                else:
                    # Call publisher.publish() with scheduled_time
                    try:
                        logger.info(
                            "Scheduling %s at %s (slot %d)",
                            product_id,
                            next_time,
                            next_idx,
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
                                logger.warning("No account for %s", p.value)
                                continue
                            platform_dicts.append(
                                {
                                    "platform": p.value,
                                    "account_id": account_id,
                                }
                            )

                        if not platform_dicts:
                            raise ValueError("No valid accounts for platforms")

                        # Upload video first (publisher needs media_id)
                        media_id = await publisher.upload_media(video)

                        # Build per-platform content from metadata files.
                        # Explicit str keys so mypy doesn't infer Literal[...] from
                        # Platform enum values used as keys below.
                        platform_contents: dict[str, dict[str, Any]] = {}

                        # Try unified metadata.json first
                        unified_meta_path = video.parent / "metadata.json"
                        unified_meta = None
                        if unified_meta_path.exists():
                            unified_meta = json.loads(unified_meta_path.read_text())
                            logger.debug(
                                "Using unified metadata: %s", unified_meta_path
                            )

                        # The producer records whether the render has a
                        # material connection to disclose. Read it rather than
                        # deriving one here, and default to disclosing: a
                        # metadata file written before the key existed carries
                        # no opinion, and a missing disclosure is the costly
                        # direction to be wrong in.
                        carries_affiliate: dict[str, bool] = {}

                        for p in platforms:
                            meta = None

                            # Use unified metadata if available
                            if unified_meta:
                                meta = unified_meta
                            else:
                                # Fallback to platform-specific metadata
                                meta_file = f"metadata_{p.value}.json"
                                platform_meta = video.parent / meta_file
                                if platform_meta.exists():
                                    meta = json.loads(platform_meta.read_text())

                            if meta:
                                carries_affiliate[p.value] = bool(
                                    meta.get("carries_affiliate_content", True)
                                )
                                desc = caption_from_metadata(meta, product_id)
                                if p.value == "youtube" and meta.get("title"):
                                    platform_contents[p.value] = {
                                        "content": desc,
                                        "title": _trim_on_word_boundary(
                                            meta.get("title") or "", 100
                                        ),
                                    }
                                else:
                                    platform_contents[p.value] = {
                                        "content": desc,
                                        "title": _trim_on_word_boundary(
                                            meta.get("title", ""), 100
                                        ),
                                    }
                            else:
                                # Fallback to data.json
                                fallback_path = video.parent / "data.json"
                                if fallback_path.exists():
                                    fb = json.loads(fallback_path.read_text())
                                    if isinstance(fb, list) and fb:
                                        fb = fb[0]
                                    # The raw scraped title, routinely past
                                    # YouTube's 100-character cap. This path
                                    # never builds a PublishMetadata, so the
                                    # clamp the other paths get from
                                    # `clamp_to_limits` has to be applied here.
                                    title = _trim_on_word_boundary(
                                        fb.get("title", "Product Video"), 100
                                    )
                                    desc = fb.get("description", "")
                                    # The title is carried as well as
                                    # concatenated: without it the YouTube
                                    # payload has none, and the platform
                                    # derives one from the caption's first
                                    # line.
                                    platform_contents[p.value] = {
                                        "content": f"{title}\n\n{desc}",
                                        "title": title,
                                    }
                                else:
                                    platform_contents[p.value] = {
                                        "content": f"Product video for {product_id}",
                                        "title": f"Product video for {product_id}",
                                    }

                        # Inject first comments into platform_contents
                        fc_config = getattr(publisher, "first_comment_config", None)
                        if fc_config and fc_config.enabled and outputs_dir:
                            for p in platforms:
                                comment = build_first_comment(
                                    fc_config,
                                    p.value,
                                    product_id,
                                    outputs_dir,
                                )
                                if comment:
                                    platform_contents.setdefault(p.value, {})[
                                        "first_comment"
                                    ] = comment

                        # Per-platform (platform, post_id) legs to record in
                        # local tracking/registry before cleanup removes the dir.
                        scheduled_legs: list[tuple[str, str]] = []

                        if self.config.use_platform_specific_content:
                            # Platform-specific mode: Create separate posts per platform
                            # with optimized metadata for each platform
                            for platform_dict in platform_dicts:
                                p_name = platform_dict["platform"]
                                p_content_data = platform_contents.get(p_name, {})
                                p_content = p_content_data.get("content", "")

                                result = await publisher.publish(
                                    media_id=media_id,
                                    platforms=[platform_dict],
                                    content=p_content,
                                    platform_contents={p_name: p_content_data},
                                    scheduled_time=next_time,
                                    carries_affiliate_content=(
                                        carries_affiliate.get(p_name, True)
                                    ),
                                )

                                # Create separate entry for this platform
                                platform_entry = ScheduleEntry(
                                    product_id=product_id,
                                    scheduled_time=next_time,
                                    platforms=[Platform(p_name)],
                                    post_id=str(result.get("post_id"))
                                    if result.get("post_id")
                                    else None,
                                    status="scheduled",
                                    created_at=datetime.now(UTC),
                                    slot_index=next_idx,
                                )

                                self.entries.append(platform_entry)
                                if platform_entry.post_id:
                                    scheduled_legs.append(
                                        (p_name, platform_entry.post_id)
                                    )
                                logger.info(
                                    "Scheduled %s on %s (post: %s)",
                                    product_id,
                                    p_name,
                                    platform_entry.post_id,
                                )

                            # Mark slot as occupied
                            occupied_slot_times.add(
                                next_time.replace(second=0, microsecond=0)
                            )
                        else:
                            # Unified mode (default): Create single post for all
                            # platforms with shared metadata
                            unified_content = ""
                            unified_platform_contents = {}

                            # Use first available platform's content as unified
                            if platform_contents:
                                first_platform = next(iter(platform_contents))
                                unified_content = platform_contents[first_platform].get(
                                    "content", ""
                                )
                                # Copy same content for all platforms
                                for p_dict in platform_dicts:
                                    p_name = p_dict["platform"]
                                    unified_platform_contents[p_name] = {
                                        "content": unified_content
                                    }
                                    # YouTube title if available
                                    if (
                                        p_name == "youtube"
                                        and "title" in platform_contents.get(p_name, {})
                                    ):
                                        unified_platform_contents[p_name]["title"] = (
                                            platform_contents[p_name]["title"]
                                        )

                            result = await publisher.publish(
                                media_id=media_id,
                                platforms=platform_dicts,  # All platforms in one post
                                content=unified_content,
                                platform_contents=unified_platform_contents,
                                scheduled_time=next_time,
                                # One post covers every platform, so it
                                # discloses if any leg has something to
                                # disclose.
                                carries_affiliate_content=(
                                    any(carries_affiliate.values())
                                    if carries_affiliate
                                    else True
                                ),
                            )

                            # Create single entry with all platforms
                            unified_entry = ScheduleEntry(
                                product_id=product_id,
                                scheduled_time=next_time,
                                platforms=[
                                    Platform(p["platform"]) for p in platform_dicts
                                ],
                                post_id=str(result.get("post_id"))
                                if result.get("post_id")
                                else None,
                                status="scheduled",
                                created_at=datetime.now(UTC),
                                slot_index=next_idx,
                            )

                            self.entries.append(unified_entry)
                            if unified_entry.post_id:
                                for p_dict in platform_dicts:
                                    scheduled_legs.append(
                                        (p_dict["platform"], unified_entry.post_id)
                                    )
                            platform_names = ", ".join(
                                p["platform"] for p in platform_dicts
                            )
                            logger.info(
                                "Scheduled %s on %s (post: %s)",
                                product_id,
                                platform_names,
                                unified_entry.post_id,
                            )

                            # Mark slot as occupied
                            occupied_slot_times.add(
                                next_time.replace(second=0, microsecond=0)
                            )

                        self._save_schedule()
                        scheduled_count += 1

                        # Record local tracking + registry BEFORE cleanup removes
                        # the dir (add_to_registry reads data.json). Mirrors the
                        # single publish path so scheduled posts keep a local
                        # record and the duplicate-publish guard sees them.
                        if outputs_dir and not dry_run:
                            for leg_platform, leg_post_id in scheduled_legs:
                                try:
                                    record_publish(
                                        product_id,
                                        leg_platform,
                                        leg_post_id,
                                        outputs_dir,
                                    )
                                except OSError as track_error:
                                    logger.error(
                                        "Failed to record publish %s:%s: %s",
                                        product_id,
                                        leg_platform,
                                        track_error,
                                    )
                            try:
                                add_to_registry(product_id, outputs_dir)
                            except (OSError, ValueError) as reg_error:
                                logger.warning(
                                    "Failed to update registry for %s: %s",
                                    product_id,
                                    reg_error,
                                )

                            # Link-in-bio before cleanup (reads data.json;
                            # non-blocking, enabled by default)
                            await update_link_in_bio_safe(
                                product_id, outputs_dir, link_in_bio_config
                            )

                        # Cleanup product directory if enabled
                        if cleanup_manager and not dry_run:
                            try:
                                cleanup_result = await cleanup_manager.cleanup(
                                    product_id, platforms, dry_run=False
                                )
                                if cleanup_result.get("success"):
                                    cleaned_count += 1
                                    logger.info(
                                        "Cleaned up %s: %s",
                                        product_id,
                                        cleanup_result.get("message", "success"),
                                    )
                                else:
                                    logger.warning(
                                        "Cleanup skipped for %s: %s",
                                        product_id,
                                        cleanup_result.get("message", "unknown"),
                                    )
                            except (OSError, ValueError) as cleanup_error:
                                logger.warning(
                                    "Cleanup failed for %s: %s",
                                    product_id,
                                    cleanup_error,
                                )

                    except (PublishError, OSError, TimeoutError) as e:
                        logger.error("Failed to schedule %s: %s", product_id, e)
                        # Create failed entry for tracking
                        failed_entry = ScheduleEntry(
                            product_id=product_id,
                            scheduled_time=next_time,
                            platforms=platforms.copy(),
                            post_id=None,
                            status="failed",
                            created_at=datetime.now(UTC),
                            slot_index=next_idx,
                        )
                        self.entries.append(failed_entry)
                        self._save_schedule()
                        failed_count += 1
                        continue

                # Move to next slot for next video
                current_slot = (next_idx + 1) % len(self.config.slots)
                # Update current_time to after this scheduled post
                current_time = next_time

            except Exception as e:  # Per-video boundary
                logger.error("Unexpected error processing %s: %s", video, e)
                failed_count += 1
                continue

        # Log summary
        summary_parts = [
            f"scheduled={scheduled_count}",
            f"skipped={skipped_count}",
            f"failed={failed_count}",
            f"cleaned={cleaned_count}",
        ]
        if conflicts_resolved_count > 0:
            summary_parts.append(f"conflicts_resolved={conflicts_resolved_count}")
        logger.info("Auto-schedule complete: %s", ", ".join(summary_parts))

        return {
            "scheduled": scheduled_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "cleaned": cleaned_count,
            "conflicts_resolved": conflicts_resolved_count,
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
                "Validation failed for %s: %s", entry.product_id, error_message
            )
            raise ValueError(f"Entry validation failed: {error_message}")

        # Add to entries list
        self.entries.append(entry)
        logger.debug(
            "Adding entry for %s scheduled at %s",
            entry.product_id,
            entry.scheduled_time,
        )

        # Atomic write to disk
        try:
            self._save_schedule()
            logger.info(
                "Successfully added entry for %s (total entries: %d)",
                entry.product_id,
                len(self.entries),
            )
        except OSError as e:
            # Roll back on write failure
            self.entries.pop()
            logger.error("Failed to save schedule after adding entry: %s", e)
            raise OSError(f"Failed to save schedule: {e}") from e

    def remove_entries(
        self,
        product_id: str,
        platform: Platform | str | None = None,
    ) -> int:
        """Remove schedule entries for a product.

        Removes all entries matching the product_id, optionally filtered by platform.
        Saves changes atomically after removal.

        Args:
        ----
            product_id: Product ID to match
            platform: Optional platform filter (removes only entries for this platform)

        Returns:
        -------
            Number of entries removed

        Example:
        -------
            >>> # Remove all entries for a product
            >>> count = manager.remove_entries("B0TEST001")
            >>> # Remove only YouTube entries
            >>> count = manager.remove_entries("B0TEST001", platform="youtube")

        """
        # Convert string to Platform enum
        if platform is not None and isinstance(platform, str):
            try:
                platform = Platform(platform.lower())
            except ValueError:
                logger.warning("Invalid platform '%s'", platform)
                return 0

        original_count = len(self.entries)

        # Filter out matching entries
        if platform is not None:
            self.entries = [
                e
                for e in self.entries
                if not (e.product_id == product_id and platform in e.platforms)
            ]
        else:
            self.entries = [e for e in self.entries if e.product_id != product_id]

        removed_count = original_count - len(self.entries)

        if removed_count > 0:
            self._save_schedule()
            platform_suffix = f" on {platform.value}" if platform else ""
            logger.info(
                "Removed %d entries for %s%s",
                removed_count,
                product_id,
                platform_suffix,
            )

        return removed_count

    def find_duplicates(self) -> list[tuple[ScheduleEntry, ScheduleEntry]]:
        """Find duplicate entries in the schedule.

        A duplicate is defined as two entries with:
        - Same product_id
        - Overlapping platforms
        - Same scheduled_time

        Returns
        -------
            List of tuples containing duplicate entry pairs

        """
        duplicates: list[tuple[ScheduleEntry, ScheduleEntry]] = []

        for i, entry in enumerate(self.entries):
            for other in self.entries[i + 1 :]:
                # Check product_id match
                if entry.product_id != other.product_id:
                    continue

                # Check scheduled_time match
                if entry.scheduled_time != other.scheduled_time:
                    continue

                # Check for overlapping platforms
                if set(entry.platforms) & set(other.platforms):
                    duplicates.append((entry, other))

        return duplicates

    def remove_duplicates(self, keep: str = "first") -> int:
        """Remove duplicate entries from the schedule.

        Args:
        ----
            keep: Which duplicate to keep - "first" or "last"

        Returns:
        -------
            Number of duplicate entries removed

        """
        duplicates = self.find_duplicates()

        if not duplicates:
            logger.info("No duplicates found")
            return 0

        # Collect entries to remove
        to_remove: set[int] = set()
        for entry, other in duplicates:
            # Find indices
            try:
                entry_idx = self.entries.index(entry)
                other_idx = self.entries.index(other)
            except ValueError:
                continue

            # Mark for removal based on keep strategy
            if keep == "first":
                to_remove.add(other_idx)
            else:
                to_remove.add(entry_idx)

        # Remove entries (in reverse order to preserve indices)
        for idx in sorted(to_remove, reverse=True):
            removed = self.entries.pop(idx)
            logger.debug("Removed duplicate: %s", removed.product_id)

        if to_remove:
            self._save_schedule()
            logger.info("Removed %d duplicate entries", len(to_remove))

        return len(to_remove)
