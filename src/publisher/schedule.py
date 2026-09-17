"""Schedule management for video publishing.

Provides calendar view, recurring schedule slots, and batch scheduling
capabilities for the publisher module.
"""

import json
import logging
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

from src.publisher.base import PublishError
from src.publisher.constants import (
    DEFAULT_OUTPUTS_DIR,
    SCHEDULE_ALTERNATIVE_SEARCH_MULTIPLIER,
    SCHEDULE_MAX_SLOT_SEARCH_ATTEMPTS,
)
from src.publisher.first_comment import build_first_comment
from src.publisher.link_in_bio.manager import update_link_in_bio_safe
from src.publisher.models import (
    CleanupConfig,
    ConflictResolution,
    LinkInBioConfig,
    Platform,
    PublishMetadata,
    RecurringSlot,
    ScheduleConfig,
    ScheduleEntry,
    _trim_on_word_boundary,
    strip_disclosure_tokens,
)
from src.publisher.product_registry import add_to_registry
from src.publisher.schedule_validator import ScheduleValidator
from src.publisher.tracking import is_already_published, record_publish
from src.scraper.base.models import carries_affiliate_content
from src.utils.outputs_paths import durable_state_path

if TYPE_CHECKING:
    from src.publisher.base import BasePublisher

logger = logging.getLogger(__name__)


def metadata_from_file(
    meta: dict,
    product_id: str | None,
    platform: Platform,
) -> PublishMetadata:
    """Build the `PublishMetadata` that `schedule auto` publishes from.

    Returns the metadata object rather than a caption string, so each
    branch clamps at its own point of use with its own target list in
    scope -- the caption-for-one-platform-reaching-another defect took four
    review rounds across this function's paths before the collapse (#408).
    Nothing here clamps; use `clamped_for(targets).format_content()`.

    Routed through `PublishMetadata` rather than assembling parts by hand:
    hand-assembly is what left this path without the leading disclosure
    line every other publish path gets.

    A record `PublishMetadata` refuses -- an empty description, a YouTube
    entry with no title -- is repaired rather than bypassed, because a
    bypass that assembles its own caption is how this function once
    returned above the clamp. Losing the scheduling run to one malformed
    file would be worse than a generic title.
    """
    hashtags = list(meta.get("hashtags", []))
    discloses = bool(meta.get("carries_affiliate_content", True))

    # The caption prompts end their worked examples with `#ad`, so the model
    # writes it into the body whatever this render carries. Removed either
    # way: on a disclosing render the leading line is the disclosure and a
    # second copy below the fold is noise, and on a topic render it is the
    # false statement #295 was about. `single` gets the same outcome from the
    # loader's trailing-hashtag rule, which this path never had.
    description, hashtags = strip_disclosure_tokens(
        str(meta.get("description", "") or ""), hashtags
    )

    try:
        return PublishMetadata(
            platform=platform,
            title=str(meta.get("title") or ""),
            description=description,
            hashtags=hashtags,
            keywords=list(meta.get("keywords", [])),
            product_id=product_id,
            carries_affiliate_content=discloses,
        )
    except ValueError as e:
        logger.warning(
            "Repairing metadata for %s on %s (%s)",
            product_id,
            platform.value,
            e,
        )
        fallback = f"Product video for {product_id}" if product_id else "Product video"
        tags = list(hashtags)
        if product_id and product_id not in tags:
            tags.append(product_id)
        return PublishMetadata(
            platform=platform,
            title=str(meta.get("title") or "") or fallback,
            description=description or fallback,
            hashtags=tags,
            keywords=list(meta.get("keywords", [])),
            product_id=product_id,
            carries_affiliate_content=discloses,
        )


def record_scheduled_posts(
    product_id: str,
    publish_results: list[dict],
    platforms: list[dict],
    schedule_time: datetime,
    slot_index: int | None,
    schedule_mgr: "ScheduleManager",
) -> int:
    """Write each scheduled post to the local schedule, as `auto_schedule` does.

    Shared by the `single` command and the global batch: both used to
    publish a scheduled post and record only `publish_history.json`, so
    `calendar`, which reads `schedule.json` and nothing else, never saw it.
    One entry per publish result: the unified mode's single post carries
    every platform, the platform-specific mode's posts carry one each.
    Returns the number recorded; a write failure is logged and does not fail
    the publish, since the post already exists on the provider.
    """
    recorded = 0
    for pub_result in publish_results:
        result_data = pub_result["result"]
        post_id = result_data.get("post_id")
        entry_platforms = (
            [Platform(p["platform"]) for p in platforms]
            if pub_result["platform"] == "all"
            else [Platform(pub_result["platform"])]
        )
        entry = ScheduleEntry(
            product_id=product_id,
            scheduled_time=schedule_time,
            platforms=entry_platforms,
            post_id=str(post_id) if post_id else None,
            status="scheduled",
            created_at=datetime.now(UTC),
            slot_index=slot_index,
        )
        try:
            schedule_mgr.record_entry(entry)
            recorded += 1
        except OSError as e:
            logger.error("Failed to record %s in the local schedule: %s", product_id, e)
    return recorded


class _VideoOutcome(NamedTuple):
    """What happened to one video, for the caller to tally.

    A slot is reported only for a scheduled video, because only a scheduled
    one moves the cursor: a skip or a failure leaves the slot for the next.
    """

    result: str  # "scheduled" | "skipped" | "failed"
    cleaned: bool = False
    conflict_resolved: bool = False
    # (time, slot index) for a scheduled video, and nothing otherwise, so the
    # cursor advances exactly when there is a slot to advance to.
    slot: tuple[datetime, int] | None = None


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
        schedule_path: Path | str | None = None,
        config: ScheduleConfig | None = None,
    ):
        """Initialize schedule manager.

        Args:
        ----
            schedule_path: Path to schedule.json file. None resolves the
                durable default under outputs/state/ (migrating a legacy
                root copy).
            config: Schedule configuration (uses defaults if None)

        """
        if schedule_path is None:
            self.schedule_path = durable_state_path(
                DEFAULT_OUTPUTS_DIR, "schedule.json"
            )
        else:
            self.schedule_path = Path(schedule_path)
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

    async def build_occupancy(
        self, publisher: "BasePublisher", current_time: datetime
    ) -> set[datetime]:
        """Slot times already taken, from the API and the local schedule.

        Local entries are NOT created from API posts: the API does not return
        the product id, and entries with placeholder ids broke duplicate
        detection. Only the times are tracked.
        """
        occupied_slot_times: set[datetime] = set()
        try:
            logger.debug("Fetching existing posts from API (all statuses)...")
            # Scheduled and published alike, or a published post's slot is
            # offered again.
            api_posts = await publisher.list_posts()
            logger.debug("Found %d posts on API", len(api_posts))

            for api_post in api_posts:
                scheduled_time = api_post.get("scheduledFor")
                if not scheduled_time:
                    continue

                if isinstance(scheduled_time, str):
                    time_str = scheduled_time.replace("+00:00", "")
                    scheduled_dt = datetime.fromisoformat(time_str)
                else:
                    scheduled_dt = scheduled_time

                if scheduled_dt.tzinfo is None:
                    scheduled_dt = scheduled_dt.replace(tzinfo=UTC)

                occupied_slot_times.add(scheduled_dt.replace(second=0, microsecond=0))

            logger.info("Found %d occupied slots from API", len(occupied_slot_times))

            for entry in self.entries:
                occupied_slot_times.add(
                    entry.scheduled_time.replace(second=0, microsecond=0)
                )

            logger.info(
                "Total %d occupied slots (API + local)", len(occupied_slot_times)
            )

            # The search still starts at now, so gaps before the latest post
            # are filled; the latest is logged only to explain the skipping.
            if occupied_slot_times:
                logger.info("Latest post on API: %s", max(occupied_slot_times))
                logger.info(
                    "Searching for next available slot from now (%s), "
                    "skipping %d occupied slots",
                    current_time,
                    len(occupied_slot_times),
                )

        except (PublishError, OSError, TimeoutError) as e:
            logger.warning("Failed to check API schedule: %s", e)

        return occupied_slot_times

    def next_free_slot(
        self,
        product_id: str,
        current_time: datetime,
        current_slot: int,
        occupied_slot_times: set[datetime],
    ) -> tuple[datetime, int]:
        """The next slot no post already holds.

        Raises ValueError once the search budget is spent, which the caller
        counts as a failed product rather than aborting the run.
        """
        search_time = current_time
        attempts = 0

        while attempts < SCHEDULE_MAX_SLOT_SEARCH_ATTEMPTS:
            next_time, next_idx = self.get_next_slot(
                slots=self.config.slots,
                after=search_time,
                slot_index=current_slot,
            )

            if next_time.replace(second=0, microsecond=0) not in occupied_slot_times:
                logger.debug(
                    "Next slot for %s: %s (slot %d)", product_id, next_time, next_idx
                )
                return next_time, next_idx

            logger.debug("Slot %s occupied by API post, trying next slot", next_time)
            search_time = next_time
            attempts += 1

        raise ValueError(
            f"No available slot after {SCHEDULE_MAX_SLOT_SEARCH_ATTEMPTS} attempts"
        )

    def _settle_conflict(
        self,
        product_id: str,
        next_time: datetime,
        next_idx: int,
        platforms: list[Platform],
        occupied_slot_times: set[datetime],
        auto_resolve: bool,
    ) -> tuple[datetime, bool] | None:
        """Validate the slot, resolving a conflict when asked to.

        Returns the time to use and whether a conflict was resolved, or None
        when the product cannot be placed -- which the caller counts as
        failed. Without `--auto-resolve` a conflict is reported with
        alternatives and the product is dropped, as before.
        """
        temp_entry = ScheduleEntry(
            product_id=product_id,
            scheduled_time=next_time,
            # One validation covers the product; separate posts per platform
            # are created later.
            platforms=[platforms[0]],
            post_id=None,
            status="pending",
            created_at=datetime.now(UTC),
            slot_index=next_idx,
        )

        validator = ScheduleValidator(self.config, self.entries)
        is_valid, error_message = validator.validate(temp_entry)
        if is_valid:
            return next_time, False

        if auto_resolve:
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
                resolved_time = resolution.resolved_time
                occupied_slot_times.add(resolved_time.replace(second=0, microsecond=0))
                return resolved_time, True

            logger.warning(
                "Could not resolve conflict for %s: %s", product_id, error_message
            )
            if resolution.alternatives:
                logger.info(
                    "Available alternatives: %s",
                    ", ".join(t.isoformat() for t in resolution.alternatives[:3]),
                )
            return None

        logger.warning("Validation failed for %s: %s", product_id, error_message)
        resolution = self.find_alternatives(
            preferred_time=next_time,
            platforms=platforms,
            occupied_slots=occupied_slot_times,
        )
        if resolution.alternatives:
            logger.info(
                "Suggested alternatives for %s: %s",
                product_id,
                ", ".join(t.isoformat() for t in resolution.alternatives[:3]),
            )
            logger.info("Use --auto-resolve to automatically use first")
        return None

    async def _publish_targets(
        self, publisher: "BasePublisher", platforms: list[Platform]
    ) -> list[dict[str, str]]:
        """The platform/account pairs the post can actually go to."""
        accounts = await publisher.get_accounts()
        account_map = {acc["platform"]: acc["account_id"] for acc in accounts}

        platform_dicts = []
        for p in platforms:
            account_id = account_map.get(p.value)
            if not account_id:
                logger.warning("No account for %s", p.value)
                continue
            platform_dicts.append({"platform": p.value, "account_id": account_id})

        if not platform_dicts:
            raise ValueError("No valid accounts for platforms")
        return platform_dicts

    def _captions_for(
        self, video: Path, product_id: str, platforms: list[Platform]
    ) -> tuple[dict[str, PublishMetadata], dict[str, str], dict[str, bool]]:
        """Per-platform metadata, titles and disclosure decisions.

        The metadata is returned unclamped: each posting branch clamps at its
        own point of use, for exactly the platforms that post carries (#408).
        Titles ride separately, because the caption clamp's trimmed title is
        deliberately not what the payload carries.
        """
        platform_metas: dict[str, PublishMetadata] = {}
        titles: dict[str, str] = {}
        # The producer records whether the render has a material connection to
        # disclose. Read it rather than deriving one here, and default to
        # disclosing: a metadata file written before the key existed carries
        # no opinion, and a missing disclosure is the costly direction.
        carries_affiliate: dict[str, bool] = {}

        unified_meta_path = video.parent / "metadata.json"
        unified_meta = None
        if unified_meta_path.exists():
            unified_meta = json.loads(unified_meta_path.read_text())
            logger.debug("Using unified metadata: %s", unified_meta_path)

        for p in platforms:
            meta = unified_meta
            if not meta:
                platform_meta = video.parent / f"metadata_{p.value}.json"
                if platform_meta.exists():
                    meta = json.loads(platform_meta.read_text())

            if meta:
                carries_affiliate[p.value] = bool(
                    meta.get("carries_affiliate_content", True)
                )
                platform_metas[p.value] = metadata_from_file(meta, product_id, p)
                titles[p.value] = _trim_on_word_boundary(meta.get("title") or "", 100)
                continue

            fallback_path = video.parent / "data.json"
            if fallback_path.exists():
                fb = json.loads(fallback_path.read_text())
                if isinstance(fb, list) and fb:
                    fb = fb[0]
                # The raw scraped title, routinely past YouTube's 100-character
                # cap. This branch carries the title separately from the
                # caption the builder returns, so it is trimmed here.
                title = _trim_on_word_boundary(fb.get("title", "Product Video"), 100)
                desc = fb.get("description", "")
                # Routed through the same builder so this branch leads with the
                # disclosure too, and the decision is read off the record:
                # `data.json` carries the two fields the rule uses, so
                # defaulting would stamp `#ad` on a topic whose own record says
                # there is nothing to disclose.
                fb_discloses = carries_affiliate_content(SimpleNamespace(**fb))
                carries_affiliate[p.value] = fb_discloses
                platform_metas[p.value] = metadata_from_file(
                    {
                        "title": title,
                        "description": f"{title}\n\n{desc}",
                        "carries_affiliate_content": fb_discloses,
                    },
                    product_id,
                    p,
                )
                titles[p.value] = title
                continue

            # Nothing on disk at all. Routed through the builder like every
            # other branch, so this caption is recorded and clamped the same
            # way (#408); it used to be a bare literal that skipped both. It
            # neither discloses nor votes in `carries_affiliate`: with no
            # record, asserting a material connection would stamp a false
            # `brand_organic` on a unified topic post whose siblings all say
            # there is nothing to disclose.
            literal = f"Product video for {product_id}"
            platform_metas[p.value] = metadata_from_file(
                {
                    "title": literal,
                    "description": literal,
                    "carries_affiliate_content": False,
                },
                product_id,
                p,
            )
            titles[p.value] = literal

        return platform_metas, titles, carries_affiliate

    def _first_comments_for(
        self,
        publisher: "BasePublisher",
        platforms: list[Platform],
        product_id: str,
        outputs_dir: Path | None,
    ) -> dict[str, str]:
        """First comments, attached by each posting branch.

        The unified branch used to drop them, because its per-platform payload
        copied only content and title.
        """
        first_comments: dict[str, str] = {}
        fc_config = getattr(publisher, "first_comment_config", None)
        if not (fc_config and fc_config.enabled and outputs_dir):
            return first_comments
        for p in platforms:
            comment = build_first_comment(fc_config, p.value, product_id, outputs_dir)
            if comment:
                first_comments[p.value] = comment
        return first_comments

    async def _post_per_platform(
        self,
        *,
        publisher: "BasePublisher",
        platform_dicts: list[dict[str, str]],
        platform_metas: dict[str, PublishMetadata],
        titles: dict[str, str],
        carries_affiliate: dict[str, bool],
        first_comments: dict[str, str],
        media_id: str,
        product_id: str,
        next_time: datetime,
        next_idx: int,
    ) -> list[tuple[str, str]]:
        """One post per platform, each with its own metadata."""
        scheduled_legs: list[tuple[str, str]] = []
        for platform_dict in platform_dicts:
            p_name = platform_dict["platform"]
            p_meta = platform_metas.get(p_name)
            p_content_data: dict[str, Any] = {}
            p_content = ""
            if p_meta is not None:
                # Clamped here, for exactly this post's one destination (#408).
                p_content = p_meta.clamped_for([Platform(p_name)]).format_content()
                p_content_data = {
                    "content": p_content,
                    "title": titles.get(p_name, ""),
                }
                if p_name in first_comments:
                    p_content_data["first_comment"] = first_comments[p_name]

            result = await publisher.publish(
                media_id=media_id,
                platforms=[platform_dict],
                content=p_content,
                platform_contents={p_name: p_content_data},
                scheduled_time=next_time,
                carries_affiliate_content=carries_affiliate.get(p_name, True),
            )

            platform_entry = ScheduleEntry(
                product_id=product_id,
                scheduled_time=next_time,
                platforms=[Platform(p_name)],
                post_id=str(result.get("post_id")) if result.get("post_id") else None,
                status="scheduled",
                created_at=datetime.now(UTC),
                slot_index=next_idx,
            )

            self.entries.append(platform_entry)
            if platform_entry.post_id:
                scheduled_legs.append((p_name, platform_entry.post_id))
            logger.info(
                "Scheduled %s on %s (post: %s)",
                product_id,
                p_name,
                platform_entry.post_id,
            )
        return scheduled_legs

    async def _post_unified(
        self,
        *,
        publisher: "BasePublisher",
        platform_dicts: list[dict[str, str]],
        platform_metas: dict[str, PublishMetadata],
        titles: dict[str, str],
        carries_affiliate: dict[str, bool],
        first_comments: dict[str, str],
        media_id: str,
        product_id: str,
        next_time: datetime,
        next_idx: int,
    ) -> list[tuple[str, str]]:
        """One post carrying one caption to every platform."""
        scheduled_legs: list[tuple[str, str]] = []
        unified_content = ""
        unified_platform_contents: dict[str, dict[str, Any]] = {}

        if platform_metas:
            first_platform = next(iter(platform_metas))
            # One post carries a single caption to every target, so it is
            # clamped for all of them at once -- reusing a caption clamped for
            # one platform is the four-round defect the collapse removed
            # (#403, #408).
            unified_targets = [
                Platform(d["platform"])
                for d in platform_dicts
                if d.get("platform") in {pl.value for pl in Platform}
            ] or [Platform(first_platform)]
            unified_content = (
                platform_metas[first_platform]
                .clamped_for(unified_targets)
                .format_content()
            )
            for p_dict in platform_dicts:
                p_name = p_dict["platform"]
                payload: dict[str, Any] = {"content": unified_content}
                if p_name == "youtube" and titles.get(p_name):
                    payload["title"] = titles[p_name]
                if p_name in first_comments:
                    payload["first_comment"] = first_comments[p_name]
                unified_platform_contents[p_name] = payload

        result = await publisher.publish(
            media_id=media_id,
            platforms=platform_dicts,  # All platforms in one post
            content=unified_content,
            platform_contents=unified_platform_contents,
            scheduled_time=next_time,
            # One post covers every platform, so it discloses if any leg has
            # something to disclose.
            carries_affiliate_content=(
                any(carries_affiliate.values()) if carries_affiliate else True
            ),
        )

        unified_entry = ScheduleEntry(
            product_id=product_id,
            scheduled_time=next_time,
            platforms=[Platform(p["platform"]) for p in platform_dicts],
            post_id=str(result.get("post_id")) if result.get("post_id") else None,
            status="scheduled",
            created_at=datetime.now(UTC),
            slot_index=next_idx,
        )

        self.entries.append(unified_entry)
        if unified_entry.post_id:
            for p_dict in platform_dicts:
                scheduled_legs.append((p_dict["platform"], unified_entry.post_id))
        logger.info(
            "Scheduled %s on %s (post: %s)",
            product_id,
            ", ".join(p["platform"] for p in platform_dicts),
            unified_entry.post_id,
        )
        return scheduled_legs

    async def _record_scheduled_legs(
        self,
        product_id: str,
        scheduled_legs: list[tuple[str, str]],
        outputs_dir: Path,
        link_in_bio_config: LinkInBioConfig | None,
    ) -> None:
        """Local tracking, registry and the bio link, before cleanup.

        `add_to_registry` and the bio link both read `data.json`, so this runs
        before the directory is removed. Mirrors the single publish path, so a
        scheduled post keeps a local record and the duplicate-publish guard
        sees it.
        """
        for leg_platform, leg_post_id in scheduled_legs:
            try:
                record_publish(product_id, leg_platform, leg_post_id, outputs_dir)
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
                "Failed to update registry for %s: %s", product_id, reg_error
            )

        await update_link_in_bio_safe(product_id, outputs_dir, link_in_bio_config)

    async def _cleanup_scheduled(
        self,
        cleanup_manager: Any,
        product_id: str,
        platforms: list[Platform],
    ) -> bool:
        """Remove the product directory; True when it was actually removed."""
        try:
            cleanup_result = await cleanup_manager.cleanup(
                product_id, platforms, dry_run=False
            )
            if cleanup_result.get("success"):
                logger.info(
                    "Cleaned up %s: %s",
                    product_id,
                    cleanup_result.get("message", "success"),
                )
                return True
            logger.warning(
                "Cleanup skipped for %s: %s",
                product_id,
                cleanup_result.get("message", "unknown"),
            )
        except (OSError, ValueError) as cleanup_error:
            logger.warning("Cleanup failed for %s: %s", product_id, cleanup_error)
        return False

    async def _schedule_one(
        self,
        *,
        video: Path,
        product_id: str,
        platforms: list[Platform],
        publisher: "BasePublisher",
        next_time: datetime,
        next_idx: int,
        occupied_slot_times: set[datetime],
        outputs_dir: Path | None,
        cleanup_manager: Any,
        link_in_bio_config: LinkInBioConfig | None,
    ) -> bool | None:
        """Publish one product into its settled slot.

        Returns whether the product directory was cleaned, or None when the
        product failed -- in which case a failed entry is already recorded.
        """
        try:
            logger.info(
                "Scheduling %s at %s (slot %d)", product_id, next_time, next_idx
            )

            platform_dicts = await self._publish_targets(publisher, platforms)
            # The upload comes before the captions: a media failure should not
            # be paid for after reading every metadata file.
            media_id = await publisher.upload_media(video)

            platform_metas, titles, carries_affiliate = self._captions_for(
                video, product_id, platforms
            )
            first_comments = self._first_comments_for(
                publisher, platforms, product_id, outputs_dir
            )

            post = (
                self._post_per_platform
                if self.config.use_platform_specific_content
                else self._post_unified
            )
            scheduled_legs = await post(
                publisher=publisher,
                platform_dicts=platform_dicts,
                platform_metas=platform_metas,
                titles=titles,
                carries_affiliate=carries_affiliate,
                first_comments=first_comments,
                media_id=media_id,
                product_id=product_id,
                next_time=next_time,
                next_idx=next_idx,
            )
            occupied_slot_times.add(next_time.replace(second=0, microsecond=0))

            self._save_schedule()

            if outputs_dir:
                await self._record_scheduled_legs(
                    product_id, scheduled_legs, outputs_dir, link_in_bio_config
                )

            if cleanup_manager:
                return await self._cleanup_scheduled(
                    cleanup_manager, product_id, platforms
                )
            return False

        except (PublishError, OSError, TimeoutError) as e:
            logger.error("Failed to schedule %s: %s", product_id, e)
            self.entries.append(
                ScheduleEntry(
                    product_id=product_id,
                    scheduled_time=next_time,
                    platforms=platforms.copy(),
                    post_id=None,
                    status="failed",
                    created_at=datetime.now(UTC),
                    slot_index=next_idx,
                )
            )
            self._save_schedule()
            return None

    async def _schedule_video(
        self,
        *,
        video: Path,
        platforms: list[Platform],
        publisher: "BasePublisher",
        current_time: datetime,
        current_slot: int,
        occupied_slot_times: set[datetime],
        dry_run: bool,
        force: bool,
        auto_resolve: bool,
        outputs_dir: Path | None,
        cleanup_manager: Any,
        link_in_bio_config: LinkInBioConfig | None,
    ) -> "_VideoOutcome":
        """Place one video: skip check, slot, conflict, then publish.

        Every exit says what happened rather than mutating counters, so the
        caller holds the tally and the cursor in one place. Only a scheduled
        video reports a slot, because only a scheduled one moves the cursor.
        """
        # "outputs/B0ABC123/video_B0ABC123.mp4" -> "B0ABC123"
        product_id = video.parent.name
        logger.debug("Processing video: %s", product_id)

        if not force:
            already_published = [
                platform.value
                for platform in platforms
                if is_already_published(product_id, platform.value)
            ]
            if already_published:
                logger.info(
                    "Skipping %s: already published to %s",
                    product_id,
                    ", ".join(already_published),
                )
                return _VideoOutcome("skipped")

        try:
            next_time, next_idx = self.next_free_slot(
                product_id, current_time, current_slot, occupied_slot_times
            )
        except (ValueError, KeyError) as e:
            logger.error("Failed to calculate next slot: %s", e)
            return _VideoOutcome("failed")

        settled = self._settle_conflict(
            product_id,
            next_time,
            next_idx,
            platforms,
            occupied_slot_times,
            auto_resolve,
        )
        if settled is None:
            return _VideoOutcome("failed")
        next_time, was_resolved = settled

        if dry_run:
            logger.info(
                "[DRY RUN] Would schedule %s at %s (slot %d)",
                product_id,
                next_time,
                next_idx,
            )
            return _VideoOutcome(
                "scheduled",
                conflict_resolved=was_resolved,
                slot=(next_time, next_idx),
            )

        try:
            cleaned = await self._schedule_one(
                video=video,
                product_id=product_id,
                platforms=platforms,
                publisher=publisher,
                next_time=next_time,
                next_idx=next_idx,
                occupied_slot_times=occupied_slot_times,
                outputs_dir=outputs_dir,
                cleanup_manager=cleanup_manager,
                link_in_bio_config=link_in_bio_config,
            )
        except Exception as e:
            # `_schedule_one` catches the three publish failures; anything
            # else -- a truncated metadata file reaching `json.loads`, an SDK
            # error from `get_accounts` -- would otherwise reach the loop's
            # own boundary, which cannot see that a conflict was resolved.
            # Same message as that boundary, so the log is unchanged.
            logger.error("Unexpected error processing %s: %s", video, e)
            return _VideoOutcome("failed", conflict_resolved=was_resolved)

        if cleaned is None:
            # The conflict was still resolved, whatever became of the post:
            # the base counted it at resolution time, and this is a refactor.
            return _VideoOutcome("failed", conflict_resolved=was_resolved)
        return _VideoOutcome(
            "scheduled",
            cleaned=cleaned,
            conflict_resolved=was_resolved,
            slot=(next_time, next_idx),
        )

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

        current_time = datetime.now(UTC)
        occupied_slot_times = await self.build_occupancy(publisher, current_time)

        scheduled_count = 0
        skipped_count = 0
        failed_count = 0
        cleaned_count = 0
        conflicts_resolved_count = 0

        cleanup_manager = None
        if cleanup_config is None:
            cleanup_config = CleanupConfig()  # Default: enabled=True
        if cleanup_config.enabled and outputs_dir:
            from src.publisher.cleanup import CleanupManager

            cleanup_manager = CleanupManager(outputs_dir, cleanup_config, publisher)
            logger.info("Cleanup enabled - will cleanup after successful scheduling")

        current_slot = start_slot  # Wraps around

        for video in videos:
            try:
                outcome = await self._schedule_video(
                    video=video,
                    platforms=platforms,
                    publisher=publisher,
                    current_time=current_time,
                    current_slot=current_slot,
                    occupied_slot_times=occupied_slot_times,
                    dry_run=dry_run,
                    force=force,
                    auto_resolve=auto_resolve,
                    outputs_dir=outputs_dir,
                    cleanup_manager=cleanup_manager,
                    link_in_bio_config=link_in_bio_config,
                )
            except Exception as e:  # Per-video boundary
                logger.error("Unexpected error processing %s: %s", video, e)
                failed_count += 1
                continue

            # Counted before the outcome is read: a conflict resolved into a
            # post that then failed to publish was still resolved, which is
            # what the number meant before the split.
            conflicts_resolved_count += int(outcome.conflict_resolved)

            if outcome.result == "skipped":
                skipped_count += 1
                continue
            if outcome.result == "failed":
                failed_count += 1
                continue

            scheduled_count += 1
            cleaned_count += int(outcome.cleaned)
            if outcome.slot is not None:
                # Only a scheduled product moves the cursor: a skip or a
                # failure leaves the slot for the next video, as before.
                scheduled_time, slot_index = outcome.slot
                current_slot = (slot_index + 1) % len(self.config.slots)
                current_time = scheduled_time

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

    def record_entry(self, entry: ScheduleEntry) -> None:
        """Record a post that already exists on the provider, without validation.

        `add_entry` runs the pre-flight validator (spacing, duplicates, daily
        limit) and refuses what it would not have scheduled. That is the
        wrong check for a post the provider has already accepted: refusing
        to record it leaves local state blind to a real post, which is how
        `calendar` came to report nothing while the provider held a week.
        This is the path `auto_schedule` uses for its own entries.

        Raises
        ------
            OSError: If the schedule file write fails; the entry is rolled back.

        """
        self.entries.append(entry)
        try:
            self._save_schedule()
        except OSError as e:
            self.entries.pop()
            logger.error("Failed to save schedule after recording entry: %s", e)
            raise OSError(f"Failed to save schedule: {e}") from e
        logger.info(
            "Recorded %s scheduled for %s (total entries: %d)",
            entry.product_id,
            entry.scheduled_time.isoformat(),
            len(self.entries),
        )

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
