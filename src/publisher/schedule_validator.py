"""Schedule validation for publisher scheduling system.

This module provides comprehensive validation for schedule entries to prevent
conflicts, enforce spacing rules, and maintain schedule integrity.
"""

import logging
from datetime import UTC, datetime, timedelta

from src.publisher.models import Platform, ScheduleConfig, ScheduleEntry

logger = logging.getLogger(__name__)


class ScheduleValidator:
    """Validates schedule entries against configuration rules.

    Enforces scheduling constraints including:
    - Timezone-aware datetime validation
    - Duplicate detection (product_id + platform + time)
    - Minimum post spacing on same platform
    - Maximum daily post limits
    - Past schedule restrictions

    Attributes:
    ----------
        config: Schedule configuration with validation rules
        existing_entries: List of existing schedule entries to check against

    Example:
    -------
        >>> validator = ScheduleValidator(config, existing_entries)
        >>> is_valid, message = validator.validate(new_entry)
        >>> if not is_valid:
        ...     logger.error(f"Validation failed: {message}")

    """

    def __init__(
        self,
        config: ScheduleConfig,
        existing_entries: list[ScheduleEntry],
    ):
        """Initialize validator with configuration and existing entries.

        Args:
        ----
            config: Schedule configuration with validation rules
            existing_entries: List of existing schedule entries to validate against

        """
        self.config = config
        self.existing_entries = existing_entries
        logger.debug(
            f"ScheduleValidator initialized with "
            f"{len(existing_entries)} existing entries"
        )

    def validate(self, entry: ScheduleEntry) -> tuple[bool, str]:
        """Validate a schedule entry against all rules.

        Main validation entry point that checks:
        1. Timezone awareness (scheduled_time has tzinfo)
        2. Past schedule restrictions (if allow_past_schedules=False)
        3. Duplicate detection (if prevent_duplicates=True)
        4. Minimum post spacing on same platform
        5. Maximum daily post limit

        Args:
        ----
            entry: Schedule entry to validate

        Returns:
        -------
            Tuple of (is_valid, error_message)
            - is_valid: True if entry passes all validations
            - error_message: Descriptive error with suggestions if invalid,
              empty string if valid

        Example:
        -------
            >>> entry = ScheduleEntry(...)
            >>> is_valid, message = validator.validate(entry)
            >>> if not is_valid:
            ...     print(f"Validation failed: {message}")

        """
        # 1. Validate timezone awareness
        if entry.scheduled_time.tzinfo is None:
            return (
                False,
                "scheduled_time must be timezone-aware. "
                "Use datetime.now(UTC) or datetime(..., tzinfo=UTC)",
            )

        # 2. Check past schedules if not allowed
        if not self.config.allow_past_schedules:
            now = datetime.now(UTC)
            if entry.scheduled_time < now:
                return (
                    False,
                    f"Cannot schedule in the past "
                    f"({entry.scheduled_time} is before {now}). "
                    "Set allow_past_schedules=true in config to override.",
                )

        # 3. Check duplicates if enabled
        if self.config.prevent_duplicates and self._is_duplicate(entry):
            platforms_str = [p.value for p in entry.platforms]
            return (
                False,
                f"Duplicate entry detected: {entry.product_id} already scheduled "
                f"at {entry.scheduled_time} for platforms {platforms_str}. "
                "Use different time or set prevent_duplicates=false.",
            )

        # 4. Check minimum post spacing on same platform
        spacing_valid, spacing_message = self._check_spacing(entry)
        if not spacing_valid:
            return False, spacing_message

        # 5. Check daily post limit
        limit_valid, limit_message = self._check_daily_limit(entry)
        if not limit_valid:
            return False, limit_message

        logger.debug(f"Entry validation passed for {entry.product_id}")
        return True, ""

    def _is_duplicate(self, entry: ScheduleEntry) -> bool:
        """Check if entry is a duplicate of an existing entry.

        An entry is considered a duplicate if an existing entry has:
        - Same product_id
        - Overlapping platforms (at least one platform in common)
        - Same scheduled_time (exact match)

        Args:
        ----
            entry: Schedule entry to check

        Returns:
        -------
            True if duplicate detected, False otherwise

        """
        for existing in self.existing_entries:
            # Check product_id match
            if existing.product_id != entry.product_id:
                continue

            # Check scheduled_time match (timezone-aware comparison)
            if existing.scheduled_time != entry.scheduled_time:
                continue

            # Check for overlapping platforms
            existing_platforms = set(existing.platforms)
            entry_platforms = set(entry.platforms)
            if existing_platforms & entry_platforms:
                logger.debug(
                    f"Duplicate detected: {entry.product_id} at {entry.scheduled_time}"
                )
                return True

        return False

    def _check_spacing(self, entry: ScheduleEntry) -> tuple[bool, str]:
        """Check minimum post spacing on same platform.

        Enforces min_post_spacing_hours between posts on the same platform.
        Spacing is checked independently per platform.

        Args:
        ----
            entry: Schedule entry to check

        Returns:
        -------
            Tuple of (is_valid, error_message)
            - is_valid: True if spacing requirements met
            - error_message: Descriptive error with conflicting entry details

        """
        if self.config.min_post_spacing_hours == 0:
            # No spacing requirement
            return True, ""

        spacing_delta = timedelta(hours=self.config.min_post_spacing_hours)

        # Check each platform in the new entry
        for platform in entry.platforms:
            for existing in self.existing_entries:
                # Only check entries on the same platform
                if platform not in existing.platforms:
                    continue

                # Calculate time difference (absolute value)
                time_diff = abs(entry.scheduled_time - existing.scheduled_time)

                # Check if too close
                if time_diff < spacing_delta:
                    hours_diff = time_diff.total_seconds() / 3600
                    min_hours = self.config.min_post_spacing_hours
                    return (
                        False,
                        f"Post spacing violation on {platform.value}: "
                        f"{entry.product_id} scheduled too close to "
                        f"{existing.product_id} "
                        f"({hours_diff:.1f}h < {min_hours}h). "
                        f"Existing post at {existing.scheduled_time}, "
                        f"schedule at least {min_hours}h apart.",
                    )

        return True, ""

    def _check_daily_limit(self, entry: ScheduleEntry) -> tuple[bool, str]:
        """Check maximum daily post limit.

        Enforces max_posts_per_day by counting posts on the same date
        (date component only, ignoring time). A value of 0 means unlimited.

        Args:
        ----
            entry: Schedule entry to check

        Returns:
        -------
            Tuple of (is_valid, error_message)
            - is_valid: True if daily limit not exceeded
            - error_message: Descriptive error with current count

        """
        if self.config.max_posts_per_day == 0:
            # No daily limit
            return True, ""

        # Get date component of scheduled_time (ignore time)
        entry_date = entry.scheduled_time.date()

        # Count existing entries on the same date
        same_day_count = sum(
            1
            for existing in self.existing_entries
            if existing.scheduled_time.date() == entry_date
        )

        # Check if adding this entry would exceed limit
        if same_day_count >= self.config.max_posts_per_day:
            return (
                False,
                f"Daily post limit exceeded: {same_day_count} posts already scheduled "
                f"on {entry_date} (limit: {self.config.max_posts_per_day}). "
                f"Schedule on a different date or increase max_posts_per_day.",
            )

        return True, ""
