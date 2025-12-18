"""Unit tests for schedule validator."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.publisher.models import Platform, ScheduleConfig, ScheduleEntry
from src.publisher.schedule_validator import ScheduleValidator


# Fixtures for test data
@pytest.fixture
def base_config():
    """Create a default schedule config for testing."""
    return ScheduleConfig(
        enabled=True,
        min_post_spacing_hours=2,
        prevent_duplicates=True,
        allow_past_schedules=False,
        max_posts_per_day=10,
        timezone="UTC",
    )


@pytest.fixture
def permissive_config():
    """Create a permissive config (no spacing, duplicates allowed, past allowed)."""
    return ScheduleConfig(
        enabled=True,
        min_post_spacing_hours=0,
        prevent_duplicates=False,
        allow_past_schedules=True,
        max_posts_per_day=0,  # Unlimited
        timezone="UTC",
    )


@pytest.fixture
def existing_entries():
    """Create sample existing schedule entries with known deterministic timestamps."""
    # Use fixed deterministic timestamps for predictable tests
    base_time = datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC)

    return [
        ScheduleEntry(
            product_id="B0EXISTING1",
            scheduled_time=base_time,
            platforms=[Platform.YOUTUBE, Platform.TIKTOK],
            status="pending",
        ),
        ScheduleEntry(
            product_id="B0EXISTING2",
            scheduled_time=base_time + timedelta(hours=3),
            platforms=[Platform.INSTAGRAM],
            status="pending",
        ),
        ScheduleEntry(
            product_id="B0EXISTING3",
            scheduled_time=base_time + timedelta(hours=5),
            platforms=[Platform.YOUTUBE],
            status="pending",
        ),
    ]


# Test validate() - main entry point
class TestValidate:
    """Tests for validate() method."""

    def test_validate_success(self, base_config, existing_entries):
        """Test validation passes for valid entry."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Create entry that passes all rules
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 21, 14, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator.validate(entry)

        assert is_valid is True
        assert message == ""

    def test_validate_timezone_naive_fails(self, base_config, existing_entries):
        """Test validation fails for timezone-naive datetime."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Create entry with naive datetime (no tzinfo)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 21, 14, 0, 0, tzinfo=UTC),  # Start with aware
            platforms=[Platform.YOUTUBE],
        )
        # Override with naive datetime (bypass __post_init__ validation)
        object.__setattr__(entry, "scheduled_time", datetime(2026, 1, 21, 14, 0, 0))

        is_valid, message = validator.validate(entry)

        assert is_valid is False
        assert "timezone-aware" in message
        assert "datetime.now(UTC)" in message

    def test_validate_past_schedule_disallowed(self, base_config, existing_entries):
        """Test validation fails for past schedule when disallowed."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Create entry in the past
        past_time = datetime.now(UTC) - timedelta(hours=1)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=past_time,
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator.validate(entry)

        assert is_valid is False
        assert "Cannot schedule in the past" in message
        assert "allow_past_schedules=true" in message

    def test_validate_past_schedule_allowed(self, permissive_config, existing_entries):
        """Test validation passes for past schedule when allowed."""
        validator = ScheduleValidator(permissive_config, existing_entries)

        # Create entry in the past
        past_time = datetime.now(UTC) - timedelta(hours=1)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=past_time,
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator.validate(entry)

        assert is_valid is True
        assert message == ""

    def test_validate_duplicate_fails(self, base_config, existing_entries):
        """Test validation fails for duplicate entry."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Create duplicate of first existing entry
        entry = ScheduleEntry(
            product_id="B0EXISTING1",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],  # Overlaps with existing
        )

        is_valid, message = validator.validate(entry)

        assert is_valid is False
        assert "Duplicate entry detected" in message
        assert "B0EXISTING1" in message
        assert "prevent_duplicates=false" in message

    def test_validate_duplicate_allowed(self, permissive_config, existing_entries):
        """Test validation passes for duplicate when allowed."""
        validator = ScheduleValidator(permissive_config, existing_entries)

        # Create duplicate entry
        entry = ScheduleEntry(
            product_id="B0EXISTING1",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator.validate(entry)

        assert is_valid is True
        assert message == ""

    def test_validate_spacing_violation(self, base_config, existing_entries):
        """Test validation fails for spacing violation."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Create entry too close to existing YOUTUBE post (B0EXISTING1 at 10:00)
        # New entry at 10:30 (0.5h < 2h minimum)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 10, 30, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator.validate(entry)

        assert is_valid is False
        assert "Post spacing violation" in message
        assert "youtube" in message
        assert "0.5h < 2h" in message

    def test_validate_daily_limit_exceeded(self, base_config):
        """Test validation fails when daily limit exceeded."""
        # Create existing entries at daily limit (10 posts on same date)
        # Use 1-hour spacing to keep all 10 on same day (8:00-17:00)
        base_time = datetime(2026, 1, 20, 8, 0, 0, tzinfo=UTC)
        existing = [
            ScheduleEntry(
                product_id=f"B0EXIST{i}",
                scheduled_time=base_time + timedelta(hours=i),
                platforms=[Platform.YOUTUBE],
            )
            for i in range(10)
        ]

        validator = ScheduleValidator(base_config, existing)

        # Try to add 11th entry on same date
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 23, 0, 0, tzinfo=UTC),
            platforms=[Platform.INSTAGRAM],  # Different platform
        )

        is_valid, message = validator.validate(entry)

        assert is_valid is False
        assert "Daily post limit exceeded" in message
        assert "10 posts already scheduled" in message
        assert "limit: 10" in message


# Test _is_duplicate() - duplicate detection logic
class TestIsDuplicate:
    """Tests for _is_duplicate() method."""

    def test_exact_duplicate(self, base_config, existing_entries):
        """Test exact duplicate detection (same product_id, time, platforms)."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Exact duplicate of first entry
        entry = ScheduleEntry(
            product_id="B0EXISTING1",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE, Platform.TIKTOK],
        )

        assert validator._is_duplicate(entry) is True

    def test_partial_platform_overlap(self, base_config, existing_entries):
        """Test duplicate detection with overlapping platforms."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Same product_id and time, but only one overlapping platform
        entry = ScheduleEntry(
            product_id="B0EXISTING1",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE, Platform.INSTAGRAM],  # YOUTUBE overlaps
        )

        assert validator._is_duplicate(entry) is True

    def test_same_product_different_time(self, base_config, existing_entries):
        """Test no duplicate when same product but different time."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Same product_id and platforms, but different time
        entry = ScheduleEntry(
            product_id="B0EXISTING1",
            scheduled_time=datetime(2026, 1, 20, 11, 0, 0, tzinfo=UTC),  # +1 hour
            platforms=[Platform.YOUTUBE],
        )

        assert validator._is_duplicate(entry) is False

    def test_same_time_different_product(self, base_config, existing_entries):
        """Test no duplicate when same time but different product."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Same time and platforms, but different product_id
        entry = ScheduleEntry(
            product_id="B0DIFFERENT",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        assert validator._is_duplicate(entry) is False

    def test_same_product_time_different_platforms(self, base_config, existing_entries):
        """Test no duplicate when no platform overlap."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Same product_id and time, but completely different platforms
        entry = ScheduleEntry(
            product_id="B0EXISTING1",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
            platforms=[Platform.INSTAGRAM],  # No overlap with YOUTUBE/TIKTOK
        )

        assert validator._is_duplicate(entry) is False

    def test_empty_existing_entries(self, base_config):
        """Test no duplicate when existing entries list is empty."""
        validator = ScheduleValidator(base_config, [])

        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        assert validator._is_duplicate(entry) is False

    def test_timezone_aware_comparison(self, base_config):
        """Test timezone-aware duplicate detection."""
        # Create existing entry in UTC
        existing = [
            ScheduleEntry(
                product_id="B0TEST",
                scheduled_time=datetime(2026, 1, 20, 15, 0, 0, tzinfo=UTC),
                platforms=[Platform.YOUTUBE],
            )
        ]

        validator = ScheduleValidator(base_config, existing)

        # Create entry in EST (same actual time: 15:00 UTC = 10:00 EST)
        est = ZoneInfo("America/New_York")
        entry = ScheduleEntry(
            product_id="B0TEST",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=est),
            platforms=[Platform.YOUTUBE],
        )

        # Should detect as duplicate (same actual time)
        assert validator._is_duplicate(entry) is True


# Test _check_spacing() - post spacing enforcement
class TestCheckSpacing:
    """Tests for _check_spacing() method."""

    def test_spacing_valid_far_apart(self, base_config, existing_entries):
        """Test spacing validation passes when posts far enough apart."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Entry 8 hours after first existing (> 2h minimum, after last entry at 15:00)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 18, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator._check_spacing(entry)

        assert is_valid is True
        assert message == ""

    def test_spacing_violation_same_platform(self, base_config, existing_entries):
        """Test spacing violation on same platform."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Entry 1 hour after first existing (< 2h minimum) on YOUTUBE
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 11, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator._check_spacing(entry)

        assert is_valid is False
        assert "Post spacing violation" in message
        assert "youtube" in message
        assert "1.0h < 2h" in message
        assert "B0EXISTING1" in message  # Conflicting entry

    def test_spacing_valid_different_platforms(self, base_config, existing_entries):
        """Test spacing not enforced across different platforms."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Entry 30 minutes after YOUTUBE post, but on INSTAGRAM (different platform)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 10, 30, 0, tzinfo=UTC),
            platforms=[Platform.INSTAGRAM],  # Different from YOUTUBE
        )

        is_valid, message = validator._check_spacing(entry)

        assert is_valid is True
        assert message == ""

    def test_spacing_boundary_exact_minimum(self, base_config, existing_entries):
        """Test boundary condition: exactly min_post_spacing_hours apart."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Entry exactly 2 hours after first existing (= 2h minimum, should pass)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 12, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator._check_spacing(entry)

        assert is_valid is True
        assert message == ""

    def test_spacing_boundary_one_second_less(self, base_config, existing_entries):
        """Test boundary condition: 1 second less than minimum spacing."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Entry 2 hours minus 1 second after first existing (< 2h, should fail)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 11, 59, 59, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator._check_spacing(entry)

        assert is_valid is False
        assert "Post spacing violation" in message

    def test_spacing_zero_requirement(self, permissive_config, existing_entries):
        """Test zero spacing requirement always passes."""
        validator = ScheduleValidator(permissive_config, existing_entries)

        # Entry at exact same time as existing (0h spacing, should pass)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator._check_spacing(entry)

        assert is_valid is True
        assert message == ""

    def test_spacing_before_existing(self, base_config, existing_entries):
        """Test spacing check works for entries scheduled before existing ones."""
        validator = ScheduleValidator(base_config, existing_entries)

        # Entry 1 hour BEFORE first existing (< 2h minimum)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 9, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator._check_spacing(entry)

        assert is_valid is False
        assert "Post spacing violation" in message
        assert "1.0h < 2h" in message


# Test _check_daily_limit() - daily post limit enforcement
class TestCheckDailyLimit:
    """Tests for _check_daily_limit() method."""

    def test_daily_limit_under_limit(self, base_config):
        """Test validation passes when under daily limit."""
        # Create 5 existing entries on same date (under limit of 10)
        base_time = datetime(2026, 1, 20, 8, 0, 0, tzinfo=UTC)
        existing = [
            ScheduleEntry(
                product_id=f"B0EXIST{i}",
                scheduled_time=base_time + timedelta(hours=i * 2),
                platforms=[Platform.YOUTUBE],
            )
            for i in range(5)
        ]

        validator = ScheduleValidator(base_config, existing)

        # Add 6th entry on same date
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 20, 0, 0, tzinfo=UTC),
            platforms=[Platform.INSTAGRAM],
        )

        is_valid, message = validator._check_daily_limit(entry)

        assert is_valid is True
        assert message == ""

    def test_daily_limit_at_limit(self, base_config):
        """Test validation fails when at daily limit."""
        # Create 10 existing entries on same date (at limit)
        # Use 1-hour spacing to keep all 10 on same day (8:00-17:00)
        base_time = datetime(2026, 1, 20, 8, 0, 0, tzinfo=UTC)
        existing = [
            ScheduleEntry(
                product_id=f"B0EXIST{i}",
                scheduled_time=base_time + timedelta(hours=i),
                platforms=[Platform.YOUTUBE],
            )
            for i in range(10)
        ]

        validator = ScheduleValidator(base_config, existing)

        # Try to add 11th entry on same date
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 23, 0, 0, tzinfo=UTC),
            platforms=[Platform.INSTAGRAM],
        )

        is_valid, message = validator._check_daily_limit(entry)

        assert is_valid is False
        assert "Daily post limit exceeded" in message
        assert "10 posts already scheduled" in message
        assert "2026-01-20" in message

    def test_daily_limit_boundary_exactly_limit_minus_one(self, base_config):
        """Test boundary condition: exactly limit-1 existing entries."""
        # Create 9 existing entries on same date (limit-1)
        base_time = datetime(2026, 1, 20, 8, 0, 0, tzinfo=UTC)
        existing = [
            ScheduleEntry(
                product_id=f"B0EXIST{i}",
                scheduled_time=base_time + timedelta(hours=i * 2),
                platforms=[Platform.YOUTUBE],
            )
            for i in range(9)
        ]

        validator = ScheduleValidator(base_config, existing)

        # Add 10th entry (should pass)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 23, 0, 0, tzinfo=UTC),
            platforms=[Platform.INSTAGRAM],
        )

        is_valid, message = validator._check_daily_limit(entry)

        assert is_valid is True
        assert message == ""

    def test_daily_limit_zero_unlimited(self, permissive_config, existing_entries):
        """Test zero daily limit means unlimited posts."""
        # Add 100 existing entries on same date
        base_time = datetime(2026, 1, 20, 8, 0, 0, tzinfo=UTC)
        existing = [
            ScheduleEntry(
                product_id=f"B0EXIST{i}",
                scheduled_time=base_time + timedelta(minutes=i * 5),
                platforms=[Platform.YOUTUBE],
            )
            for i in range(100)
        ]

        validator = ScheduleValidator(permissive_config, existing)

        # Add 101st entry (should pass with unlimited)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 20, 23, 0, 0, tzinfo=UTC),
            platforms=[Platform.INSTAGRAM],
        )

        is_valid, message = validator._check_daily_limit(entry)

        assert is_valid is True
        assert message == ""

    def test_daily_limit_different_dates(self, base_config, existing_entries):
        """Test daily limit not enforced across different dates."""
        # Add 10 entries on Jan 20
        base_time = datetime(2026, 1, 20, 8, 0, 0, tzinfo=UTC)
        existing = [
            ScheduleEntry(
                product_id=f"B0EXIST{i}",
                scheduled_time=base_time + timedelta(hours=i * 2),
                platforms=[Platform.YOUTUBE],
            )
            for i in range(10)
        ]

        validator = ScheduleValidator(base_config, existing)

        # Add entry on Jan 21 (different date, should pass)
        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 21, 8, 0, 0, tzinfo=UTC),
            platforms=[Platform.INSTAGRAM],
        )

        is_valid, message = validator._check_daily_limit(entry)

        assert is_valid is True
        assert message == ""

    def test_daily_limit_same_date_different_times(self, base_config):
        """Test daily limit counts all posts on same date regardless of time."""
        # Create entries throughout the day on Jan 20
        existing = [
            ScheduleEntry(
                product_id="B0MORNING",
                scheduled_time=datetime(2026, 1, 20, 6, 0, 0, tzinfo=UTC),
                platforms=[Platform.YOUTUBE],
            ),
            ScheduleEntry(
                product_id="B0NOON",
                scheduled_time=datetime(2026, 1, 20, 12, 0, 0, tzinfo=UTC),
                platforms=[Platform.INSTAGRAM],
            ),
            ScheduleEntry(
                product_id="B0EVENING",
                scheduled_time=datetime(2026, 1, 20, 18, 0, 0, tzinfo=UTC),
                platforms=[Platform.TIKTOK],
            ),
        ]

        validator = ScheduleValidator(base_config, existing)

        # Add entry late at night (same date)
        entry = ScheduleEntry(
            product_id="B0NIGHT",
            scheduled_time=datetime(2026, 1, 20, 23, 59, 59, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator._check_daily_limit(entry)

        # Should pass (3 existing + 1 new = 4, under limit of 10)
        assert is_valid is True
        assert message == ""

    def test_daily_limit_timezone_aware_date_extraction(self, base_config):
        """Test daily limit uses date() correctly with timezone-aware datetimes."""
        # Create entry in UTC at 2025-01-20 23:00
        existing = [
            ScheduleEntry(
                product_id="B0UTC",
                scheduled_time=datetime(2026, 1, 20, 23, 0, 0, tzinfo=UTC),
                platforms=[Platform.YOUTUBE],
            )
        ]

        validator = ScheduleValidator(base_config, existing)

        # Create entry in EST at 2025-01-21 00:00 EST (= 2025-01-21 05:00 UTC)
        # This is next day in both timezones
        est = ZoneInfo("America/New_York")
        entry = ScheduleEntry(
            product_id="B0EST",
            scheduled_time=datetime(2026, 1, 21, 0, 0, 0, tzinfo=est),
            platforms=[Platform.YOUTUBE],
        )

        is_valid, message = validator._check_daily_limit(entry)

        # Should pass (different dates: Jan 20 vs Jan 21)
        assert is_valid is True
        assert message == ""
