"""Unit tests for schedule models."""

import json
from datetime import UTC, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.publisher.models import (
    CleanupConfig,
    Platform,
    RecurringSlot,
    ScheduleConfig,
    ScheduleEntry,
)


# Fixtures for test data
@pytest.fixture
def recurring_slot_monday():
    """Create a recurring slot for Monday 9:00 AM UTC."""
    return RecurringSlot(
        day_of_week="monday",
        time="09:00:00",
        timezone="UTC",
    )


@pytest.fixture
def recurring_slot_friday():
    """Create a recurring slot for Friday 2:30 PM EST."""
    return RecurringSlot(
        day_of_week="friday",
        time="14:30:00",
        timezone="America/New_York",
    )


@pytest.fixture
def schedule_entry_pending():
    """Create a pending schedule entry."""
    return ScheduleEntry(
        product_id="B0ABC123",
        scheduled_time=datetime(2026, 1, 20, 14, 0, 0, tzinfo=UTC),
        platforms=[Platform.YOUTUBE, Platform.TIKTOK],
        status="pending",
    )


@pytest.fixture
def schedule_entry_published():
    """Create a published schedule entry."""
    return ScheduleEntry(
        product_id="B0DEF456",
        scheduled_time=datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC),
        platforms=[Platform.INSTAGRAM],
        post_id="post_12345",
        status="published",
        slot_index=0,
    )


@pytest.fixture
def schedule_config_default():
    """Create a default schedule config."""
    return ScheduleConfig(
        enabled=True,
        min_post_spacing_hours=2,
        prevent_duplicates=True,
        allow_past_schedules=False,
        max_posts_per_day=10,
        timezone="UTC",
    )


# RecurringSlot tests
class TestRecurringSlot:
    """Tests for RecurringSlot model."""

    def test_next_occurrence_same_week(self, recurring_slot_monday):
        """Test next_occurrence within the same week."""
        # Wednesday Jan 14, 2026 at 8:00 AM
        after = datetime(2026, 1, 14, 8, 0, 0, tzinfo=UTC)

        next_time = recurring_slot_monday.next_occurrence(after)

        # Should return next Monday (Jan 19) at 9:00 AM
        assert next_time.year == 2026
        assert next_time.month == 1
        assert next_time.day == 19
        assert next_time.hour == 9
        assert next_time.minute == 0
        assert next_time.weekday() == 0  # Monday

    def test_next_occurrence_next_week(self, recurring_slot_monday):
        """Test next_occurrence crossing week boundary."""
        # Monday Jan 19, 2026 at 10:00 AM (after the 9:00 AM slot)
        after = datetime(2026, 1, 19, 10, 0, 0, tzinfo=UTC)

        next_time = recurring_slot_monday.next_occurrence(after)

        # Should return next Monday (Jan 26) at 9:00 AM
        assert next_time.year == 2026
        assert next_time.month == 1
        assert next_time.day == 26
        assert next_time.hour == 9
        assert next_time.minute == 0

    def test_next_occurrence_with_timezone(self, recurring_slot_friday):
        """Test next_occurrence with non-UTC timezone."""
        # Thursday Jan 15, 2026 at 12:00 PM EST
        tz = ZoneInfo("America/New_York")
        after = datetime(2026, 1, 15, 12, 0, 0, tzinfo=tz)

        next_time = recurring_slot_friday.next_occurrence(after)

        # Should return Friday Jan 16 at 2:30 PM EST
        assert next_time.year == 2026
        assert next_time.month == 1
        assert next_time.day == 16
        assert next_time.hour == 14
        assert next_time.minute == 30
        assert next_time.weekday() == 4  # Friday

    def test_next_occurrence_month_boundary(self, recurring_slot_monday):
        """Test next_occurrence crossing month boundary."""
        # Monday Jan 26, 2026 at 10:00 AM
        after = datetime(2026, 1, 26, 10, 0, 0, tzinfo=UTC)

        next_time = recurring_slot_monday.next_occurrence(after)

        # Should return Monday Feb 2 at 9:00 AM
        assert next_time.year == 2026
        assert next_time.month == 2
        assert next_time.day == 2
        assert next_time.hour == 9

    def test_next_occurrence_year_boundary(self):
        """Test next_occurrence crossing year boundary."""
        slot = RecurringSlot(
            day_of_week="monday",
            time="09:00:00",
            timezone="UTC",
        )

        # Monday Dec 29, 2025 at 10:00 AM
        after = datetime(2025, 12, 29, 10, 0, 0, tzinfo=UTC)

        next_time = slot.next_occurrence(after)

        # Should return Monday Jan 5, 2026 at 9:00 AM
        assert next_time.year == 2026
        assert next_time.month == 1
        assert next_time.day == 5

    def test_to_dict(self, recurring_slot_monday):
        """Test to_dict serialization."""
        data = recurring_slot_monday.to_dict()

        assert data["day_of_week"] == "monday"
        assert data["time"] == "09:00:00"
        assert data["timezone"] == "UTC"

        # Ensure it's JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None

    def test_frozen_dataclass(self, recurring_slot_monday):
        """Test that RecurringSlot is immutable."""
        with pytest.raises(
            (AttributeError, TypeError)
        ):  # Frozen dataclass raises AttributeError
            recurring_slot_monday.time = "10:00"


# ScheduleEntry tests
class TestScheduleEntry:
    """Tests for ScheduleEntry model."""

    def test_valid_pending_entry(self, schedule_entry_pending):
        """Test valid pending schedule entry."""
        assert schedule_entry_pending.product_id == "B0ABC123"
        assert schedule_entry_pending.status == "pending"
        assert len(schedule_entry_pending.platforms) == 2
        assert schedule_entry_pending.post_id is None

    def test_valid_published_entry(self, schedule_entry_published):
        """Test valid published schedule entry."""
        assert schedule_entry_published.status == "published"
        assert schedule_entry_published.post_id == "post_12345"
        assert schedule_entry_published.slot_index == 0

    def test_created_at_defaults_to_now(self):
        """Test created_at defaults to current UTC time."""
        before = datetime.now(UTC)
        entry = ScheduleEntry(
            product_id="B0TEST",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )
        after = datetime.now(UTC)

        assert entry.created_at.tzinfo is not None
        assert before <= entry.created_at <= after

    def test_to_dict(self, schedule_entry_pending):
        """Test to_dict serialization."""
        data = schedule_entry_pending.to_dict()

        assert data["product_id"] == "B0ABC123"
        assert data["status"] == "pending"
        assert data["platforms"] == ["youtube", "tiktok"]
        assert "scheduled_time" in data
        assert data["post_id"] is None

        # Ensure it's JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None

    def test_to_dict_with_post_id(self, schedule_entry_published):
        """Test to_dict with post_id and slot_index."""
        data = schedule_entry_published.to_dict()

        assert data["post_id"] == "post_12345"
        assert data["slot_index"] == 0

    def test_timezone_aware_scheduled_time(self):
        """Test that scheduled_time must be timezone-aware."""
        # This should work - timezone-aware
        entry = ScheduleEntry(
            product_id="B0TEST",
            scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )
        assert entry.scheduled_time.tzinfo is not None

    def test_validation_non_empty_platforms(self):
        """Test validation requires at least one platform."""
        with pytest.raises(ValueError, match="platforms cannot be empty"):
            ScheduleEntry(
                product_id="B0TEST",
                scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
                platforms=[],  # Empty platforms list
            )

    def test_validation_product_id_not_empty(self):
        """Test validation requires non-empty product_id."""
        with pytest.raises(ValueError, match="product_id cannot be empty"):
            ScheduleEntry(
                product_id="",  # Empty product_id
                scheduled_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=UTC),
                platforms=[Platform.YOUTUBE],
            )

    def test_validation_timezone_aware(self):
        """Test validation requires timezone-aware datetime."""
        with pytest.raises(
            ValueError, match="scheduled_time must include timezone information"
        ):
            ScheduleEntry(
                product_id="B0TEST",
                scheduled_time=datetime(2026, 1, 20, 10, 0, 0),  # Naive datetime
                platforms=[Platform.YOUTUBE],
            )


# ScheduleConfig tests
class TestScheduleConfig:
    """Tests for ScheduleConfig model."""

    def test_valid_config(self, schedule_config_default):
        """Test valid schedule config."""
        assert schedule_config_default.enabled is True
        assert schedule_config_default.min_post_spacing_hours == 2
        assert schedule_config_default.max_posts_per_day == 10
        assert schedule_config_default.timezone == "UTC"

    def test_default_values(self):
        """Test default values for ScheduleConfig."""
        config = ScheduleConfig()

        assert config.enabled is False
        assert config.min_post_spacing_hours == 2
        assert config.prevent_duplicates is True
        assert config.allow_past_schedules is False
        assert config.max_posts_per_day == 10
        assert config.timezone == "UTC"
        assert len(config.slots) == 0

    def test_with_recurring_slots(self, recurring_slot_monday, recurring_slot_friday):
        """Test config with recurring slots."""
        config = ScheduleConfig(
            enabled=True,
            slots=[recurring_slot_monday, recurring_slot_friday],
        )

        assert len(config.slots) == 2
        assert config.slots[0].day_of_week == "monday"
        assert config.slots[1].day_of_week == "friday"

    def test_to_dict(self, schedule_config_default):
        """Test to_dict serialization."""
        data = schedule_config_default.to_dict()

        assert data["enabled"] is True
        assert data["min_post_spacing_hours"] == 2
        assert data["prevent_duplicates"] is True
        assert data["max_posts_per_day"] == 10
        assert data["slots"] == []

        # Ensure it's JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None

    def test_to_dict_with_slots(self, recurring_slot_monday):
        """Test to_dict with recurring slots."""
        config = ScheduleConfig(
            enabled=True,
            slots=[recurring_slot_monday],
        )

        data = config.to_dict()

        assert len(data["slots"]) == 1
        assert data["slots"][0]["day_of_week"] == "monday"

    def test_validation_non_negative_spacing(self):
        """Test validation requires non-negative min_post_spacing_hours."""
        with pytest.raises(
            ValueError, match="min_post_spacing_hours must be non-negative"
        ):
            ScheduleConfig(min_post_spacing_hours=-1)

    def test_validation_non_negative_max_posts(self):
        """Test validation requires non-negative max_posts_per_day."""
        with pytest.raises(ValueError, match="max_posts_per_day must be non-negative"):
            ScheduleConfig(max_posts_per_day=-1)


# CleanupConfig tests (bonus - since it's part of the same models file)
class TestCleanupConfig:
    """Tests for CleanupConfig model."""

    def test_default_values(self):
        """Test default values for CleanupConfig."""
        config = CleanupConfig()

        assert config.enabled is True  # Updated: default changed to True
        assert config.verify_before_delete is True
        assert config.require_all_platforms is True
        assert config.archive_before_delete is False
        assert config.keep_published_days == 0
        assert config.preserve_metadata is False
        assert config.preserve_logs is True

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = CleanupConfig(
            enabled=True,
            verify_before_delete=True,
            archive_before_delete=True,
        )

        data = config.to_dict()

        assert data["enabled"] is True
        assert data["verify_before_delete"] is True
        assert data["archive_before_delete"] is True

        # Ensure it's JSON serializable
        json_str = json.dumps(data, default=str)  # Use default=str for Path
        assert json_str is not None

    def test_validation_non_negative_keep_days(self):
        """Test validation requires non-negative keep_published_days."""
        with pytest.raises(
            ValueError, match="keep_published_days must be non-negative"
        ):
            CleanupConfig(keep_published_days=-1)
