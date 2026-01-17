"""Tests for schedule conflict resolution."""

from datetime import UTC, datetime, timedelta

import pytest

from src.publisher.models import (
    ConflictResolution,
    Platform,
    RecurringSlot,
    ScheduleConfig,
    ScheduleEntry,
)
from src.publisher.schedule import ScheduleManager


def get_future_monday() -> datetime:
    """Get a future Monday at 10:00 UTC for testing."""
    now = datetime.now(UTC)
    # Find next Monday
    days_until_monday = (7 - now.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7  # Always go to next week
    future_monday = now + timedelta(days=days_until_monday)
    return future_monday.replace(hour=10, minute=0, second=0, microsecond=0)


def get_future_wednesday() -> datetime:
    """Get a future Wednesday at 14:00 UTC for testing."""
    now = datetime.now(UTC)
    # Find next Wednesday
    days_until_wednesday = (2 - now.weekday()) % 7
    if days_until_wednesday == 0:
        days_until_wednesday = 7  # Always go to next week
    future_wednesday = now + timedelta(days=days_until_wednesday)
    return future_wednesday.replace(hour=14, minute=0, second=0, microsecond=0)


@pytest.fixture
def schedule_config() -> ScheduleConfig:
    """Create schedule config with recurring slots."""
    return ScheduleConfig(
        enabled=True,
        slots=[
            RecurringSlot("monday", "10:00:00", "UTC"),
            RecurringSlot("wednesday", "14:00:00", "UTC"),
            RecurringSlot("friday", "16:00:00", "UTC"),
        ],
        min_post_spacing_hours=2,
        prevent_duplicates=True,
        allow_past_schedules=False,
        max_posts_per_day=5,
        conflict_alternatives_count=5,
    )


@pytest.fixture
def schedule_manager(tmp_path, schedule_config) -> ScheduleManager:
    """Create schedule manager with temp schedule file."""
    schedule_path = tmp_path / "schedule.json"
    return ScheduleManager(schedule_path=schedule_path, config=schedule_config)


class TestConflictResolutionDataclass:
    """Tests for ConflictResolution dataclass."""

    def test_basic_creation(self):
        """Test creating a basic conflict resolution."""
        now = datetime.now(UTC)
        resolution = ConflictResolution(
            original_time=now,
            conflict_reason="Slot occupied",
        )

        assert resolution.original_time == now
        assert resolution.conflict_reason == "Slot occupied"
        assert resolution.alternatives == []
        assert resolution.auto_resolved is False
        assert resolution.resolved_time is None

    def test_with_alternatives(self):
        """Test conflict resolution with alternatives."""
        now = datetime.now(UTC)
        alt1 = now + timedelta(hours=2)
        alt2 = now + timedelta(hours=4)

        resolution = ConflictResolution(
            original_time=now,
            conflict_reason="Duplicate entry",
            alternatives=[alt1, alt2],
        )

        assert len(resolution.alternatives) == 2
        assert alt1 in resolution.alternatives
        assert alt2 in resolution.alternatives

    def test_auto_resolved(self):
        """Test auto-resolved conflict."""
        now = datetime.now(UTC)
        resolved = now + timedelta(hours=2)

        resolution = ConflictResolution(
            original_time=now,
            conflict_reason="Slot occupied",
            alternatives=[resolved],
            auto_resolved=True,
            resolved_time=resolved,
        )

        assert resolution.auto_resolved is True
        assert resolution.resolved_time == resolved

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime(2025, 1, 20, 10, 0, tzinfo=UTC)
        alt = datetime(2025, 1, 20, 14, 0, tzinfo=UTC)

        resolution = ConflictResolution(
            original_time=now,
            conflict_reason="Slot occupied",
            alternatives=[alt],
            auto_resolved=True,
            resolved_time=alt,
        )

        result = resolution.to_dict()

        assert result["original_time"] == "2025-01-20T10:00:00+00:00"
        assert result["conflict_reason"] == "Slot occupied"
        assert len(result["alternatives"]) == 1
        assert result["auto_resolved"] is True
        assert result["resolved_time"] == "2025-01-20T14:00:00+00:00"

    def test_to_dict_no_resolved_time(self):
        """Test serialization when not auto-resolved."""
        now = datetime(2025, 1, 20, 10, 0, tzinfo=UTC)

        resolution = ConflictResolution(
            original_time=now,
            conflict_reason="Slot occupied",
        )

        result = resolution.to_dict()

        assert result["resolved_time"] is None


class TestFindAlternatives:
    """Tests for ScheduleManager.find_alternatives method."""

    def test_find_alternatives_basic(self, schedule_manager):
        """Test finding basic alternatives."""
        preferred_time = get_future_monday()
        occupied = {preferred_time}

        resolution = schedule_manager.find_alternatives(
            preferred_time=preferred_time,
            platforms=[Platform.YOUTUBE],
            occupied_slots=occupied,
            count=3,
        )

        assert resolution.original_time == preferred_time
        assert len(resolution.alternatives) <= 3
        # Alternatives should not include occupied slot
        for alt in resolution.alternatives:
            normalized = alt.replace(second=0, microsecond=0)
            assert normalized not in occupied

    def test_find_alternatives_with_multiple_occupied(self, schedule_manager):
        """Test finding alternatives with multiple occupied slots."""
        future_monday = get_future_monday()
        future_wednesday = get_future_wednesday()

        # Occupy several slots
        occupied = {future_monday, future_wednesday}

        resolution = schedule_manager.find_alternatives(
            preferred_time=future_monday,
            platforms=[Platform.YOUTUBE],
            occupied_slots=occupied,
            count=5,
        )

        # Should find alternatives that skip occupied slots
        for alt in resolution.alternatives:
            normalized = alt.replace(second=0, microsecond=0)
            assert normalized not in occupied

    def test_find_alternatives_sorted_by_proximity(self, schedule_manager):
        """Test that alternatives are sorted by time proximity."""
        preferred_time = get_future_wednesday()
        occupied: set[datetime] = set()  # No occupied slots

        resolution = schedule_manager.find_alternatives(
            preferred_time=preferred_time,
            platforms=[Platform.YOUTUBE],
            occupied_slots=occupied,
            count=3,
        )

        # Alternatives should be sorted by proximity to preferred time
        if len(resolution.alternatives) >= 2:
            for i in range(len(resolution.alternatives) - 1):
                dist_i = abs(
                    (resolution.alternatives[i] - preferred_time).total_seconds()
                )
                dist_j = abs(
                    (resolution.alternatives[i + 1] - preferred_time).total_seconds()
                )
                assert dist_i <= dist_j

    def test_find_alternatives_respects_count(self, schedule_manager):
        """Test that find_alternatives respects the count parameter."""
        preferred_time = get_future_monday()
        occupied: set[datetime] = set()

        resolution = schedule_manager.find_alternatives(
            preferred_time=preferred_time,
            platforms=[Platform.YOUTUBE],
            occupied_slots=occupied,
            count=2,
        )

        assert len(resolution.alternatives) <= 2

    def test_find_alternatives_uses_config_count(self, schedule_manager):
        """Test that find_alternatives uses config count when not specified."""
        preferred_time = get_future_monday()
        occupied: set[datetime] = set()

        resolution = schedule_manager.find_alternatives(
            preferred_time=preferred_time,
            platforms=[Platform.YOUTUBE],
            occupied_slots=occupied,
        )

        # Config has conflict_alternatives_count=5
        assert len(resolution.alternatives) <= 5

    def test_find_alternatives_no_slots_configured(self, tmp_path):
        """Test behavior when no slots are configured."""
        config = ScheduleConfig(enabled=True, slots=[])
        manager = ScheduleManager(
            schedule_path=tmp_path / "schedule.json", config=config
        )

        preferred_time = get_future_monday()

        resolution = manager.find_alternatives(
            preferred_time=preferred_time,
            platforms=[Platform.YOUTUBE],
            occupied_slots=set(),
        )

        assert resolution.alternatives == []
        assert "No recurring slots" in resolution.conflict_reason


class TestResolveConflict:
    """Tests for ScheduleManager.resolve_conflict method."""

    def test_resolve_conflict_without_auto_resolve(self, schedule_manager):
        """Test resolve_conflict without auto_resolve."""
        preferred_time = get_future_monday()
        occupied = {preferred_time}

        resolution = schedule_manager.resolve_conflict(
            preferred_time=preferred_time,
            platforms=[Platform.YOUTUBE],
            occupied_slots=occupied,
            auto_resolve=False,
        )

        assert resolution.auto_resolved is False
        assert resolution.resolved_time is None
        # Should still have alternatives
        assert len(resolution.alternatives) > 0

    def test_resolve_conflict_with_auto_resolve(self, schedule_manager):
        """Test resolve_conflict with auto_resolve enabled."""
        preferred_time = get_future_monday()
        occupied = {preferred_time}

        resolution = schedule_manager.resolve_conflict(
            preferred_time=preferred_time,
            platforms=[Platform.YOUTUBE],
            occupied_slots=occupied,
            auto_resolve=True,
        )

        assert resolution.auto_resolved is True
        assert resolution.resolved_time is not None
        # Resolved time should be first alternative
        assert resolution.resolved_time == resolution.alternatives[0]

    def test_resolve_conflict_no_alternatives_available(self, tmp_path):
        """Test resolve_conflict when no alternatives available."""
        config = ScheduleConfig(enabled=True, slots=[])
        manager = ScheduleManager(
            schedule_path=tmp_path / "schedule.json", config=config
        )

        preferred_time = get_future_monday()

        resolution = manager.resolve_conflict(
            preferred_time=preferred_time,
            platforms=[Platform.YOUTUBE],
            occupied_slots=set(),
            auto_resolve=True,
        )

        # Should not auto-resolve if no alternatives
        assert resolution.auto_resolved is False
        assert resolution.resolved_time is None


class TestScheduleConfigConflictSettings:
    """Tests for ScheduleConfig conflict settings."""

    def test_default_conflict_alternatives_count(self):
        """Test default conflict_alternatives_count value."""
        config = ScheduleConfig()
        assert config.conflict_alternatives_count == 5

    def test_custom_conflict_alternatives_count(self):
        """Test custom conflict_alternatives_count."""
        config = ScheduleConfig(conflict_alternatives_count=10)
        assert config.conflict_alternatives_count == 10

    def test_invalid_conflict_alternatives_count(self):
        """Test validation of conflict_alternatives_count."""
        with pytest.raises(ValueError, match="conflict_alternatives_count"):
            ScheduleConfig(conflict_alternatives_count=0)

    def test_to_dict_includes_conflict_alternatives_count(self):
        """Test that to_dict includes conflict_alternatives_count."""
        config = ScheduleConfig(conflict_alternatives_count=7)
        result = config.to_dict()
        assert result["conflict_alternatives_count"] == 7


class TestConflictResolutionIntegration:
    """Integration tests for conflict resolution in scheduling."""

    def test_existing_entries_cause_conflict(self, schedule_manager):
        """Test that existing entries are detected as conflicts."""
        # Add an existing entry
        existing_time = get_future_monday()
        existing_entry = ScheduleEntry(
            product_id="B0EXISTING",
            scheduled_time=existing_time,
            platforms=[Platform.YOUTUBE],
            status="scheduled",
            created_at=datetime.now(UTC),
        )
        schedule_manager.entries.append(existing_entry)

        # Try to find alternatives for same time
        resolution = schedule_manager.find_alternatives(
            preferred_time=existing_time,
            platforms=[Platform.YOUTUBE],
            occupied_slots={existing_time},
        )

        # Should find alternatives
        assert len(resolution.alternatives) > 0
        # Alternatives should be different from existing time
        for alt in resolution.alternatives:
            assert alt != existing_time

    def test_resolve_then_schedule(self, schedule_manager):
        """Test resolving conflict then using resolved time."""
        preferred_time = get_future_monday()
        occupied = {preferred_time}

        # Resolve conflict
        resolution = schedule_manager.resolve_conflict(
            preferred_time=preferred_time,
            platforms=[Platform.YOUTUBE],
            occupied_slots=occupied,
            auto_resolve=True,
        )

        assert resolution.auto_resolved is True
        resolved_time = resolution.resolved_time
        assert resolved_time is not None

        # Create entry with resolved time
        entry = ScheduleEntry(
            product_id="B0RESOLVED",
            scheduled_time=resolved_time,
            platforms=[Platform.YOUTUBE],
            status="pending",
            created_at=datetime.now(UTC),
        )

        # Should be valid
        schedule_manager.add_entry(entry)
        assert len(schedule_manager.entries) == 1
        assert schedule_manager.entries[0].scheduled_time == resolved_time
