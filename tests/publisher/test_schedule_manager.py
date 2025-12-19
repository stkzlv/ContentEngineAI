"""Unit tests for ScheduleManager."""

import json
from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.publisher.models import Platform, RecurringSlot, ScheduleConfig, ScheduleEntry
from src.publisher.schedule import ScheduleManager


# Fixtures
@pytest.fixture
def tmp_schedule_path(tmp_path):
    """Create temporary schedule path."""
    return tmp_path / "schedule.json"


@pytest.fixture
def sample_slots():
    """Create sample recurring slots."""
    return [
        RecurringSlot(day_of_week="monday", time="10:00:00", timezone="UTC"),
        RecurringSlot(day_of_week="wednesday", time="14:30:00", timezone="UTC"),
        RecurringSlot(day_of_week="friday", time="16:00:00", timezone="UTC"),
    ]


@pytest.fixture
def sample_config(sample_slots):
    """Create sample schedule config."""
    return ScheduleConfig(
        enabled=True,
        slots=sample_slots,
        min_post_spacing_hours=2,
        max_posts_per_day=5,
    )


@pytest.fixture
def sample_entries():
    """Create sample schedule entries."""
    return [
        ScheduleEntry(
            product_id="B0ABC123",
            scheduled_time=datetime(2026, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
            status="pending",
        ),
        ScheduleEntry(
            product_id="B0DEF456",
            scheduled_time=datetime(2026, 1, 22, 14, 30, tzinfo=UTC),
            platforms=[Platform.TIKTOK, Platform.INSTAGRAM],
            status="scheduled",
        ),
        ScheduleEntry(
            product_id="B0GHI789",
            scheduled_time=datetime(2026, 1, 24, 16, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
            post_id="post_12345",
            status="published",
        ),
    ]


@pytest.fixture
def mock_publisher():
    """Create mock publisher."""
    publisher = AsyncMock()
    publisher.get_accounts = AsyncMock(
        return_value=[
            {"platform": "youtube", "account_id": "yt_account_123"},
            {"platform": "tiktok", "account_id": "tt_account_456"},
            {"platform": "instagram", "account_id": "ig_account_789"},
        ]
    )
    publisher.upload_media = AsyncMock(return_value="media_url_123")
    publisher.publish = AsyncMock(
        return_value={"post_id": "mock_post_123", "status": "scheduled"}
    )
    return publisher


# Test _load_schedule and _save_schedule
class TestScheduleFileOperations:
    """Tests for schedule file I/O operations."""

    def test_load_schedule_missing_file(self, tmp_schedule_path):
        """Test loading when schedule file doesn't exist."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)

        assert manager.entries == []
        assert not tmp_schedule_path.exists()

    def test_load_schedule_empty_file(self, tmp_schedule_path):
        """Test loading empty schedule file."""
        tmp_schedule_path.write_text(json.dumps({"entries": []}))

        manager = ScheduleManager(schedule_path=tmp_schedule_path)

        assert manager.entries == []

    def test_load_schedule_valid_entries(self, tmp_schedule_path):
        """Test loading valid schedule entries."""
        data = {
            "entries": [
                {
                    "product_id": "B0ABC123",
                    "scheduled_time": "2025-01-20T10:00:00+00:00",
                    "platforms": ["youtube"],
                    "post_id": None,
                    "status": "pending",
                    "created_at": "2025-01-15T08:00:00+00:00",
                    "slot_index": None,
                }
            ]
        }
        tmp_schedule_path.write_text(json.dumps(data))

        manager = ScheduleManager(schedule_path=tmp_schedule_path)

        assert len(manager.entries) == 1
        assert manager.entries[0].product_id == "B0ABC123"
        assert manager.entries[0].platforms == [Platform.YOUTUBE]

    def test_load_schedule_corrupted_json(self, tmp_schedule_path):
        """Test loading corrupted JSON file."""
        tmp_schedule_path.write_text("{invalid json")

        manager = ScheduleManager(schedule_path=tmp_schedule_path)

        # Should start with empty list, not crash
        assert manager.entries == []

    def test_load_schedule_invalid_format(self, tmp_schedule_path):
        """Test loading invalid format (not a dict)."""
        tmp_schedule_path.write_text(json.dumps(["not", "a", "dict"]))

        manager = ScheduleManager(schedule_path=tmp_schedule_path)

        assert manager.entries == []

    def test_load_schedule_invalid_entry(self, tmp_schedule_path):
        """Test loading with some invalid entries (should skip bad ones)."""
        data = {
            "entries": [
                {
                    "product_id": "B0ABC123",
                    "scheduled_time": "2025-01-20T10:00:00+00:00",
                    "platforms": ["youtube"],
                    "status": "pending",
                    "created_at": "2025-01-15T08:00:00+00:00",
                },
                {
                    "product_id": "B0INVALID",
                    # Missing required fields
                    "platforms": [],
                },
            ]
        }
        tmp_schedule_path.write_text(json.dumps(data))

        manager = ScheduleManager(schedule_path=tmp_schedule_path)

        # Should load valid entry, skip invalid one
        assert len(manager.entries) == 1
        assert manager.entries[0].product_id == "B0ABC123"

    def test_save_schedule_creates_directory(self, tmp_path):
        """Test save creates parent directory if missing."""
        schedule_path = tmp_path / "subdir" / "schedule.json"
        manager = ScheduleManager(schedule_path=schedule_path)

        # Add entry and save
        entry = ScheduleEntry(
            product_id="B0TEST",
            scheduled_time=datetime(2026, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )
        manager.entries.append(entry)
        manager._save_schedule()

        assert schedule_path.exists()
        assert schedule_path.parent.exists()

    def test_save_schedule_atomic_write(self, tmp_schedule_path):
        """Test atomic write creates temp file then renames."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)

        entry = ScheduleEntry(
            product_id="B0TEST",
            scheduled_time=datetime(2026, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )
        manager.entries.append(entry)

        # Save should create temp file then rename
        manager._save_schedule()

        # Final file should exist
        assert tmp_schedule_path.exists()

        # No temp files should remain
        temp_files = list(tmp_schedule_path.parent.glob(".schedule_*.tmp"))
        assert len(temp_files) == 0

        # Verify content
        data = json.loads(tmp_schedule_path.read_text())
        assert len(data["entries"]) == 1
        assert data["entries"][0]["product_id"] == "B0TEST"
        assert "last_updated" in data

    def test_save_and_reload_roundtrip(self, tmp_schedule_path, sample_entries):
        """Test save and reload maintains data integrity."""
        # Create manager and add entries
        manager1 = ScheduleManager(schedule_path=tmp_schedule_path)
        manager1.entries = sample_entries
        manager1._save_schedule()

        # Load in new manager
        manager2 = ScheduleManager(schedule_path=tmp_schedule_path)

        # Should have same number of entries
        assert len(manager2.entries) == len(sample_entries)

        # Verify all entries match
        for original, loaded in zip(sample_entries, manager2.entries, strict=False):
            assert loaded.product_id == original.product_id
            assert loaded.scheduled_time == original.scheduled_time
            assert loaded.platforms == original.platforms
            assert loaded.status == original.status


# Test get_next_slot
class TestGetNextSlot:
    """Tests for get_next_slot method."""

    def test_get_next_slot_empty_list(self):
        """Test get_next_slot with empty slots list."""
        manager = ScheduleManager()
        after = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

        with pytest.raises(ValueError, match="slots list cannot be empty"):
            manager.get_next_slot([], after)

    def test_get_next_slot_naive_datetime(self, sample_slots):
        """Test get_next_slot with timezone-naive datetime."""
        manager = ScheduleManager()
        after = datetime(2026, 1, 15, 12, 0)  # Naive

        with pytest.raises(ValueError, match="must be timezone-aware"):
            manager.get_next_slot(sample_slots, after)

    def test_get_next_slot_invalid_index(self, sample_slots):
        """Test get_next_slot with invalid slot_index."""
        manager = ScheduleManager()
        after = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

        with pytest.raises(ValueError, match="slot_index must be between"):
            manager.get_next_slot(sample_slots, after, slot_index=10)

    def test_get_next_slot_finds_earliest(self, sample_slots):
        """Test get_next_slot returns earliest available slot across all slots."""
        manager = ScheduleManager()

        # Wednesday Jan 14, 2026 at 9:00 AM
        after = datetime(2026, 1, 14, 9, 0, tzinfo=UTC)

        next_time, next_idx = manager.get_next_slot(sample_slots, after)

        # Returns Wednesday slot (index 1) - earliest is same day at 14:30
        assert next_time.weekday() == 2  # Wednesday
        assert next_time.day == 14  # Jan 14, 2026 (same day)
        assert next_time.hour == 14
        assert next_time.minute == 30
        assert next_idx == 1

    def test_get_next_slot_wraps_around(self, sample_slots):
        """Test get_next_slot wraps around to next week."""
        manager = ScheduleManager()

        # Friday Jan 16, 2026 at 17:00 (after Friday 16:00 slot)
        after = datetime(2026, 1, 16, 17, 0, tzinfo=UTC)

        next_time, next_idx = manager.get_next_slot(sample_slots, after)

        # Should wrap to next Monday (Jan 19) at 10:00
        assert next_time.weekday() == 0  # Monday
        assert next_time.day == 19
        assert next_time.hour == 10
        assert next_idx == 0

    def test_get_next_slot_with_start_index(self, sample_slots):
        """Test get_next_slot still finds earliest even with start_index."""
        manager = ScheduleManager()

        # Monday Jan 12, 2026 at 9:00 AM
        after = datetime(2026, 1, 12, 9, 0, tzinfo=UTC)

        # Start from index 1 (Wednesday) - but still finds earliest (Monday)
        next_time, next_idx = manager.get_next_slot(sample_slots, after, slot_index=1)

        # Should find Monday (Jan 12) at 10:00 - earliest across all slots
        assert next_time.weekday() == 0  # Monday
        assert next_time.day == 12
        assert next_idx == 0

    def test_get_next_slot_timezone_conversion(self):
        """Test get_next_slot handles timezone conversions."""
        # Slot in EST
        slots = [
            RecurringSlot(
                day_of_week="monday", time="10:00:00", timezone="America/New_York"
            )
        ]
        manager = ScheduleManager()

        # Reference time in UTC
        after = datetime(
            2026, 1, 12, 16, 0, tzinfo=UTC
        )  # Monday 16:00 UTC (after 10:00 EST = 15:00 UTC)

        next_time, next_idx = manager.get_next_slot(slots, after)

        # Monday 10:00 EST = 15:00 UTC, so next occurrence should be Jan 19
        assert next_time.weekday() == 0  # Monday
        assert next_time.day == 19


# Test list_scheduled
class TestListScheduled:
    """Tests for list_scheduled method."""

    def test_list_scheduled_all_entries(self, tmp_schedule_path, sample_entries):
        """Test listing all entries without filters."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)
        manager.entries = sample_entries

        result = manager.list_scheduled()

        assert len(result) == 3
        # Should be sorted by scheduled_time
        assert (
            result[0].scheduled_time
            < result[1].scheduled_time
            < result[2].scheduled_time
        )

    def test_list_scheduled_empty(self, tmp_schedule_path):
        """Test listing with no entries."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)

        result = manager.list_scheduled()

        assert result == []

    def test_list_scheduled_filter_platform(self, tmp_schedule_path, sample_entries):
        """Test filtering by platform."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)
        manager.entries = sample_entries

        result = manager.list_scheduled(platform="youtube")

        # Should return 2 entries with YouTube
        assert len(result) == 2
        assert all(Platform.YOUTUBE in entry.platforms for entry in result)

    def test_list_scheduled_filter_invalid_platform(
        self, tmp_schedule_path, sample_entries
    ):
        """Test filtering by invalid platform returns empty."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)
        manager.entries = sample_entries

        result = manager.list_scheduled(platform="invalid_platform")

        assert result == []

    def test_list_scheduled_filter_status(self, tmp_schedule_path, sample_entries):
        """Test filtering by status."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)
        manager.entries = sample_entries

        result = manager.list_scheduled(status="pending")

        assert len(result) == 1
        assert result[0].status == "pending"

    def test_list_scheduled_filter_date_from(self, tmp_schedule_path, sample_entries):
        """Test filtering by date_from (inclusive)."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)
        manager.entries = sample_entries

        # Filter from Jan 22
        date_from = datetime(2026, 1, 22, 0, 0, tzinfo=UTC)
        result = manager.list_scheduled(date_from=date_from)

        # Should return 2 entries on or after Jan 22
        assert len(result) == 2
        assert all(entry.scheduled_time >= date_from for entry in result)

    def test_list_scheduled_filter_date_to(self, tmp_schedule_path, sample_entries):
        """Test filtering by date_to (inclusive)."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)
        manager.entries = sample_entries

        # Filter up to Jan 22
        date_to = datetime(2026, 1, 22, 23, 59, 59, tzinfo=UTC)
        result = manager.list_scheduled(date_to=date_to)

        # Should return 2 entries on or before Jan 22
        assert len(result) == 2
        assert all(entry.scheduled_time <= date_to for entry in result)

    def test_list_scheduled_filter_date_range(self, tmp_schedule_path, sample_entries):
        """Test filtering by date range."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)
        manager.entries = sample_entries

        date_from = datetime(2026, 1, 21, 0, 0, tzinfo=UTC)
        date_to = datetime(2026, 1, 23, 23, 59, 59, tzinfo=UTC)
        result = manager.list_scheduled(date_from=date_from, date_to=date_to)

        # Should return only 1 entry (Jan 22)
        assert len(result) == 1
        assert result[0].scheduled_time.day == 22

    def test_list_scheduled_multiple_filters(self, tmp_schedule_path, sample_entries):
        """Test combining multiple filters."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)
        manager.entries = sample_entries

        result = manager.list_scheduled(
            platform="youtube",
            status="published",
            date_from=datetime(2026, 1, 20, 0, 0, tzinfo=UTC),
        )

        # Should return 1 entry (YouTube, published, after Jan 20)
        assert len(result) == 1
        assert result[0].product_id == "B0GHI789"

    def test_list_scheduled_all_none_filters(self, tmp_schedule_path, sample_entries):
        """Test with all filters set to None."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)
        manager.entries = sample_entries

        result = manager.list_scheduled(
            platform=None, status=None, date_from=None, date_to=None
        )

        # Should return all entries
        assert len(result) == 3

    def test_list_scheduled_naive_datetime_warning(
        self, tmp_schedule_path, sample_entries
    ):
        """Test naive datetime in filters gets converted to UTC."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)
        manager.entries = sample_entries

        # Use naive datetime (should log warning and treat as UTC)
        date_from = datetime(2026, 1, 22, 0, 0)  # Naive

        result = manager.list_scheduled(date_from=date_from)

        # Should still work, treating as UTC
        assert len(result) == 2


# Test add_entry
class TestAddEntry:
    """Tests for add_entry method."""

    def test_add_entry_valid(self, tmp_schedule_path):
        """Test adding valid entry."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)

        entry = ScheduleEntry(
            product_id="B0NEW",
            scheduled_time=datetime(2026, 1, 25, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        manager.add_entry(entry)

        # Should be in entries list
        assert len(manager.entries) == 1
        assert manager.entries[0].product_id == "B0NEW"

        # Should be saved to disk
        assert tmp_schedule_path.exists()
        data = json.loads(tmp_schedule_path.read_text())
        assert len(data["entries"]) == 1

    def test_add_entry_multiple(self, tmp_schedule_path):
        """Test adding multiple entries."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path)

        for i in range(3):
            entry = ScheduleEntry(
                product_id=f"B0TEST{i}",
                scheduled_time=datetime(2026, 1, 20 + i, 10, 0, tzinfo=UTC),
                platforms=[Platform.YOUTUBE],
            )
            manager.add_entry(entry)

        assert len(manager.entries) == 3

        # Reload from disk
        manager2 = ScheduleManager(schedule_path=tmp_schedule_path)
        assert len(manager2.entries) == 3

    def test_add_entry_with_validation(self, tmp_schedule_path, sample_config):
        """Test add_entry respects validation rules."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path, config=sample_config)

        # Add first entry
        entry1 = ScheduleEntry(
            product_id="B0TEST1",
            scheduled_time=datetime(2026, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )
        manager.add_entry(entry1)

        # Try to add duplicate (same product, platform, time)
        entry2 = ScheduleEntry(
            product_id="B0TEST1",
            scheduled_time=datetime(2026, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
        )

        # Should raise validation error
        with pytest.raises(ValueError, match="duplicate"):
            manager.add_entry(entry2)


# Test auto_schedule (with mocked publisher)
@pytest.mark.asyncio
class TestAutoSchedule:
    """Tests for auto_schedule method."""

    async def test_auto_schedule_basic(
        self, tmp_schedule_path, sample_config, mock_publisher, tmp_path
    ):
        """Test basic auto-schedule functionality."""
        # Disable validation rules for basic test
        config = ScheduleConfig(
            enabled=True,
            slots=sample_config.slots,
            min_post_spacing_hours=0,
            prevent_duplicates=False,
            allow_past_schedules=True,
            max_posts_per_day=0,
        )
        manager = ScheduleManager(schedule_path=tmp_schedule_path, config=config)

        # Create mock video files with data.json
        videos = [
            tmp_path / "B0TEST1" / "video_test1.mp4",
            tmp_path / "B0TEST2" / "video_test2.mp4",
        ]
        for i, video in enumerate(videos):
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()
            # Create data.json for metadata
            (video.parent / "data.json").write_text(
                json.dumps(
                    {
                        "title": f"Test Product {i+1}",
                        "description": f"Description {i+1}",
                    }
                )
            )

        summary = await manager.auto_schedule(
            videos=videos,
            platforms=[Platform.YOUTUBE],
            publisher=mock_publisher,
            start_slot=0,
            dry_run=False,
        )

        # Should schedule both videos
        assert summary["scheduled"] == 2
        assert summary["skipped"] == 0
        assert summary["failed"] == 0

        # Verify publisher.publish was called twice
        assert mock_publisher.publish.call_count == 2

    async def test_auto_schedule_dry_run(
        self, tmp_schedule_path, sample_config, mock_publisher, tmp_path
    ):
        """Test auto-schedule in dry-run mode."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path, config=sample_config)

        videos = [tmp_path / "B0TEST1" / "video_test1.mp4"]
        videos[0].parent.mkdir(parents=True, exist_ok=True)
        videos[0].touch()

        summary = await manager.auto_schedule(
            videos=videos,
            platforms=[Platform.YOUTUBE],
            publisher=mock_publisher,
            start_slot=0,
            dry_run=True,
        )

        # Should not call publisher in dry-run
        assert mock_publisher.publish.call_count == 0

        # Should still report scheduled
        assert summary["scheduled"] == 1

    async def test_auto_schedule_no_slots(
        self, tmp_schedule_path, mock_publisher, tmp_path
    ):
        """Test auto-schedule with no configured slots raises ValueError."""
        config = ScheduleConfig(enabled=True, slots=[])
        manager = ScheduleManager(schedule_path=tmp_schedule_path, config=config)

        videos = [tmp_path / "B0TEST1" / "video_test1.mp4"]
        videos[0].parent.mkdir(parents=True, exist_ok=True)
        videos[0].touch()

        # Should raise ValueError when no slots configured
        with pytest.raises(ValueError, match="No recurring slots configured"):
            await manager.auto_schedule(
                videos=videos,
                platforms=[Platform.YOUTUBE],
                publisher=mock_publisher,
            )

    async def test_auto_schedule_respects_start_slot(
        self, tmp_schedule_path, sample_config, mock_publisher, tmp_path
    ):
        """Test auto-schedule respects start_slot parameter."""
        manager = ScheduleManager(schedule_path=tmp_schedule_path, config=sample_config)

        videos = [tmp_path / "B0TEST1" / "video_test1.mp4"]
        videos[0].parent.mkdir(parents=True, exist_ok=True)
        videos[0].touch()

        mock_publisher.upload_media = AsyncMock(return_value="media_url_123")

        # Start from slot 2 (Friday)
        await manager.auto_schedule(
            videos=videos,
            platforms=[Platform.YOUTUBE],
            publisher=mock_publisher,
            start_slot=2,
            dry_run=False,
        )

        # Verify entry was created with slot_index=2
        assert len(manager.entries) == 1
        assert manager.entries[0].slot_index == 2
