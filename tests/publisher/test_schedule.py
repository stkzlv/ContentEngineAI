"""Tests for schedule management functionality."""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.publisher.models import Platform, RecurringSlot, ScheduleConfig, ScheduleEntry
from src.publisher.schedule import ScheduleManager


@pytest.fixture
def temp_schedule_file():
    """Create a temporary schedule file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def schedule_config_with_slots():
    """Schedule config with recurring slots enabled, validation rules disabled."""
    return ScheduleConfig(
        enabled=True,
        slots=[
            RecurringSlot("monday", "10:00:00", "UTC"),
            RecurringSlot("wednesday", "14:00:00", "UTC"),
            RecurringSlot("friday", "18:00:00", "UTC"),
        ],
        timezone="UTC",
        min_post_spacing_hours=0,  # Disable spacing validation for tests
        prevent_duplicates=False,  # Disable duplicate validation for tests
        allow_past_schedules=True,  # Allow past schedules for tests
        max_posts_per_day=0,  # Disable daily limit for tests
    )


@pytest.fixture
def mock_publisher():
    """Mock publisher instance."""
    publisher = AsyncMock()
    publisher.get_accounts = AsyncMock(
        return_value=[
            {"platform": "youtube", "account_id": "yt_account_123"},
            {"platform": "tiktok", "account_id": "tt_account_456"},
            {"platform": "instagram", "account_id": "ig_account_789"},
        ]
    )
    publisher.upload_media = AsyncMock(return_value="media_123")
    publisher.publish = AsyncMock(
        return_value={"post_id": "post_456", "status": "scheduled"}
    )
    return publisher


@pytest.fixture
def mock_video_files(tmp_path):
    """Create mock video files with data.json."""
    videos = []
    for i, product_id in enumerate(["B0TEST001", "B0TEST002", "B0TEST003"]):
        product_dir = tmp_path / "outputs" / product_id
        product_dir.mkdir(parents=True)

        # Create video file
        video_path = product_dir / f"video_{product_id}.mp4"
        video_path.write_text(f"mock video {i}")
        videos.append(video_path)

        # Create metadata
        metadata = {
            "title": f"Test Product {i+1}",
            "description": f"Description for product {i+1}",
        }
        (product_dir / "data.json").write_text(json.dumps(metadata))

    return videos


class TestAutoSchedule:
    """Tests for auto_schedule() method."""

    @pytest.mark.asyncio
    async def test_requires_enabled_recurring_schedule(
        self, temp_schedule_file, mock_publisher, mock_video_files
    ):
        """Test that auto_schedule() requires recurring schedule to be enabled."""
        # Config with disabled recurring schedule
        config = ScheduleConfig(enabled=False)
        manager = ScheduleManager(temp_schedule_file, config)

        with pytest.raises(ValueError, match="Recurring schedule is not enabled"):
            await manager.auto_schedule(
                videos=mock_video_files,
                platforms=[Platform.YOUTUBE],
                publisher=mock_publisher,
            )

    @pytest.mark.asyncio
    async def test_requires_configured_slots(
        self, temp_schedule_file, mock_publisher, mock_video_files
    ):
        """Test that auto_schedule() requires slots to be configured."""
        # Config with enabled=True but no slots
        config = ScheduleConfig(enabled=True, slots=[])
        manager = ScheduleManager(temp_schedule_file, config)

        with pytest.raises(ValueError, match="No recurring slots configured"):
            await manager.auto_schedule(
                videos=mock_video_files,
                platforms=[Platform.YOUTUBE],
                publisher=mock_publisher,
            )

    @pytest.mark.asyncio
    async def test_dry_run_mode(
        self,
        temp_schedule_file,
        schedule_config_with_slots,
        mock_publisher,
        mock_video_files,
    ):
        """Test dry run mode doesn't actually publish."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        result = await manager.auto_schedule(
            videos=mock_video_files,
            platforms=[Platform.YOUTUBE, Platform.TIKTOK],
            publisher=mock_publisher,
            dry_run=True,
        )

        # Should count scheduled but not call publisher
        assert result["scheduled"] == 3
        assert result["skipped"] == 0
        assert result["failed"] == 0

        # Publisher should not be called in dry run
        mock_publisher.upload_media.assert_not_called()
        mock_publisher.publish.assert_not_called()

        # Schedule file should not have entries added
        assert len(manager.entries) == 0

    @pytest.mark.asyncio
    async def test_skips_already_published_videos(
        self,
        temp_schedule_file,
        schedule_config_with_slots,
        mock_publisher,
        mock_video_files,
    ):
        """Test that already published videos are skipped."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        # Mock is_already_published to return True for first video
        with patch("src.publisher.tracking.is_already_published") as mock_check:
            # First video is already published to YouTube
            def side_effect(product_id, platform):
                return product_id == "B0TEST001" and platform == "youtube"

            mock_check.side_effect = side_effect

            result = await manager.auto_schedule(
                videos=mock_video_files,
                platforms=[Platform.YOUTUBE],
                publisher=mock_publisher,
            )

            # First video skipped, other two scheduled
            assert result["scheduled"] == 2
            assert result["skipped"] == 1
            assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_successful_scheduling(
        self,
        temp_schedule_file,
        schedule_config_with_slots,
        mock_publisher,
        mock_video_files,
    ):
        """Test successful video scheduling with unified posting (default)."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        with patch("src.publisher.tracking.is_already_published", return_value=False):
            result = await manager.auto_schedule(
                videos=mock_video_files,
                platforms=[Platform.YOUTUBE, Platform.TIKTOK],
                publisher=mock_publisher,
            )

            # All videos scheduled
            assert result["scheduled"] == 3
            assert result["skipped"] == 0
            assert result["failed"] == 0

            # Unified mode: Publisher called once per video (upload + publish)
            # 3 videos × 1 upload = 3 upload calls
            # 3 videos × 1 publish (all platforms) = 3 publish calls
            assert mock_publisher.upload_media.call_count == 3
            assert mock_publisher.publish.call_count == 3

            # Entries added to schedule (one per video with all platforms)
            assert len(manager.entries) == 3

            # Verify entries have correct structure (unified posts)
            for entry in manager.entries:
                assert entry.status == "scheduled"
                assert entry.post_id == "post_456"
                # Each entry contains all platforms in one post
                assert len(entry.platforms) == 2
                assert Platform.YOUTUBE in entry.platforms
                assert Platform.TIKTOK in entry.platforms

    @pytest.mark.asyncio
    async def test_platform_specific_scheduling(
        self,
        temp_schedule_file,
        mock_publisher,
        mock_video_files,
    ):
        """Test platform-specific scheduling mode (optional behavior)."""
        # Config with platform-specific content enabled
        config = ScheduleConfig(
            enabled=True,
            slots=[
                RecurringSlot("monday", "10:00:00", "UTC"),
                RecurringSlot("wednesday", "14:00:00", "UTC"),
            ],
            timezone="UTC",
            use_platform_specific_content=True,  # Enable platform-specific mode
            min_post_spacing_hours=0,
            prevent_duplicates=False,
            allow_past_schedules=True,
            max_posts_per_day=0,
        )
        manager = ScheduleManager(temp_schedule_file, config)

        with patch("src.publisher.tracking.is_already_published", return_value=False):
            result = await manager.auto_schedule(
                videos=mock_video_files,
                platforms=[Platform.YOUTUBE, Platform.TIKTOK],
                publisher=mock_publisher,
            )

            # All videos scheduled
            assert result["scheduled"] == 3
            assert result["skipped"] == 0
            assert result["failed"] == 0

            # Platform-specific mode: Publisher called once per platform per video
            # 3 videos × 1 upload = 3 upload calls
            # 3 videos × 2 platforms = 6 publish calls
            assert mock_publisher.upload_media.call_count == 3
            assert mock_publisher.publish.call_count == 6

            # Entries added to schedule (one per platform per video)
            # 3 videos × 2 platforms = 6 entries
            assert len(manager.entries) == 6

            # Verify entries have correct structure (separate per platform)
            for entry in manager.entries:
                assert entry.status == "scheduled"
                assert entry.post_id == "post_456"
                # Each entry is for a single platform
                assert len(entry.platforms) == 1
                assert entry.platforms[0] in [Platform.YOUTUBE, Platform.TIKTOK]

    @pytest.mark.asyncio
    async def test_slot_wrapping_with_start_slot(
        self,
        temp_schedule_file,
        schedule_config_with_slots,
        mock_publisher,
        mock_video_files,
    ):
        """Test that slot assignment uses earliest available slots sequentially.

        The implementation finds the earliest available slot from the current time,
        scheduling in chronological order: Monday (slot 0), Wednesday (slot 1),
        Friday (slot 2). The start_slot parameter suggests a starting point but
        chronological ordering takes precedence.
        """
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        with patch("src.publisher.tracking.is_already_published", return_value=False):
            result = await manager.auto_schedule(
                videos=mock_video_files,
                platforms=[Platform.YOUTUBE],
                publisher=mock_publisher,
                start_slot=2,  # start_slot hint (chronological order takes precedence)
            )

            assert result["scheduled"] == 3

            # Verify slot indices in entries
            # Slots are scheduled in chronological order from current time:
            # Monday (slot 0) -> Wednesday (slot 1) -> Friday (slot 2)
            slot_indices = [e.slot_index for e in manager.entries]
            assert slot_indices == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_handles_publish_failures(
        self,
        temp_schedule_file,
        schedule_config_with_slots,
        mock_publisher,
        mock_video_files,
    ):
        """Test that publish failures are tracked correctly."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        # Make second publish call fail
        call_count = 0

        async def publish_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Publish failed")
            return {"post_id": f"post_{call_count}", "status": "scheduled"}

        mock_publisher.publish.side_effect = publish_side_effect

        with patch("src.publisher.tracking.is_already_published", return_value=False):
            result = await manager.auto_schedule(
                videos=mock_video_files,
                platforms=[Platform.YOUTUBE],
                publisher=mock_publisher,
            )

            # Two successful, one failed
            assert result["scheduled"] == 2
            assert result["skipped"] == 0
            assert result["failed"] == 1

            # All entries recorded (including failed)
            assert len(manager.entries) == 3

            # Failed entry has failed status
            failed_entry = manager.entries[1]
            assert failed_entry.status == "failed"
            assert failed_entry.product_id == "B0TEST002"

    @pytest.mark.asyncio
    async def test_uses_metadata_for_content(
        self,
        temp_schedule_file,
        schedule_config_with_slots,
        mock_publisher,
        mock_video_files,
    ):
        """Test that metadata from data.json is used for post content."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        with patch("src.publisher.tracking.is_already_published", return_value=False):
            await manager.auto_schedule(
                videos=[mock_video_files[0]],
                platforms=[Platform.YOUTUBE],
                publisher=mock_publisher,
            )

            # Check that publish was called with platform_contents from metadata
            publish_call = mock_publisher.publish.call_args
            platform_contents = publish_call[1]["platform_contents"]

            assert "youtube" in platform_contents
            youtube_content = platform_contents["youtube"]["content"]
            assert "Test Product 1" in youtube_content
            assert "Description for product 1" in youtube_content


class TestAddEntry:
    """Tests for add_entry() method."""

    def test_adds_valid_entry(self, temp_schedule_file, schedule_config_with_slots):
        """Test adding a valid schedule entry."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        entry = ScheduleEntry(
            product_id="B0TEST001",
            scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE, Platform.TIKTOK],
            post_id=None,
            status="pending",
            created_at=datetime.now(UTC),
        )

        manager.add_entry(entry)

        # Entry should be added
        assert len(manager.entries) == 1
        assert manager.entries[0].product_id == "B0TEST001"
        assert manager.entries[0].status == "pending"

        # Schedule file should be updated
        assert temp_schedule_file.exists()

    def test_rejects_missing_product_id(
        self, temp_schedule_file, schedule_config_with_slots
    ):
        """Test that entry without product_id is rejected."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        # ScheduleEntry __post_init__ validation rejects empty product_id
        with pytest.raises(ValueError, match="product_id cannot be empty"):
            ScheduleEntry(
                product_id="",
                scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
                platforms=[Platform.YOUTUBE],
                post_id=None,
                status="pending",
                created_at=datetime.now(UTC),
            )

        # Entry should not be added
        assert len(manager.entries) == 0

    def test_rejects_timezone_naive_datetime(
        self, temp_schedule_file, schedule_config_with_slots
    ):
        """Test that timezone-naive scheduled_time is rejected."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        # ScheduleEntry __post_init__ validation rejects timezone-naive datetime
        with pytest.raises(ValueError, match="scheduled_time must include timezone"):
            ScheduleEntry(
                product_id="B0TEST001",
                scheduled_time=datetime(2025, 1, 20, 10, 0),  # No tzinfo
                platforms=[Platform.YOUTUBE],
                post_id=None,
                status="pending",
                created_at=datetime.now(UTC),
            )

        assert len(manager.entries) == 0

    def test_rejects_empty_platforms(
        self, temp_schedule_file, schedule_config_with_slots
    ):
        """Test that entry without platforms is rejected."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        # ScheduleEntry __post_init__ validation rejects empty platforms
        with pytest.raises(ValueError, match="platforms cannot be empty"):
            ScheduleEntry(
                product_id="B0TEST001",
                scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
                platforms=[],
                post_id=None,
                status="pending",
                created_at=datetime.now(UTC),
            )

        assert len(manager.entries) == 0

    def test_rejects_invalid_status(
        self, temp_schedule_file, schedule_config_with_slots
    ):
        """Test that entry with invalid status is rejected."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        # ScheduleEntry __post_init__ validation rejects invalid status
        with pytest.raises(ValueError, match="status must be one of"):
            ScheduleEntry(
                product_id="B0TEST001",
                scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
                platforms=[Platform.YOUTUBE],
                post_id=None,
                status="invalid_status",
                created_at=datetime.now(UTC),
            )

        assert len(manager.entries) == 0

    def test_rejects_duplicate_entry(self, temp_schedule_file):
        """Test that duplicate entries are rejected."""
        # Config with duplicate detection enabled
        config = ScheduleConfig(
            enabled=True,
            prevent_duplicates=True,
            min_post_spacing_hours=0,
            allow_past_schedules=True,
            max_posts_per_day=0,
        )
        manager = ScheduleManager(temp_schedule_file, config)

        entry1 = ScheduleEntry(
            product_id="B0TEST001",
            scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE, Platform.TIKTOK],
            post_id=None,
            status="pending",
            created_at=datetime.now(UTC),
        )

        # Add first entry
        manager.add_entry(entry1)
        assert len(manager.entries) == 1

        # Try to add duplicate
        entry2 = ScheduleEntry(
            product_id="B0TEST001",
            scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[
                Platform.TIKTOK,
                Platform.YOUTUBE,
            ],  # Same platforms, different order
            post_id=None,
            status="pending",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(ValueError, match="Duplicate entry detected"):
            manager.add_entry(entry2)

        # Only first entry should remain
        assert len(manager.entries) == 1

    def test_rollback_on_write_failure(
        self, temp_schedule_file, schedule_config_with_slots, monkeypatch
    ):
        """Test that entry is rolled back if save fails."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        # Make _save_schedule raise an exception
        def mock_save():
            raise OSError("Disk full")

        monkeypatch.setattr(manager, "_save_schedule", mock_save)

        entry = ScheduleEntry(
            product_id="B0TEST001",
            scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
            post_id=None,
            status="pending",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(IOError, match="Failed to save schedule"):
            manager.add_entry(entry)

        # Entry should be rolled back
        assert len(manager.entries) == 0

    def test_multiple_entries(self, temp_schedule_file, schedule_config_with_slots):
        """Test adding multiple different entries."""
        manager = ScheduleManager(temp_schedule_file, schedule_config_with_slots)

        # Add three different entries
        for i in range(3):
            entry = ScheduleEntry(
                product_id=f"B0TEST00{i+1}",
                scheduled_time=datetime(2025, 1, 20 + i, 10, 0, tzinfo=UTC),
                platforms=[Platform.YOUTUBE],
                post_id=None,
                status="pending",
                created_at=datetime.now(UTC),
            )
            manager.add_entry(entry)

        # All entries should be added
        assert len(manager.entries) == 3
        assert manager.entries[0].product_id == "B0TEST001"
        assert manager.entries[1].product_id == "B0TEST002"
        assert manager.entries[2].product_id == "B0TEST003"
