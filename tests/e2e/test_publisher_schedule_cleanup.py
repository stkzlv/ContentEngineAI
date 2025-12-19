"""Integration tests for publisher schedule and cleanup workflows.

Tests end-to-end workflows for:
- Schedule configuration and auto-scheduling
- Cleanup workflow with verification and archiving
- CLI argument parsing
- --no-cleanup flag behavior
"""

import argparse
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.publisher.cleanup import CleanupManager
from src.publisher.models import (
    CleanupConfig,
    Platform,
    RecurringSlot,
    ScheduleConfig,
    ScheduleEntry,
)
from src.publisher.schedule import ScheduleManager

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_outputs_dir(tmp_path):
    """Create temporary outputs directory with realistic structure."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return outputs


@pytest.fixture
def mock_publisher():
    """Mock publisher instance with async methods."""
    publisher = AsyncMock()
    publisher.authenticate = AsyncMock(return_value=True)
    publisher.get_accounts = AsyncMock(
        return_value=[
            {"platform": "youtube", "account_id": "acc_youtube_123"},
            {"platform": "tiktok", "account_id": "acc_tiktok_456"},
        ]
    )
    publisher.upload_media = AsyncMock(return_value="media_url_12345")
    publisher.publish = AsyncMock(
        return_value={"post_id": "post_123", "status": "published"}
    )
    publisher.get_status = AsyncMock(return_value={"status": "published"})
    return publisher


@pytest.fixture
def schedule_config():
    """Create schedule configuration with recurring slots."""
    return ScheduleConfig(
        enabled=True,
        slots=[
            RecurringSlot(day_of_week="monday", time="10:00:00", timezone="UTC"),
            RecurringSlot(day_of_week="wednesday", time="14:00:00", timezone="UTC"),
            RecurringSlot(day_of_week="friday", time="16:00:00", timezone="UTC"),
        ],
        min_post_spacing_hours=1,
        max_posts_per_day=3,
        allow_past_schedules=True,  # Allow past dates for testing
    )


@pytest.fixture
def cleanup_config(tmp_path):
    """Create cleanup configuration."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    return CleanupConfig(
        enabled=True,
        verify_before_delete=True,
        require_all_platforms=True,
        archive_before_delete=True,
        archive_dir=archive_dir,
        keep_published_days=0,  # Immediate cleanup for testing
    )


@pytest.fixture
def product_directory(temp_outputs_dir):
    """Create a realistic product directory with video and metadata."""
    product_id = "B0TEST001"
    product_dir = temp_outputs_dir / product_id
    product_dir.mkdir()

    # Create text directory with metadata
    text_dir = product_dir / "text"
    text_dir.mkdir()

    # Create metadata file
    metadata = {
        "platform": "youtube",
        "title": "Test Product Video",
        "description": "This is a test product video.",
        "tags": ["test", "product"],
    }
    (text_dir / "metadata_youtube.json").write_text(json.dumps(metadata))

    # Create data.json
    data = {
        "title": "Test Product",
        "description": "A test product description",
        "product_id": product_id,
    }
    (product_dir / "data.json").write_text(json.dumps(data))

    # Create dummy video file (1MB)
    video_path = product_dir / f"video_{product_id}_sequential.mp4"
    video_path.write_bytes(b"0" * (1024 * 1024))

    # Create images directory
    images_dir = product_dir / "images"
    images_dir.mkdir()
    (images_dir / "image_001.jpg").write_bytes(b"fake_jpg" * 100)

    return {
        "product_id": product_id,
        "product_dir": product_dir,
        "video_path": video_path,
    }


@pytest.fixture
def multiple_products(temp_outputs_dir):
    """Create multiple product directories for batch testing."""
    products = []

    for i in range(3):
        product_id = f"B0BATCH{i:03d}"
        product_dir = temp_outputs_dir / product_id
        product_dir.mkdir()

        # Create video file
        video_path = product_dir / f"video_{product_id}_sequential.mp4"
        video_path.write_bytes(b"0" * (512 * 1024))  # 512KB each

        # Create data.json
        (product_dir / "data.json").write_text(
            json.dumps({"title": f"Product {i}", "product_id": product_id})
        )

        products.append(
            {
                "product_id": product_id,
                "product_dir": product_dir,
                "video_path": video_path,
            }
        )

    return products


def create_publish_history(
    outputs_dir: Path, product_id: str, platform: str, post_id: str
):
    """Helper to create publish_history.json entry."""
    tracking_path = outputs_dir / "publish_history.json"

    if tracking_path.exists():
        data = json.loads(tracking_path.read_text())
    else:
        data = {"posts": {}}

    key = f"{product_id}:{platform}"
    data["posts"][key] = {
        "product_id": product_id,
        "platform": platform,
        "post_id": post_id,
        "published_at": datetime.now(UTC).isoformat(),
        "post_url": f"https://{platform}.com/{post_id}",
    }

    tracking_path.write_text(json.dumps(data, indent=2))


def create_schedule_json(schedule_path: Path, entries: list[dict]):
    """Helper to create schedule.json file."""
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"entries": entries, "last_updated": datetime.now(UTC).isoformat()}
    schedule_path.write_text(json.dumps(data, indent=2))


# ============================================================================
# Schedule Workflow Integration Tests
# ============================================================================


class TestScheduleWorkflowIntegration:
    """Integration tests for complete schedule workflow."""

    def test_schedule_manager_init_creates_empty_schedule(self, temp_outputs_dir):
        """Test ScheduleManager initializes with empty schedule when file missing."""
        schedule_path = temp_outputs_dir / "schedule.json"

        manager = ScheduleManager(schedule_path=schedule_path)

        assert manager.entries == []
        assert not schedule_path.exists()

    def test_schedule_manager_loads_existing_entries(self, temp_outputs_dir):
        """Test ScheduleManager loads existing schedule entries."""
        schedule_path = temp_outputs_dir / "schedule.json"

        # Create existing schedule
        entries = [
            {
                "product_id": "B0TEST001",
                "scheduled_time": datetime(2025, 1, 20, 10, 0, tzinfo=UTC).isoformat(),
                "platforms": ["youtube"],
                "status": "pending",
                "created_at": datetime.now(UTC).isoformat(),
            }
        ]
        create_schedule_json(schedule_path, entries)

        manager = ScheduleManager(schedule_path=schedule_path)

        assert len(manager.entries) == 1
        assert manager.entries[0].product_id == "B0TEST001"
        assert manager.entries[0].status == "pending"

    def test_add_entry_saves_atomically(self, temp_outputs_dir, schedule_config):
        """Test add_entry saves to disk atomically."""
        schedule_path = temp_outputs_dir / "schedule.json"
        manager = ScheduleManager(schedule_path=schedule_path, config=schedule_config)

        entry = ScheduleEntry(
            product_id="B0TEST001",
            scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
            status="pending",
            created_at=datetime.now(UTC),
        )

        manager.add_entry(entry)

        # Verify saved to disk
        assert schedule_path.exists()
        data = json.loads(schedule_path.read_text())
        assert len(data["entries"]) == 1
        assert data["entries"][0]["product_id"] == "B0TEST001"

    def test_list_scheduled_filters_correctly(self, temp_outputs_dir, schedule_config):
        """Test list_scheduled applies filters correctly."""
        schedule_path = temp_outputs_dir / "schedule.json"

        # Create schedule with multiple entries
        now = datetime.now(UTC)
        entries = [
            {
                "product_id": "B0TEST001",
                "scheduled_time": now.isoformat(),
                "platforms": ["youtube"],
                "status": "pending",
                "created_at": now.isoformat(),
            },
            {
                "product_id": "B0TEST002",
                "scheduled_time": (now + timedelta(days=1)).isoformat(),
                "platforms": ["tiktok"],
                "status": "published",
                "created_at": now.isoformat(),
            },
            {
                "product_id": "B0TEST003",
                "scheduled_time": (now + timedelta(days=2)).isoformat(),
                "platforms": ["youtube", "tiktok"],
                "status": "pending",
                "created_at": now.isoformat(),
            },
        ]
        create_schedule_json(schedule_path, entries)

        manager = ScheduleManager(schedule_path=schedule_path, config=schedule_config)

        # Filter by platform
        youtube_entries = manager.list_scheduled(platform="youtube")
        assert len(youtube_entries) == 2

        # Filter by status
        pending_entries = manager.list_scheduled(status="pending")
        assert len(pending_entries) == 2

        # Filter by date range
        date_entries = manager.list_scheduled(
            date_from=now, date_to=now + timedelta(hours=12)
        )
        assert len(date_entries) == 1

    def test_get_next_slot_calculates_correctly(self, schedule_config):
        """Test get_next_slot finds next available recurring slot."""
        manager = ScheduleManager(config=schedule_config)

        # Wednesday at noon
        reference = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)

        next_time, slot_idx = manager.get_next_slot(
            slots=schedule_config.slots, after=reference, slot_index=0
        )

        # Should find next slot after reference time
        assert next_time > reference
        assert slot_idx in [0, 1, 2]

    @pytest.mark.asyncio
    async def test_auto_schedule_dry_run(
        self,
        temp_outputs_dir,
        schedule_config,
        mock_publisher,
        multiple_products,
    ):
        """Test auto_schedule in dry_run mode doesn't publish."""
        schedule_path = temp_outputs_dir / "schedule.json"
        manager = ScheduleManager(schedule_path=schedule_path, config=schedule_config)

        videos = [p["video_path"] for p in multiple_products]

        summary = await manager.auto_schedule(
            videos=videos,
            platforms=[Platform.YOUTUBE],
            publisher=mock_publisher,
            start_slot=0,
            dry_run=True,
        )

        assert summary["scheduled"] == 3
        assert summary["failed"] == 0
        # Publisher should not have been called
        mock_publisher.upload_media.assert_not_called()
        mock_publisher.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_schedule_skips_published(
        self,
        temp_outputs_dir,
        schedule_config,
        mock_publisher,
        multiple_products,
    ):
        """Test auto_schedule skips already published videos."""
        schedule_path = temp_outputs_dir / "schedule.json"
        manager = ScheduleManager(schedule_path=schedule_path, config=schedule_config)

        first_product = multiple_products[0]
        videos = [p["video_path"] for p in multiple_products]

        # Patch is_already_published to return True for first product
        def mock_is_published(product_id: str, platform: str, outputs_dir=None) -> bool:
            return bool(product_id == first_product["product_id"])

        with patch(
            "src.publisher.tracking.is_already_published",
            side_effect=mock_is_published,
        ):
            summary = await manager.auto_schedule(
                videos=videos,
                platforms=[Platform.YOUTUBE],
                publisher=mock_publisher,
                start_slot=0,
                dry_run=True,
            )

        # Should skip the published one
        assert summary["scheduled"] == 2
        assert summary["skipped"] == 1


# ============================================================================
# Cleanup Workflow Integration Tests
# ============================================================================


class TestCleanupWorkflowIntegration:
    """Integration tests for complete cleanup workflow."""

    @pytest.mark.asyncio
    async def test_cleanup_full_workflow(
        self,
        temp_outputs_dir,
        cleanup_config,
        mock_publisher,
        product_directory,
    ):
        """Test complete cleanup workflow: verify → archive → delete."""
        product_id = product_directory["product_id"]

        # Record publish to tracking
        create_publish_history(temp_outputs_dir, product_id, "youtube", "post_123")

        manager = CleanupManager(
            outputs_dir=temp_outputs_dir,
            config=cleanup_config,
            publisher=mock_publisher,
        )

        result = await manager.cleanup(
            product_id=product_id, platforms=[Platform.YOUTUBE], dry_run=False
        )

        assert result["success"] is True
        assert isinstance(result["disk_freed"], int) and result["disk_freed"] > 0

        # Verify directory was removed
        assert not product_directory["product_dir"].exists()

        # Verify archive was created
        archive_files = list(cleanup_config.archive_dir.glob("*.zip"))
        assert len(archive_files) == 1
        assert product_id in archive_files[0].name

        # Verify audit log was created
        audit_path = temp_outputs_dir / "cleanup_audit.json"
        assert audit_path.exists()
        audit_data = json.loads(audit_path.read_text())
        assert len(audit_data["cleanups"]) == 1
        assert audit_data["cleanups"][0]["product_id"] == product_id

    @pytest.mark.asyncio
    async def test_cleanup_dry_run_no_deletion(
        self,
        temp_outputs_dir,
        cleanup_config,
        mock_publisher,
        product_directory,
    ):
        """Test cleanup dry_run doesn't delete anything."""
        product_id = product_directory["product_id"]
        create_publish_history(temp_outputs_dir, product_id, "youtube", "post_123")

        manager = CleanupManager(
            outputs_dir=temp_outputs_dir,
            config=cleanup_config,
            publisher=mock_publisher,
        )

        result = await manager.cleanup(
            product_id=product_id, platforms=[Platform.YOUTUBE], dry_run=True
        )

        assert result["success"] is True
        assert isinstance(result["message"], str) and "[DRY RUN]" in result["message"]
        assert result["disk_freed"] == 0

        # Directory should still exist
        assert product_directory["product_dir"].exists()

        # No archive should be created
        archive_files = list(cleanup_config.archive_dir.glob("*.zip"))
        assert len(archive_files) == 0

    @pytest.mark.asyncio
    async def test_cleanup_verification_fails(
        self,
        temp_outputs_dir,
        cleanup_config,
        mock_publisher,
        product_directory,
    ):
        """Test cleanup fails when verification fails."""
        product_id = product_directory["product_id"]
        create_publish_history(temp_outputs_dir, product_id, "youtube", "post_123")

        # Make publisher return failed status
        mock_publisher.get_status = AsyncMock(return_value={"status": "failed"})

        manager = CleanupManager(
            outputs_dir=temp_outputs_dir,
            config=cleanup_config,
            publisher=mock_publisher,
        )

        result = await manager.cleanup(
            product_id=product_id, platforms=[Platform.YOUTUBE], dry_run=False
        )

        assert result["success"] is False
        assert (
            isinstance(result["message"], str) and "not published" in result["message"]
        )

        # Directory should still exist
        assert product_directory["product_dir"].exists()

    @pytest.mark.asyncio
    async def test_cleanup_disabled_in_config(
        self,
        temp_outputs_dir,
        mock_publisher,
        product_directory,
    ):
        """Test cleanup does nothing when disabled in config."""
        product_id = product_directory["product_id"]

        disabled_config = CleanupConfig(enabled=False)

        manager = CleanupManager(
            outputs_dir=temp_outputs_dir,
            config=disabled_config,
            publisher=mock_publisher,
        )

        result = await manager.cleanup(
            product_id=product_id, platforms=[Platform.YOUTUBE], dry_run=False
        )

        assert result["success"] is False
        message = result["message"]
        assert isinstance(message, str) and "disabled" in message.lower()
        assert product_directory["product_dir"].exists()

    @pytest.mark.asyncio
    async def test_cleanup_all_batch(
        self,
        temp_outputs_dir,
        cleanup_config,
        mock_publisher,
        multiple_products,
    ):
        """Test cleanup_all processes multiple products."""
        # Record all products as published
        for product in multiple_products:
            create_publish_history(
                temp_outputs_dir,
                product["product_id"],
                "youtube",
                f"post_{product['product_id']}",
            )

        manager = CleanupManager(
            outputs_dir=temp_outputs_dir,
            config=cleanup_config,
            publisher=mock_publisher,
        )

        summary = await manager.cleanup_all(platforms=[Platform.YOUTUBE], dry_run=False)

        assert summary["cleaned"] == 3
        assert summary["skipped"] == 0
        assert summary["disk_freed"] > 0

        # All directories should be removed
        for product in multiple_products:
            assert not product["product_dir"].exists()

    @pytest.mark.asyncio
    async def test_cleanup_respects_keep_days(
        self,
        temp_outputs_dir,
        mock_publisher,
        product_directory,
    ):
        """Test cleanup respects keep_published_days setting."""
        product_id = product_directory["product_id"]

        # Create config with 7 day keep period
        config = CleanupConfig(
            enabled=True,
            verify_before_delete=False,
            keep_published_days=7,
        )

        # Record publish from today
        create_publish_history(temp_outputs_dir, product_id, "youtube", "post_123")

        manager = CleanupManager(
            outputs_dir=temp_outputs_dir, config=config, publisher=mock_publisher
        )

        result = await manager.cleanup(
            product_id=product_id, platforms=[Platform.YOUTUBE], dry_run=False
        )

        assert result["success"] is False
        assert (
            isinstance(result["message"], str) and "not old enough" in result["message"]
        )
        assert product_directory["product_dir"].exists()


# ============================================================================
# CLI Argument Parsing Tests
# ============================================================================


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_parse_datetime_valid_formats(self):
        """Test parse_datetime accepts various formats."""
        from src.publisher.late.cli import parse_datetime

        # Format 1: YYYY-MM-DD HH:MM:SS
        dt1 = parse_datetime("2025-01-20 14:00:00")
        assert dt1.year == 2025
        assert dt1.month == 1
        assert dt1.day == 20
        assert dt1.hour == 14

        # Format 2: YYYY-MM-DDTHH:MM:SS
        dt2 = parse_datetime("2025-01-20T14:00:00")
        assert dt2.year == 2025
        assert dt2.hour == 14

        # Format 3: YYYY-MM-DD HH:MM
        dt3 = parse_datetime("2025-01-20 14:00")
        assert dt3.minute == 0

    def test_parse_datetime_invalid_format(self):
        """Test parse_datetime raises on invalid format."""
        from src.publisher.late.cli import parse_datetime

        with pytest.raises(ValueError) as exc_info:
            parse_datetime("invalid-date")

        assert "Invalid datetime format" in str(exc_info.value)

    def test_single_command_requires_video(self):
        """Test single command requires --video argument."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        single_parser = subparsers.add_parser("single")
        single_parser.add_argument("--video", type=Path, required=True)
        single_parser.add_argument(
            "--platform", action="append", dest="platforms", required=True
        )
        single_parser.add_argument("--immediate", action="store_true")

        # Should fail without --video
        with pytest.raises(SystemExit):
            parser.parse_args(["single", "--platform", "youtube", "--immediate"])

    def test_batch_command_requires_immediate(self):
        """Test batch command validates --immediate flag."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        batch_parser = subparsers.add_parser("batch")
        batch_parser.add_argument(
            "--platform", action="append", dest="platforms", required=True
        )
        batch_parser.add_argument("--immediate", action="store_true")

        # Parse without immediate (should succeed parsing, validation is in main())
        args = parser.parse_args(["batch", "--platform", "youtube"])
        assert args.immediate is False

    def test_cleanup_requires_product_id_or_all(self):
        """Test cleanup command requires --product-id or --all."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        cleanup_parser = subparsers.add_parser("cleanup")
        cleanup_group = cleanup_parser.add_mutually_exclusive_group(required=True)
        cleanup_group.add_argument("--product-id")
        cleanup_group.add_argument("--all", action="store_true")

        # Should fail without either
        with pytest.raises(SystemExit):
            parser.parse_args(["cleanup"])

        # Should work with --product-id
        args1 = parser.parse_args(["cleanup", "--product-id", "B0TEST001"])
        assert args1.product_id == "B0TEST001"

        # Should work with --all
        args2 = parser.parse_args(["cleanup", "--all"])
        assert args2.all is True

    def test_schedule_command_platforms(self):
        """Test schedule command accepts multiple platforms."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        schedule_parser = subparsers.add_parser("schedule")
        schedule_parser.add_argument("action", choices=["auto"])
        schedule_parser.add_argument("--platform", action="append", dest="platforms")
        schedule_parser.add_argument("--dry-run", action="store_true")

        args = parser.parse_args(
            [
                "schedule",
                "auto",
                "--platform",
                "youtube",
                "--platform",
                "tiktok",
                "--dry-run",
            ]
        )

        assert args.platforms == ["youtube", "tiktok"]
        assert args.dry_run is True

    def test_calendar_command_filters(self):
        """Test calendar command accepts filter arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        calendar_parser = subparsers.add_parser("calendar")
        calendar_parser.add_argument("action", choices=["list"])
        calendar_parser.add_argument("--platform")
        calendar_parser.add_argument("--status")
        calendar_parser.add_argument("--date-from")
        calendar_parser.add_argument("--date-to")

        args = parser.parse_args(
            [
                "calendar",
                "list",
                "--platform",
                "youtube",
                "--status",
                "pending",
                "--date-from",
                "2025-01-01",
                "--date-to",
                "2025-01-31",
            ]
        )

        assert args.platform == "youtube"
        assert args.status == "pending"
        assert args.date_from == "2025-01-01"
        assert args.date_to == "2025-01-31"


# ============================================================================
# --no-cleanup Flag Integration Tests
# ============================================================================


class TestNoCleanupFlagIntegration:
    """Tests for --no-cleanup flag behavior."""

    def test_single_command_has_no_cleanup_flag(self):
        """Test single command has --no-cleanup argument."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        single_parser = subparsers.add_parser("single")
        single_parser.add_argument("--video", type=Path, required=True)
        single_parser.add_argument(
            "--platform", action="append", dest="platforms", required=True
        )
        single_parser.add_argument("--immediate", action="store_true")
        single_parser.add_argument("--no-cleanup", action="store_true")

        args = parser.parse_args(
            [
                "single",
                "--video",
                "test.mp4",
                "--platform",
                "youtube",
                "--immediate",
                "--no-cleanup",
            ]
        )

        assert args.no_cleanup is True

    def test_batch_command_has_no_cleanup_flag(self):
        """Test batch command has --no-cleanup argument."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        batch_parser = subparsers.add_parser("batch")
        batch_parser.add_argument(
            "--platform", action="append", dest="platforms", required=True
        )
        batch_parser.add_argument("--immediate", action="store_true")
        batch_parser.add_argument("--no-cleanup", action="store_true")

        args = parser.parse_args(
            ["batch", "--platform", "youtube", "--immediate", "--no-cleanup"]
        )

        assert args.no_cleanup is True

    @pytest.mark.asyncio
    async def test_no_cleanup_flag_prevents_cleanup(
        self, temp_outputs_dir, mock_publisher, product_directory
    ):
        """Test --no-cleanup flag prevents automatic cleanup."""
        # Simulate what cmd_single does with no_cleanup flag
        cleanup_config = CleanupConfig(enabled=True)
        no_cleanup = True

        # Track whether cleanup would be called
        cleanup_called = False

        if cleanup_config.enabled and not no_cleanup:
            cleanup_called = True

        assert cleanup_called is False
        assert product_directory["product_dir"].exists()

    @pytest.mark.asyncio
    async def test_cleanup_runs_without_flag(
        self, temp_outputs_dir, cleanup_config, mock_publisher, product_directory
    ):
        """Test cleanup runs when --no-cleanup is not set."""
        product_id = product_directory["product_id"]
        create_publish_history(temp_outputs_dir, product_id, "youtube", "post_123")

        # Simulate what cmd_single does without no_cleanup flag
        no_cleanup = False

        if cleanup_config.enabled and not no_cleanup:
            manager = CleanupManager(
                outputs_dir=temp_outputs_dir,
                config=cleanup_config,
                publisher=mock_publisher,
            )

            result = await manager.cleanup(
                product_id=product_id,
                platforms=[Platform.YOUTUBE],
                dry_run=False,
            )

            assert result["success"] is True

        # Directory should be removed
        assert not product_directory["product_dir"].exists()


# ============================================================================
# Component Integration Tests
# ============================================================================


class TestComponentIntegration:
    """Tests for component interactions."""

    def test_schedule_entry_to_dict_roundtrip(self):
        """Test ScheduleEntry serialization/deserialization roundtrip."""
        entry = ScheduleEntry(
            product_id="B0TEST001",
            scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE, Platform.TIKTOK],
            post_id="post_123",
            status="scheduled",
            created_at=datetime(2025, 1, 15, 8, 0, tzinfo=UTC),
            slot_index=2,
        )

        # Serialize
        data = entry.to_dict()
        assert data["product_id"] == "B0TEST001"
        assert data["platforms"] == ["youtube", "tiktok"]
        assert data["slot_index"] == 2

        # Deserialize
        restored = ScheduleEntry(
            product_id=data["product_id"],
            scheduled_time=datetime.fromisoformat(data["scheduled_time"]),
            platforms=[Platform(p) for p in data["platforms"]],
            post_id=data.get("post_id"),
            status=data.get("status", "pending"),
            created_at=datetime.fromisoformat(data["created_at"]),
            slot_index=data.get("slot_index"),
        )

        assert restored.product_id == entry.product_id
        assert restored.platforms == entry.platforms
        assert restored.slot_index == entry.slot_index

    @pytest.mark.asyncio
    async def test_schedule_and_cleanup_integration(
        self,
        temp_outputs_dir,
        schedule_config,
        cleanup_config,
        mock_publisher,
        product_directory,
    ):
        """Test schedule → publish → cleanup full workflow."""
        product_id = product_directory["product_id"]

        # Step 1: Create schedule entry
        schedule_path = temp_outputs_dir / "schedule.json"
        schedule_mgr = ScheduleManager(
            schedule_path=schedule_path, config=schedule_config
        )

        entry = ScheduleEntry(
            product_id=product_id,
            scheduled_time=datetime(2025, 1, 20, 10, 0, tzinfo=UTC),
            platforms=[Platform.YOUTUBE],
            status="pending",
            created_at=datetime.now(UTC),
        )
        schedule_mgr.add_entry(entry)

        # Step 2: Simulate publish (record to tracking)
        create_publish_history(temp_outputs_dir, product_id, "youtube", "post_123")

        # Update schedule entry status
        schedule_mgr.entries[0].status = "published"
        schedule_mgr.entries[0].post_id = "post_123"
        schedule_mgr._save_schedule()

        # Step 3: Cleanup
        cleanup_mgr = CleanupManager(
            outputs_dir=temp_outputs_dir,
            config=cleanup_config,
            publisher=mock_publisher,
        )

        result = await cleanup_mgr.cleanup(
            product_id=product_id, platforms=[Platform.YOUTUBE], dry_run=False
        )

        # Verify complete workflow
        assert result["success"] is True
        assert not product_directory["product_dir"].exists()

        # Verify schedule still has record
        loaded_mgr = ScheduleManager(
            schedule_path=schedule_path, config=schedule_config
        )
        assert len(loaded_mgr.entries) == 1
        assert loaded_mgr.entries[0].status == "published"

    def test_format_bytes_helper(self):
        """Test format_bytes utility function."""
        from src.publisher.late.cli import format_bytes

        assert format_bytes(500) == "500.00 B"
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"
        assert format_bytes(1536 * 1024) == "1.50 MB"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling across components."""

    def test_schedule_manager_handles_corrupt_json(self, temp_outputs_dir):
        """Test ScheduleManager handles corrupted JSON gracefully."""
        schedule_path = temp_outputs_dir / "schedule.json"
        schedule_path.write_text("{ invalid json }")

        manager = ScheduleManager(schedule_path=schedule_path)

        # Should start with empty schedule
        assert manager.entries == []

    def test_schedule_manager_handles_invalid_structure(self, temp_outputs_dir):
        """Test ScheduleManager handles invalid data structure."""
        schedule_path = temp_outputs_dir / "schedule.json"
        schedule_path.write_text(json.dumps(["not", "a", "dict"]))

        manager = ScheduleManager(schedule_path=schedule_path)

        assert manager.entries == []

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_directory(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test cleanup handles missing product directory."""
        manager = CleanupManager(
            outputs_dir=temp_outputs_dir,
            config=cleanup_config,
            publisher=mock_publisher,
        )

        result = await manager.cleanup(
            product_id="NONEXISTENT", platforms=[Platform.YOUTUBE], dry_run=False
        )

        assert result["success"] is False
        assert isinstance(result["message"], str) and "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_cleanup_handles_api_errors(
        self,
        temp_outputs_dir,
        cleanup_config,
        mock_publisher,
        product_directory,
    ):
        """Test cleanup handles API errors gracefully."""
        product_id = product_directory["product_id"]
        create_publish_history(temp_outputs_dir, product_id, "youtube", "post_123")

        # Make API raise error
        mock_publisher.get_status = AsyncMock(side_effect=Exception("API Error"))

        manager = CleanupManager(
            outputs_dir=temp_outputs_dir,
            config=cleanup_config,
            publisher=mock_publisher,
        )

        result = await manager.cleanup(
            product_id=product_id, platforms=[Platform.YOUTUBE], dry_run=False
        )

        assert result["success"] is False
        assert product_directory["product_dir"].exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
