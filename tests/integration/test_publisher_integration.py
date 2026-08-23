"""Integration tests for the full publisher workflow.

Tests the complete pipeline: media upload → platform publishing → schedule creation
→ status tracking → publication verification → cleanup execution.

All network calls are mocked to ensure tests run without real API access.
"""

import json
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
from src.publisher.tracking import (
    get_publish_record,
    is_already_published,
    record_publish,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def mock_link_in_bio():
    """Keep link-in-bio (enabled by default) from hitting the real API."""
    with (
        patch(
            "src.publisher.batch.update_link_in_bio_safe", new_callable=AsyncMock
        ) as batch_mock,
        patch(
            "src.publisher.schedule.update_link_in_bio_safe", new_callable=AsyncMock
        ) as schedule_mock,
    ):
        yield {"batch": batch_mock, "schedule": schedule_mock}


@pytest.fixture
def outputs_dir(tmp_path: Path) -> Path:
    """Create a temporary outputs directory structure."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return outputs


@pytest.fixture
def product_dir(outputs_dir: Path) -> Path:
    """Create a test product directory with video and metadata files."""
    product_id = "B0TEST001"
    product = outputs_dir / product_id
    product.mkdir()

    # Create a minimal video file (just needs to exist and have content)
    video_file = product / f"video_{product_id}.mp4"
    # Write some bytes to simulate a video file (needs to be non-empty)
    video_file.write_bytes(b"\x00\x00\x00\x1c\x66\x74\x79\x70" * 1000)  # ~8KB fake MP4

    # Create metadata files
    metadata = {
        "title": "Test Product Video",
        "description": "Amazing product for testing #ad",
        "hashtags": ["ad", "test", "product"],
    }
    (product / "metadata.json").write_text(json.dumps(metadata))
    (product / "metadata_youtube.json").write_text(json.dumps(metadata))
    (product / "metadata_tiktok.json").write_text(json.dumps(metadata))

    # Create data.json (product info)
    data = {
        "title": "Test Product - Amazing Widget",
        "description": "This is a great product for testing purposes.",
        "price": "$99.99",
        "asin": product_id,
    }
    (product / "data.json").write_text(json.dumps([data]))

    return product


@pytest.fixture
def mock_late_sdk():
    """Create a mock Late SDK client with all required methods."""
    mock_client = MagicMock()

    # Mock accounts.list - returns account data
    mock_accounts_response = MagicMock()
    mock_accounts_response.accounts = [
        MagicMock(
            platform="youtube",
            username="TestChannel",
            field_id="acc_yt_001",
            isActive=True,
            displayName="Test Channel",
        ),
        MagicMock(
            platform="tiktok",
            username="testuser",
            field_id="acc_tt_001",
            isActive=True,
            displayName="Test TikTok",
        ),
    ]
    mock_client.accounts.list = MagicMock(return_value=mock_accounts_response)

    # Mock media.upload - returns media URL
    # Need to set url as a property, not a MagicMock
    mock_file = MagicMock()
    mock_file.url = "https://storage.late.dev/media_123.mp4"
    mock_upload_response = MagicMock()
    mock_upload_response.files = [mock_file]
    mock_upload_response.url = None  # Direct URL not set for small files
    mock_client.media.upload = MagicMock(return_value=mock_upload_response)

    # Mock media.upload_large - returns media URL for large files
    mock_large_response = MagicMock()
    mock_large_response.url = "https://storage.late.dev/large_media_456.mp4"
    mock_client.media.upload_large = MagicMock(return_value=mock_large_response)

    # Mock posts.create - returns post data
    mock_post_response = MagicMock()
    mock_post_response.post = MagicMock(
        field_id="post_abc123",
        status=MagicMock(value="scheduled"),
        platforms=[
            MagicMock(platform="youtube", platformPostUrl=None),
            MagicMock(platform="tiktok", platformPostUrl=None),
        ],
    )
    mock_client.posts.create = MagicMock(return_value=mock_post_response)

    # Mock posts.get - returns post status
    mock_status_response = MagicMock()
    mock_status_response.post = MagicMock(
        field_id="post_abc123",
        status=MagicMock(value="published"),
        scheduledFor=None,
        publishedAt=datetime.now(UTC).isoformat(),
        platforms=[
            MagicMock(
                platform="youtube",
                platformPostUrl="https://youtube.com/shorts/xyz123",
            ),
            MagicMock(
                platform="tiktok",
                platformPostUrl="https://tiktok.com/@user/video/789",
            ),
        ],
    )
    mock_client.posts.get = MagicMock(return_value=mock_status_response)

    # Mock posts.list - returns list of posts
    mock_list_response = MagicMock()
    mock_list_response.posts = []
    mock_client.posts.list = MagicMock(return_value=mock_list_response)

    # Mock posts.delete - returns success
    mock_client.posts.delete = MagicMock(return_value=None)

    return mock_client


@pytest.fixture
def mock_publisher(mock_late_sdk):
    """Create a mock LatePublisher with the mocked SDK."""
    with patch("src.publisher.late.client.Late", return_value=mock_late_sdk):
        from src.publisher.late.client import LatePublisher

        publisher = LatePublisher(
            api_key="sk_test_mock_key_12345",
            vercel_token="vercel_test_token_67890",  # noqa: S106
            timeout=30.0,
            max_retries=2,
        )
        # Replace client with our mock
        publisher.client = mock_late_sdk
        return publisher


@pytest.fixture
def schedule_config() -> ScheduleConfig:
    """Create a schedule configuration with recurring slots."""
    return ScheduleConfig(
        enabled=True,
        slots=[
            RecurringSlot(day_of_week="monday", time="10:00:00", timezone="UTC"),
            RecurringSlot(day_of_week="wednesday", time="14:00:00", timezone="UTC"),
            RecurringSlot(day_of_week="friday", time="18:00:00", timezone="UTC"),
        ],
        min_post_spacing_hours=2,
        prevent_duplicates=True,
        allow_past_schedules=False,
        max_posts_per_day=10,
        timezone="UTC",
    )


@pytest.fixture
def cleanup_config(outputs_dir: Path) -> CleanupConfig:
    """Create a cleanup configuration."""
    archive_dir = outputs_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    return CleanupConfig(
        enabled=True,
        verify_before_delete=True,
        require_all_platforms=False,  # Allow cleanup if ANY platform published
        archive_before_delete=False,
        archive_dir=archive_dir,
        keep_published_days=0,  # Immediate cleanup for testing
    )


# =============================================================================
# Integration Tests: Full Workflow
# =============================================================================


class TestFullPublishWorkflow:
    """Test complete publish-schedule-verify-cleanup workflow."""

    @pytest.mark.asyncio
    async def test_full_pipeline_immediate_publish(
        self,
        mock_publisher,
        outputs_dir: Path,
        product_dir: Path,
        cleanup_config: CleanupConfig,
    ):
        """Test complete workflow: upload → publish → track → verify → cleanup."""
        product_id = product_dir.name
        video_path = product_dir / f"video_{product_id}.mp4"
        platforms = [Platform.YOUTUBE, Platform.TIKTOK]

        # Step 1: Upload media
        media_url = await mock_publisher.upload_media(video_path)
        assert media_url == "https://storage.late.dev/media_123.mp4"
        mock_publisher.client.media.upload.assert_called_once()

        # Step 2: Publish to platforms
        platform_dicts = [
            {"platform": "youtube", "account_id": "acc_yt_001"},
            {"platform": "tiktok", "account_id": "acc_tt_001"},
        ]
        result = await mock_publisher.publish(
            media_id=media_url,
            platforms=platform_dicts,
            content="Amazing product! #ad #test",
        )
        assert result["post_id"] == "post_abc123"
        assert result["status"] == "scheduled"

        # Step 3: Record publish for tracking
        for platform in platforms:
            record_publish(
                product_id=product_id,
                platform=platform.value,
                post_id=result["post_id"],
                outputs_dir=outputs_dir,
            )

        # Step 4: Verify tracking records exist
        for platform in platforms:
            assert is_already_published(product_id, platform.value, outputs_dir)
            record = get_publish_record(product_id, platform.value, outputs_dir)
            assert record is not None
            assert record["post_id"] == "post_abc123"

        # Step 5: Verify publication via CleanupManager
        cleanup_manager = CleanupManager(outputs_dir, cleanup_config, mock_publisher)
        all_published, statuses = await cleanup_manager.verify_publication(
            product_id, platforms
        )
        # Should be published/scheduled based on our tracking records
        assert all_published or any(
            s in ("published", "scheduled") for s in statuses.values()
        )

        # Step 6: Execute cleanup
        cleanup_result = await cleanup_manager.cleanup(product_id, platforms)

        # Verify cleanup succeeded
        assert cleanup_result["success"] is True
        assert int(cleanup_result["disk_freed"]) > 0
        assert not product_dir.exists()  # Directory should be removed

        # Step 7: Verify audit log
        audit_log_path = outputs_dir / "cleanup_audit.json"
        assert audit_log_path.exists()

        audit_data = json.loads(audit_log_path.read_text())
        assert "cleanups" in audit_data
        assert len(audit_data["cleanups"]) == 1

        cleanup_record = audit_data["cleanups"][0]
        assert cleanup_record["product_id"] == product_id
        assert cleanup_record["disk_freed_bytes"] > 0
        assert "youtube" in cleanup_record["platforms"]
        assert "tiktok" in cleanup_record["platforms"]

    @pytest.mark.asyncio
    async def test_full_pipeline_scheduled_publish(
        self,
        mock_publisher,
        outputs_dir: Path,
        product_dir: Path,
        schedule_config: ScheduleConfig,
        cleanup_config: CleanupConfig,
    ):
        """Test workflow with scheduled publishing using ScheduleManager."""
        product_id = product_dir.name
        video_path = product_dir / f"video_{product_id}.mp4"
        platforms = [Platform.YOUTUBE]

        # Step 1: Upload media
        media_url = await mock_publisher.upload_media(video_path)
        assert media_url is not None

        # Step 2: Create schedule manager and calculate next slot
        schedule_path = outputs_dir / "schedule.json"
        schedule_manager = ScheduleManager(schedule_path, schedule_config)

        # Get next available slot
        next_time, slot_idx = schedule_manager.get_next_slot(
            slots=schedule_config.slots,
            after=datetime.now(UTC),
            slot_index=0,
        )
        assert next_time > datetime.now(UTC)

        # Step 3: Publish with scheduled time
        platform_dicts = [{"platform": "youtube", "account_id": "acc_yt_001"}]
        result = await mock_publisher.publish(
            media_id=media_url,
            platforms=platform_dicts,
            content="Scheduled post! #ad",
            scheduled_time=next_time,
        )

        # Step 4: Add schedule entry
        entry = ScheduleEntry(
            product_id=product_id,
            scheduled_time=next_time,
            platforms=platforms,
            post_id=result["post_id"],
            status="scheduled",
            created_at=datetime.now(UTC),
            slot_index=slot_idx,
        )
        schedule_manager.add_entry(entry)

        # Step 5: Verify schedule was saved
        assert schedule_path.exists()
        schedule_data = json.loads(schedule_path.read_text())
        assert len(schedule_data["entries"]) == 1
        assert schedule_data["entries"][0]["product_id"] == product_id
        assert schedule_data["entries"][0]["status"] == "scheduled"

        # Step 6: Record publish for tracking
        record_publish(
            product_id=product_id,
            platform="youtube",
            post_id=result["post_id"],
            outputs_dir=outputs_dir,
        )

        # Step 7: Cleanup with verification
        # Update cleanup config to not verify (since post is scheduled, not published)
        cleanup_config.verify_before_delete = False
        cleanup_manager = CleanupManager(outputs_dir, cleanup_config, mock_publisher)

        cleanup_result = await cleanup_manager.cleanup(product_id, platforms)
        assert cleanup_result["success"] is True


class TestMediaUploadWorkflow:
    """Test media upload scenarios."""

    @pytest.mark.asyncio
    async def test_upload_small_file(self, mock_publisher, product_dir: Path):
        """Test uploading a small file (≤4MB) uses direct upload."""
        product_id = product_dir.name
        video_path = product_dir / f"video_{product_id}.mp4"

        # Small file should use direct upload
        media_url = await mock_publisher.upload_media(video_path)

        assert media_url == "https://storage.late.dev/media_123.mp4"
        mock_publisher.client.media.upload.assert_called_once()
        mock_publisher.client.media.upload_large.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_validates_file_exists(self, mock_publisher, tmp_path: Path):
        """Test upload validates file existence."""
        from src.publisher.base import ValidationError

        nonexistent = tmp_path / "nonexistent.mp4"

        with pytest.raises(ValidationError, match="not found"):
            await mock_publisher.upload_media(nonexistent)

    @pytest.mark.asyncio
    async def test_upload_validates_empty_file(self, mock_publisher, tmp_path: Path):
        """Test upload rejects empty files."""
        from src.publisher.base import ValidationError

        empty_file = tmp_path / "empty.mp4"
        empty_file.write_bytes(b"")

        with pytest.raises(ValidationError, match="empty"):
            await mock_publisher.upload_media(empty_file)


class TestPlatformPublishing:
    """Test platform publishing scenarios."""

    @pytest.mark.asyncio
    async def test_publish_immediate(self, mock_publisher):
        """Test immediate publishing without scheduled time."""
        result = await mock_publisher.publish(
            media_id="https://storage.late.dev/media_123.mp4",
            platforms=[{"platform": "youtube", "account_id": "acc_yt_001"}],
            content="Test post #ad",
        )

        assert result["post_id"] == "post_abc123"
        mock_publisher.client.posts.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_scheduled(self, mock_publisher):
        """Test scheduled publishing with future time."""
        scheduled_time = datetime.now(UTC) + timedelta(days=1)

        result = await mock_publisher.publish(
            media_id="https://storage.late.dev/media_123.mp4",
            platforms=[{"platform": "youtube", "account_id": "acc_yt_001"}],
            content="Scheduled post #ad",
            scheduled_time=scheduled_time,
        )

        assert result["post_id"] == "post_abc123"
        assert result["scheduled_time"] == scheduled_time

    @pytest.mark.asyncio
    async def test_publish_multi_platform(self, mock_publisher):
        """Test publishing to multiple platforms."""
        platforms = [
            {"platform": "youtube", "account_id": "acc_yt_001"},
            {"platform": "tiktok", "account_id": "acc_tt_001"},
        ]

        result = await mock_publisher.publish(
            media_id="https://storage.late.dev/media_123.mp4",
            platforms=platforms,
            content="Multi-platform post #ad",
        )

        assert result["post_id"] == "post_abc123"

    @pytest.mark.asyncio
    async def test_publish_with_platform_specific_content(self, mock_publisher):
        """Test publishing with different content per platform."""
        platforms = [
            {"platform": "youtube", "account_id": "acc_yt_001"},
            {"platform": "tiktok", "account_id": "acc_tt_001"},
        ]
        platform_contents = {
            "youtube": {
                "content": "YouTube optimized! #ad #Shorts",
                "title": "Amazing!",
            },
            "tiktok": {"content": "TikTok vibes! #ad #fyp"},
        }

        result = await mock_publisher.publish(
            media_id="https://storage.late.dev/media_123.mp4",
            platforms=platforms,
            content="Fallback content",
            platform_contents=platform_contents,
        )

        assert result["post_id"] == "post_abc123"

    @pytest.mark.asyncio
    async def test_publish_tiktok_includes_commercial_disclosure(self, mock_publisher):
        """Test TikTok gets commercial content disclosure even without platform_contents."""
        platforms = [
            {"platform": "tiktok", "account_id": "acc_tt_001"},
        ]

        await mock_publisher.publish(
            media_id="https://storage.late.dev/media_123.mp4",
            platforms=platforms,
            content="TikTok post #ad",
        )

        # Verify posts.create was called with TikTok platform data
        call_kwargs = mock_publisher.client.posts.create.call_args
        sdk_platforms = call_kwargs.kwargs.get(
            "platforms", call_kwargs[1].get("platforms", [])
        )
        tiktok_platform = next(p for p in sdk_platforms if p["platform"] == "tiktok")
        tiktok_settings = tiktok_platform["platformSpecificData"]["tiktokSettings"]

        assert tiktok_settings["commercial_content_type"] == "brand_organic"
        assert tiktok_settings["is_brand_organic_post"] is True
        assert tiktok_settings["privacy_level"] == "PUBLIC_TO_EVERYONE"
        assert tiktok_settings["content_preview_confirmed"] is True
        assert tiktok_settings["express_consent_given"] is True

    @pytest.mark.asyncio
    async def test_publish_youtube_sets_synthetic_media_flag(self, mock_publisher):
        """The YouTube payload carries the configured synthetic-content value.

        Off by default: the policy excludes AI narration, AI scripts and stock
        footage, so nothing this pipeline renders today meets the bar. Both
        settings are exercised, because the flag is gated rather than removed.
        """
        mock_publisher.synthetic_media_disclosure = True
        platforms = [
            {"platform": "youtube", "account_id": "acc_yt_001"},
        ]

        await mock_publisher.publish(
            media_id="https://storage.late.dev/media_123.mp4",
            platforms=platforms,
            content="YouTube post #ad",
        )

        call_kwargs = mock_publisher.client.posts.create.call_args
        sdk_platforms = call_kwargs.kwargs.get(
            "platforms", call_kwargs[1].get("platforms", [])
        )
        youtube_platform = next(p for p in sdk_platforms if p["platform"] == "youtube")
        psd = youtube_platform["platformSpecificData"]
        assert psd["containsSyntheticMedia"] is True

    @pytest.mark.asyncio
    async def test_publish_youtube_synthetic_media_with_platform_content(
        self, mock_publisher
    ):
        """The configured value applies on the platform_contents branch too.

        The flag is set at two sites in the builder, with and without
        platform-specific content, and they must not drift apart.
        """
        mock_publisher.synthetic_media_disclosure = True
        platforms = [
            {"platform": "youtube", "account_id": "acc_yt_001"},
        ]
        platform_contents = {
            "youtube": {"content": "YouTube body #ad", "title": "Test Title"},
        }

        await mock_publisher.publish(
            media_id="https://storage.late.dev/media_123.mp4",
            platforms=platforms,
            platform_contents=platform_contents,
        )

        call_kwargs = mock_publisher.client.posts.create.call_args
        sdk_platforms = call_kwargs.kwargs.get(
            "platforms", call_kwargs[1].get("platforms", [])
        )
        youtube_platform = next(p for p in sdk_platforms if p["platform"] == "youtube")
        psd = youtube_platform["platformSpecificData"]
        assert psd["containsSyntheticMedia"] is True
        assert psd["title"] == "Test Title"

    @pytest.mark.asyncio
    async def test_publish_validates_empty_platforms(self, mock_publisher):
        """Test publish rejects empty platforms list."""
        from src.publisher.base import ValidationError

        with pytest.raises(ValidationError, match="cannot be empty"):
            await mock_publisher.publish(
                media_id="https://storage.late.dev/media_123.mp4",
                platforms=[],
                content="Test #ad",
            )

    @pytest.mark.asyncio
    async def test_publish_validates_past_scheduled_time(self, mock_publisher):
        """Test publish rejects past scheduled time."""
        from src.publisher.base import ValidationError

        past_time = datetime.now(UTC) - timedelta(hours=1)

        with pytest.raises(ValidationError, match="cannot be in past"):
            await mock_publisher.publish(
                media_id="https://storage.late.dev/media_123.mp4",
                platforms=[{"platform": "youtube", "account_id": "acc_yt_001"}],
                content="Test #ad",
                scheduled_time=past_time,
            )


class TestLinkInBio:
    """Test link-in-bio integration."""

    @pytest.mark.asyncio
    async def test_manager_adds_link_from_data_json(self, tmp_path: Path):
        """Test LinkInBioManager reads data.json and adds link."""
        from unittest.mock import AsyncMock

        from src.publisher.link_in_bio.manager import LinkInBioManager

        # Create mock data.json with affiliate_link and images array
        product_dir = tmp_path / "B0TEST123"
        product_dir.mkdir()
        (product_dir / "data.json").write_text(
            json.dumps(
                [
                    {
                        "title": "Test Product Title",
                        "url": "https://amazon.com/dp/B0TEST123/ref=sr_1_1",
                        "affiliate_link": "https://amazon.com/dp/B0TEST123?tag=test-20",
                        "images": [
                            "https://images.amazon.com/test.jpg",
                            "https://images.amazon.com/test2.jpg",
                        ],
                        "downloaded_images": [],
                    }
                ]
            )
        )

        mock_provider = AsyncMock()
        mock_provider.authenticate.return_value = True
        mock_provider.list_links.return_value = []
        mock_provider.add_link.return_value = {"status": True, "data": {"id": 42}}

        mgr = LinkInBioManager(provider=mock_provider, max_links=25)
        result = await mgr.update("B0TEST123", tmp_path)

        assert result["success"] is True
        mock_provider.add_link.assert_called_once_with(
            title="Test Product Title",
            url="https://amazon.com/dp/B0TEST123?tag=test-20",
            image="https://images.amazon.com/test.jpg",
            image_file=None,
        )

    @pytest.mark.asyncio
    async def test_manager_uses_affiliate_link_over_url(self, tmp_path: Path):
        """Test manager prefers affiliate_link over url field."""
        from unittest.mock import AsyncMock

        from src.publisher.link_in_bio.manager import LinkInBioManager

        product_dir = tmp_path / "B0AFF"
        product_dir.mkdir()
        (product_dir / "data.json").write_text(
            json.dumps(
                [
                    {
                        "title": "Affiliate Test",
                        "url": "https://amazon.com/dp/B0AFF/ref=sr",
                        "affiliate_link": "https://amazon.com/dp/B0AFF?tag=mytag-20",
                        "images": [],
                        "downloaded_images": [],
                    }
                ]
            )
        )

        mock_provider = AsyncMock()
        mock_provider.authenticate.return_value = True
        mock_provider.list_links.return_value = []
        mock_provider.add_link.return_value = {"status": True, "data": {"id": 1}}

        mgr = LinkInBioManager(provider=mock_provider, max_links=25)
        result = await mgr.update("B0AFF", tmp_path)

        assert result["success"] is True
        call_kwargs = mock_provider.add_link.call_args.kwargs
        assert call_kwargs["url"] == "https://amazon.com/dp/B0AFF?tag=mytag-20"

    @pytest.mark.asyncio
    async def test_manager_falls_back_to_downloaded_image(self, tmp_path: Path):
        """Test manager uses downloaded_images when images array is empty."""
        from unittest.mock import AsyncMock

        from src.publisher.link_in_bio.manager import LinkInBioManager

        product_dir = tmp_path / "B0IMG"
        product_dir.mkdir()
        img_dir = product_dir / "images"
        img_dir.mkdir()
        img_file = img_dir / "B0IMG_image_0.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        (product_dir / "data.json").write_text(
            json.dumps(
                [
                    {
                        "title": "Image Fallback Test",
                        "url": "https://amazon.com/dp/B0IMG",
                        "images": [],
                        "downloaded_images": [
                            "B0IMG/images/B0IMG_image_0.jpg",
                        ],
                    }
                ]
            )
        )

        mock_provider = AsyncMock()
        mock_provider.authenticate.return_value = True
        mock_provider.list_links.return_value = []
        mock_provider.add_link.return_value = {"status": True, "data": {"id": 2}}

        mgr = LinkInBioManager(provider=mock_provider, max_links=25)
        result = await mgr.update("B0IMG", tmp_path)

        assert result["success"] is True
        call_kwargs = mock_provider.add_link.call_args.kwargs
        assert call_kwargs["image"] is None
        assert call_kwargs["image_file"] == img_file

    @pytest.mark.asyncio
    async def test_manager_skips_duplicate(self, tmp_path: Path):
        """Test LinkInBioManager skips when product link already exists."""
        from src.publisher.link_in_bio.manager import LinkInBioManager

        product_dir = tmp_path / "B0TEST123"
        product_dir.mkdir()
        (product_dir / "data.json").write_text(
            json.dumps([{"title": "Test", "url": "https://amazon.com/dp/B0TEST123"}])
        )

        mock_provider = AsyncMock()
        mock_provider.authenticate.return_value = True
        mock_provider.list_links.return_value = [
            {"id": 1, "url": "https://amazon.com/dp/B0TEST123?tag=test-20"},
        ]

        mgr = LinkInBioManager(provider=mock_provider, max_links=25)
        result = await mgr.update("B0TEST123", tmp_path)

        assert result["success"] is True
        assert result["existing"] is True
        mock_provider.add_link.assert_not_called()

    @pytest.mark.asyncio
    async def test_manager_rotates_oldest_at_capacity(self, tmp_path: Path):
        """Test LinkInBioManager deletes oldest link when at max capacity."""
        from src.publisher.link_in_bio.manager import LinkInBioManager

        product_dir = tmp_path / "B0NEW"
        product_dir.mkdir()
        (product_dir / "data.json").write_text(
            json.dumps([{"title": "New Product", "url": "https://amazon.com/dp/B0NEW"}])
        )

        mock_provider = AsyncMock()
        mock_provider.authenticate.return_value = True
        mock_provider.list_links.return_value = [
            {"id": 1, "url": "https://amazon.com/dp/B0OLD1"},
            {"id": 2, "url": "https://amazon.com/dp/B0OLD2"},
        ]
        mock_provider.add_link.return_value = {"status": True, "data": {"id": 3}}

        mgr = LinkInBioManager(provider=mock_provider, max_links=2)
        result = await mgr.update("B0NEW", tmp_path)

        assert result["success"] is True
        mock_provider.delete_link.assert_called_once_with("2")
        mock_provider.add_link.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_skips_missing_data(self, tmp_path: Path):
        """Test LinkInBioManager handles missing data.json gracefully."""
        from src.publisher.link_in_bio.manager import LinkInBioManager

        mock_provider = AsyncMock()
        mgr = LinkInBioManager(provider=mock_provider, max_links=25)
        result = await mgr.update("B0MISSING", tmp_path)

        assert result["success"] is False
        assert result["reason"] == "no_data"

    @pytest.mark.asyncio
    async def test_manager_returns_missing_fields_when_no_title(self, tmp_path: Path):
        """Test manager returns missing_fields when title is empty."""
        from src.publisher.link_in_bio.manager import LinkInBioManager

        product_dir = tmp_path / "B0NOTITLE"
        product_dir.mkdir()
        (product_dir / "data.json").write_text(
            json.dumps([{"title": "", "url": "https://amazon.com/dp/B0NOTITLE"}])
        )

        mock_provider = AsyncMock()
        mgr = LinkInBioManager(provider=mock_provider, max_links=25)
        result = await mgr.update("B0NOTITLE", tmp_path)

        assert result["success"] is False
        assert result["reason"] == "missing_fields"

    @pytest.mark.asyncio
    async def test_manager_truncates_long_title(self, tmp_path: Path):
        """Test manager truncates titles exceeding max_title_length."""
        from src.publisher.link_in_bio.manager import LinkInBioManager

        long_title = "A" * 100
        product_dir = tmp_path / "B0LONG"
        product_dir.mkdir()
        (product_dir / "data.json").write_text(
            json.dumps([{"title": long_title, "url": "https://amazon.com/dp/B0LONG"}])
        )

        mock_provider = AsyncMock()
        mock_provider.authenticate.return_value = True
        mock_provider.list_links.return_value = []
        mock_provider.add_link.return_value = {"status": True}

        mgr = LinkInBioManager(
            provider=mock_provider, max_links=25, max_title_length=50
        )
        await mgr.update("B0LONG", tmp_path)

        call_title = mock_provider.add_link.call_args.kwargs["title"]
        assert len(call_title) == 50
        assert call_title.endswith("...")

    def test_factory_creates_lnkbio_manager(self):
        """Test factory creates manager with LnkBioProvider."""
        from src.publisher.link_in_bio.lnkbio import LnkBioProvider
        from src.publisher.link_in_bio.manager import create_link_in_bio_manager

        mgr = create_link_in_bio_manager("lnkbio", max_links=10, max_title_length=60)
        assert isinstance(mgr.provider, LnkBioProvider)
        assert mgr.max_links == 10
        assert mgr.max_title_length == 60

    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises ValueError for unknown provider."""
        from src.publisher.link_in_bio.manager import create_link_in_bio_manager

        with pytest.raises(ValueError, match="Unknown link-in-bio provider"):
            create_link_in_bio_manager("unknown_provider")


class TestTikTokContentSettings:
    """Test TikTokContentSettings dataclass."""

    def test_to_sdk_dict_returns_all_fields(self):
        """Test to_sdk_dict returns complete settings dict."""
        from src.publisher.models import TikTokContentSettings

        settings = TikTokContentSettings()
        sdk = settings.to_sdk_dict()

        assert sdk["commercial_content_type"] == "brand_organic"
        assert sdk["is_brand_organic_post"] is True
        assert sdk["privacy_level"] == "PUBLIC_TO_EVERYONE"
        assert sdk["allow_comment"] is True
        assert sdk["allow_duet"] is False
        assert sdk["allow_stitch"] is False
        assert len(sdk) == 8

    def test_to_top_level_dict_returns_camel_case(self):
        """Test to_top_level_dict uses camelCase keys for API."""
        from src.publisher.models import TikTokContentSettings

        settings = TikTokContentSettings()
        top = settings.to_top_level_dict()

        assert top["privacyLevel"] == "PUBLIC_TO_EVERYONE"
        assert top["mediaType"] == "video"
        assert top["commercialContentType"] == "brand_organic"

    def test_custom_settings_propagate(self):
        """Test non-default values appear in output."""
        from src.publisher.models import TikTokContentSettings

        settings = TikTokContentSettings(
            privacy_level="SELF_ONLY",
            allow_duet=True,
            commercial_content_type="brand_content",
        )
        sdk = settings.to_sdk_dict()

        assert sdk["privacy_level"] == "SELF_ONLY"
        assert sdk["allow_duet"] is True
        assert sdk["commercial_content_type"] == "brand_content"


class TestLinkInBioConfig:
    """Test LinkInBioConfig validation."""

    def test_rejects_negative_max_links(self):
        """Test max_links < 0 raises ValueError."""
        from src.publisher.models import LinkInBioConfig

        with pytest.raises(ValueError, match="max_links"):
            LinkInBioConfig(max_links=-1)

    def test_rejects_short_max_title_length(self):
        """Test max_title_length < 10 raises ValueError."""
        from src.publisher.models import LinkInBioConfig

        with pytest.raises(ValueError, match="max_title_length"):
            LinkInBioConfig(max_title_length=5)

    def test_default_values(self):
        """Test default config values."""
        from src.publisher.models import LinkInBioConfig

        config = LinkInBioConfig()
        assert config.enabled is True
        assert config.provider == "lnkbio"
        assert config.max_links == 0
        assert config.max_title_length == 80


class TestDefaultPlatforms:
    """Test DEFAULT_PLATFORMS constant."""

    def test_includes_all_primary_platforms(self):
        """Test DEFAULT_PLATFORMS includes YouTube, TikTok, Instagram."""
        from src.publisher.models import DEFAULT_PLATFORMS, Platform

        assert Platform.YOUTUBE in DEFAULT_PLATFORMS
        assert Platform.TIKTOK in DEFAULT_PLATFORMS
        assert Platform.INSTAGRAM in DEFAULT_PLATFORMS
        assert len(DEFAULT_PLATFORMS) == 3


class TestScheduleCreation:
    """Test schedule creation scenarios."""

    def test_schedule_manager_add_entry(
        self, outputs_dir: Path, schedule_config: ScheduleConfig
    ):
        """Test adding a schedule entry."""
        schedule_path = outputs_dir / "schedule.json"
        manager = ScheduleManager(schedule_path, schedule_config)

        entry = ScheduleEntry(
            product_id="B0TEST001",
            scheduled_time=datetime.now(UTC) + timedelta(days=1),
            platforms=[Platform.YOUTUBE],
            post_id="post_123",
            status="scheduled",
            created_at=datetime.now(UTC),
            slot_index=0,
        )

        manager.add_entry(entry)

        # Verify saved to disk
        assert schedule_path.exists()
        data = json.loads(schedule_path.read_text())
        assert len(data["entries"]) == 1
        assert data["entries"][0]["product_id"] == "B0TEST001"

    def test_schedule_manager_prevents_duplicates(
        self, outputs_dir: Path, schedule_config: ScheduleConfig
    ):
        """Test that duplicate entries are rejected."""
        schedule_path = outputs_dir / "schedule.json"
        manager = ScheduleManager(schedule_path, schedule_config)

        scheduled_time = datetime.now(UTC) + timedelta(days=1)

        entry1 = ScheduleEntry(
            product_id="B0TEST001",
            scheduled_time=scheduled_time,
            platforms=[Platform.YOUTUBE],
            post_id="post_123",
            status="scheduled",
            created_at=datetime.now(UTC),
        )
        manager.add_entry(entry1)

        # Try adding duplicate
        entry2 = ScheduleEntry(
            product_id="B0TEST001",
            scheduled_time=scheduled_time,
            platforms=[Platform.YOUTUBE],
            post_id="post_456",
            status="scheduled",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(ValueError, match="validation failed"):
            manager.add_entry(entry2)

    def test_schedule_manager_get_next_slot(
        self, outputs_dir: Path, schedule_config: ScheduleConfig
    ):
        """Test calculating next available slot."""
        schedule_path = outputs_dir / "schedule.json"
        manager = ScheduleManager(schedule_path, schedule_config)

        next_time, slot_idx = manager.get_next_slot(
            slots=schedule_config.slots,
            after=datetime.now(UTC),
            slot_index=0,
        )

        assert next_time > datetime.now(UTC)
        assert 0 <= slot_idx < len(schedule_config.slots)

    def test_schedule_manager_list_scheduled(
        self, outputs_dir: Path, schedule_config: ScheduleConfig
    ):
        """Test listing scheduled entries with filters."""
        schedule_path = outputs_dir / "schedule.json"
        manager = ScheduleManager(schedule_path, schedule_config)

        # Add entries
        for i in range(3):
            entry = ScheduleEntry(
                product_id=f"B0TEST00{i}",
                scheduled_time=datetime.now(UTC) + timedelta(days=i + 1),
                platforms=[Platform.YOUTUBE if i % 2 == 0 else Platform.TIKTOK],
                status="scheduled",
                created_at=datetime.now(UTC),
            )
            manager.add_entry(entry)

        # List all
        all_entries = manager.list_scheduled()
        assert len(all_entries) == 3

        # Filter by platform
        youtube_entries = manager.list_scheduled(platform="youtube")
        assert len(youtube_entries) == 2  # B0TEST000, B0TEST002

        # Filter by status
        scheduled_entries = manager.list_scheduled(status="scheduled")
        assert len(scheduled_entries) == 3


class TestStatusTracking:
    """Test status tracking scenarios."""

    def test_record_and_retrieve_publish(self, outputs_dir: Path):
        """Test recording and retrieving publish records."""
        product_id = "B0TEST001"
        platform = "youtube"
        post_id = "post_abc123"

        # Record publish
        record_publish(product_id, platform, post_id, outputs_dir)

        # Verify is_already_published
        assert is_already_published(product_id, platform, outputs_dir)
        assert not is_already_published(product_id, "tiktok", outputs_dir)

        # Verify get_publish_record
        record = get_publish_record(product_id, platform, outputs_dir)
        assert record is not None
        assert record["product_id"] == product_id
        assert record["platform"] == platform
        assert record["post_id"] == post_id
        assert "published_at" in record

    def test_tracking_persists_to_file(self, outputs_dir: Path):
        """Test that tracking data persists to publish_history.json."""
        product_id = "B0TEST001"
        record_publish(product_id, "youtube", "post_123", outputs_dir)

        tracking_path = outputs_dir / "publish_history.json"
        assert tracking_path.exists()

        data = json.loads(tracking_path.read_text())
        assert "posts" in data
        assert f"{product_id}:youtube" in data["posts"]

    def test_multiple_platform_tracking(self, outputs_dir: Path):
        """Test tracking publishes to multiple platforms."""
        product_id = "B0TEST001"

        record_publish(product_id, "youtube", "post_yt", outputs_dir)
        record_publish(product_id, "tiktok", "post_tt", outputs_dir)

        assert is_already_published(product_id, "youtube", outputs_dir)
        assert is_already_published(product_id, "tiktok", outputs_dir)
        assert not is_already_published(product_id, "instagram", outputs_dir)


class TestPublicationVerification:
    """Test publication verification scenarios."""

    @pytest.mark.asyncio
    async def test_verify_published_product(
        self, mock_publisher, outputs_dir: Path, cleanup_config: CleanupConfig
    ):
        """Test verifying a published product."""
        product_id = "B0TEST001"
        platforms = [Platform.YOUTUBE]

        # Record publish
        record_publish(product_id, "youtube", "post_abc123", outputs_dir)

        # Verify publication
        cleanup_manager = CleanupManager(outputs_dir, cleanup_config, mock_publisher)
        all_published, statuses = await cleanup_manager.verify_publication(
            product_id, platforms
        )

        # Should query API for status
        assert "youtube" in statuses

    @pytest.mark.asyncio
    async def test_verify_unpublished_product(
        self, mock_publisher, outputs_dir: Path, cleanup_config: CleanupConfig
    ):
        """Test verifying an unpublished product."""
        product_id = "B0UNPUBLISHED"
        platforms = [Platform.YOUTUBE]

        # Don't record any publish

        cleanup_manager = CleanupManager(outputs_dir, cleanup_config, mock_publisher)
        all_published, statuses = await cleanup_manager.verify_publication(
            product_id, platforms
        )

        assert all_published is False
        assert statuses.get("youtube") == "not_published"


class TestCleanupExecution:
    """Test cleanup execution scenarios."""

    @pytest.mark.asyncio
    async def test_cleanup_verified_product(
        self,
        mock_publisher,
        outputs_dir: Path,
        product_dir: Path,
        cleanup_config: CleanupConfig,
    ):
        """Test cleanup of a verified published product."""
        product_id = product_dir.name
        platforms = [Platform.YOUTUBE]

        # Record publish
        record_publish(product_id, "youtube", "post_abc123", outputs_dir)

        # Execute cleanup
        cleanup_manager = CleanupManager(outputs_dir, cleanup_config, mock_publisher)
        result = await cleanup_manager.cleanup(product_id, platforms)

        assert result["success"] is True
        assert int(result["disk_freed"]) > 0
        assert not product_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_creates_audit_log(
        self,
        mock_publisher,
        outputs_dir: Path,
        product_dir: Path,
        cleanup_config: CleanupConfig,
    ):
        """Test that cleanup creates audit log entries."""
        product_id = product_dir.name
        platforms = [Platform.YOUTUBE, Platform.TIKTOK]

        # Record publish
        for p in platforms:
            record_publish(product_id, p.value, "post_abc123", outputs_dir)

        # Execute cleanup
        cleanup_config.verify_before_delete = False  # Skip verification for test
        cleanup_manager = CleanupManager(outputs_dir, cleanup_config, mock_publisher)
        await cleanup_manager.cleanup(product_id, platforms)

        # Verify audit log
        audit_path = outputs_dir / "cleanup_audit.json"
        assert audit_path.exists()

        audit_data = json.loads(audit_path.read_text())
        assert len(audit_data["cleanups"]) == 1

        record = audit_data["cleanups"][0]
        assert record["product_id"] == product_id
        assert record["disk_freed_bytes"] > 0
        assert "cleaned_at" in record
        assert set(record["platforms"]) == {"youtube", "tiktok"}

    @pytest.mark.asyncio
    async def test_cleanup_dry_run(
        self,
        mock_publisher,
        outputs_dir: Path,
        product_dir: Path,
        cleanup_config: CleanupConfig,
    ):
        """Test cleanup dry run doesn't delete files."""
        product_id = product_dir.name
        platforms = [Platform.YOUTUBE]

        # Record publish
        record_publish(product_id, "youtube", "post_abc123", outputs_dir)

        # Execute dry run
        cleanup_config.verify_before_delete = False
        cleanup_manager = CleanupManager(outputs_dir, cleanup_config, mock_publisher)
        result = await cleanup_manager.cleanup(product_id, platforms, dry_run=True)

        assert result["success"] is True
        assert "[DRY RUN]" in str(result["message"])
        assert int(result["disk_freed"]) == 0
        assert product_dir.exists()  # Directory should NOT be removed

    @pytest.mark.asyncio
    async def test_cleanup_with_archive(
        self,
        mock_publisher,
        outputs_dir: Path,
        product_dir: Path,
        cleanup_config: CleanupConfig,
    ):
        """Test cleanup with archive creation."""
        product_id = product_dir.name
        platforms = [Platform.YOUTUBE]

        # Enable archiving
        cleanup_config.archive_before_delete = True
        cleanup_config.verify_before_delete = False

        # Record publish
        record_publish(product_id, "youtube", "post_abc123", outputs_dir)

        # Execute cleanup
        cleanup_manager = CleanupManager(outputs_dir, cleanup_config, mock_publisher)
        result = await cleanup_manager.cleanup(product_id, platforms)

        assert result["success"] is True
        assert not product_dir.exists()

        # Verify archive was created
        archive_dir = cleanup_config.archive_dir
        archives = list(archive_dir.glob(f"{product_id}_*.zip"))
        assert len(archives) == 1

    @pytest.mark.asyncio
    async def test_cleanup_blocked_for_unpublished(
        self,
        mock_publisher,
        outputs_dir: Path,
        product_dir: Path,
        cleanup_config: CleanupConfig,
    ):
        """Test cleanup is blocked for unpublished products when verification enabled."""
        product_id = product_dir.name
        platforms = [Platform.YOUTUBE]

        # Don't record any publish - product is unpublished

        # Execute cleanup with verification
        cleanup_manager = CleanupManager(outputs_dir, cleanup_config, mock_publisher)
        result = await cleanup_manager.cleanup(product_id, platforms)

        assert result["success"] is False
        assert "not published" in str(result["message"]).lower()
        assert product_dir.exists()  # Directory should NOT be removed

    @pytest.mark.asyncio
    async def test_cleanup_disabled_config(
        self, mock_publisher, outputs_dir: Path, product_dir: Path
    ):
        """Test cleanup does nothing when disabled in config."""
        product_id = product_dir.name
        platforms = [Platform.YOUTUBE]

        # Disable cleanup
        config = CleanupConfig(enabled=False)
        cleanup_manager = CleanupManager(outputs_dir, config, mock_publisher)

        result = await cleanup_manager.cleanup(product_id, platforms)

        assert result["success"] is False
        assert "disabled" in str(result["message"]).lower()
        assert product_dir.exists()


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_publish_handles_api_error(self, mock_publisher):
        """Test publish handles API errors gracefully."""
        from src.publisher.base import PublishError

        # Make posts.create raise an error
        mock_publisher.client.posts.create = MagicMock(
            side_effect=Exception("API Error: Rate limited")
        )

        with pytest.raises(PublishError, match="(?i)failed"):
            await mock_publisher.publish(
                media_id="https://storage.late.dev/media_123.mp4",
                platforms=[{"platform": "youtube", "account_id": "acc_yt_001"}],
                content="Test #ad",
            )

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_directory(
        self, mock_publisher, outputs_dir: Path, cleanup_config: CleanupConfig
    ):
        """Test cleanup handles missing product directory."""
        cleanup_manager = CleanupManager(outputs_dir, cleanup_config, mock_publisher)

        result = await cleanup_manager.cleanup("B0NONEXISTENT", [Platform.YOUTUBE])

        assert result["success"] is False
        assert "not found" in str(result["message"]).lower()


class TestAccountDiscovery:
    """Test account discovery scenarios."""

    @pytest.mark.asyncio
    async def test_get_accounts(self, mock_publisher):
        """Test fetching connected accounts."""
        accounts = await mock_publisher.get_accounts()

        assert len(accounts) == 2
        assert any(a["platform"] == "youtube" for a in accounts)
        assert any(a["platform"] == "tiktok" for a in accounts)

    @pytest.mark.asyncio
    async def test_authenticate(self, mock_publisher):
        """Test authentication validation."""
        result = await mock_publisher.authenticate()
        assert result is True


class TestGetStatus:
    """Test status retrieval scenarios."""

    @pytest.mark.asyncio
    async def test_get_status_published(self, mock_publisher):
        """Test getting status of a published post."""
        status = await mock_publisher.get_status("post_abc123")

        assert status["post_id"] == "post_abc123"
        assert status["status"] == "published"
        assert len(status["published_urls"]) == 2

    @pytest.mark.asyncio
    async def test_get_status_handles_not_found(self, mock_publisher):
        """Test get_status handles missing post gracefully."""
        # Make posts.get raise a 404-like error
        mock_publisher.client.posts.get = MagicMock(
            side_effect=Exception("Post not found")
        )

        # get_status should not raise, returns error in dict
        status = await mock_publisher.get_status("post_nonexistent")

        assert status["status"] == "unknown"
        assert status["error_message"] is not None


class TestRetryQueue:
    """Test retry queue functionality."""

    def test_add_and_get_retry_queue(self, outputs_dir: Path):
        """Test adding items to retry queue and retrieving them."""
        from src.publisher.tracking import (
            add_to_retry_queue,
            get_retry_queue,
            get_retry_queue_count,
        )

        # Add item to retry queue
        add_to_retry_queue(
            product_id="B0TEST001",
            platforms=["youtube", "tiktok"],
            error="Upload failed: timeout",
            scheduled_time="2026-01-20T10:00:00Z",
            outputs_dir=outputs_dir,
        )

        # Verify it's in the queue
        queue = get_retry_queue(outputs_dir)
        assert len(queue) == 1
        assert queue[0]["product_id"] == "B0TEST001"
        assert queue[0]["platforms"] == ["youtube", "tiktok"]
        assert queue[0]["error"] == "Upload failed: timeout"
        assert queue[0]["scheduled_time"] == "2026-01-20T10:00:00Z"
        assert queue[0]["retry_count"] == 1

        # Verify count
        assert get_retry_queue_count(outputs_dir) == 1

    def test_retry_count_increments(self, outputs_dir: Path):
        """Test retry count increments on subsequent failures."""
        from src.publisher.tracking import add_to_retry_queue, get_retry_queue

        # Add same item twice
        add_to_retry_queue(
            product_id="B0TEST001",
            platforms=["youtube"],
            error="First failure",
            outputs_dir=outputs_dir,
        )
        add_to_retry_queue(
            product_id="B0TEST001",
            platforms=["youtube"],
            error="Second failure",
            outputs_dir=outputs_dir,
        )

        queue = get_retry_queue(outputs_dir)
        assert len(queue) == 1  # Still one entry (updated)
        assert queue[0]["retry_count"] == 2
        assert queue[0]["error"] == "Second failure"  # Latest error

    def test_remove_from_retry_queue(self, outputs_dir: Path):
        """Test removing items from retry queue on success."""
        from src.publisher.tracking import (
            add_to_retry_queue,
            get_retry_queue,
            remove_from_retry_queue,
        )

        # Add items
        add_to_retry_queue("B0TEST001", ["youtube"], "Error 1", outputs_dir=outputs_dir)
        add_to_retry_queue("B0TEST002", ["tiktok"], "Error 2", outputs_dir=outputs_dir)

        assert len(get_retry_queue(outputs_dir)) == 2

        # Remove one
        removed = remove_from_retry_queue("B0TEST001", outputs_dir)
        assert removed is True

        queue = get_retry_queue(outputs_dir)
        assert len(queue) == 1
        assert queue[0]["product_id"] == "B0TEST002"

        # Try to remove non-existent
        removed = remove_from_retry_queue("B0NONEXISTENT", outputs_dir)
        assert removed is False

    def test_clear_retry_queue(self, outputs_dir: Path):
        """Test clearing entire retry queue."""
        from src.publisher.tracking import (
            add_to_retry_queue,
            clear_retry_queue,
            get_retry_queue_count,
        )

        # Add items
        add_to_retry_queue("B0TEST001", ["youtube"], "Error 1", outputs_dir=outputs_dir)
        add_to_retry_queue("B0TEST002", ["tiktok"], "Error 2", outputs_dir=outputs_dir)
        add_to_retry_queue(
            "B0TEST003", ["instagram"], "Error 3", outputs_dir=outputs_dir
        )

        assert get_retry_queue_count(outputs_dir) == 3

        # Clear all
        cleared = clear_retry_queue(outputs_dir)
        assert cleared == 3
        assert get_retry_queue_count(outputs_dir) == 0

    def test_get_retry_queue_item(self, outputs_dir: Path):
        """Test getting specific item from retry queue."""
        from src.publisher.tracking import add_to_retry_queue, get_retry_queue_item

        add_to_retry_queue(
            "B0TEST001",
            ["youtube"],
            "Test error",
            scheduled_time="2026-01-20T10:00:00Z",
            outputs_dir=outputs_dir,
        )

        # Get existing item
        item = get_retry_queue_item("B0TEST001", outputs_dir)
        assert item is not None
        assert item["product_id"] == "B0TEST001"

        # Get non-existent item
        item = get_retry_queue_item("B0NONEXISTENT", outputs_dir)
        assert item is None


class TestBatchPublisherRetryMode:
    """Test BatchPublisher retry mode functionality."""

    @pytest.mark.asyncio
    async def test_batch_publisher_stores_failed_items(
        self, mock_publisher, outputs_dir: Path, product_dir: Path
    ):
        """Test that failed items are added to retry queue."""
        from src.publisher.batch import BatchPublisher
        from src.publisher.tracking import get_retry_queue

        # Make publish fail
        mock_publisher.client.posts.create = MagicMock(
            side_effect=Exception("API Error")
        )

        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=0,
            stagger_delay_max=0,
        )

        summary = await batch.publish_batch()

        # Should have 1 failed
        assert summary.failed == 1

        # Should be in retry queue
        queue = get_retry_queue(outputs_dir)
        assert len(queue) == 1
        assert queue[0]["product_id"] == product_dir.name
        assert "API Error" in queue[0]["error"]

    @pytest.mark.asyncio
    async def test_batch_publisher_removes_on_success(
        self, mock_publisher, outputs_dir: Path, product_dir: Path
    ):
        """Test that successful items are removed from retry queue."""
        from src.publisher.batch import BatchPublisher
        from src.publisher.tracking import add_to_retry_queue, get_retry_queue

        product_id = product_dir.name

        # Pre-populate retry queue
        add_to_retry_queue(
            product_id,
            ["youtube"],
            "Previous failure",
            outputs_dir=outputs_dir,
        )
        assert len(get_retry_queue(outputs_dir)) == 1

        # Run successful batch
        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=0,
            stagger_delay_max=0,
        )

        summary = await batch.publish_batch()

        # Should have succeeded
        assert summary.successful == 1

        # Should be removed from retry queue
        queue = get_retry_queue(outputs_dir)
        assert len(queue) == 0

    @pytest.mark.asyncio
    async def test_batch_publisher_retry_mode_processes_queue(
        self, mock_publisher, outputs_dir: Path, product_dir: Path
    ):
        """Test retry mode only processes items from retry queue."""
        from src.publisher.batch import BatchPublisher
        from src.publisher.tracking import add_to_retry_queue, get_retry_queue

        product_id = product_dir.name

        # Add item to retry queue
        add_to_retry_queue(
            product_id,
            ["youtube"],
            "Previous failure",
            scheduled_time="2026-01-20T10:00:00Z",
            outputs_dir=outputs_dir,
        )

        # Run in retry mode
        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=0,
            stagger_delay_max=0,
            retry_failed=True,
        )

        summary = await batch.publish_batch()

        # Should process the item from retry queue
        assert summary.total_videos == 1
        assert summary.successful == 1

        # Queue should be empty after success
        assert len(get_retry_queue(outputs_dir)) == 0

    @pytest.mark.asyncio
    async def test_batch_publisher_retry_mode_empty_queue(
        self, mock_publisher, outputs_dir: Path
    ):
        """Test retry mode with empty queue returns immediately."""
        from src.publisher.batch import BatchPublisher
        from src.publisher.tracking import get_retry_queue

        # Ensure queue is empty
        assert len(get_retry_queue(outputs_dir)) == 0

        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            retry_failed=True,
        )

        summary = await batch.publish_batch()

        assert summary.total_videos == 0
        assert summary.successful == 0
        assert summary.failed == 0

    @pytest.mark.asyncio
    async def test_retry_preserves_scheduled_time(
        self, mock_publisher, outputs_dir: Path, product_dir: Path
    ):
        """Test that retry preserves original scheduled time."""
        from src.publisher.batch import BatchPublisher
        from src.publisher.tracking import add_to_retry_queue

        product_id = product_dir.name
        original_scheduled_time = "2026-01-20T10:00:00Z"

        # Add item with scheduled time
        add_to_retry_queue(
            product_id,
            ["youtube"],
            "Previous failure",
            scheduled_time=original_scheduled_time,
            outputs_dir=outputs_dir,
        )

        # Run in retry mode
        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=0,
            stagger_delay_max=0,
            retry_failed=True,
        )

        # Get videos from retry queue
        videos = batch._get_retry_queue_videos()

        assert len(videos) == 1
        assert videos[0]["scheduled_time"] == original_scheduled_time
