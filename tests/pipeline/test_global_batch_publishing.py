"""Tests for global batch pipeline publishing phase with auto-scheduling and cleanup.

These tests validate:
1. Auto-scheduling with occupied slot detection
2. Post-publication cleanup functionality
3. Publisher configuration loading
4. Vercel token handling for large files

Run with: pytest tests/pipeline/test_global_batch_publishing.py -v
"""

import tempfile
from datetime import UTC, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.pipeline.config import GlobalBatchConfig, PublishingPhaseSummary
from src.pipeline.global_batch import GlobalPipelineOrchestrator
from src.scraper.amazon.models import SearchParameters

# Test markers
pytestmark = pytest.mark.integration


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_outputs_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory(prefix="test_publish_outputs_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_publisher_config():
    """Mock publisher.yaml configuration."""
    return {
        "immediate_publish": False,
        "recurring_schedule": {
            "enabled": True,
            "timezone": "UTC",
            "slots": [
                {"day_of_week": "monday", "time": "10:00:00"},
                {"day_of_week": "tuesday", "time": "10:00:00"},
                {"day_of_week": "wednesday", "time": "10:00:00"},
            ],
        },
        "default_platforms": ["youtube", "tiktok", "instagram"],
        "stagger_delay_min": 30,
        "stagger_delay_max": 60,
        "cleanup": {
            "enabled": True,
            "verify_before_delete": True,
            "require_all_platforms": True,
        },
    }


@pytest.fixture
def mock_video_config():
    """Mock video configuration."""
    return SimpleNamespace(
        pipeline_timeout_sec=300,
        llm_settings=SimpleNamespace(api_key_env_var=None),
    )


# ============================================================================
# AUTO-SCHEDULING TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_auto_scheduling_finds_first_unoccupied_slot(
    temp_outputs_dir, mock_publisher_config, mock_video_config
):
    """Test auto-scheduling finds first unoccupied slot by querying API."""
    config = GlobalBatchConfig(
        product_ids=["B0TEST1"],
        keywords=[],
        max_products=1,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        outputs_dir=temp_outputs_dir,
        skip_publish=False,  # Enable publishing
        platforms=["youtube"],
    )

    # Create product directory with video
    product_dir = temp_outputs_dir / "B0TEST1"
    product_dir.mkdir(parents=True)
    video_path = product_dir / "video.mp4"
    video_path.write_text("fake video")
    metadata_path = product_dir / "metadata.json"
    metadata_path.write_text('{"title": "Test", "description": "Test"}')

    orchestrator = GlobalPipelineOrchestrator(config)

    # Mock existing posts (Monday and Tuesday occupied)
    occupied_posts = [
        {
            "scheduledFor": datetime(2025, 12, 29, 10, 0, tzinfo=UTC),  # Mon
        },
        {
            "scheduledFor": datetime(2025, 12, 30, 10, 0, tzinfo=UTC),  # Tue
        },
    ]

    with (
        patch(
            "builtins.open",
            side_effect=[
                tempfile._TemporaryFileWrapper(
                    tempfile.NamedTemporaryFile(mode="w", delete=False),
                    name="publisher.yaml",
                )
            ],
        ),
        patch("yaml.safe_load", return_value=mock_publisher_config),
        patch.dict(
            "os.environ",
            {
                "LATE_API_KEY": "test_key",
                "LATE_VERCEL_TOKEN": "test_vercel_token",
            },
        ),
        patch("src.publisher.create_publisher") as mock_create_publisher,
        patch("src.publisher.metadata.load_platform_metadata") as mock_load_metadata,
    ):
        # Mock publisher for slot checking
        temp_publisher = AsyncMock()
        temp_publisher.authenticate = AsyncMock()
        temp_publisher.list_posts = AsyncMock(return_value=occupied_posts)

        # Mock publisher for actual publishing
        main_publisher = AsyncMock()
        main_publisher.authenticate = AsyncMock()
        main_publisher.get_accounts = AsyncMock(
            return_value=[{"platform": "youtube", "account_id": "acc1"}]
        )
        main_publisher.upload_media = AsyncMock(return_value="media_123")
        main_publisher.publish = AsyncMock()

        mock_create_publisher.side_effect = [temp_publisher, main_publisher]

        # Mock metadata
        mock_metadata = Mock()
        mock_metadata.format_content = Mock(return_value="Test content")
        mock_load_metadata.return_value = mock_metadata

        # Execute publishing phase
        produced_videos = [(video_path, "B0TEST1")]
        await orchestrator._execute_publishing_phase(produced_videos)

        # Verify slot checking happened
        temp_publisher.list_posts.assert_called_once()

        # Verify publish was called with Wednesday slot (first unoccupied)
        main_publisher.publish.assert_called_once()
        call_kwargs = main_publisher.publish.call_args[1]
        scheduled_time = call_kwargs["scheduled_time"]

        # Should schedule to Wednesday (neither Mon nor Tue)
        assert scheduled_time.weekday() == 2  # Wednesday


@pytest.mark.asyncio
async def test_auto_scheduling_falls_back_to_immediate_when_all_slots_occupied(
    temp_outputs_dir, mock_publisher_config, mock_video_config
):
    """Test auto-scheduling publishes immediately when all slots occupied."""
    config = GlobalBatchConfig(
        product_ids=["B0TEST1"],
        keywords=[],
        max_products=1,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        outputs_dir=temp_outputs_dir,
        skip_publish=False,
        platforms=["youtube"],
    )

    product_dir = temp_outputs_dir / "B0TEST1"
    product_dir.mkdir(parents=True)
    video_path = product_dir / "video.mp4"
    video_path.write_text("fake video")
    metadata_path = product_dir / "metadata.json"
    metadata_path.write_text('{"title": "Test", "description": "Test"}')

    orchestrator = GlobalPipelineOrchestrator(config)

    # Mock all slots occupied for next 8 weeks
    now = datetime.now(UTC)
    occupied_posts = [
        {"scheduledFor": now.replace(hour=10, minute=0, second=0, microsecond=0)}
        for _ in range(100)  # More than 8 weeks worth
    ]

    with (
        patch("yaml.safe_load", return_value=mock_publisher_config),
        patch.dict(
            "os.environ",
            {
                "LATE_API_KEY": "test_key",
                "LATE_VERCEL_TOKEN": "test_vercel_token",
            },
        ),
        patch("src.publisher.create_publisher") as mock_create_publisher,
        patch("src.publisher.metadata.load_platform_metadata") as mock_load_metadata,
        patch("pathlib.Path.exists", return_value=True),
    ):
        temp_publisher = AsyncMock()
        temp_publisher.authenticate = AsyncMock()
        temp_publisher.list_posts = AsyncMock(return_value=occupied_posts)

        main_publisher = AsyncMock()
        main_publisher.authenticate = AsyncMock()
        main_publisher.get_accounts = AsyncMock(
            return_value=[{"platform": "youtube", "account_id": "acc1"}]
        )
        main_publisher.upload_media = AsyncMock(return_value="media_123")
        main_publisher.publish = AsyncMock()

        mock_create_publisher.side_effect = [temp_publisher, main_publisher]

        mock_metadata = Mock()
        mock_metadata.format_content = Mock(return_value="Test content")
        mock_load_metadata.return_value = mock_metadata

        produced_videos = [(video_path, "B0TEST1")]
        await orchestrator._execute_publishing_phase(produced_videos)

        # Verify publish called with None (immediate publish)
        main_publisher.publish.assert_called_once()
        # Scheduled time should be None when all slots occupied


# ============================================================================
# CLEANUP TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_cleanup_removes_directory_after_successful_publish(
    temp_outputs_dir, mock_publisher_config
):
    """Test cleanup removes product directory after successful multi-platform publish."""
    config = GlobalBatchConfig(
        product_ids=["B0TEST1"],
        keywords=[],
        max_products=1,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        outputs_dir=temp_outputs_dir,
        skip_publish=False,
        platforms=["youtube", "tiktok", "instagram"],
    )

    # Create product directory
    product_dir = temp_outputs_dir / "B0TEST1"
    product_dir.mkdir(parents=True)
    video_path = product_dir / "video.mp4"
    video_path.write_text("fake video")
    (product_dir / "metadata.json").write_text('{"title": "Test"}')

    orchestrator = GlobalPipelineOrchestrator(config)

    with (
        patch("yaml.safe_load", return_value=mock_publisher_config),
        patch.dict("os.environ", {"LATE_API_KEY": "test_key"}),
        patch("src.publisher.create_publisher") as mock_create_publisher,
        patch("src.publisher.metadata.load_platform_metadata") as mock_load_metadata,
    ):
        publisher = AsyncMock()
        publisher.authenticate = AsyncMock()
        publisher.get_accounts = AsyncMock(
            return_value=[
                {"platform": "youtube", "account_id": "acc1"},
                {"platform": "tiktok", "account_id": "acc2"},
                {"platform": "instagram", "account_id": "acc3"},
            ]
        )
        publisher.upload_media = AsyncMock(return_value="media_123")
        publisher.publish = AsyncMock()  # Succeeds for all platforms

        mock_create_publisher.return_value = publisher

        mock_metadata = Mock()
        mock_metadata.format_content = Mock(return_value="Test")
        mock_load_metadata.return_value = mock_metadata

        # Directory exists before
        assert product_dir.exists()

        produced_videos = [(video_path, "B0TEST1")]
        summary = await orchestrator._execute_publishing_phase(produced_videos)

        # Directory should be removed after successful publish
        assert not product_dir.exists()
        assert summary.successful == 1


@pytest.mark.asyncio
async def test_cleanup_preserves_directory_on_partial_failure(
    temp_outputs_dir, mock_publisher_config
):
    """Test cleanup preserves product directory when not all platforms succeed."""
    config = GlobalBatchConfig(
        product_ids=["B0TEST1"],
        keywords=[],
        max_products=1,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        outputs_dir=temp_outputs_dir,
        skip_publish=False,
        platforms=["youtube", "tiktok"],
    )

    product_dir = temp_outputs_dir / "B0TEST1"
    product_dir.mkdir(parents=True)
    video_path = product_dir / "video.mp4"
    video_path.write_text("fake video")
    (product_dir / "metadata.json").write_text('{"title": "Test"}')

    orchestrator = GlobalPipelineOrchestrator(config)

    with (
        patch("yaml.safe_load", return_value=mock_publisher_config),
        patch.dict("os.environ", {"LATE_API_KEY": "test_key"}),
        patch("src.publisher.create_publisher") as mock_create_publisher,
        patch("src.publisher.metadata.load_platform_metadata") as mock_load_metadata,
    ):
        publisher = AsyncMock()
        publisher.authenticate = AsyncMock()
        publisher.get_accounts = AsyncMock(
            return_value=[
                {"platform": "youtube", "account_id": "acc1"},
                {"platform": "tiktok", "account_id": "acc2"},
            ]
        )
        publisher.upload_media = AsyncMock(return_value="media_123")

        # First publish succeeds, second fails
        publisher.publish = AsyncMock(
            side_effect=[
                None,  # YouTube success
                RuntimeError("TikTok failed"),  # TikTok failure
            ]
        )

        mock_create_publisher.return_value = publisher

        mock_metadata = Mock()
        mock_metadata.format_content = Mock(return_value="Test")
        mock_load_metadata.return_value = mock_metadata

        produced_videos = [(video_path, "B0TEST1")]
        summary = await orchestrator._execute_publishing_phase(produced_videos)

        # Directory should still exist (partial failure)
        assert product_dir.exists()
        assert summary.failed == 1


# ============================================================================
# VERCEL TOKEN TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_vercel_token_loaded_from_environment(
    temp_outputs_dir, mock_publisher_config
):
    """Test Vercel token is loaded from environment and passed to publisher."""
    config = GlobalBatchConfig(
        product_ids=["B0TEST1"],
        keywords=[],
        max_products=1,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        outputs_dir=temp_outputs_dir,
        skip_publish=False,
    )

    product_dir = temp_outputs_dir / "B0TEST1"
    product_dir.mkdir(parents=True)
    (product_dir / "video.mp4").write_text("fake video")
    (product_dir / "metadata.json").write_text('{"title": "Test"}')

    orchestrator = GlobalPipelineOrchestrator(config)

    with (
        patch("yaml.safe_load", return_value=mock_publisher_config),
        patch.dict(
            "os.environ",
            {
                "LATE_API_KEY": "test_api_key",
                "LATE_VERCEL_TOKEN": "test_vercel_token",
            },
        ),
        patch("src.publisher.create_publisher") as mock_create_publisher,
    ):
        publisher = AsyncMock()
        publisher.authenticate = AsyncMock()
        mock_create_publisher.return_value = publisher

        produced_videos = [(product_dir / "video.mp4", "B0TEST1")]
        await orchestrator._execute_publishing_phase(produced_videos)

        # Verify publisher created with Vercel token
        calls = mock_create_publisher.call_args_list
        for call in calls:
            assert call[1]["vercel_token"] == "test_vercel_token"  # noqa: S105
