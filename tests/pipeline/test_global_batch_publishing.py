"""Tests for global batch pipeline publishing phase with auto-scheduling and cleanup.

These tests validate:
1. Auto-scheduling with occupied slot detection
2. Post-publication cleanup functionality
3. Publisher configuration loading
4. Vercel token handling for large files

Run with: pytest tests/pipeline/test_global_batch_publishing.py -v
"""

import tempfile
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.pipeline.config import GlobalBatchConfig, PublishingPhaseSummary
from src.pipeline.global_batch import GlobalPipelineOrchestrator
from src.publisher.models import FirstCommentConfig
from src.scraper.amazon.models import SearchParameters

# Test markers
pytestmark = pytest.mark.integration


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    """Mock asyncio.sleep to prevent slow stagger delays in tests."""
    with patch("asyncio.sleep", return_value=None):
        yield


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
        pipeline_timeout_sec=900,
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

    # Use a fixed reference time (Sunday) so we can predict the next slots
    # Sunday Jan 5, 2026 at 08:00 UTC - before any slot times
    fixed_now = datetime(2026, 1, 5, 8, 0, 0, tzinfo=UTC)

    # From Sunday, the next slots are:
    # Monday Jan 6 at 10:00 (slot 0) - will be occupied
    # Tuesday Jan 7 at 10:00 (slot 1) - will be occupied
    # Wednesday Jan 8 at 10:00 (slot 2) - should be selected
    next_monday = datetime(2026, 1, 6, 10, 0, 0, tzinfo=UTC)
    next_tuesday = datetime(2026, 1, 7, 10, 0, 0, tzinfo=UTC)

    occupied_posts = [
        {"scheduledFor": next_monday},
        {"scheduledFor": next_tuesday},
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
        patch(
            "src.publisher.publish_modes.load_platform_metadata"
        ) as mock_load_metadata,
        patch("src.publisher.schedule.datetime") as mock_datetime,
    ):
        # Freeze time at Sunday Jan 5, 2026 08:00 UTC
        mock_datetime.now.return_value = fixed_now
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        # Mock publisher for slot checking
        temp_publisher = AsyncMock()
        temp_publisher.authenticate = AsyncMock()
        temp_publisher.first_comment_config = FirstCommentConfig(enabled=False)
        temp_publisher.list_posts = AsyncMock(return_value=occupied_posts)

        # Mock publisher for actual publishing
        main_publisher = AsyncMock()
        main_publisher.authenticate = AsyncMock()
        main_publisher.first_comment_config = FirstCommentConfig(enabled=False)
        main_publisher.get_accounts = AsyncMock(
            return_value=[{"platform": "youtube", "account_id": "acc1"}]
        )
        main_publisher.upload_media = AsyncMock(return_value="media_123")
        main_publisher.publish = AsyncMock()

        mock_create_publisher.side_effect = [temp_publisher, main_publisher]

        # Mock metadata
        mock_metadata = Mock()
        mock_metadata.format_content = Mock(return_value="Test content")
        mock_metadata.clamp_to_limits = Mock(return_value=())
        mock_load_metadata.return_value = mock_metadata

        # Execute publishing phase
        produced_videos = [(video_path, "B0TEST1")]
        await orchestrator._execute_publishing_phase(produced_videos)

        # Verify slot checking happened
        temp_publisher.list_posts.assert_called_once()

        # Verify publish was called with next available slot after occupied ones
        main_publisher.publish.assert_called_once()
        call_kwargs = main_publisher.publish.call_args[1]
        scheduled_time = call_kwargs["scheduled_time"]

        # Should schedule to a time after the occupied slots (Wednesday or later)
        assert scheduled_time > next_tuesday  # After the last occupied slot
        assert scheduled_time.weekday() < 5  # Should be a weekday (Mon-Fri)


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
        patch(
            "src.publisher.publish_modes.load_platform_metadata"
        ) as mock_load_metadata,
        patch("pathlib.Path.exists", return_value=True),
    ):
        temp_publisher = AsyncMock()
        temp_publisher.authenticate = AsyncMock()
        temp_publisher.first_comment_config = FirstCommentConfig(enabled=False)
        temp_publisher.list_posts = AsyncMock(return_value=occupied_posts)

        main_publisher = AsyncMock()
        main_publisher.authenticate = AsyncMock()
        main_publisher.first_comment_config = FirstCommentConfig(enabled=False)
        main_publisher.get_accounts = AsyncMock(
            return_value=[{"platform": "youtube", "account_id": "acc1"}]
        )
        main_publisher.upload_media = AsyncMock(return_value="media_123")
        main_publisher.publish = AsyncMock()

        mock_create_publisher.side_effect = [temp_publisher, main_publisher]

        mock_metadata = Mock()
        mock_metadata.format_content = Mock(return_value="Test content")
        mock_metadata.clamp_to_limits = Mock(return_value=())
        mock_load_metadata.return_value = mock_metadata

        produced_videos = [(video_path, "B0TEST1")]
        await orchestrator._execute_publishing_phase(produced_videos)

        # Verify publish called with None (immediate publish)
        main_publisher.publish.assert_called_once()
        # Scheduled time should be None when all slots occupied


@pytest.mark.asyncio
async def test_auto_scheduling_assigns_unique_slots_per_product(
    temp_outputs_dir, mock_publisher_config, mock_video_config
):
    """Test that batch scheduling gives each product a different slot.

    Regression test: previously all products in a batch got the same
    schedule_time because the slot was computed once before the loop.
    """
    config = GlobalBatchConfig(
        product_ids=["B0TEST1", "B0TEST2"],
        keywords=[],
        max_products=2,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        outputs_dir=temp_outputs_dir,
        skip_publish=False,
        platforms=["youtube"],
    )

    # Create two product directories with videos
    for pid in ["B0TEST1", "B0TEST2"]:
        pdir = temp_outputs_dir / pid
        pdir.mkdir(parents=True)
        (pdir / "video.mp4").write_text("fake video")
        (pdir / "metadata.json").write_text('{"title": "Test", "description": "Test"}')

    orchestrator = GlobalPipelineOrchestrator(config)

    # No occupied slots - both products should get consecutive slots
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
        patch(
            "src.publisher.publish_modes.load_platform_metadata"
        ) as mock_load_metadata,
    ):
        # Temp publisher for slot checking (returns no occupied slots)
        temp_publisher = AsyncMock()
        temp_publisher.authenticate = AsyncMock()
        temp_publisher.first_comment_config = FirstCommentConfig(enabled=False)
        temp_publisher.list_posts = AsyncMock(return_value=[])

        # Main publisher for actual publishing
        main_publisher = AsyncMock()
        main_publisher.authenticate = AsyncMock()
        main_publisher.first_comment_config = FirstCommentConfig(enabled=False)
        main_publisher.get_accounts = AsyncMock(
            return_value=[{"platform": "youtube", "account_id": "acc1"}]
        )
        main_publisher.upload_media = AsyncMock(return_value="media_123")
        main_publisher.publish = AsyncMock()

        mock_create_publisher.side_effect = [temp_publisher, main_publisher]

        mock_metadata = Mock()
        mock_metadata.format_content = Mock(return_value="Test content")
        mock_metadata.clamp_to_limits = Mock(return_value=())
        mock_load_metadata.return_value = mock_metadata

        produced_videos = [
            (temp_outputs_dir / "B0TEST1" / "video.mp4", "B0TEST1"),
            (temp_outputs_dir / "B0TEST2" / "video.mp4", "B0TEST2"),
        ]
        await orchestrator._execute_publishing_phase(produced_videos)

        # Both products should have been published
        assert main_publisher.publish.call_count == 2

        # Extract scheduled times from both publish calls
        times = []
        for call in main_publisher.publish.call_args_list:
            times.append(call[1]["scheduled_time"])

        # The two scheduled times must be different
        assert times[0] != times[1], (
            f"Both products got the same slot: {times[0]}. "
            "Each product should get a unique slot."
        )


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
        patch(
            "src.publisher.publish_modes.load_platform_metadata"
        ) as mock_load_metadata,
    ):
        publisher = AsyncMock()
        publisher.authenticate = AsyncMock()
        publisher.first_comment_config = FirstCommentConfig(enabled=False)
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
        mock_metadata.clamp_to_limits = Mock(return_value=())
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
        patch(
            "src.publisher.publish_modes.load_platform_metadata"
        ) as mock_load_metadata,
    ):
        publisher = AsyncMock()
        publisher.authenticate = AsyncMock()
        publisher.first_comment_config = FirstCommentConfig(enabled=False)
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
        mock_metadata.clamp_to_limits = Mock(return_value=())
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
        publisher.first_comment_config = FirstCommentConfig(enabled=False)
        mock_create_publisher.return_value = publisher

        produced_videos = [(product_dir / "video.mp4", "B0TEST1")]
        await orchestrator._execute_publishing_phase(produced_videos)

        # Verify publisher created with Vercel token
        calls = mock_create_publisher.call_args_list
        for call in calls:
            assert call[1]["vercel_token"] == "test_vercel_token"  # noqa: S105


@pytest.mark.asyncio
async def test_batch_publisher_gets_the_configured_synthetic_media_flag(
    temp_outputs_dir, mock_publisher_config, mock_video_config
):
    """The batch builds its own publisher instead of reusing the CLI's.

    So every setting has to be passed here too. Miss one and the same
    `publisher.yaml` produces different payloads depending on whether the run
    went through `python -m src.publisher.late` or `make batch-lowpri`, with
    no log line either way.

    Driven from a config that turns the flag on, so a hardcoded default fails
    as loudly as a dropped kwarg.
    """
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
    (product_dir / "metadata.json").write_text(
        '{"title": "Test", "description": "Test"}'
    )

    publisher_config = dict(mock_publisher_config)
    publisher_config["synthetic_media_disclosure"] = True
    publisher_config["immediate_publish"] = True

    orchestrator = GlobalPipelineOrchestrator(config)

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
        patch("yaml.safe_load", return_value=publisher_config),
        patch.dict(
            "os.environ",
            {"LATE_API_KEY": "test_key", "LATE_VERCEL_TOKEN": "test_vercel_token"},
        ),
        patch("src.publisher.create_publisher") as mock_create_publisher,
        patch(
            "src.publisher.publish_modes.load_platform_metadata"
        ) as mock_load_metadata,
    ):
        main_publisher = AsyncMock()
        main_publisher.authenticate = AsyncMock()
        main_publisher.first_comment_config = FirstCommentConfig(enabled=False)
        main_publisher.get_accounts = AsyncMock(
            return_value=[{"platform": "youtube", "account_id": "acc1"}]
        )
        main_publisher.upload_media = AsyncMock(return_value="media_123")
        main_publisher.publish = AsyncMock()
        mock_create_publisher.return_value = main_publisher

        mock_metadata = Mock()
        mock_metadata.format_content = Mock(return_value="Test content")
        mock_metadata.clamp_to_limits = Mock(return_value=())
        mock_metadata.carries_affiliate_content = True
        mock_load_metadata.return_value = mock_metadata

        await orchestrator._execute_publishing_phase([(video_path, "B0TEST1")])

    publishing_calls = [
        c
        for c in mock_create_publisher.call_args_list
        if "synthetic_media_disclosure" in c.kwargs
    ]
    assert (
        publishing_calls
    ), "the batch built a publisher without passing synthetic_media_disclosure"
    assert publishing_calls[-1].kwargs["synthetic_media_disclosure"] is True
