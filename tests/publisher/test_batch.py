"""Tests for BatchPublisher class and cleanup integration.

Tests batch publishing operations including:
- Batch video discovery and publishing
- Cleanup integration with immediate mode (schedule --immediate)
- --no-cleanup flag behavior
- Error handling and failure isolation
"""

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.publisher.batch import BatchPublisher
from src.publisher.models import BatchPublishSummary, Platform


@pytest.fixture(autouse=True)
def mock_link_in_bio():
    """Keep link-in-bio (enabled by default) from hitting the real API."""
    with patch(
        "src.publisher.batch.update_link_in_bio_safe", new_callable=AsyncMock
    ) as mock:
        yield mock


@pytest.fixture
def mock_publisher():
    """Create mock publisher with common behaviors."""
    publisher = AsyncMock()
    publisher.authenticate = AsyncMock(return_value=True)
    publisher.upload_media = AsyncMock(return_value="media_12345")
    publisher.get_accounts = AsyncMock(
        return_value=[
            {"platform": "youtube", "account_id": "yt_acc_1", "username": "test_yt"},
            {"platform": "tiktok", "account_id": "tt_acc_1", "username": "test_tt"},
            {"platform": "instagram", "account_id": "ig_acc_1", "username": "test_ig"},
        ]
    )
    publisher.publish = AsyncMock(
        return_value={
            "post_id": "post_abc123",
            "status": "scheduled",
            "published_urls": ["https://example.com/post/123"],
        }
    )
    publisher.get_status = AsyncMock(
        return_value={"status": "published", "published_urls": []}
    )
    return publisher


@pytest.fixture
def outputs_dir(tmp_path):
    """Create outputs directory with sample product folders and videos."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    # Create product directories with videos
    for product_id in ["B0TEST001", "B0TEST002", "B0TEST003"]:
        product_dir = outputs / product_id
        product_dir.mkdir()

        # Create video file
        video_file = product_dir / f"video_{product_id}_sequential.mp4"
        video_file.write_bytes(b"fake video content")

        # Create metadata files
        data_file = product_dir / "data.json"
        data_file.write_text('{"title": "Test Product"}')

    return outputs


@pytest.fixture
def empty_outputs_dir(tmp_path):
    """Create empty outputs directory."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return outputs


class TestBatchPublisherInit:
    """Test BatchPublisher initialization."""

    def test_init_with_defaults(self, mock_publisher):
        """Test initialization with default parameters."""
        batch = BatchPublisher(publisher=mock_publisher)

        assert batch.publisher == mock_publisher
        assert batch.outputs_dir == Path("outputs")
        assert batch.platforms == [
            Platform.YOUTUBE,
            Platform.TIKTOK,
            Platform.INSTAGRAM,
        ]
        assert batch.stagger_delay_min == 30
        assert batch.stagger_delay_max == 60
        assert batch.fail_fast is False

    def test_init_with_custom_params(self, mock_publisher, outputs_dir):
        """Test initialization with custom parameters."""
        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE, Platform.TIKTOK],
            stagger_delay_min=10,
            stagger_delay_max=20,
            fail_fast=True,
        )

        assert batch.outputs_dir == outputs_dir
        assert batch.platforms == [Platform.YOUTUBE, Platform.TIKTOK]
        assert batch.stagger_delay_min == 10
        assert batch.stagger_delay_max == 20
        assert batch.fail_fast is True

    def test_init_with_string_path(self, mock_publisher):
        """Test initialization with string path (should convert to Path)."""
        batch = BatchPublisher(publisher=mock_publisher, outputs_dir="custom/outputs")

        assert batch.outputs_dir == Path("custom/outputs")


class TestVideoDiscovery:
    """Test video discovery functionality."""

    def test_discover_videos_finds_all(self, mock_publisher, outputs_dir):
        """Test that all videos are discovered."""
        batch = BatchPublisher(publisher=mock_publisher, outputs_dir=outputs_dir)

        videos = batch._discover_videos()

        assert len(videos) == 3
        product_ids = {v["product_id"] for v in videos}
        assert product_ids == {"B0TEST001", "B0TEST002", "B0TEST003"}

    def test_discover_videos_empty_directory(self, mock_publisher, empty_outputs_dir):
        """Test discovery with empty directory."""
        batch = BatchPublisher(publisher=mock_publisher, outputs_dir=empty_outputs_dir)

        videos = batch._discover_videos()

        assert videos == []

    def test_discover_videos_nonexistent_directory(self, mock_publisher, tmp_path):
        """Test discovery with nonexistent directory."""
        batch = BatchPublisher(
            publisher=mock_publisher, outputs_dir=tmp_path / "nonexistent"
        )

        videos = batch._discover_videos()

        assert videos == []

    def test_discover_videos_ignores_non_video_files(self, mock_publisher, outputs_dir):
        """Test that non-video files are ignored."""
        # Add non-video file
        product_dir = outputs_dir / "B0TEST001"
        (product_dir / "thumbnail.jpg").write_bytes(b"fake image")
        (product_dir / "metadata.json").write_text('{"foo": "bar"}')

        batch = BatchPublisher(publisher=mock_publisher, outputs_dir=outputs_dir)

        videos = batch._discover_videos()

        # Should only find video files
        for video in videos:
            assert video["path"].suffix == ".mp4"
            assert video["path"].stem.startswith("video_")


class TestBatchPublishing:
    """Test batch publishing operations."""

    @pytest.mark.asyncio
    async def test_publish_batch_empty_directory(
        self, mock_publisher, empty_outputs_dir
    ):
        """Test batch publishing with no videos."""
        batch = BatchPublisher(publisher=mock_publisher, outputs_dir=empty_outputs_dir)

        summary = await batch.publish_batch()

        assert summary.total_videos == 0
        assert summary.successful == 0
        assert summary.failed == 0
        assert summary.skipped == 0

    @pytest.mark.asyncio
    async def test_publish_batch_success(self, mock_publisher, outputs_dir):
        """Test successful batch publishing."""
        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=0,
            stagger_delay_max=0,
        )

        # Mock platform metadata
        with patch("src.publisher.batch.load_platform_metadata") as mock_metadata:
            mock_meta = MagicMock()
            mock_meta.format_content.return_value = "Test content"
            mock_metadata.return_value = mock_meta

            summary = await batch.publish_batch()

        assert summary.total_videos == 3
        assert summary.successful == 3
        assert summary.failed == 0
        assert mock_publisher.upload_media.call_count == 3
        assert mock_publisher.publish.call_count == 3

    @pytest.mark.asyncio
    async def test_publish_batch_with_failures(self, mock_publisher, outputs_dir):
        """Test batch publishing with some failures."""
        # Make first publish fail
        mock_publisher.publish = AsyncMock(
            side_effect=[
                Exception("API error"),
                {
                    "post_id": "post_2",
                    "status": "scheduled",
                    "published_urls": [],
                },
                {
                    "post_id": "post_3",
                    "status": "scheduled",
                    "published_urls": [],
                },
            ]
        )

        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=0,
            stagger_delay_max=0,
            fail_fast=False,
        )

        with patch("src.publisher.batch.load_platform_metadata") as mock_metadata:
            mock_meta = MagicMock()
            mock_meta.format_content.return_value = "Test content"
            mock_metadata.return_value = mock_meta

            summary = await batch.publish_batch()

        assert summary.total_videos == 3
        assert summary.failed >= 1
        assert len(summary.errors) >= 1

    @pytest.mark.asyncio
    async def test_publish_batch_fail_fast(self, mock_publisher, outputs_dir):
        """Test fail-fast mode stops on first failure."""
        # Make first publish fail
        mock_publisher.upload_media = AsyncMock(
            side_effect=[Exception("Upload failed"), "media_2", "media_3"]
        )

        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=0,
            stagger_delay_max=0,
            fail_fast=True,
        )

        summary = await batch.publish_batch()

        # Should stop after first failure
        assert summary.failed >= 1
        # Not all videos should be attempted
        assert mock_publisher.upload_media.call_count == 1

    @pytest.mark.asyncio
    async def test_publish_batch_skips_missing_metadata(
        self, mock_publisher, outputs_dir
    ):
        """Test that videos without metadata are skipped."""
        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=0,
            stagger_delay_max=0,
        )

        # Return None for metadata (missing)
        with patch("src.publisher.batch.load_platform_metadata") as mock_metadata:
            mock_metadata.return_value = None

            summary = await batch.publish_batch()

        assert summary.skipped >= 1


class TestBatchSummary:
    """Test BatchPublishSummary functionality."""

    def test_summary_initialization(self):
        """Test summary initialization."""
        summary = BatchPublishSummary(
            total_videos=10,
            successful=7,
            failed=2,
            skipped=1,
        )

        assert summary.total_videos == 10
        assert summary.successful == 7
        assert summary.failed == 2
        assert summary.skipped == 1

    def test_summary_success_rate(self):
        """Test success rate calculation."""
        summary = BatchPublishSummary(
            total_videos=10,
            successful=8,
            failed=2,
            skipped=0,
        )

        rate = summary.get_success_rate()
        assert rate == 80.0

    def test_summary_success_rate_zero_videos(self):
        """Test success rate with zero videos."""
        summary = BatchPublishSummary(
            total_videos=0,
            successful=0,
            failed=0,
            skipped=0,
        )

        rate = summary.get_success_rate()
        assert rate == 0.0

    def test_summary_add_error(self):
        """Test adding errors to summary."""
        summary = BatchPublishSummary(
            total_videos=1,
            successful=0,
            failed=1,
            skipped=0,
        )

        summary.add_error("B0TEST001", "API connection failed")

        assert len(summary.errors) == 1
        assert summary.errors[0]["video_id"] == "B0TEST001"
        assert "API connection failed" in summary.errors[0]["error"]

    def test_summary_add_platform_result(self):
        """Test adding platform-specific results."""
        summary = BatchPublishSummary(
            total_videos=1,
            successful=1,
            failed=0,
            skipped=0,
        )

        summary.add_platform_result(Platform.YOUTUBE, success=True)
        summary.add_platform_result(Platform.TIKTOK, success=False)

        assert summary.platform_results[Platform.YOUTUBE]["successful"] == 1
        assert summary.platform_results[Platform.TIKTOK]["failed"] == 1


class TestCleanupIntegration:
    """Test cleanup integration with immediate publishing.

    These tests verify that cleanup runs correctly after immediate publishing
    (schedule --immediate) when enabled, and respects the --no-cleanup flag.
    """

    @pytest.fixture
    def mock_config(self):
        """Create mock config with cleanup enabled."""
        config = MagicMock()
        config.provider = "late"
        config.api_key = "test_api_key"  # noqa: S105
        config.vercel_token = "test_vercel_token"  # noqa: S105
        config.timeout = 30
        config.max_retries = 3
        config.stagger_delay_min = 0
        config.stagger_delay_max = 0

        config.cleanup_config = MagicMock()
        config.cleanup_config.enabled = True
        config.cleanup_config.verify_before_delete = True
        config.cleanup_config.archive_before_delete = False
        config.cleanup_config.require_all_platforms = True

        return config

    @pytest.fixture
    def mock_args(self, outputs_dir):
        """Create mock args for immediate publish."""
        args = MagicMock()
        args.platforms = [Platform.YOUTUBE]
        args.outputs_dir = outputs_dir
        args.fail_fast = False
        args.retry_failed = False
        args.no_cleanup = False
        args.debug = False
        return args

    @pytest.mark.asyncio
    async def test_cleanup_runs_after_successful_batch(
        self, mock_publisher, mock_config, mock_args, outputs_dir
    ):
        """Test that cleanup runs after successful immediate publish."""
        mock_summary = BatchPublishSummary(
            total_videos=3,
            successful=3,
            failed=0,
            skipped=0,
        )

        with (
            patch("src.publisher.late.cli.BatchPublisher") as mock_batch_class,
            patch("src.publisher.late.cli.CleanupManager") as mock_cleanup_class,
        ):
            mock_batch_instance = AsyncMock()
            mock_batch_instance.publish_batch = AsyncMock(return_value=mock_summary)
            mock_batch_class.return_value = mock_batch_instance

            mock_cleanup_instance = AsyncMock()
            mock_cleanup_instance.cleanup_all = AsyncMock(
                return_value={"cleaned": 3, "skipped": 0, "disk_freed": 1024000}
            )
            mock_cleanup_class.return_value = mock_cleanup_instance

            from src.publisher.late.cli import _run_immediate_batch

            with contextlib.suppress(SystemExit):
                await _run_immediate_batch(mock_args, mock_config, mock_publisher)

            mock_cleanup_class.assert_called_once()
            mock_cleanup_instance.cleanup_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_cleanup_flag_prevents_cleanup(
        self, mock_publisher, mock_config, mock_args, outputs_dir
    ):
        """Test that --no-cleanup flag prevents cleanup from running."""
        mock_args.no_cleanup = True

        mock_summary = BatchPublishSummary(
            total_videos=3,
            successful=3,
            failed=0,
            skipped=0,
        )

        with (
            patch("src.publisher.late.cli.BatchPublisher") as mock_batch_class,
            patch("src.publisher.late.cli.CleanupManager") as mock_cleanup_class,
        ):
            mock_batch_instance = AsyncMock()
            mock_batch_instance.publish_batch = AsyncMock(return_value=mock_summary)
            mock_batch_class.return_value = mock_batch_instance

            from src.publisher.late.cli import _run_immediate_batch

            with contextlib.suppress(SystemExit):
                await _run_immediate_batch(mock_args, mock_config, mock_publisher)

            mock_cleanup_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_disabled_in_config_skips_cleanup(
        self, mock_publisher, mock_config, mock_args, outputs_dir
    ):
        """Test that cleanup is skipped when disabled in config."""
        mock_config.cleanup_config.enabled = False

        mock_summary = BatchPublishSummary(
            total_videos=3,
            successful=3,
            failed=0,
            skipped=0,
        )

        with (
            patch("src.publisher.late.cli.BatchPublisher") as mock_batch_class,
            patch("src.publisher.late.cli.CleanupManager") as mock_cleanup_class,
        ):
            mock_batch_instance = AsyncMock()
            mock_batch_instance.publish_batch = AsyncMock(return_value=mock_summary)
            mock_batch_class.return_value = mock_batch_instance

            from src.publisher.late.cli import _run_immediate_batch

            with contextlib.suppress(SystemExit):
                await _run_immediate_batch(mock_args, mock_config, mock_publisher)

            mock_cleanup_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_skipped_when_no_successful_publishes(
        self, mock_publisher, mock_config, mock_args, outputs_dir
    ):
        """Test that cleanup is skipped when no publishes succeeded."""
        mock_summary = BatchPublishSummary(
            total_videos=3,
            successful=0,
            failed=3,
            skipped=0,
        )

        with (
            patch("src.publisher.late.cli.BatchPublisher") as mock_batch_class,
            patch("src.publisher.late.cli.CleanupManager") as mock_cleanup_class,
        ):
            mock_batch_instance = AsyncMock()
            mock_batch_instance.publish_batch = AsyncMock(return_value=mock_summary)
            mock_batch_class.return_value = mock_batch_instance

            from src.publisher.late.cli import _run_immediate_batch

            with contextlib.suppress(SystemExit):
                await _run_immediate_batch(mock_args, mock_config, mock_publisher)

            mock_cleanup_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_fail_batch(
        self, mock_publisher, mock_config, mock_args, outputs_dir
    ):
        """Test that cleanup failure doesn't fail the publish operation."""
        mock_summary = BatchPublishSummary(
            total_videos=3,
            successful=3,
            failed=0,
            skipped=0,
        )

        with (
            patch("src.publisher.late.cli.BatchPublisher") as mock_batch_class,
            patch("src.publisher.late.cli.CleanupManager") as mock_cleanup_class,
            patch("src.publisher.late.cli.logger") as mock_logger,
        ):
            mock_batch_instance = AsyncMock()
            mock_batch_instance.publish_batch = AsyncMock(return_value=mock_summary)
            mock_batch_class.return_value = mock_batch_instance

            mock_cleanup_instance = AsyncMock()
            mock_cleanup_instance.cleanup_all = AsyncMock(
                side_effect=Exception("Cleanup failed")
            )
            mock_cleanup_class.return_value = mock_cleanup_instance

            from src.publisher.late.cli import _run_immediate_batch

            with contextlib.suppress(SystemExit):
                await _run_immediate_batch(mock_args, mock_config, mock_publisher)

            mock_cleanup_instance.cleanup_all.assert_called_once()
            mock_logger.warning.assert_called()


class TestStaggerDelay:
    """Test staggered delay between posts."""

    @pytest.mark.asyncio
    async def test_stagger_delay_applied(self, mock_publisher, outputs_dir):
        """Test that stagger delay is applied between videos."""
        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=1,
            stagger_delay_max=1,
        )

        with (
            patch("src.publisher.batch.load_platform_metadata") as mock_metadata,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_meta = MagicMock()
            mock_meta.format_content.return_value = "Test content"
            mock_metadata.return_value = mock_meta

            await batch.publish_batch()

        # Sleep should be called between videos (n-1 times for n videos)
        assert mock_sleep.call_count == 2  # 3 videos, 2 delays

    @pytest.mark.asyncio
    async def test_no_stagger_delay_after_last_video(self, mock_publisher, outputs_dir):
        """Test that no delay is applied after the last video."""
        # Create directory with single video
        for item in outputs_dir.iterdir():
            if item.name != "B0TEST001":
                import shutil

                shutil.rmtree(item)

        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=1,
            stagger_delay_max=1,
        )

        with (
            patch("src.publisher.batch.load_platform_metadata") as mock_metadata,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_meta = MagicMock()
            mock_meta.format_content.return_value = "Test content"
            mock_metadata.return_value = mock_meta

            await batch.publish_batch()

        # No sleep for single video
        mock_sleep.assert_not_called()


class TestBatchCarriesTheDisclosureDecision:
    """The batch builds its own publish call rather than reusing the single
    path, so the render's disclosure decision has to be passed here too.

    A dropped kwarg is silent: `publish()` defaults it to True and a topic
    render declares commercial content to TikTok again with nothing failing.
    """

    @pytest.mark.asyncio
    async def test_a_topic_render_does_not_declare_commercial_content(
        self, mock_publisher, outputs_dir
    ):
        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=0,
            stagger_delay_max=0,
        )

        with patch("src.publisher.batch.load_platform_metadata") as mock_metadata:
            mock_meta = MagicMock()
            mock_meta.format_content.return_value = "Body."
            mock_meta.carries_affiliate_content = False
            mock_metadata.return_value = mock_meta

            await batch.publish_batch()

        assert mock_publisher.publish.call_args_list
        for call in mock_publisher.publish.call_args_list:
            assert call.kwargs["carries_affiliate_content"] is False

    @pytest.mark.asyncio
    async def test_an_affiliate_render_still_discloses(
        self, mock_publisher, outputs_dir
    ):
        batch = BatchPublisher(
            publisher=mock_publisher,
            outputs_dir=outputs_dir,
            platforms=[Platform.YOUTUBE],
            stagger_delay_min=0,
            stagger_delay_max=0,
        )

        with patch("src.publisher.batch.load_platform_metadata") as mock_metadata:
            mock_meta = MagicMock()
            mock_meta.format_content.return_value = "Body."
            mock_meta.carries_affiliate_content = True
            mock_metadata.return_value = mock_meta

            await batch.publish_batch()

        assert mock_publisher.publish.call_args_list
        for call in mock_publisher.publish.call_args_list:
            assert call.kwargs["carries_affiliate_content"] is True
