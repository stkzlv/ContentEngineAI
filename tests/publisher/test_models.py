"""Unit tests for publisher data models."""

from datetime import UTC, datetime, timezone

import pytest

from src.publisher.models import (
    BatchPublishSummary,
    Platform,
    PublisherConfig,
    PublishMetadata,
    PublishResult,
    PublishStatus,
)


class TestPublishStatus:
    """Test PublishStatus enum."""

    def test_status_values(self):
        """Test that status enum has expected values."""
        assert PublishStatus.PENDING.value == "pending"
        assert PublishStatus.SCHEDULED.value == "scheduled"
        assert PublishStatus.PUBLISHED.value == "published"
        assert PublishStatus.FAILED.value == "failed"

    def test_status_from_string(self):
        """Test creating status from string value."""
        status = PublishStatus("published")
        assert status == PublishStatus.PUBLISHED


class TestPlatform:
    """Test Platform enum."""

    def test_platform_values(self):
        """Test that platform enum has expected values."""
        assert Platform.YOUTUBE.value == "youtube"
        assert Platform.TIKTOK.value == "tiktok"
        assert Platform.INSTAGRAM.value == "instagram"
        assert Platform.FACEBOOK.value == "facebook"
        assert Platform.TWITTER.value == "twitter"
        assert Platform.LINKEDIN.value == "linkedin"

    def test_platform_from_string(self):
        """Test creating platform from string value."""
        platform = Platform("youtube")
        assert platform == Platform.YOUTUBE


class TestPublishResult:
    """Test PublishResult dataclass."""

    def test_publish_result_creation(self):
        """Test creating a PublishResult object."""
        result = PublishResult(
            post_id="post_123",
            status=PublishStatus.PUBLISHED,
            platforms=(Platform.YOUTUBE, Platform.TIKTOK),
            published_urls=("https://youtube.com/watch?v=abc",),
            metadata={"test_key": "test_value"},
        )

        assert result.post_id == "post_123"
        assert result.status == PublishStatus.PUBLISHED
        assert len(result.platforms) == 2
        assert Platform.YOUTUBE in result.platforms
        assert Platform.TIKTOK in result.platforms
        assert len(result.published_urls) == 1
        assert result.metadata["test_key"] == "test_value"

    def test_publish_result_immutability(self):
        """Test that PublishResult is immutable (frozen=True)."""
        result = PublishResult(
            post_id="post_123",
            status=PublishStatus.PUBLISHED,
            platforms=(Platform.YOUTUBE,),
        )

        with pytest.raises(AttributeError, match="can't set attribute|has no setter"):
            result.post_id = "post_456"

    def test_publish_result_with_scheduled_time(self):
        """Test PublishResult with scheduled_time."""
        scheduled_time = datetime(2025, 1, 20, 14, 0, 0, tzinfo=UTC)
        result = PublishResult(
            post_id="post_123",
            status=PublishStatus.SCHEDULED,
            platforms=(Platform.YOUTUBE,),
            scheduled_time=scheduled_time,
        )

        assert result.scheduled_time == scheduled_time
        assert result.status == PublishStatus.SCHEDULED

    def test_publish_result_with_error(self):
        """Test PublishResult with error_message."""
        result = PublishResult(
            post_id="post_123",
            status=PublishStatus.FAILED,
            platforms=(Platform.YOUTUBE,),
            error_message="Rate limit exceeded",
        )

        assert result.status == PublishStatus.FAILED
        assert result.error_message == "Rate limit exceeded"

    def test_publish_result_default_values(self):
        """Test PublishResult default values."""
        result = PublishResult(
            post_id="post_123",
            status=PublishStatus.PUBLISHED,
            platforms=(Platform.YOUTUBE,),
        )

        assert result.scheduled_time is None
        assert result.published_urls == ()
        assert result.error_message is None
        assert result.metadata == {}


class TestPublishMetadata:
    """Test PublishMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating a PublishMetadata object."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Amazing Product Review",
            description="Check out this amazing product!",
            hashtags=["#ad", "#tech", "#review"],
            keywords=["product", "review", "tech"],
        )

        assert metadata.platform == Platform.YOUTUBE
        assert metadata.title == "Amazing Product Review"
        assert metadata.description == "Check out this amazing product!"
        assert len(metadata.hashtags) == 3
        assert len(metadata.keywords) == 3

    def test_metadata_without_title(self):
        """Test creating metadata without title (allowed for some platforms)."""
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="TikTok caption #viral",
        )

        assert metadata.title is None
        assert metadata.description == "TikTok caption #viral"

    def test_validate_limits_youtube_valid(self):
        """Test validate_limits for YouTube with valid content."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Valid Title",
            description="Valid description within limits",
            hashtags=["#tag1", "#tag2", "#tag3"],
        )

        is_valid, message = metadata.validate_limits()
        assert is_valid is True
        assert message == "Content within limits"

    def test_validate_limits_youtube_title_too_long(self):
        """Test validate_limits for YouTube with title exceeding 100 chars."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="A" * 101,  # 101 chars
            description="Valid description",
        )

        is_valid, message = metadata.validate_limits()
        assert is_valid is False
        assert "Title exceeds" in message
        assert "100" in message

    def test_validate_limits_youtube_description_too_long(self):
        """Test validate_limits for YouTube with description exceeding 5000 chars."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Valid Title",
            description="A" * 5001,  # 5001 chars
        )

        is_valid, message = metadata.validate_limits()
        assert is_valid is False
        assert "Description exceeds" in message
        assert "5000" in message

    def test_validate_limits_youtube_too_few_hashtags(self):
        """Test validate_limits for YouTube with fewer than 3 hashtags."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Valid Title",
            description="Valid description",
            hashtags=["#tag1", "#tag2"],  # Only 2 hashtags
        )

        is_valid, message = metadata.validate_limits()
        assert is_valid is False
        assert "Hashtags must be between 3 and 15" in message

    def test_validate_limits_youtube_too_many_hashtags(self):
        """Test validate_limits for YouTube with more than 15 hashtags."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Valid Title",
            description="Valid description",
            hashtags=[f"#tag{i}" for i in range(16)],  # 16 hashtags
        )

        is_valid, message = metadata.validate_limits()
        assert is_valid is False
        assert "Hashtags must be between 3 and 15" in message

    def test_validate_limits_tiktok_valid(self):
        """Test validate_limits for TikTok with valid content."""
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="Short caption #viral",
            hashtags=["#viral", "#fyp", "#trending"],
        )

        is_valid, message = metadata.validate_limits()
        assert is_valid is True

    def test_validate_limits_tiktok_description_too_long(self):
        """Test validate_limits for TikTok with description exceeding 150 chars."""
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="A" * 151,  # 151 chars
        )

        is_valid, message = metadata.validate_limits()
        assert is_valid is False
        assert "Description exceeds" in message

    def test_validate_limits_instagram_valid(self):
        """Test validate_limits for Instagram with valid content."""
        metadata = PublishMetadata(
            platform=Platform.INSTAGRAM,
            title=None,
            description="Instagram caption with hashtags",
            hashtags=["#insta", "#photo", "#art", "#love", "#instagood"],
        )

        is_valid, message = metadata.validate_limits()
        assert is_valid is True

    def test_format_content_youtube(self):
        """Test format_content for YouTube."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="My Video Title",
            description="This is the description.",
            hashtags=["#tag1", "#tag2", "#tag3"],
        )

        content = metadata.format_content()
        assert "My Video Title" in content
        assert "This is the description." in content
        assert "#tag1 #tag2 #tag3" in content

    def test_format_content_tiktok(self):
        """Test format_content for TikTok."""
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="TikTok caption",
            hashtags=["#viral", "#fyp"],
        )

        content = metadata.format_content()
        assert "TikTok caption" in content
        assert "#viral #fyp" in content

    def test_format_content_no_hashtags(self):
        """Test format_content without hashtags."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Title",
            description="Description only",
        )

        content = metadata.format_content()
        assert "Title" in content
        assert "Description only" in content


class TestPublisherConfig:
    """Test PublisherConfig dataclass."""

    def test_config_creation(self):
        """Test creating a PublisherConfig object."""
        config = PublisherConfig(
            provider="late",
            api_key="sk_test_123",
            vercel_token="vercel_token_456",
            default_platforms=[Platform.YOUTUBE, Platform.TIKTOK],
            immediate_publish=True,
            max_retries=5,
            timeout=60.0,
        )

        assert config.provider == "late"
        assert config.api_key == "sk_test_123"
        assert config.vercel_token == "vercel_token_456"
        assert len(config.default_platforms) == 2
        assert config.immediate_publish is True
        assert config.max_retries == 5
        assert config.timeout == 60.0

    def test_config_default_values(self):
        """Test PublisherConfig default values."""
        config = PublisherConfig(
            provider="late",
            api_key="sk_test_123",
        )

        assert config.vercel_token is None
        assert config.default_platforms == []
        assert config.immediate_publish is True
        assert config.max_retries == 3
        assert config.timeout == 30.0
        assert config.stagger_delay_min == 30
        assert config.stagger_delay_max == 60

    def test_config_with_privacy_settings(self):
        """Test PublisherConfig with privacy settings."""
        config = PublisherConfig(
            provider="late",
            api_key="sk_test_123",
            privacy_settings={
                "youtube": "public",
                "tiktok": "public",
                "instagram": "everyone",
            },
        )

        assert config.privacy_settings is not None
        assert config.privacy_settings["youtube"] == "public"


class TestBatchPublishSummary:
    """Test BatchPublishSummary dataclass."""

    def test_summary_creation(self):
        """Test creating a BatchPublishSummary object."""
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

    def test_summary_default_values(self):
        """Test BatchPublishSummary default values."""
        summary = BatchPublishSummary(
            total_videos=5,
            successful=5,
            failed=0,
            skipped=0,
        )

        assert summary.platform_results == {}
        assert summary.errors == []
        assert summary.duration_seconds == 0.0

    def test_get_success_rate(self):
        """Test get_success_rate calculation."""
        summary = BatchPublishSummary(
            total_videos=10,
            successful=8,
            failed=2,
            skipped=0,
        )

        assert summary.get_success_rate() == 80.0

    def test_get_success_rate_zero_videos(self):
        """Test get_success_rate with zero videos."""
        summary = BatchPublishSummary(
            total_videos=0,
            successful=0,
            failed=0,
            skipped=0,
        )

        assert summary.get_success_rate() == 0.0

    def test_add_platform_result_success(self):
        """Test add_platform_result for successful publish."""
        summary = BatchPublishSummary(
            total_videos=1,
            successful=1,
            failed=0,
            skipped=0,
        )

        summary.add_platform_result(Platform.YOUTUBE, success=True)

        assert Platform.YOUTUBE in summary.platform_results
        assert summary.platform_results[Platform.YOUTUBE]["successful"] == 1
        assert summary.platform_results[Platform.YOUTUBE]["failed"] == 0

    def test_add_platform_result_failure(self):
        """Test add_platform_result for failed publish."""
        summary = BatchPublishSummary(
            total_videos=1,
            successful=0,
            failed=1,
            skipped=0,
        )

        summary.add_platform_result(Platform.YOUTUBE, success=False)

        assert Platform.YOUTUBE in summary.platform_results
        assert summary.platform_results[Platform.YOUTUBE]["successful"] == 0
        assert summary.platform_results[Platform.YOUTUBE]["failed"] == 1

    def test_add_platform_result_multiple_calls(self):
        """Test add_platform_result with multiple calls for same platform."""
        summary = BatchPublishSummary(
            total_videos=3,
            successful=2,
            failed=1,
            skipped=0,
        )

        summary.add_platform_result(Platform.YOUTUBE, success=True)
        summary.add_platform_result(Platform.YOUTUBE, success=True)
        summary.add_platform_result(Platform.YOUTUBE, success=False)

        assert summary.platform_results[Platform.YOUTUBE]["successful"] == 2
        assert summary.platform_results[Platform.YOUTUBE]["failed"] == 1

    def test_add_error(self):
        """Test add_error method."""
        summary = BatchPublishSummary(
            total_videos=1,
            successful=0,
            failed=1,
            skipped=0,
        )

        summary.add_error("B0ABC123", "Upload failed")

        assert len(summary.errors) == 1
        assert summary.errors[0]["video_id"] == "B0ABC123"
        assert summary.errors[0]["error"] == "Upload failed"

    def test_add_multiple_errors(self):
        """Test add_error with multiple errors."""
        summary = BatchPublishSummary(
            total_videos=3,
            successful=1,
            failed=2,
            skipped=0,
        )

        summary.add_error("B0ABC123", "Upload failed")
        summary.add_error("B0DEF456", "Rate limit exceeded")

        assert len(summary.errors) == 2
        assert summary.errors[0]["video_id"] == "B0ABC123"
        assert summary.errors[1]["video_id"] == "B0DEF456"

    def test_summary_with_duration(self):
        """Test summary with duration tracking."""
        summary = BatchPublishSummary(
            total_videos=5,
            successful=5,
            failed=0,
            skipped=0,
            duration_seconds=120.5,
        )

        assert summary.duration_seconds == 120.5
        assert summary.get_success_rate() == 100.0
