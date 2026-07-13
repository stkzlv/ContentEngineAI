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

        with pytest.raises(
            (AttributeError, Exception),
            match="can't set attribute|has no setter|cannot assign",
        ):
            result.post_id = "post_456"  # type: ignore[misc]

    def test_publish_result_with_scheduled_time(self):
        """Test PublishResult with scheduled_time."""
        from datetime import timedelta

        scheduled_time = datetime.now(UTC) + timedelta(days=7)
        result = PublishResult(
            post_id="post_123",
            status=PublishStatus.SCHEDULED,
            platforms=(Platform.YOUTUBE,),
            scheduled_time=scheduled_time,
        )

        assert result.scheduled_time == scheduled_time
        assert result.status == PublishStatus.SCHEDULED

    def test_publish_result_with_past_scheduled_time(self):
        """Test PublishResult accepts past scheduled_time (historical results)."""
        from datetime import timedelta

        past_time = datetime.now(UTC) - timedelta(days=30)
        result = PublishResult(
            post_id="post_123",
            status=PublishStatus.PUBLISHED,
            platforms=(Platform.YOUTUBE,),
            scheduled_time=past_time,
        )

        assert result.scheduled_time == past_time

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
        # '#ad' is dropped from hashtags because it duplicates the disclosure
        # which now leads the formatted caption.
        assert metadata.hashtags == ["tech", "review"]
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
        assert "Hashtags: 3-15 required" in message

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
        assert "Hashtags: 3-15 required" in message

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
        """Test validate_limits for TikTok with description exceeding 2200 chars.

        2200 is TikTok's actual platform hard cap; the 150-char "optimal" soft
        target lives in the prompt config, not the publisher validator.
        """
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="A" * 2201,
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
        """Test format_content for YouTube (description + hashtags, no title)."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="My Video Title",
            description="This is the description.",
            hashtags=["tag1", "tag2", "tag3"],
        )

        content = metadata.format_content()
        # Title is NOT included in format_content - it's passed separately to platform APIs
        assert "My Video Title" not in content
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
        """Test format_content without hashtags (disclosure + description only)."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Title",
            description="Description only",
        )

        content = metadata.format_content()
        # Title is NOT included - it's passed separately to platform APIs
        assert "Title" not in content
        # Disclosure leads, then description, separated by blank line
        assert content == "#ad\n\nDescription only"

    def test_format_content_with_affiliate_disclosure(self):
        """Affiliate disclosure phrase appears between #ad and description."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Title",
            description="Check out this product.",
            affiliate_disclosure="As an Amazon Associate I earn from qualifying purchases",
        )
        content = metadata.format_content()
        assert content.startswith("#ad\n")
        assert "As an Amazon Associate I earn from qualifying purchases" in content
        assert "Check out this product." in content
        lines = content.split("\n\n")
        assert lines[0] == "#ad"
        assert "As an Amazon Associate I earn from qualifying purchases" in lines[1]
        assert lines[2] == "Check out this product."

    def test_format_content_disclosure_leads(self):
        """Disclosure must be the first line of the formatted content."""
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="Tested this $15 cable for a week.",
            hashtags=["techfinds", "amazonfinds"],
        )

        content = metadata.format_content()
        first_line = content.split("\n", 1)[0]
        assert first_line == "#ad"

    def test_format_content_dedupes_disclosure_from_hashtags(self):
        """If '#ad' is also in hashtags, drop it (disclosure leads instead)."""
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="Caption text.",
            hashtags=["ad", "techfinds"],
        )

        content = metadata.format_content()
        # The leading line is #ad
        assert content.startswith("#ad\n")
        # The trailing hashtag block does not repeat #ad
        assert "#ad #techfinds" not in content
        assert "#techfinds" in content
        # Only one '#ad' total
        assert content.count("#ad") == 1

    def test_format_content_custom_disclosure(self):
        """Disclosure can be overridden, e.g. for Spanish-language renders."""
        metadata = PublishMetadata(
            platform=Platform.INSTAGRAM,
            title=None,
            description="Texto en español.",
            hashtags=["gadgetstech"],
            disclosure="#publi",
        )

        content = metadata.format_content()
        assert content.startswith("#publi\n")
        assert "#gadgetstech" in content

    def test_format_content_empty_disclosure_omits_line(self):
        """An empty disclosure string skips the leading line entirely."""
        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Title",
            description="Description.",
            hashtags=["tag1"],
            disclosure="",
        )

        content = metadata.format_content()
        # No leading #ad
        assert not content.startswith("#ad")
        assert content == "Description.\n\n#tag1"

    @pytest.mark.parametrize("variant", ["ad", "AD", "Ad", "aD", "#ad", "#AD"])
    def test_format_content_dedupes_ad_case_insensitive(self, variant):
        """Disclosure dedup matches case-insensitively against the bare token."""
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="Caption.",
            hashtags=[variant, "techfinds"],
        )
        # The disclosure leads on its own line; the supplied #ad/AD/Ad variant
        # is dropped from hashtags so it doesn't appear twice in the caption.
        assert metadata.hashtags == ["techfinds"]
        assert metadata.format_content().count("#ad") == 1

    def test_to_dict_includes_disclosure(self):
        """to_dict serialization carries the disclosure for round-trip."""
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="Body.",
            disclosure="#publi",
        )
        as_dict = metadata.to_dict()
        assert as_dict["disclosure"] == "#publi"


class TestPublisherConfig:
    """Test PublisherConfig dataclass."""

    def test_config_creation(self):
        """Test creating a PublisherConfig object."""
        config = PublisherConfig(
            provider="late",
            api_key="sk_test_123",
            vercel_token="vercel_token_456",  # noqa: S106
            default_platforms=[Platform.YOUTUBE, Platform.TIKTOK],
            immediate_publish=True,
            max_retries=5,
            timeout=60.0,
        )

        assert config.provider == "late"
        assert config.api_key == "sk_test_123"
        assert config.vercel_token == "vercel_token_456"  # noqa: S105
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
        assert config.default_platforms == [
            Platform.YOUTUBE,
            Platform.TIKTOK,
            Platform.INSTAGRAM,
        ]
        assert config.immediate_publish is False
        assert config.max_retries == 3
        assert config.timeout == 120.0
        assert config.stagger_delay_min == 30
        assert config.stagger_delay_max == 60

    def test_config_with_privacy_settings(self):
        """Test PublisherConfig with privacy settings."""
        config = PublisherConfig(
            provider="late",
            api_key="sk_test_123",
            privacy_settings={
                Platform.YOUTUBE: "public",
                Platform.TIKTOK: "public",
                Platform.INSTAGRAM: "everyone",
            },
        )

        assert config.privacy_settings is not None
        assert config.privacy_settings[Platform.YOUTUBE] == "public"


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


class TestTrimOnWordBoundary:
    """Module-level helper used by clamp_to_limits (#109)."""

    def test_no_op_when_under_limit(self):
        from src.publisher.models import _trim_on_word_boundary

        assert _trim_on_word_boundary("hello world", 100) == "hello world"

    def test_breaks_on_word_boundary(self):
        from src.publisher.models import _trim_on_word_boundary

        text = "one two three four five six seven eight nine ten"
        out = _trim_on_word_boundary(text, 20)
        assert len(out) <= 20
        assert out.endswith("...")
        # Trimmed at a space, so no partial word before the ellipsis.
        assert " " in out
        body = out[:-3].rstrip()
        assert not body.endswith("o")  # not in the middle of "two"

    def test_hard_cut_when_no_whitespace(self):
        from src.publisher.models import _trim_on_word_boundary

        text = "abcdefghijklmnopqrstuvwxyz" * 5
        out = _trim_on_word_boundary(text, 20)
        assert len(out) == 20
        assert out.endswith("...")


class TestClampToLimits:
    """PublishMetadata.clamp_to_limits trims long title/description (#109)."""

    def test_clamps_youtube_title(self):
        long_title = (
            "Anker SOLIX C1000 Gen 2 Portable Power Station, "
            "2,000W Peak 3,000W Surge LFP Battery for Camping RV "
            "Home Backup and Outdoor Emergency Use"
        )
        assert len(long_title) > 100

        meta = PublishMetadata(
            platform=Platform.YOUTUBE,
            title=long_title,
            description="desc",
        )
        trimmed = meta.clamp_to_limits()

        assert "title" in trimmed
        assert meta.title is not None
        assert len(meta.title) <= 100
        assert meta.title.endswith("...")
        assert meta.character_counts["title"] == len(meta.title)

    def test_no_op_when_under_limit(self):
        meta = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Short title",
            description="Short description",
        )
        original_title = meta.title
        original_desc = meta.description

        trimmed = meta.clamp_to_limits()

        assert trimmed == ()
        assert meta.title == original_title
        assert meta.description == original_desc

    def test_clamps_tiktok_description(self):
        long_desc = "word " * 500  # ~2500 chars
        assert len(long_desc) > 2200

        meta = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description=long_desc,
        )
        trimmed = meta.clamp_to_limits()

        assert "description" in trimmed
        assert len(meta.description) <= 2200
        assert meta.character_counts["description"] == len(meta.description)

    def test_clamps_both_title_and_description(self):
        meta = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="x " * 60,  # 120 chars
            description="y " * 3000,  # 6000 chars
        )
        trimmed = meta.clamp_to_limits()

        assert set(trimmed) == {"title", "description"}
        assert meta.title is not None
        assert len(meta.title) <= 100
        assert len(meta.description) <= 5000
