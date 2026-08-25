"""Tests for publish mode helpers (unified and platform-specific)."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.publisher.models import FirstCommentConfig, Platform
from src.publisher.publish_modes import publish_product


@pytest.fixture
def mock_publisher():
    """Create mock publisher with publish method.

    ``first_comment_config`` is a real config object rather than an auto-created
    mock attribute. On an ``AsyncMock`` every attribute call returns a coroutine,
    so ``platforms.get(...)`` would hand the builder a coroutine where its
    contract says ``str``. These tests don't exercise first comments, so the
    faithful stand-in is a disabled config.
    """
    publisher = AsyncMock()
    publisher.publish.return_value = {"post_id": "post_123", "status": "published"}
    publisher.first_comment_config = FirstCommentConfig(enabled=False)
    return publisher


@pytest.fixture
def platforms():
    """Standard platform list."""
    return [
        {"platform": "youtube", "account_id": "acc_yt"},
        {"platform": "tiktok", "account_id": "acc_tt"},
    ]


@pytest.fixture
def mock_metadata():
    """Create mock PublishMetadata."""
    from src.publisher.models import PublishMetadata

    return PublishMetadata(
        platform=Platform.YOUTUBE,
        title="Test Video",
        description="Test description",
        hashtags=["test"],
        keywords=["test"],
        product_id="B0TEST001",
    )


class TestPublishProductUnified:
    """Test unified publishing mode."""

    @pytest.mark.asyncio
    async def test_unified_mode_calls_publish_once(
        self, mock_publisher, platforms, mock_metadata
    ):
        """Unified mode should make a single publish call with all platforms."""
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=mock_metadata,
        ):
            results = await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                platform_specific=False,
            )

        assert len(results) == 1
        assert results[0]["platform"] == "all"
        mock_publisher.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_unified_mode_passes_all_platforms(
        self, mock_publisher, platforms, mock_metadata
    ):
        """Unified mode passes the full platforms list to publish()."""
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=mock_metadata,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
            )

        call_kwargs = mock_publisher.publish.call_args.kwargs
        assert call_kwargs["platforms"] == platforms

    @pytest.mark.asyncio
    async def test_unified_mode_no_metadata_raises(self, mock_publisher, platforms):
        """Unified mode raises ValueError if no metadata found."""
        with (
            patch(
                "src.publisher.publish_modes.load_platform_metadata",
                return_value=None,
            ),
            pytest.raises(ValueError, match="No metadata found"),
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
            )

    @pytest.mark.asyncio
    async def test_unified_mode_with_schedule_time(
        self, mock_publisher, platforms, mock_metadata
    ):
        """Schedule time is passed through to publisher."""
        schedule = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=mock_metadata,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                schedule_time=schedule,
            )

        call_kwargs = mock_publisher.publish.call_args.kwargs
        assert call_kwargs["scheduled_time"] == schedule


class TestPublishProductPlatformSpecific:
    """Test platform-specific publishing mode."""

    @pytest.mark.asyncio
    async def test_platform_specific_calls_publish_per_platform(
        self, mock_publisher, platforms, mock_metadata
    ):
        """Platform-specific mode makes one publish call per platform."""
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=mock_metadata,
        ):
            results = await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                platform_specific=True,
            )

        assert len(results) == 2
        assert results[0]["platform"] == "youtube"
        assert results[1]["platform"] == "tiktok"
        assert mock_publisher.publish.call_count == 2

    @pytest.mark.asyncio
    async def test_platform_specific_each_call_has_single_platform(
        self, mock_publisher, platforms, mock_metadata
    ):
        """Each publish call in platform-specific mode has exactly one platform."""
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=mock_metadata,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                platform_specific=True,
            )

        for call in mock_publisher.publish.call_args_list:
            assert len(call.kwargs["platforms"]) == 1

    @pytest.mark.asyncio
    async def test_platform_specific_fallback_metadata(self, mock_publisher, platforms):
        """Falls back to other platform metadata when specific not found."""
        from src.publisher.models import PublishMetadata

        fallback = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Fallback",
            description="Fallback desc",
            hashtags=[],
            keywords=[],
            product_id="B0TEST001",
        )

        def load_side_effect(product_id, platform, outputs_dir):
            # Return None for tiktok, fallback for youtube
            if isinstance(platform, Platform):
                platform = platform.value
            if platform == "youtube":
                return fallback
            return None

        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            side_effect=load_side_effect,
        ):
            results = await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                platform_specific=True,
            )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_platform_specific_no_fallback_raises(self, mock_publisher):
        """Raises ValueError if no metadata found for any platform."""
        platforms = [{"platform": "youtube", "account_id": "acc_yt"}]
        with (
            patch(
                "src.publisher.publish_modes.load_platform_metadata",
                return_value=None,
            ),
            pytest.raises(ValueError, match="No metadata found"),
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                platform_specific=True,
            )

    @pytest.mark.asyncio
    async def test_outputs_dir_converted_to_path(
        self, mock_publisher, platforms, mock_metadata
    ):
        """String outputs_dir is converted to Path."""
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=mock_metadata,
        ) as mock_load:
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="custom/outputs",
            )

        call_args = mock_load.call_args
        assert isinstance(call_args.args[2], Path)


class TestAffiliateDisclosure:
    """Affiliate program literal phrase is threaded into captions."""

    PHRASE = "As an Amazon Associate I earn from qualifying purchases"

    @pytest.mark.asyncio
    async def test_unified_includes_disclosure_phrase(
        self, mock_publisher, platforms, mock_metadata
    ):
        """Unified publish content includes the affiliate phrase."""
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=mock_metadata,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                disclosure_phrase=self.PHRASE,
            )

        content = mock_publisher.publish.call_args.kwargs["content"]
        assert self.PHRASE in content
        assert content.startswith("#ad\n")

    @pytest.mark.asyncio
    async def test_unified_omits_phrase_when_none(
        self, mock_publisher, platforms, mock_metadata
    ):
        """No affiliate phrase is inserted when disclosure_phrase is None."""
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=mock_metadata,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                disclosure_phrase=None,
            )

        content = mock_publisher.publish.call_args.kwargs["content"]
        assert self.PHRASE not in content
        assert content.startswith("#ad\n")

    @pytest.mark.asyncio
    async def test_platform_specific_includes_disclosure_phrase(
        self, mock_publisher, platforms, mock_metadata
    ):
        """Platform-specific mode includes the phrase in every post."""
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=mock_metadata,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                platform_specific=True,
                disclosure_phrase=self.PHRASE,
            )

        for call in mock_publisher.publish.call_args_list:
            assert self.PHRASE in call.kwargs["content"]


class TestPublishClampsMetadata:
    """Metadata exceeding platform limits is clamped before publish (#109)."""

    @pytest.mark.asyncio
    async def test_unified_mode_clamps_oversized_title(self, mock_publisher, platforms):
        from src.publisher.models import PublishMetadata

        long_title = "Word " * 30  # 150 chars, over the 100 cap
        oversized = PublishMetadata(
            platform=Platform.YOUTUBE,
            title=long_title,
            description="ok",
            hashtags=[],
            keywords=[],
            product_id="B0TEST001",
        )
        assert len(oversized.title) > 100

        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=oversized,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                platform_specific=False,
            )

        # clamp_to_limits mutates the metadata before format_content runs,
        # so the publisher.publish call sees a clamped title length.
        assert oversized.title is not None
        assert len(oversized.title) <= 100

    @pytest.mark.asyncio
    async def test_platform_specific_clamps_oversized_description(
        self, mock_publisher, platforms
    ):
        from src.publisher.models import PublishMetadata

        long_desc = "word " * 600  # ~3000 chars, over TikTok's 2200 cap
        oversized = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description=long_desc,
            hashtags=[],
            keywords=[],
            product_id="B0TEST001",
        )
        assert len(oversized.description) > 2200

        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=oversized,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=[{"platform": "tiktok", "account_id": "acc_tt"}],
                outputs_dir="outputs",
                platform_specific=True,
            )

        assert len(oversized.description) <= 2200


class TestPlatformContentsCarryFullPayload:
    """Entries must carry content and title, not only the comment (#195).

    The consumer treats this dict as authoritative and reads ``content`` and
    ``title`` from it, so an entry holding only ``first_comment`` blanked the
    caption and sent no YouTube title.
    """

    def _publisher_with_comments(self, template: str = "{closing_line}"):
        publisher = AsyncMock()
        publisher.first_comment_config = FirstCommentConfig(
            enabled=True, platforms={"youtube": template}
        )
        return publisher

    def _youtube_metadata(self):
        from src.publisher.models import PublishMetadata

        return PublishMetadata(
            platform=Platform.YOUTUBE,
            title="A real product title",
            description="Body copy here.",
            hashtags=["tech"],
        )

    def test_entry_carries_content_and_title(self, tmp_path):
        from src.publisher.publish_modes import _build_platform_contents_with_comments

        meta = self._youtube_metadata()
        with patch(
            "src.publisher.publish_modes.build_first_comment", return_value="a comment"
        ):
            out = _build_platform_contents_with_comments(
                self._publisher_with_comments(),
                [{"platform": "youtube", "account_id": "acc"}],
                "B0TEST",
                tmp_path,
                metadata_by_platform={"youtube": meta},
            )

        assert out is not None
        entry = out["youtube"]
        assert entry["first_comment"] == "a comment"
        assert entry["title"] == "A real product title"
        # Content is the fully formatted caption, disclosure included.
        assert entry["content"] == meta.format_content()
        assert entry["content"].startswith("#ad")

    def test_the_payload_survives_a_platform_with_no_comment(self, tmp_path):
        """A comment that could not be built must not take the title with it.

        The title is the platform's primary ranking signal, and with none
        sent the platform derives one from the caption -- whose first line is
        the `#ad` disclosure.
        """
        from src.publisher.publish_modes import _build_platform_contents_with_comments

        with patch("src.publisher.publish_modes.build_first_comment", return_value=""):
            out = _build_platform_contents_with_comments(
                self._publisher_with_comments(),
                [{"platform": "youtube", "account_id": "acc"}],
                "B0TEST",
                tmp_path,
                metadata_by_platform={"youtube": self._youtube_metadata()},
            )

        assert out is not None
        assert out["youtube"]["title"]
        assert out["youtube"]["content"]
        assert "first_comment" not in out["youtube"]

    def test_the_payload_is_built_with_comments_disabled(self, tmp_path):
        """Turning comments off must not turn the title off with them.

        This was the path that masked the bug: no platform_contents meant the
        consumer's per-platform branch never ran, so the title went missing
        on exactly the publishes that generated no comment.
        """
        from src.publisher.publish_modes import _build_platform_contents_with_comments

        publisher = AsyncMock()
        publisher.first_comment_config = FirstCommentConfig(enabled=False)

        out = _build_platform_contents_with_comments(
            publisher,
            [{"platform": "youtube", "account_id": "acc"}],
            "B0TEST",
            tmp_path,
            metadata_by_platform={"youtube": self._youtube_metadata()},
        )

        assert out is not None
        assert out["youtube"]["title"]
        assert "first_comment" not in out["youtube"]


class TestMaterialConnectionReachesThePublisher:
    """The render's disclosure decision must survive the call.

    Both modes read it off the metadata and pass it to `publish()`, which
    defaults it to True. A dropped kwarg is therefore silent: every topic
    render goes back to declaring commercial content to TikTok, with nothing
    failing. These pin the argument at each call site rather than the payload
    it eventually produces.
    """

    @pytest.fixture
    def topic_metadata(self):
        """A render with nothing to disclose: no affiliate relationship."""
        from src.publisher.models import PublishMetadata

        meta = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Why your phone charges slowly",
            description="Body.",
            hashtags=["tech"],
            product_id="topic-why-your-phone-charges-slowly",
        )
        meta.carries_affiliate_content = False
        return meta

    @pytest.mark.asyncio
    async def test_unified_mode_passes_the_decision(
        self, mock_publisher, platforms, topic_metadata
    ):
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=topic_metadata,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="topic-x",
                platforms=platforms,
                outputs_dir="outputs",
                platform_specific=False,
            )

        call = mock_publisher.publish.call_args
        assert call.kwargs["carries_affiliate_content"] is False

    @pytest.mark.asyncio
    async def test_platform_specific_mode_passes_the_decision(
        self, mock_publisher, platforms, topic_metadata
    ):
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=topic_metadata,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="topic-x",
                platforms=platforms,
                outputs_dir="outputs",
                platform_specific=True,
            )

        assert mock_publisher.publish.call_count == len(platforms)
        for call in mock_publisher.publish.call_args_list:
            assert call.kwargs["carries_affiliate_content"] is False

    @pytest.mark.asyncio
    async def test_an_affiliate_render_still_discloses(
        self, mock_publisher, platforms, mock_metadata
    ):
        """The gate must not become a blanket suppression."""
        with patch(
            "src.publisher.publish_modes.load_platform_metadata",
            return_value=mock_metadata,
        ):
            await publish_product(
                publisher=mock_publisher,
                media_id="media_123",
                product_id="B0TEST001",
                platforms=platforms,
                outputs_dir="outputs",
                platform_specific=False,
            )

        call = mock_publisher.publish.call_args
        assert call.kwargs["carries_affiliate_content"] is True
