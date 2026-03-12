"""Tests for first comment builder."""

import json
from pathlib import Path

import pytest

from src.publisher.first_comment import build_first_comment
from src.publisher.models import FirstCommentConfig, Platform, PublishMetadata


@pytest.fixture
def enabled_config():
    """Config with YouTube and Instagram templates."""
    return FirstCommentConfig(
        enabled=True,
        platforms={
            "youtube": "Get it here: {affiliate_link}\n\nSubscribe!",
            "instagram": "{affiliate_link}",
        },
    )


@pytest.fixture
def product_data():
    """Standard product data with affiliate link."""
    return {
        "title": "Wireless Earbuds",
        "affiliate_link": "https://amzn.to/abc123",
        "shortened_affiliate_link": "https://short.link/x",
    }


@pytest.fixture
def outputs_dir(tmp_path, product_data):
    """Create outputs dir with data.json."""
    product_dir = tmp_path / "B0TEST001"
    product_dir.mkdir()
    (product_dir / "data.json").write_text(json.dumps(product_data))
    return tmp_path


class TestBuildFirstComment:
    """Test build_first_comment()."""

    def test_youtube_template_rendered(self, enabled_config, outputs_dir):
        result = build_first_comment(
            enabled_config, "youtube", "B0TEST001", outputs_dir
        )
        assert result == "Get it here: https://short.link/x\n\nSubscribe!"

    def test_instagram_template_rendered(self, enabled_config, outputs_dir):
        result = build_first_comment(
            enabled_config, "instagram", "B0TEST001", outputs_dir
        )
        assert result == "https://short.link/x"

    def test_tiktok_always_skipped(self, outputs_dir):
        config = FirstCommentConfig(
            enabled=True, platforms={"tiktok": "{affiliate_link}"}
        )
        result = build_first_comment(config, "tiktok", "B0TEST001", outputs_dir)
        assert result is None

    def test_disabled_config_returns_none(self, outputs_dir):
        config = FirstCommentConfig(
            enabled=False, platforms={"youtube": "{affiliate_link}"}
        )
        result = build_first_comment(config, "youtube", "B0TEST001", outputs_dir)
        assert result is None

    def test_unconfigured_platform_returns_none(self, enabled_config, outputs_dir):
        result = build_first_comment(
            enabled_config, "facebook", "B0TEST001", outputs_dir
        )
        assert result is None

    def test_missing_data_json_returns_none(self, enabled_config, tmp_path):
        (tmp_path / "B0MISSING").mkdir()
        result = build_first_comment(enabled_config, "youtube", "B0MISSING", tmp_path)
        assert result is None

    def test_no_affiliate_link_returns_none(self, enabled_config, tmp_path):
        product_dir = tmp_path / "B0NOLINK"
        product_dir.mkdir()
        (product_dir / "data.json").write_text(json.dumps({"title": "No link"}))
        result = build_first_comment(enabled_config, "youtube", "B0NOLINK", tmp_path)
        assert result is None

    def test_falls_back_to_affiliate_link_when_no_shortened(
        self, enabled_config, tmp_path
    ):
        product_dir = tmp_path / "B0LONG"
        product_dir.mkdir()
        data = {"title": "Test", "affiliate_link": "https://amzn.to/long"}
        (product_dir / "data.json").write_text(json.dumps(data))

        result = build_first_comment(enabled_config, "youtube", "B0LONG", tmp_path)
        assert "https://amzn.to/long" in result

    def test_data_json_as_list(self, enabled_config, tmp_path):
        """data.json can be a list (scraper output), first element used."""
        product_dir = tmp_path / "B0LIST"
        product_dir.mkdir()
        data = [{"title": "First", "affiliate_link": "https://amzn.to/first"}]
        (product_dir / "data.json").write_text(json.dumps(data))

        result = build_first_comment(enabled_config, "youtube", "B0LIST", tmp_path)
        assert "https://amzn.to/first" in result

    def test_invalid_json_returns_none(self, enabled_config, tmp_path):
        product_dir = tmp_path / "B0BAD"
        product_dir.mkdir()
        (product_dir / "data.json").write_text("not json{{{")
        result = build_first_comment(enabled_config, "youtube", "B0BAD", tmp_path)
        assert result is None

    def test_product_title_placeholder(self, tmp_path):
        config = FirstCommentConfig(
            enabled=True,
            platforms={"youtube": "Check out {product_title}: {affiliate_link}"},
        )
        product_dir = tmp_path / "B0TITLE"
        product_dir.mkdir()
        data = {"title": "Cool Gadget", "affiliate_link": "https://amzn.to/x"}
        (product_dir / "data.json").write_text(json.dumps(data))

        result = build_first_comment(config, "youtube", "B0TITLE", tmp_path)
        assert result == "Check out Cool Gadget: https://amzn.to/x"


class TestMoveHashtagsToComment:
    """Test hashtag migration to first comment."""

    def test_hashtags_moved_for_instagram(self, tmp_path):
        config = FirstCommentConfig(
            enabled=True,
            move_hashtags_to_comment=True,
            platforms={"instagram": "{affiliate_link} {hashtags}"},
        )
        product_dir = tmp_path / "B0HASH"
        product_dir.mkdir()
        data = {"title": "Test", "affiliate_link": "https://amzn.to/h"}
        (product_dir / "data.json").write_text(json.dumps(data))

        metadata = PublishMetadata(
            platform=Platform.INSTAGRAM,
            title="Test",
            description="desc",
            hashtags=["earbuds", "tech"],
            keywords=[],
            product_id="B0HASH",
        )

        result = build_first_comment(
            config, "instagram", "B0HASH", tmp_path, metadata=metadata
        )
        assert "#earbuds" in result
        assert "#tech" in result
        assert "https://amzn.to/h" in result

    def test_hashtags_not_moved_for_youtube(self, tmp_path):
        config = FirstCommentConfig(
            enabled=True,
            move_hashtags_to_comment=True,
            platforms={"youtube": "{affiliate_link} {hashtags}"},
        )
        product_dir = tmp_path / "B0YT"
        product_dir.mkdir()
        data = {"title": "Test", "affiliate_link": "https://amzn.to/y"}
        (product_dir / "data.json").write_text(json.dumps(data))

        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Test",
            description="desc",
            hashtags=["tech"],
            keywords=[],
            product_id="B0YT",
        )

        result = build_first_comment(
            config, "youtube", "B0YT", tmp_path, metadata=metadata
        )
        # Hashtags only moved for instagram, so {hashtags} is empty string
        assert "#tech" not in result

    def test_hashtags_not_double_prefixed(self, tmp_path):
        """Hashtags already starting with # don't get double-prefixed."""
        config = FirstCommentConfig(
            enabled=True,
            move_hashtags_to_comment=True,
            platforms={"instagram": "{hashtags}"},
        )
        product_dir = tmp_path / "B0DUP"
        product_dir.mkdir()
        data = {"title": "Test", "affiliate_link": "https://amzn.to/d"}
        (product_dir / "data.json").write_text(json.dumps(data))

        metadata = PublishMetadata(
            platform=Platform.INSTAGRAM,
            title="Test",
            description="desc",
            hashtags=["#already", "nohash"],
            keywords=[],
            product_id="B0DUP",
        )

        result = build_first_comment(
            config, "instagram", "B0DUP", tmp_path, metadata=metadata
        )
        assert "#already" in result
        assert "#nohash" in result
        assert "##already" not in result

    def test_no_metadata_means_empty_hashtags(self, tmp_path):
        config = FirstCommentConfig(
            enabled=True,
            move_hashtags_to_comment=True,
            platforms={"instagram": "{affiliate_link}{hashtags}"},
        )
        product_dir = tmp_path / "B0NOMETA"
        product_dir.mkdir()
        data = {"title": "Test", "affiliate_link": "https://amzn.to/n"}
        (product_dir / "data.json").write_text(json.dumps(data))

        result = build_first_comment(
            config, "instagram", "B0NOMETA", tmp_path, metadata=None
        )
        assert result == "https://amzn.to/n"
