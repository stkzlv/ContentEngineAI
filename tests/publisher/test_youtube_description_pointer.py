"""A product's YouTube description carries no URL and points at the profile link.

REQ-CNT-150. Optimized metadata had both from its own prompt and parser
(REQ-PUB-148); unified metadata, the bundled default, shares one description
across platforms and had neither. The publisher now builds YouTube's own
entry with them, and leaves the other platforms' captions alone.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.publisher.models import Platform, PublishMetadata
from src.publisher.publish_modes import (
    _build_platform_contents_with_comments,
    youtube_metadata,
)
from src.utils.youtube_description import with_profile_pointer

PLATFORMS = [
    {"platform": name, "account_id": "acc"}
    for name in ("youtube", "tiktok", "instagram")
]


def _meta(description: str, product_id: str = "B0TESTPTR1") -> PublishMetadata:
    return PublishMetadata(
        platform=Platform.YOUTUBE,
        title="Tilted cat bowl",
        description=description,
        hashtags=["cat"],
        keywords=[],
        product_id=product_id,
        disclosure="#ad",
        carries_affiliate_content=False,
    )


def _entries(meta: PublishMetadata, product_id: str, *, pointer: bool = True):
    publisher = MagicMock()
    publisher.first_comment_config = None
    publisher.youtube_profile_pointer = pointer
    out = _build_platform_contents_with_comments(
        publisher,
        PLATFORMS,
        product_id,
        "unused",
        metadata_by_platform={p["platform"]: meta for p in PLATFORMS},
    )
    assert out is not None
    return {name: entry["content"] for name, entry in out.items()}


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-150")
class TestYouTubeEntry:
    def test_a_url_is_removed_and_the_pointer_added(self):
        meta = _meta("A comfy tilted bowl.\n\nShop: https://amzn.to/x")

        content = _entries(meta, "B0TESTPTR1")

        assert content["youtube"].startswith("A comfy tilted bowl.\n\nLink in bio.")
        assert "amzn.to" not in content["youtube"]

    def test_other_platforms_keep_the_shared_caption(self):
        meta = _meta("A comfy tilted bowl.")

        content = _entries(meta, "B0TESTPTR1")

        assert content["tiktok"] == content["instagram"] == meta.format_content()
        assert "Link in bio." not in content["tiktok"]

    def test_an_existing_pointer_is_not_repeated(self):
        meta = _meta("A comfy tilted bowl. Link in my bio!")

        content = _entries(meta, "B0TESTPTR1")

        assert content["youtube"].count("bio") == 1

    def test_a_topic_gets_no_pointer(self):
        meta = _meta("Turn on the 80% charging limit.", "topic-charging-limit")

        content = _entries(meta, "topic-charging-limit")

        assert "Link in bio." not in content["youtube"]

    def test_the_switch_turns_the_pointer_off_but_not_the_url_removal(self):
        meta = _meta("A comfy tilted bowl.\n\nShop: https://amzn.to/x")

        content = _entries(meta, "B0TESTPTR1", pointer=False)

        assert "Link in bio." not in content["youtube"]
        assert "amzn.to" not in content["youtube"]

    def test_a_long_caption_still_fits_youtube(self):
        meta = _meta("word " * 1100)

        youtube = youtube_metadata(meta, "B0TESTPTR1", pointer=True)

        assert len(youtube.format_content()) <= 5000


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-150")
def test_the_bundled_config_turns_the_pointer_on(monkeypatch):
    from src.publisher.config import load_publisher_config

    monkeypatch.delenv("LATE_API_KEY", raising=False)
    config = load_publisher_config(cli_overrides={"api_key": "sk_test_" + "x" * 40})

    assert config.youtube_profile_pointer is True


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-150")
@pytest.mark.parametrize(
    "text",
    ["Link in bio.", "links in bio", "Check the link in our bio if you want one."],
)
def test_pointer_phrasings_are_recognised(text):
    assert with_profile_pointer(text) == text
