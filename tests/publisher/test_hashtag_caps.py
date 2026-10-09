"""A caption's hashtags stay within each platform's cap (REQ-PUB-108, #567)."""

from __future__ import annotations

import re

import pytest

from src.publisher.models import Platform, PublishMetadata

ALL = [Platform.TIKTOK, Platform.YOUTUBE, Platform.INSTAGRAM]


def _tags(meta: PublishMetadata) -> list[str]:
    return re.findall(r"(?<![\w#])#\w+", meta.format_content())


def _meta(tags: list[str], product_id: str = "B0CAPTEST1") -> PublishMetadata:
    return PublishMetadata(
        platform=Platform.INSTAGRAM,
        title="Tripod",
        description="A tall tripod.",
        hashtags=tags,
        product_id=product_id,
    )


@pytest.mark.req("REQ-PUB-108")
def test_instagram_caps_the_caption_at_five_with_disclosure_and_id() -> None:
    meta = _meta(["Tripod", "Phone", "Travel", "Camera", "Selfie", "Gear"])

    assert meta.clamp_for_platforms(ALL) == ("hashtags",)
    tags = _tags(meta)
    assert len(tags) == 5
    assert tags[0] == "#ad" and tags[-1] == "#B0CAPTEST1"
    assert meta.hashtags == ["Tripod", "Phone", "Travel"]


@pytest.mark.req("REQ-PUB-108")
def test_a_caption_within_the_cap_is_left_alone() -> None:
    meta = _meta(["Tripod", "Phone", "Travel"])

    assert meta.clamp_for_platforms(ALL) == ()
    assert len(_tags(meta)) == 5


@pytest.mark.req("REQ-PUB-108")
def test_without_instagram_more_tags_stay() -> None:
    meta = _meta(["Tripod", "Phone", "Travel", "Camera", "Selfie", "Gear"])

    assert meta.clamp_for_platforms([Platform.TIKTOK, Platform.YOUTUBE]) == ()
    assert len(_tags(meta)) == 8


@pytest.mark.req("REQ-PUB-107")
def test_a_topic_caption_carries_no_id_tag() -> None:
    meta = _meta(["WifiTips"], product_id="topic-why-wifi-drops-1a2b3c4d")

    assert "#topic" not in meta.format_content()


def test_the_topic_prefix_matches_the_producer() -> None:
    from src.publisher import models
    from src.video.producer.topic_input import TOPIC_ID_PREFIX

    assert models._TOPIC_ID_PREFIX == TOPIC_ID_PREFIX
