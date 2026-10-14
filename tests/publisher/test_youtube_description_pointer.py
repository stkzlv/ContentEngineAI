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
from src.publisher.publish_modes import _build_platform_contents_with_comments
from src.publisher.youtube_caption import youtube_metadata
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

    def test_off_a_stale_pointer_line_is_removed(self):
        # Metadata written while the pointer was on still ends on it.
        meta = _meta("A comfy tilted bowl.\n\nTeam A or team B?\n\nLink in bio.")

        content = _entries(meta, "B0TESTPTR1", pointer=False)

        assert content["youtube"].startswith(
            "A comfy tilted bowl.\n\nTeam A or team B?"
        )
        assert "bio" not in content["youtube"]
        assert "Link in bio." in content["tiktok"]

    def test_on_a_pointer_line_is_kept(self):
        meta = _meta("A comfy tilted bowl.\n\nLink in bio.")

        content = _entries(meta, "B0TESTPTR1")

        assert content["youtube"].count("Link in bio.") == 1

    @pytest.mark.parametrize("words", [1100, 990, 996])
    def test_a_long_caption_fits_youtube_and_keeps_the_pointer(self, words):
        # 990 and 996 sit just under YouTube's budget, where appending the
        # pointer and then trimming the end used to cut the pointer itself.
        meta = _meta("word " * words).clamped_for([Platform.YOUTUBE])

        youtube = youtube_metadata(meta, "B0TESTPTR1", pointer=True)

        caption = youtube.format_content()
        assert len(caption) <= 5000
        assert youtube.description.endswith("Link in bio.")

    def test_a_description_that_is_only_a_url_becomes_the_pointer(self):
        meta = _meta("Shop now: https://amzn.to/abc")

        youtube = youtube_metadata(meta, "B0TESTPTR1", pointer=True)

        assert youtube.description == "Link in bio."

    def test_the_copy_does_not_share_the_counts(self):
        meta = _meta("word " * 1100)
        before = dict(meta.character_counts)

        youtube_metadata(meta, "B0TESTPTR1", pointer=True)

        assert meta.character_counts == before


def _schedule_publisher(pointer: bool = True):
    calls: list[dict] = []

    async def publish(**kwargs):
        calls.append(kwargs)
        return {"post_id": "p1"}

    publisher = MagicMock()
    publisher.publish = publish
    publisher.youtube_profile_pointer = pointer
    return publisher, calls


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-150")
class TestTheSchedulerPathsToo:
    """`schedule` builds its captions itself, not through publish_modes."""

    @staticmethod
    def _kwargs(tmp_path, publisher, metas):
        from datetime import UTC, datetime

        return {
            "publisher": publisher,
            "platform_dicts": [{"platform": n, "account_id": "a"} for n in metas],
            "platform_metas": metas,
            "titles": {"youtube": "Tilted cat bowl"},
            "carries_affiliate": {n: False for n in metas},
            "first_comments": {},
            "media_id": "m",
            "product_id": "B0TESTPTR1",
            "next_time": datetime(2026, 10, 20, 8, tzinfo=UTC),
            "next_idx": 0,
        }

    @pytest.mark.asyncio
    async def test_per_platform(self, tmp_path):
        from src.publisher.schedule import ScheduleManager

        manager = ScheduleManager(schedule_path=tmp_path / "schedule.json")
        publisher, calls = _schedule_publisher()
        meta = _meta("A comfy tilted bowl.\n\nShop: https://amzn.to/x")
        metas = {"youtube": meta, "tiktok": meta}

        await manager._post_per_platform(**self._kwargs(tmp_path, publisher, metas))

        sent = {c["platforms"][0]["platform"]: c["content"] for c in calls}
        assert "amzn.to" not in sent["youtube"]
        assert "Link in bio." in sent["youtube"]
        assert "Link in bio." not in sent["tiktok"]

    @pytest.mark.asyncio
    async def test_unified(self, tmp_path):
        from src.publisher.schedule import ScheduleManager

        manager = ScheduleManager(schedule_path=tmp_path / "schedule.json")
        publisher, calls = _schedule_publisher()
        meta = _meta("A comfy tilted bowl.\n\nShop: https://amzn.to/x")
        metas = {"youtube": meta, "tiktok": meta}

        await manager._post_unified(**self._kwargs(tmp_path, publisher, metas))

        entries = calls[0]["platform_contents"]
        assert "amzn.to" not in entries["youtube"]["content"]
        assert "Link in bio." in entries["youtube"]["content"]
        assert "Link in bio." not in entries["tiktok"]["content"]


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-150")
@pytest.mark.req("REQ-OPS-121")
def test_the_bundled_config_ships_the_pointer_off(monkeypatch):
    # Decision 0017: an installation with a bio link turns it on in its
    # private overlay.
    from src.publisher.config import load_publisher_config
    from src.utils.youtube_description import profile_pointer_on

    monkeypatch.delenv("LATE_API_KEY", raising=False)
    config = load_publisher_config(cli_overrides={"api_key": "sk_test_" + "x" * 40})

    assert config.youtube_profile_pointer is False
    assert profile_pointer_on() is False


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-150")
@pytest.mark.parametrize(
    "text",
    ["Link in bio.", "links in bio", "Check the link in our bio if you want one."],
)
def test_pointer_phrasings_are_recognised(text):
    assert with_profile_pointer(text) == text


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-150")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Great bowl.\n\nLink in bio.", "Great bowl."),
        ("Great bowl.\nlinks in my bio!\nMore text.", "Great bowl.\nMore text."),
        ("Great bowl.\nThe link is in the bio", "Great bowl."),
        # A sentence that only mentions the bio stays: it says something else.
        ("The link in bio has the size chart.", "The link in bio has the size chart."),
        ("No pointer here.", "No pointer here."),
    ],
)
def test_without_profile_pointer(text, expected):
    from src.utils.youtube_description import without_profile_pointer

    assert without_profile_pointer(text) == expected
