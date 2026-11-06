"""Tests for first-comment delivery verification."""

from unittest.mock import AsyncMock

import pytest

from src.publisher.comment_verify import (
    FIRST_COMMENT_PLATFORMS,
    verify_post_first_comments,
)


def _owner_comment() -> dict:
    return {"message": "Get it here: ...", "from": {"name": "@chan", "isOwner": True}}


def _viewer_comment() -> dict:
    return {"message": "nice", "from": {"name": "@viewer", "isOwner": False}}


def _client(platforms: list[dict], comments_by_id: dict[str, list]) -> AsyncMock:
    client = AsyncMock()
    client.get_post_platforms = AsyncMock(return_value=platforms)

    async def _get_comments(platform_post_id, account_id, limit=25):
        return comments_by_id.get(platform_post_id, [])

    client.get_post_comments = AsyncMock(side_effect=_get_comments)
    return client


@pytest.mark.asyncio
async def test_owner_comment_present():
    client = _client(
        [{"platform": "youtube", "platform_post_id": "vid1", "account_id": "acc"}],
        {"vid1": [_owner_comment()]},
    )
    checks = await verify_post_first_comments(client, "post1")
    assert len(checks) == 1
    assert checks[0].platform == "youtube"
    assert checks[0].present is True


@pytest.mark.asyncio
async def test_owner_comment_absent():
    # Only a viewer comment exists, no owner first comment.
    client = _client(
        [{"platform": "instagram", "platform_post_id": "ig1", "account_id": "acc"}],
        {"ig1": [_viewer_comment()]},
    )
    checks = await verify_post_first_comments(client, "post1")
    assert len(checks) == 1
    assert checks[0].present is False


@pytest.mark.asyncio
async def test_tiktok_skipped():
    client = _client(
        [{"platform": "tiktok", "platform_post_id": "tt1", "account_id": "acc"}],
        {},
    )
    checks = await verify_post_first_comments(client, "post1")
    assert checks == []
    client.get_post_comments.assert_not_called()


@pytest.mark.asyncio
async def test_not_live_skipped():
    # No platform_post_id yet -> the post isn't live, skip without an API call.
    client = _client(
        [{"platform": "youtube", "platform_post_id": None, "account_id": "acc"}],
        {},
    )
    checks = await verify_post_first_comments(client, "post1")
    assert checks == []
    client.get_post_comments.assert_not_called()


def test_first_comment_platforms_excludes_tiktok():
    assert "youtube" in FIRST_COMMENT_PLATFORMS
    assert "instagram" in FIRST_COMMENT_PLATFORMS
    assert "tiktok" not in FIRST_COMMENT_PLATFORMS


@pytest.mark.req("REQ-PUB-161", "REQ-PUB-044")
@pytest.mark.asyncio
async def test_a_leg_sent_without_a_first_comment_is_not_checked():
    # REQ-PUB-161: a post whose metadata skipped the comment carries none, so
    # its absence is not a miss. A leg that doesn't say is still checked.
    client = _client(
        [
            {
                "platform": "instagram",
                "platform_post_id": "ig1",
                "account_id": "acc",
                "first_comment_sent": False,
            },
            {"platform": "youtube", "platform_post_id": "vid1", "account_id": "acc"},
        ],
        {},
    )
    checks = await verify_post_first_comments(client, "post1")
    assert [(c.platform, c.sent, c.present) for c in checks] == [
        ("instagram", False, False),
        ("youtube", True, False),
    ]
    # The inbox is read only for the leg that was sent with a comment.
    client.get_post_comments.assert_awaited_once_with("vid1", "acc")


@pytest.mark.parametrize(
    ("specific", "sent"),
    [
        ({"firstComment": "Which one?"}, True),
        ({"firstComment": ""}, False),
        ({"title": "t"}, False),
        (None, None),
    ],
)
@pytest.mark.req("REQ-PUB-161")
def test_first_comment_sent_reads_the_leg(specific, sent):
    from src.publisher.late.client import _first_comment_sent

    assert _first_comment_sent(specific) is sent


@pytest.mark.req("REQ-PUB-161")
@pytest.mark.asyncio
async def test_sent_flag_from_the_sdk_models_get_post_platforms_reads():
    # REQ-PUB-161: real runs read pydantic models, not dicts. The SDK's
    # platformSpecificData is an undiscriminated union, so check the parsed
    # leg still exposes firstComment for each shape we send.
    from unittest.mock import patch

    from late.models import PostGetResponse

    from src.publisher.late.client import LatePublisher

    def leg(platform: str, ppid: str, data: dict) -> dict:
        return {
            "platform": platform,
            "accountId": "acc",
            "platformPostId": ppid,
            "status": "published",
            "platformSpecificData": data,
        }

    resp = PostGetResponse.model_validate(
        {
            "post": {
                "_id": "p1",
                "platforms": [
                    leg("youtube", "y1", {"title": "t", "firstComment": "Which?"}),
                    leg(
                        "youtube", "y2", {"title": "t", "containsSyntheticMedia": True}
                    ),
                    leg(
                        "instagram", "i1", {"isAiGenerated": True, "firstComment": "x"}
                    ),
                    leg("instagram", "i2", {"isAiGenerated": True}),
                    leg("tiktok", "t1", {"videoMadeWithAi": True}),
                ],
            }
        }
    )
    pub = LatePublisher(api_key="test")  # noqa: S106 - test stub
    with patch.object(pub, "_posts_get_safe", return_value=resp):
        legs = await pub.get_post_platforms("p1")
    assert [(x["platform_post_id"], x["first_comment_sent"]) for x in legs] == [
        ("y1", True),
        ("y2", False),
        ("i1", True),
        ("i2", False),
        ("t1", False),
    ]


@pytest.mark.req("REQ-PUB-044", "REQ-PUB-161")
@pytest.mark.asyncio
async def test_verify_comments_counts_unsent_legs_apart(monkeypatch, caplog, tmp_path):
    # REQ-PUB-044, REQ-PUB-161: the command, not only the function, keeps a leg
    # sent without a comment out of the missing count.
    import argparse

    from src.publisher.late.cli import cmd_verify_comments

    pub = AsyncMock()
    pub.authenticate = AsyncMock(return_value=True)
    pub.list_posts = AsyncMock(return_value=[{"id": "post1"}])
    pub.get_post_platforms = AsyncMock(
        return_value=[
            {
                "platform": "instagram",
                "platform_post_id": "ig1",
                "account_id": "acc",
                "first_comment_sent": False,
            },
            {
                "platform": "youtube",
                "platform_post_id": "vid1",
                "account_id": "acc",
                "first_comment_sent": True,
            },
        ]
    )
    pub.get_post_comments = AsyncMock(return_value=[_viewer_comment()])
    monkeypatch.setattr(
        "src.publisher.late.cli._create_publisher_from_config", lambda *_: pub
    )
    monkeypatch.setattr("src.publisher.late.cli._load_product_map", lambda _: {})

    args = argparse.Namespace(limit=5, outputs_dir=tmp_path)
    with caplog.at_level("INFO", logger="src.publisher.late.cli"):
        await cmd_verify_comments(args, config=None, session=None)

    missing = [r for r in caplog.records if "First comment MISSING" in r.message]
    unsent = [r for r in caplog.records if "No first comment sent" in r.message]
    assert [r.levelname for r in missing] == ["WARNING"]
    assert "youtube" in missing[0].getMessage()
    assert [r.levelname for r in unsent] == ["INFO"]
    assert "1 missing of 1 checked across 1 post(s); 1 leg(s) sent without one" in (
        caplog.text
    )
