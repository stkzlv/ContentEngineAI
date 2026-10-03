"""`schedule --immediate` applies the guards the other publish paths apply.

It published through a loop of its own, which skipped the duplicate check, the
publish history, the first comment and the affiliate phrase, and `--dry-run`
did not stop it. Nothing noticed because the scheduled automation never takes
this path. These drive the real path against a fake provider, with metadata
read from files rather than mocked.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.publisher.batch import BatchPublisher
from src.publisher.models import FirstCommentConfig, Platform
from src.publisher.product_registry import add_to_registry, get_registry_path
from src.publisher.tracking import is_already_published

PRODUCT = "B0IMMED001"
PHRASE = "As an Amazon Associate I earn from qualifying purchases."


@pytest.fixture(autouse=True)
def no_bio_link():
    with patch("src.publisher.batch.update_link_in_bio_safe", new_callable=AsyncMock):
        yield


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    product = tmp_path / PRODUCT
    product.mkdir()
    (product / f"video_{PRODUCT}_slideshow.mp4").write_bytes(b"video")
    (product / "data.json").write_text(
        json.dumps({"title": "Desk lamp", "affiliate_link": "https://amzn.to/x"})
    )
    (product / "metadata_youtube.json").write_text(
        json.dumps(
            {
                "title": "Desk lamp",
                "description": "A lamp for the desk.",
                "hashtags": ["lamp", "desk", "home"],
                "carries_affiliate_content": True,
            }
        )
    )
    return tmp_path


@pytest.fixture
def provider() -> AsyncMock:
    fake = AsyncMock()
    fake.upload_media.return_value = "media_1"
    fake.get_accounts.return_value = [{"platform": "youtube", "account_id": "yt_1"}]
    fake.publish.return_value = {"post_id": "post_1", "status": "published"}
    fake.first_comment_config = FirstCommentConfig(
        enabled=True, platforms={"youtube": "Link: {affiliate_link}"}
    )
    return fake


def immediate(provider: AsyncMock, outputs: Path, **kwargs) -> BatchPublisher:
    return BatchPublisher(
        publisher=provider,
        outputs_dir=outputs,
        platforms=[Platform.YOUTUBE],
        stagger_delay_min=0,
        stagger_delay_max=0,
        disclosure_phrase=PHRASE,
        **kwargs,
    )


@pytest.mark.req("REQ-PUB-043", "REQ-PUB-057", "REQ-PUB-068")
@pytest.mark.asyncio
async def test_an_immediate_post_carries_what_a_scheduled_one_does(
    provider: AsyncMock, outputs: Path
) -> None:
    summary = await immediate(provider, outputs).publish_batch()

    assert summary.successful == 1
    call = provider.publish.call_args.kwargs
    assert call["scheduled_time"] is None
    assert PHRASE in call["content"]
    assert call["platform_contents"]["youtube"]["first_comment"] == (
        "Link: https://amzn.to/x"
    )
    assert is_already_published(PRODUCT, "youtube", outputs)


@pytest.mark.req("REQ-PUB-046")
@pytest.mark.asyncio
async def test_a_second_immediate_run_does_not_post_again(
    provider: AsyncMock, outputs: Path
) -> None:
    await immediate(provider, outputs).publish_batch()
    summary = await immediate(provider, outputs).publish_batch()

    assert provider.publish.call_count == 1
    assert summary.skipped == 1
    assert summary.errors == []


@pytest.mark.req("REQ-PUB-047")
@pytest.mark.asyncio
async def test_force_posts_again(provider: AsyncMock, outputs: Path) -> None:
    await immediate(provider, outputs).publish_batch()
    await immediate(provider, outputs, force=True).publish_batch()

    assert provider.publish.call_count == 2


@pytest.mark.req("REQ-PUB-024")
@pytest.mark.asyncio
async def test_an_immediate_dry_run_contacts_no_provider(outputs: Path) -> None:
    from src.publisher.late import cli as late_cli

    args = Namespace(
        immediate=True,
        dry_run=True,
        platforms=[Platform.YOUTUBE],
        outputs_dir=outputs,
        force=False,
        debug=False,
    )
    with patch.object(late_cli, "_create_publisher_from_config") as create:
        await late_cli.cmd_schedule_auto(args, config=None, session=None)

    create.assert_not_called()


@pytest.mark.req("REQ-PUB-092")
def test_an_unreadable_registry_is_left_alone(outputs: Path) -> None:
    path = get_registry_path(outputs, "json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[{"product_id": "B0OLD00001", "title": "trunc')

    assert add_to_registry(PRODUCT, outputs) is False
    assert path.read_text() == '[{"product_id": "B0OLD00001", "title": "trunc'


@pytest.mark.req("REQ-PUB-068")
@pytest.mark.asyncio
@pytest.mark.parametrize("discloses", [True, False])
async def test_a_scheduled_post_carries_the_phrase_when_it_discloses(
    provider: AsyncMock, outputs: Path, tmp_path: Path, discloses: bool
) -> None:
    from src.publisher.models import CleanupConfig, RecurringSlot, ScheduleConfig
    from src.publisher.schedule import ScheduleManager

    (outputs / PRODUCT / "metadata.json").write_text(
        json.dumps(
            {
                "title": "Desk lamp",
                "description": "A lamp for the desk.",
                "carries_affiliate_content": discloses,
            }
        )
    )
    manager = ScheduleManager(
        tmp_path / "schedule.json",
        ScheduleConfig(
            enabled=True,
            slots=[RecurringSlot("monday", "10:00:00", "UTC")],
            timezone="UTC",
            min_post_spacing_hours=0,
            prevent_duplicates=False,
            allow_past_schedules=True,
            max_posts_per_day=0,
        ),
    )
    provider.list_posts = AsyncMock(return_value=[])
    with patch(
        "src.publisher.schedule.update_link_in_bio_safe", new_callable=AsyncMock
    ):
        await manager.auto_schedule(
            videos=[outputs / PRODUCT / f"video_{PRODUCT}_slideshow.mp4"],
            platforms=[Platform.YOUTUBE],
            publisher=provider,
            cleanup_config=CleanupConfig(enabled=False),
            outputs_dir=outputs,
            disclosure_phrase=PHRASE,
        )

    assert (PHRASE in provider.publish.call_args.kwargs["content"]) is discloses
