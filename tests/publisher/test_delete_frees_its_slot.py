"""`delete` removes the post from the local schedule too (REQ-PUB-114).

The slot finder counts a slot as taken when the local schedule holds a post
there (REQ-PUB-025), so a post deleted on the provider but left in the file
kept its slot: a replacement scheduled right after went days later.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.publisher.late import cli
from src.publisher.models import Platform, ScheduleConfig, ScheduleEntry
from src.publisher.schedule import ScheduleManager


def _entry(product_id: str, post_id: str) -> ScheduleEntry:
    return ScheduleEntry(
        product_id=product_id,
        scheduled_time=datetime(2026, 10, 23, 8, 0, tzinfo=UTC),
        platforms=[Platform.YOUTUBE],
        post_id=post_id,
        status="scheduled",
    )


def _manager(outputs: Path) -> ScheduleManager:
    return ScheduleManager(
        schedule_path=cli._schedule_path(argparse.Namespace(outputs_dir=outputs)),
        config=ScheduleConfig(),
    )


@pytest.mark.req("REQ-PUB-114")
def test_remove_post_drops_only_that_post(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.record_entry(_entry("topic-a", "post-old"))
    manager.record_entry(_entry("topic-a", "post-new"))
    manager.record_entry(_entry("B0TEST001", "post-other"))

    assert manager.remove_post("post-old") == 1
    assert manager.remove_post("post-missing") == 0
    assert [e.post_id for e in _manager(tmp_path).entries] == ["post-new", "post-other"]


@pytest.mark.req("REQ-PUB-114")
@pytest.mark.asyncio
@pytest.mark.parametrize(("deleted", "kept"), [(True, []), (False, ["post-old"])])
async def test_delete_frees_the_slot_only_when_the_provider_deleted(
    tmp_path: Path, deleted: bool, kept: list[str]
) -> None:
    _manager(tmp_path).record_entry(_entry("topic-a", "post-old"))
    publisher = SimpleNamespace(
        authenticate=AsyncMock(return_value=True),
        delete_post=AsyncMock(return_value=deleted),
    )
    args = argparse.Namespace(post_id="post-old", debug=False, outputs_dir=tmp_path)
    config = SimpleNamespace(schedule_config=ScheduleConfig())

    with patch(
        "src.publisher.late.cli._create_publisher_from_config", return_value=publisher
    ):
        if deleted:
            await cli.cmd_delete(args, config, session=None)
        else:
            with pytest.raises(SystemExit):
                await cli.cmd_delete(args, config, session=None)

    assert [e.post_id for e in _manager(tmp_path).entries] == kept
