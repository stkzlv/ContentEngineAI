"""`delete` removes the post from the local schedule too (REQ-PUB-114).

The slot finder counts a slot as taken when the local schedule holds a post
there (REQ-PUB-025), so a post deleted on the provider but left in the file
kept its slot: a replacement scheduled right after went days later.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.publisher.late import cli
from src.publisher.models import Platform, ScheduleConfig, ScheduleEntry
from src.publisher.schedule import ScheduleManager


def _entry(product_id: str, post_id: str, days_ahead: int = 3) -> ScheduleEntry:
    return ScheduleEntry(
        product_id=product_id,
        scheduled_time=datetime.now(UTC) + timedelta(days=days_ahead),
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


@pytest.mark.req("REQ-PUB-114")
def test_a_past_entry_stays_for_the_readouts(tmp_path: Path) -> None:
    """A post that went live keeps the entry that maps it to its product."""
    manager = _manager(tmp_path)
    manager.record_entry(_entry("topic-a", "post-live", days_ahead=-5))

    assert manager.remove_post("post-live") == 0
    assert [e.post_id for e in _manager(tmp_path).entries] == ["post-live"]


@pytest.mark.req("REQ-PUB-115")
@pytest.mark.asyncio
async def test_a_schedule_write_failure_does_not_fail_the_delete(
    tmp_path: Path, caplog
) -> None:
    """Exit 1 means the provider refused; here it deleted the post."""
    _manager(tmp_path).record_entry(_entry("topic-a", "post-old"))
    state = tmp_path / "state"
    state.chmod(0o555)
    publisher = SimpleNamespace(
        authenticate=AsyncMock(return_value=True),
        delete_post=AsyncMock(return_value=True),
    )
    args = argparse.Namespace(post_id="post-old", debug=False, outputs_dir=tmp_path)
    config = SimpleNamespace(schedule_config=ScheduleConfig())

    try:
        with patch(
            "src.publisher.late.cli._create_publisher_from_config",
            return_value=publisher,
        ):
            await cli.cmd_delete(args, config, session=None)
    finally:
        state.chmod(0o755)

    assert "local schedule was not updated" in caplog.text


def test_a_failed_write_leaves_no_temp_file(tmp_path: Path) -> None:
    """A write that fails mid-dump removes its temp file and raises OSError."""
    manager = _manager(tmp_path)
    manager.record_entry(_entry("topic-a", "post-old"))

    def full_disk(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    with (
        patch("src.publisher.schedule.json.dump", side_effect=full_disk),
        pytest.raises(OSError),
    ):
        manager.remove_post("post-old")

    assert [p.name for p in (tmp_path / "state").iterdir()] == ["schedule.json"]
