"""A post scheduled through `single` reaches the local schedule (#485).

Only `auto_schedule` wrote `outputs/state/schedule.json`, so `calendar`,
which reads nothing else, listed a June entry as the newest while the
provider held the current week. Slot selection was unaffected, because
`single` unions the provider's posts with the local file before choosing a slot; the wrong
number was only ever read by a person, which is why it stayed quiet.

Two changes: the single path records each scheduled post the way the batch
does, and `calendar` puts the provider's own upcoming count beside the local
one so a gap is named rather than presented as an empty schedule.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.publisher.late.cli import _provider_upcoming_count
from src.publisher.models import Platform, ScheduleConfig, ScheduleEntry
from src.publisher.schedule import ScheduleManager, record_scheduled_posts
from src.utils.outputs_paths import get_project_root

REPO = get_project_root()
CLI = REPO / "src/publisher/late/cli.py"


def _entry(product_id: str, when: datetime) -> ScheduleEntry:
    return ScheduleEntry(
        product_id=product_id,
        scheduled_time=when,
        platforms=[Platform.YOUTUBE],
        post_id="p1",
        status="scheduled",
    )


class TestRecordEntry:
    def test_it_persists_without_the_preflight_validator(self, tmp_path: Path):
        """`add_entry` refuses a duplicate or a too-close slot. A post the
        provider already accepted is neither pre-flight nor refusable.
        """
        path = tmp_path / "schedule.json"
        manager = ScheduleManager(schedule_path=path, config=ScheduleConfig())
        when = datetime(2026, 9, 20, 8, 0, tzinfo=UTC)
        manager.record_entry(_entry("B0SAME", when))
        manager.record_entry(_entry("B0SAME", when))

        reloaded = ScheduleManager(schedule_path=path, config=ScheduleConfig())
        assert [e.product_id for e in reloaded.entries] == ["B0SAME", "B0SAME"]

    def test_add_entry_would_have_refused_that(self, tmp_path: Path):
        """The contrast that justifies a second method."""
        path = tmp_path / "schedule.json"
        manager = ScheduleManager(schedule_path=path, config=ScheduleConfig())
        when = datetime(2026, 9, 20, 8, 0, tzinfo=UTC)
        manager.add_entry(_entry("B0SAME", when))
        with pytest.raises(ValueError):
            manager.add_entry(_entry("B0SAME", when))

    def test_a_failed_write_rolls_the_entry_back(self, tmp_path: Path, monkeypatch):
        manager = ScheduleManager(schedule_path=tmp_path / "s.json")

        def boom() -> None:
            raise OSError("disk full")

        monkeypatch.setattr(manager, "_save_schedule", boom)
        with pytest.raises(OSError):
            manager.record_entry(_entry("B0X", datetime.now(UTC)))
        assert manager.entries == []


class TestTheSinglePathRecordsWhatItScheduled:
    @pytest.fixture
    def manager(self, tmp_path: Path) -> ScheduleManager:
        return ScheduleManager(schedule_path=tmp_path / "schedule.json")

    def test_a_unified_post_is_one_entry_carrying_every_platform(self, manager):
        when = datetime(2026, 9, 21, 8, 0, tzinfo=UTC)
        platforms = [
            {"platform": "youtube", "account_id": "a"},
            {"platform": "tiktok", "account_id": "b"},
        ]
        results = [{"platform": "all", "result": {"post_id": "post-1"}}]

        recorded = record_scheduled_posts(
            "B0UNI", results, platforms, when, slot_index=3, schedule_mgr=manager
        )

        assert recorded == 1
        (entry,) = manager.entries
        assert entry.product_id == "B0UNI"
        assert entry.scheduled_time == when
        assert entry.platforms == [Platform.YOUTUBE, Platform.TIKTOK]
        assert entry.post_id == "post-1"
        assert entry.status == "scheduled"
        assert entry.slot_index == 3

    def test_platform_specific_posts_are_one_entry_each(self, manager):
        when = datetime(2026, 9, 21, 8, 0, tzinfo=UTC)
        platforms = [
            {"platform": "youtube", "account_id": "a"},
            {"platform": "instagram", "account_id": "c"},
        ]
        results = [
            {"platform": "youtube", "result": {"post_id": "y1"}},
            {"platform": "instagram", "result": {"post_id": "i1"}},
        ]

        recorded = record_scheduled_posts(
            "B0SPLIT", results, platforms, when, slot_index=None, schedule_mgr=manager
        )

        assert recorded == 2
        assert [(e.platforms, e.post_id) for e in manager.entries] == [
            ([Platform.YOUTUBE], "y1"),
            ([Platform.INSTAGRAM], "i1"),
        ]
        assert all(e.slot_index is None for e in manager.entries)

    def test_the_record_survives_a_reload(self, tmp_path: Path):
        """The point is what `calendar` reads next time, not this process."""
        path = tmp_path / "schedule.json"
        when = datetime(2026, 9, 22, 8, 0, tzinfo=UTC)
        record_scheduled_posts(
            "B0DUR",
            [{"platform": "all", "result": {"post_id": "d1"}}],
            [{"platform": "youtube", "account_id": "a"}],
            when,
            slot_index=0,
            schedule_mgr=ScheduleManager(schedule_path=path),
        )

        later = ScheduleManager(schedule_path=path)
        assert [e.product_id for e in later.list_scheduled(date_from=when)] == ["B0DUR"]

    def test_a_write_failure_does_not_fail_the_publish(self, manager, monkeypatch):
        """The post exists on the provider either way."""

        def boom(entry) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(manager, "record_entry", boom)
        recorded = record_scheduled_posts(
            "B0X",
            [{"platform": "all", "result": {"post_id": "x"}}],
            [{"platform": "youtube", "account_id": "a"}],
            datetime.now(UTC),
            slot_index=None,
            schedule_mgr=manager,
        )
        assert recorded == 0


class TestCalendarSeesTheProvider:
    class FakePublisher:
        def __init__(self, posts):
            self.posts = posts

        async def list_posts(self, status=None):
            return self.posts

    class BrokenPublisher:
        async def list_posts(self, status=None):
            raise OSError("no network")

    class ExpiredKeyPublisher:
        async def list_posts(self, status=None):
            from src.publisher.base import AuthenticationError

            raise AuthenticationError("Authentication expired: 401")

    @pytest.mark.asyncio
    async def test_a_rotated_key_is_none_not_a_crash(self):
        """`AuthenticationError` is a `PublisherError`, not a `PublishError`.

        The command listed fine offline before it consulted the provider; a
        rotated key must leave it listing, with a warning, not a traceback.
        """
        assert await _provider_upcoming_count(self.ExpiredKeyPublisher()) is None

    @pytest.mark.asyncio
    async def test_it_counts_only_future_posts_with_a_time(self):
        now = datetime.now(UTC)
        posts = [
            {"scheduledFor": (now + timedelta(days=1)).isoformat()},
            {"scheduledFor": (now + timedelta(hours=2)).isoformat()},
            {"scheduledFor": (now - timedelta(days=1)).isoformat()},
            {"scheduledFor": None},
            {},
            {"scheduledFor": "not a date"},
        ]
        assert await _provider_upcoming_count(self.FakePublisher(posts)) == 2

    @pytest.mark.asyncio
    async def test_an_unreachable_provider_is_none_not_zero(self):
        """Zero would read as an empty schedule, which is the defect."""
        assert await _provider_upcoming_count(self.BrokenPublisher()) is None


class TestTheCallSitesAreWired:
    """The fixes are calls inside long functions.

    Three call sites: `single`, `calendar`, and the batch's publishing
    phase, which had the same missing write and which the Module/Batch
    Alignment Rule exists to catch. Nothing drives `cmd_single` or
    `cmd_calendar` end to end, so those two are pinned here by reading the
    source; the batch phase is driven by
    `tests/pipeline/test_global_batch_publishing.py`, which also asserts on
    the file it writes.
    """

    @staticmethod
    def _calls_in(function: str, path: Path = CLI) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name == function
            ):
                return {
                    n.func.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
        raise AssertionError(f"{function} not found")

    def test_single_records_its_schedule_entries(self):
        assert "record_scheduled_posts" in self._calls_in("cmd_single")

    def test_calendar_reads_the_provider(self):
        assert "_provider_upcoming_count" in self._calls_in("cmd_calendar")

    def test_the_batch_records_its_schedule_entries_too(self):
        assert "record_scheduled_posts" in self._calls_in(
            "run_publishing_phase", REPO / "src/pipeline/phases/publishing.py"
        )
