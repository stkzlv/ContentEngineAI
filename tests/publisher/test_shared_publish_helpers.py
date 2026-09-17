"""The publisher package owns what the CLI and the global batch both do (#450).

The batch's publishing phase used to restate the CLI: its own publisher
construction (which never passed `timeout` or `max_retries`, and for several
releases not `tiktok_settings`), its own account pairing, its own history
writes, its own occupancy read. Each now lives once, here, and both paths
call it. The batch's cleanup is also here, on purpose a different policy
from `CleanupManager`, so the difference is written down rather than
scattered.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.publisher.cleanup import remove_published_product_dir
from src.publisher.models import CleanupConfig, Platform, PublisherConfig
from src.publisher.publish_modes import accounts_for_platforms
from src.publisher.registry import create_publisher_from_config
from src.utils.outputs_paths import get_project_root

REPO = get_project_root()


class TestTheFactoryCarriesEverySetting:
    def test_timeout_and_retries_reach_the_provider(self):
        """The batch built its publisher without these two for its whole life."""
        config = PublisherConfig(
            provider="late",
            api_key="sk_live_" + "0" * 48,
            timeout=17.0,
            max_retries=5,
        )
        with patch("src.publisher.registry.create_publisher") as factory:
            create_publisher_from_config(config)
        kwargs = factory.call_args.kwargs
        assert kwargs["timeout"] == 17.0
        assert kwargs["max_retries"] == 5
        assert kwargs["tiktok_settings"] is config.tiktok_settings
        assert kwargs["first_comment_config"] is config.first_comment_config
        assert kwargs["synthetic_media_disclosure"] is config.synthetic_media_disclosure

    @pytest.mark.parametrize(
        "path",
        ["src/publisher/late/cli.py", "src/pipeline/phases/publishing.py"],
    )
    def test_neither_path_builds_its_own_publisher(self, path: str):
        tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
        direct = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "create_publisher"
        ]
        assert not direct, f"{path} calls create_publisher directly"


class TestAccountPairing:
    ACCOUNTS = [
        {"platform": "YouTube", "account_id": "yt1"},
        {"platform": "tiktok", "account_id": "tt1"},
    ]

    def test_it_pairs_case_insensitively_and_names_the_missing(self):
        targets, missing = accounts_for_platforms(
            [Platform.YOUTUBE, Platform.INSTAGRAM, Platform.TIKTOK], self.ACCOUNTS
        )
        assert targets == [
            {"platform": "youtube", "account_id": "yt1"},
            {"platform": "tiktok", "account_id": "tt1"},
        ]
        assert missing == [Platform.INSTAGRAM]

    def test_no_accounts_means_nothing_to_publish(self):
        targets, missing = accounts_for_platforms([Platform.YOUTUBE], [])
        assert targets == []
        assert missing == [Platform.YOUTUBE]


class TestTheBatchCleanup:
    def test_it_removes_the_directory_when_every_platform_was_required(
        self, tmp_path: Path
    ):
        (tmp_path / "B0X").mkdir()
        (tmp_path / "B0X" / "video.mp4").write_text("x")
        removed = remove_published_product_dir(
            tmp_path, "B0X", CleanupConfig(enabled=True, require_all_platforms=True)
        )
        assert removed
        assert not (tmp_path / "B0X").exists()

    @pytest.mark.parametrize(
        "config",
        [
            CleanupConfig(enabled=False, require_all_platforms=True),
            CleanupConfig(enabled=True, require_all_platforms=False),
        ],
        ids=["disabled", "not-all-platforms"],
    )
    def test_it_leaves_the_directory_otherwise(self, tmp_path: Path, config):
        (tmp_path / "B0X").mkdir()
        assert not remove_published_product_dir(tmp_path, "B0X", config)
        assert (tmp_path / "B0X").exists()

    def test_a_missing_directory_is_not_an_error(self, tmp_path: Path):
        assert not remove_published_product_dir(
            tmp_path, "B0NONE", CleanupConfig(enabled=True)
        )


class TestThePhaseCallsThePackage:
    """The issue's done-when: no logic in the phase that the package has."""

    @staticmethod
    def _calls() -> set[str]:
        tree = ast.parse(
            (REPO / "src/pipeline/phases/publishing.py").read_text(encoding="utf-8")
        )
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    names.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    names.add(n.func.attr)
        return names

    @pytest.mark.parametrize(
        "name",
        [
            "create_publisher_from_config",
            "build_occupancy",
            "next_free_slot",
            "accounts_for_platforms",
            "publish_product",
            "record_publish_results",
            "record_scheduled_posts",
            "remove_published_product_dir",
            "run_blob_retention",
            "run_delivery_sweep",
        ],
    )
    def test_it_calls(self, name: str):
        assert name in self._calls()

    @pytest.mark.parametrize(
        "name", ["list_posts", "get_next_slot", "record_publish", "rmtree"]
    )
    def test_it_no_longer_restates(self, name: str):
        """Each of these was the phase doing the package's job by hand."""
        assert name not in self._calls()


class TestTheOccupancyReadDegradesOnePostAtATime:
    """`build_occupancy` feeds every scheduling path's slot search.

    A read that fails part-way used to abort `schedule` (the exception
    escaped) and, once the batch called it, would have returned a set with
    neither the provider's posts nor the local file's, so the first slot
    after now, already held locally, was offered again.
    """

    class _Posts:
        def __init__(self, posts):
            self.posts = posts

        async def list_posts(self):
            return self.posts

    class _Rejected:
        async def list_posts(self):
            from src.publisher.base import AuthenticationError

            raise AuthenticationError("[401] Invalid API key")

    @staticmethod
    def _manager(tmp_path: Path, local: datetime):
        from src.publisher.models import ScheduleEntry
        from src.publisher.schedule import ScheduleManager

        manager = ScheduleManager(schedule_path=tmp_path / "schedule.json")
        manager.record_entry(
            ScheduleEntry(
                product_id="B0LOCAL",
                scheduled_time=local,
                platforms=[Platform.YOUTUBE],
                post_id="p-local",
                status="scheduled",
            )
        )
        return manager

    @pytest.mark.asyncio
    async def test_one_unreadable_timestamp_drops_that_post_only(self, tmp_path):
        now = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
        local = now + timedelta(days=2)
        good = now + timedelta(days=1)
        manager = self._manager(tmp_path, local)

        occupied = await manager.build_occupancy(
            self._Posts(
                [{"scheduledFor": "not a date"}, {"scheduledFor": good.isoformat()}]
            ),
            now,
        )

        assert occupied == {good, local}

    @pytest.mark.asyncio
    async def test_a_rejected_read_still_counts_the_local_entries(self, tmp_path):
        now = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
        local = now + timedelta(days=2)
        manager = self._manager(tmp_path, local)

        occupied = await manager.build_occupancy(self._Rejected(), now)

        assert occupied == {local}


class TestEveryPathTakesItsSlotsFromTheSchedulemanager:
    """One occupancy read and one slot search, called from three places.

    `single` kept its own loop over the provider's posts after `schedule`
    and the batch had moved to the shared read, so it alone did not count
    the local schedule (#493).
    """

    @staticmethod
    def _calls_in(path: str, function: str) -> set[str]:
        tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            and n.name == function
        )
        names = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    names.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    names.add(n.func.attr)
        return names

    @pytest.mark.parametrize(
        ("path", "function"),
        [
            ("src/publisher/late/cli.py", "cmd_single"),
            ("src/pipeline/phases/publishing.py", "run_publishing_phase"),
            ("src/publisher/schedule.py", "auto_schedule"),
        ],
    )
    def test_the_path_calls_the_shared_read(self, path: str, function: str):
        calls = self._calls_in(path, function)
        assert "build_occupancy" in calls, f"{function} does not call build_occupancy"
        assert "list_posts" not in calls, f"{function} reads the provider itself"
        assert "get_next_slot" not in calls, f"{function} walks the slots itself"
