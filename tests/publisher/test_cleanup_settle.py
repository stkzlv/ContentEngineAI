"""An --immediate publish must not read "still publishing" as "did not publish".

`verify_publication` checked each platform once, immediately after
`posts.create` returned. The scheduler takes roughly 30-90s to move a leg to
`published` even on an immediate run, so every leg read `publishing`, the
verification failed, and cleanup was skipped on a product that went fully live
a minute later. The directory and its uploaded blob then stayed behind
indefinitely.

The two issues describing this both read the three log lines -- one per
platform, inside a second -- as a retry loop that gave up too fast. There was
no loop at all: the call checked once. So the fix is to add waiting, not to
slow existing waiting down.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.publisher.cleanup import CleanupManager
from src.publisher.models import CleanupConfig, Platform


def write_record(outputs_dir: Path, product_id: str, platform: str, post_id: str):
    """Write one publish_history row, the shape verify_publication reads."""
    path = outputs_dir / "publish_history.json"
    data = json.loads(path.read_text()) if path.exists() else {"posts": {}}
    data["posts"][f"{product_id}:{platform}"] = {
        "product_id": product_id,
        "platform": platform,
        "post_id": post_id,
        "published_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(data))


@pytest.fixture
def outputs_dir(tmp_path):
    d = tmp_path / "outputs"
    d.mkdir()
    return d


@pytest.fixture
def config():
    return CleanupConfig(
        enabled=True,
        verify_before_delete=True,
        require_all_platforms=True,
        settle_timeout_sec=300,
        settle_initial_delay_sec=30,
    )


class TestSettleDelays:
    """The schedule is derived, so the two knobs cannot disagree with it."""

    def test_delays_double_and_sum_to_the_budget(self, outputs_dir, config):
        manager = CleanupManager(outputs_dir, config, AsyncMock())

        assert manager._settle_delays() == [30, 60, 120, 90]
        assert sum(manager._settle_delays()) == 300

    def test_a_zero_timeout_means_check_once(self, outputs_dir):
        config = CleanupConfig(settle_timeout_sec=0)
        manager = CleanupManager(outputs_dir, config, AsyncMock())

        assert manager._settle_delays() == []

    def test_the_last_delay_is_trimmed_not_dropped(self, outputs_dir):
        """A budget that is not a sum of doublings must still be spent.

        Truncating instead would silently wait less than configured, which is
        the defect this change exists to fix, reintroduced as an off-by-a-lot.
        """
        config = CleanupConfig(settle_timeout_sec=100, settle_initial_delay_sec=30)
        manager = CleanupManager(outputs_dir, config, AsyncMock())

        assert manager._settle_delays() == [30, 60, 10]

    def test_a_non_positive_delay_with_a_budget_is_refused(self):
        """Otherwise the schedule is an unbounded list of zero-length waits."""
        with pytest.raises(ValueError, match="settle_initial_delay_sec"):
            CleanupConfig(settle_timeout_sec=300, settle_initial_delay_sec=0)

    def test_a_negative_timeout_is_refused(self):
        with pytest.raises(ValueError, match="settle_timeout_sec"):
            CleanupConfig(settle_timeout_sec=-1)


class TestVerifyPublicationWaits:
    """The behaviour the two issues asked for."""

    @pytest.mark.asyncio
    async def test_a_publishing_leg_is_waited_for(self, outputs_dir, config):
        """The regression guard: revert the loop and this fails.

        Both legs read `publishing` on the first pass, exactly as they did in
        the log the issue quotes, and both are live by the second.
        """
        write_record(outputs_dir, "B0TEST001", "youtube", "post1")
        write_record(outputs_dir, "B0TEST001", "tiktok", "post2")

        calls = {"n": 0}

        async def get_status(post_id):
            calls["n"] += 1
            # Two legs per pass, so the first pass is calls 1 and 2.
            return {"status": "publishing" if calls["n"] <= 2 else "published"}

        publisher = AsyncMock()
        publisher.get_status = AsyncMock(side_effect=get_status)
        manager = CleanupManager(outputs_dir, config, publisher)

        with patch("asyncio.sleep", new=AsyncMock()) as sleep:
            success, statuses = await manager.verify_publication(
                "B0TEST001", [Platform.YOUTUBE, Platform.TIKTOK]
            )

        assert success is True
        assert statuses == {"youtube": "published", "tiktok": "published"}
        sleep.assert_awaited_once_with(30)

    @pytest.mark.asyncio
    async def test_a_settled_run_never_sleeps(self, outputs_dir, config):
        """Waiting is conditional on a transient status, not on the config.

        Scheduled publishing is the common path and its legs are final the
        moment they are read; paying the settle budget there would add five
        minutes to every batch cleanup.
        """
        write_record(outputs_dir, "B0TEST001", "youtube", "post1")

        publisher = AsyncMock()
        publisher.get_status = AsyncMock(return_value={"status": "published"})
        manager = CleanupManager(outputs_dir, config, publisher)

        with patch("asyncio.sleep", new=AsyncMock()) as sleep:
            success, _ = await manager.verify_publication(
                "B0TEST001", [Platform.YOUTUBE]
            )

        assert success is True
        sleep.assert_not_awaited()
        assert publisher.get_status.await_count == 1

    @pytest.mark.asyncio
    async def test_a_failed_leg_is_not_waited_for(self, outputs_dir, config):
        """`failed` is final. Waiting on it burns the budget for no answer."""
        write_record(outputs_dir, "B0TEST001", "youtube", "post1")

        publisher = AsyncMock()
        publisher.get_status = AsyncMock(return_value={"status": "failed"})
        manager = CleanupManager(outputs_dir, config, publisher)

        with patch("asyncio.sleep", new=AsyncMock()) as sleep:
            success, statuses = await manager.verify_publication(
                "B0TEST001", [Platform.YOUTUBE]
            )

        assert success is False
        assert statuses["youtube"] == "failed"
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_leg_stuck_publishing_gives_up_at_the_budget(
        self, outputs_dir, config
    ):
        """Cleanup must not block forever on a leg that never settles."""
        write_record(outputs_dir, "B0TEST001", "youtube", "post1")

        publisher = AsyncMock()
        publisher.get_status = AsyncMock(return_value={"status": "publishing"})
        manager = CleanupManager(outputs_dir, config, publisher)

        with patch("asyncio.sleep", new=AsyncMock()) as sleep:
            success, statuses = await manager.verify_publication(
                "B0TEST001", [Platform.YOUTUBE]
            )

        assert success is False
        assert statuses["youtube"] == "publishing"
        assert [c.args[0] for c in sleep.await_args_list] == [30, 60, 120, 90]

    @pytest.mark.asyncio
    async def test_one_settled_leg_still_waits_for_the_other(self, outputs_dir, config):
        """The wait is per-run, and one finished leg does not end it.

        Reading "any settled" as "settled" would restore the original bug for
        every product whose platforms publish at different speeds, which is all
        of them.
        """
        write_record(outputs_dir, "B0TEST001", "youtube", "post1")
        write_record(outputs_dir, "B0TEST001", "tiktok", "post2")

        seen = {"post2": 0}

        async def get_status(post_id):
            if post_id == "post1":
                return {"status": "published"}
            seen["post2"] += 1
            return {"status": "publishing" if seen["post2"] == 1 else "published"}

        publisher = AsyncMock()
        publisher.get_status = AsyncMock(side_effect=get_status)
        manager = CleanupManager(outputs_dir, config, publisher)

        with patch("asyncio.sleep", new=AsyncMock()) as sleep:
            success, statuses = await manager.verify_publication(
                "B0TEST001", [Platform.YOUTUBE, Platform.TIKTOK]
            )

        assert success is True
        assert statuses["tiktok"] == "published"
        sleep.assert_awaited_once_with(30)


class TestConfigReachesTheManager:
    """The cleanup parser filters on an explicit key list.

    A field added to the dataclass but not to that list is dropped in silence:
    the YAML documents a five-minute wait and the run keeps the default. Same
    silent-drop shape as the profile-override field map.
    """

    def test_the_yaml_keys_round_trip(self, tmp_path):
        import yaml

        from src.publisher.config import load_publisher_config

        path = tmp_path / "publisher.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "cleanup": {
                        "enabled": True,
                        "settle_timeout_sec": 120,
                        "settle_initial_delay_sec": 10,
                    }
                }
            )
        )

        with patch.dict(
            "os.environ", {"LATE_API_KEY": "test-api-key-1234"}, clear=True
        ):
            config = load_publisher_config(path)

        assert config.cleanup_config.settle_timeout_sec == 120
        assert config.cleanup_config.settle_initial_delay_sec == 10

    def test_the_bundled_config_waits(self):
        """The default has to be on, or the fix ships disabled."""
        from src.publisher.config import load_publisher_config

        with patch.dict(
            "os.environ", {"LATE_API_KEY": "test-api-key-1234"}, clear=True
        ):
            config = load_publisher_config(Path("config/publisher.yaml"))

        assert config.cleanup_config.settle_timeout_sec > 0
        assert config.cleanup_config.settle_initial_delay_sec > 0
