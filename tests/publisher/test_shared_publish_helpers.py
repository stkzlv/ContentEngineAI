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
