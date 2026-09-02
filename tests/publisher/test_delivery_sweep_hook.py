"""The delivery sweep runs after every publish, and never fails one.

`verify-delivery` (#156) existed only as a manual subcommand, so a leg that
failed at publish time was caught by the vendor's email or by nobody. The fix
(`posts.retry`) expires once Zernio drops the CDN copy, so the sweep has to
run on its own, after each publish, over a trailing window (#201).

Two things a helper-level test cannot see are asserted here: that each publish
path actually calls the hook (an AST walk over the call sites, the same guard
the batch's `create_publisher` kwargs use), and that the hook swallows a
failing status read rather than turning an accepted publish into an error.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.publisher.models import DeliverySweepConfig
from src.publisher.partial_post_sweep import run_delivery_sweep

REPO = Path(__file__).resolve().parents[2]


def _client(posts, legs_by_id):
    client = AsyncMock()
    client.list_posts = AsyncMock(return_value=posts)
    client.get_post_platforms = AsyncMock(side_effect=lambda pid: legs_by_id[pid])
    return client


@pytest.mark.unit
class TestTheHook:
    @pytest.mark.asyncio
    async def test_it_warns_per_failing_leg_with_id_platform_and_category(self, caplog):
        client = _client(
            [{"id": "p1", "status": "partial"}],
            {
                "p1": [
                    {"platform": "youtube", "status": "published"},
                    {
                        "platform": "tiktok",
                        "status": "failed",
                        "error_message": "502 Bad Gateway",
                        "error_category": "platform_error",
                    },
                ]
            },
        )

        with caplog.at_level(logging.WARNING):
            await run_delivery_sweep(client, DeliverySweepConfig(limit=10))

        text = caplog.text
        assert "p1" in text
        assert "tiktok (platform_error)" in text
        assert "posts.retry('p1')" in text

    @pytest.mark.asyncio
    async def test_a_payload_rejection_says_update_first(self, caplog):
        """A bare retry resubmits the same rejected payload."""
        client = _client(
            [{"id": "p2", "status": "partial"}],
            {
                "p2": [
                    {
                        "platform": "tiktok",
                        "status": "failed",
                        "error_message": (
                            "Commercial content disclosure is enabled but no "
                            "option selected"
                        ),
                        "error_category": "platform_rejected",
                    }
                ]
            },
        )

        with caplog.at_level(logging.WARNING):
            await run_delivery_sweep(client, DeliverySweepConfig())

        assert "posts.update('p2'" in caplog.text

    @pytest.mark.asyncio
    async def test_it_inspects_the_configured_window(self):
        posts = [{"id": f"p{i}", "status": "published"} for i in range(40)]
        client = _client(posts, {})

        await run_delivery_sweep(client, DeliverySweepConfig(limit=7))

        client.list_posts.assert_awaited_once()
        client.get_post_platforms.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", [None, DeliverySweepConfig(enabled=False)])
    async def test_it_does_nothing_when_off(self, config):
        client = _client([], {})

        await run_delivery_sweep(client, config)

        client.list_posts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failing_status_read_never_raises(self, caplog):
        """The publish Zernio already accepted must not be reported failed."""
        client = AsyncMock()
        client.list_posts = AsyncMock(side_effect=RuntimeError("502 from Zernio"))

        with caplog.at_level(logging.WARNING):
            await run_delivery_sweep(client, DeliverySweepConfig())

        assert "Delivery sweep failed" in caplog.text


@pytest.mark.unit
class TestEveryPublishPathCallsIt:
    """Sharing a hook is not the guard; the call site is."""

    @staticmethod
    def _calls_in(path: str, function: str) -> set[str]:
        tree = ast.parse((REPO / path).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
                and node.name == function
            ):
                return {
                    n.func.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
        raise AssertionError(f"{function} not found in {path}")

    @pytest.mark.parametrize(
        ("path", "function"),
        [
            ("src/pipeline/global_batch.py", "_execute_publishing_phase"),
            ("src/publisher/late/cli.py", "cmd_single"),
            ("src/publisher/late/cli.py", "_run_immediate_batch"),
            ("src/publisher/late/cli.py", "cmd_schedule_auto"),
        ],
    )
    def test_the_path_runs_the_sweep(self, path, function):
        assert "run_delivery_sweep" in self._calls_in(path, function)

    def test_the_scheduled_path_trims_the_blob_store_too(self):
        """Documented on every publish path; this one had no call at all."""
        assert "run_blob_retention" in self._calls_in(
            "src/publisher/late/cli.py", "cmd_schedule_auto"
        )


@pytest.mark.unit
class TestTheConfig:
    @pytest.fixture(autouse=True)
    def _api_key(self, monkeypatch):
        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)

    def test_the_bundled_yaml_turns_it_on(self):
        from src.publisher.config import load_publisher_config

        config = load_publisher_config(str(REPO / "config" / "publisher.yaml"))

        assert config.delivery_sweep_config.enabled is True
        assert config.delivery_sweep_config.limit == 25

    def test_a_missing_section_falls_back_to_the_defaults(self, tmp_path):
        from src.publisher.config import load_publisher_config

        minimal = tmp_path / "publisher.yaml"
        minimal.write_text("late:\n  api_key_env_var: LATE_API_KEY\n")

        config = load_publisher_config(str(minimal))

        assert config.delivery_sweep_config == DeliverySweepConfig()
