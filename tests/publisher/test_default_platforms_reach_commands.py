"""`single` and `schedule` without `--platform` target `default_platforms`.

Both used to fall back to a hardcoded three-platform list, so the key changed
nothing for the two commands that publish. These drive `main` with the config
loader and the command stubbed, and read the platforms the command received.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.publisher.late import cli as late_cli
from src.publisher.models import Platform


@pytest.mark.req("REQ-PUB-117")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["single", "B0DEFAULT1"], "cmd_single"),
        (["schedule", "--dry-run"], "cmd_schedule_auto"),
    ],
)
async def test_no_platform_flag_takes_the_configured_platforms(
    argv: list[str], command: str
) -> None:
    config = MagicMock(default_platforms=[Platform.YOUTUBE], active_account=None)
    stub = AsyncMock()
    with (
        patch("sys.argv", ["late", *argv]),
        patch.object(late_cli, "load_publisher_config", return_value=config),
        patch.object(late_cli, command, stub),
    ):
        await late_cli.main()

    assert stub.call_args.args[0].platforms == [Platform.YOUTUBE]


@pytest.mark.req("REQ-PUB-116")
@pytest.mark.asyncio
async def test_a_platform_flag_wins_over_the_configured_platforms() -> None:
    config = MagicMock(default_platforms=[Platform.YOUTUBE], active_account=None)
    stub = AsyncMock()
    with (
        patch("sys.argv", ["late", "schedule", "--dry-run", "--platform", "tiktok"]),
        patch.object(late_cli, "load_publisher_config", return_value=config),
        patch.object(late_cli, "cmd_schedule_auto", stub),
    ):
        await late_cli.main()

    assert stub.call_args.args[0].platforms == [Platform.TIKTOK]
