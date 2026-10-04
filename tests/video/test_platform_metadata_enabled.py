"""Optimized metadata honours `platform_metadata.enabled` and each platform's.

Both switches were declared and documented but nothing read them, so optimized
mode always generated all three platforms. These drive the producer's
optimized-metadata step with the generator stubbed and read what it was asked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.video.config import load_video_config_modular
from src.video.producer import steps


def ctx_for(tmp_path, update):
    config = load_video_config_modular()
    pm = config.description_settings.platform_metadata
    assert pm is not None
    update(pm)
    return SimpleNamespace(
        config=config,
        product=SimpleNamespace(title="Desk lamp", topic=None, pillar=None),
        run_paths={
            "run_root": tmp_path,
            "description_file": tmp_path / "text" / "description.txt",
            "script_file": tmp_path / "text" / "script.txt",
        },
        state={},
        secrets={},
        session=None,
        debug_mode=False,
    )


async def asked_platforms(ctx) -> list[str] | None:
    generate = AsyncMock(return_value={})
    with patch(
        "src.ai.platform_metadata.PlatformMetadataFactory.generate_multi_platform",
        generate,
    ):
        await steps._generate_optimized_metadata(ctx)
    if not generate.called:
        return None
    return sorted(generate.call_args.kwargs["platform_settings"])


@pytest.mark.req("REQ-CNT-137")
@pytest.mark.asyncio
async def test_a_disabled_platform_is_not_generated(tmp_path) -> None:
    def off(pm):
        pm.tiktok.enabled = False

    assert await asked_platforms(ctx_for(tmp_path, off)) == ["instagram", "youtube"]


@pytest.mark.req("REQ-CNT-138")
@pytest.mark.asyncio
async def test_a_disabled_block_falls_back_to_unified(tmp_path) -> None:
    def off(pm):
        pm.enabled = False

    assert await asked_platforms(ctx_for(tmp_path, off)) is None


@pytest.mark.asyncio
async def test_the_shipped_config_generates_all_three(tmp_path) -> None:
    assert await asked_platforms(ctx_for(tmp_path, lambda pm: None)) == [
        "instagram",
        "tiktok",
        "youtube",
    ]
