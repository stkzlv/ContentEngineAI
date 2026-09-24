"""The script step records the drawn sign-off for the first-comment extractor.

The sign-off is spoken where the extractor looks for the closing beat, so
without the record the YouTube first comment would be the sign-off (#440).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.llm_settings import SignatureConfig
from src.video.config import config
from src.video.producer.context import PipelineContext
from src.video.producer.state import (
    STEP_GENERATE_SCRIPT,
    _update_state_after_step,
    get_video_run_paths,
)

SIGNOFF = "That's the find for today."
SCRIPT = f"A script about a mount. Team magnetic or plug-in? {SIGNOFF} Link in bio."


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "global_output_root_path", tmp_path)
    paths = get_video_run_paths(config, "B0SIGN0001", "slideshow_images1")
    product = MagicMock(asin="B0SIGN0001", topic=None, title="Magnetic mount")
    return PipelineContext(
        product=product,
        profile=config.video_profiles["slideshow_images1"],
        profile_name="slideshow_images1",
        config=config,
        secrets={},
        session=MagicMock(),
        run_paths=paths,
        debug_mode=False,
    )


async def _generate(ctx) -> None:
    from src.video.producer import steps

    with (
        patch.object(
            steps,
            "generate_ai_script",
            AsyncMock(return_value=(SCRIPT, "classic_promo", "Link in bio.")),
        ),
        patch.object(steps, "_ensure_fact_checked", AsyncMock()),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
    ):
        await steps.step_generate_script(ctx)


@pytest.mark.asyncio
async def test_a_drawn_signoff_is_recorded_and_survives_the_step_entry(
    ctx, monkeypatch
) -> None:
    monkeypatch.setattr(
        config.llm_settings.script_templates,
        "signature",
        SignatureConfig(use_rate=1.0, signoffs=[SIGNOFF]),
    )
    await _generate(ctx)
    assert ctx.state["signoff"] == SIGNOFF
    await _update_state_after_step(ctx, STEP_GENERATE_SCRIPT)
    assert ctx.state[STEP_GENERATE_SCRIPT]["signoff"] == SIGNOFF


@pytest.mark.asyncio
async def test_no_signature_records_nothing(ctx, monkeypatch) -> None:
    monkeypatch.setattr(
        config.llm_settings.script_templates, "signature", SignatureConfig()
    )
    await _generate(ctx)
    assert "signoff" not in ctx.state
