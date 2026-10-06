"""The script step records the drawn sign-off for the first-comment extractor.

The sign-off is spoken where the extractor looks for the closing beat, so
without the record the YouTube first comment would be the sign-off (#440).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.llm_settings import SignatureConfig, SignaturePools
from src.video.config import config
from src.video.producer.context import PipelineContext
from src.video.producer.state import (
    STEP_GENERATE_SCRIPT,
    _update_state_after_step,
    get_video_run_paths,
)


def _on(use_rate: float = 1.0, **pools: list[str]) -> SignatureConfig:
    """Signature switched on, the same pools for both arms."""
    arm = SignaturePools(use_rate=use_rate, **pools)
    return SignatureConfig(enabled=True, product=arm, topic=arm)


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
        _on(1.0, signoffs=[SIGNOFF]),
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


@pytest.mark.req("REQ-CNT-045")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("script", "found"),
    [
        ("Quick find for you, if your old tracker died. Link in bio.", True),
        ("QUICK FIND FOR YOU! Your tracker died. Link in bio.", True),
        ("A quick find for your desk. Link in bio.", False),
    ],
)
async def test_an_opener_is_found_whatever_its_punctuation(
    ctx, monkeypatch, caplog, script, found
) -> None:
    from src.video.producer import steps

    monkeypatch.setattr(
        config.llm_settings.script_templates,
        "signature",
        _on(1.0, openers=["Quick find for you."]),
    )
    caplog.set_level("INFO", logger=steps.logger.name)
    with (
        patch.object(
            steps,
            "generate_ai_script",
            AsyncMock(return_value=(script, "classic_promo", "Link in bio.")),
        ),
        patch.object(steps, "_ensure_fact_checked", AsyncMock()),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
    ):
        await steps.step_generate_script(ctx)
    verdict = "is in the script" if found else "is NOT in the script"
    assert any(verdict in r.getMessage() for r in caplog.records)


@pytest.mark.req("REQ-CNT-046")
@pytest.mark.asyncio
@pytest.mark.parametrize("topic", [False, True])
async def test_the_record_draws_from_the_renders_arm(
    ctx, monkeypatch, caplog, topic
) -> None:
    from src.ai.llm_settings import SignatureConfig, SignaturePools
    from src.video.producer import steps

    monkeypatch.setattr(
        config.llm_settings.script_templates,
        "signature",
        SignatureConfig(
            enabled=True,
            product=SignaturePools(use_rate=1.0, signoffs=["Product sign-off."]),
            topic=SignaturePools(use_rate=1.0, signoffs=["Topic sign-off."]),
        ),
    )
    ctx.product.topic = "a topic" if topic else None
    with (
        patch.object(steps, "_topic_step_list", AsyncMock(return_value=None)),
        patch.object(
            steps,
            "generate_ai_script",
            AsyncMock(return_value=(SCRIPT, "classic_promo", "Link in bio.")),
        ),
        patch.object(steps, "_ensure_fact_checked", AsyncMock()),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
    ):
        await steps.step_generate_script(ctx)
    assert ctx.state["signoff"] == ("Topic sign-off." if topic else "Product sign-off.")


@pytest.mark.req("REQ-CNT-153")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("template", "opens"),
    [("topic_answer_first", True), ("topic_symptom_cause", False)],
)
async def test_the_record_follows_the_template(
    ctx, monkeypatch, caplog, template, opens
) -> None:
    from src.ai.llm_settings import SignatureConfig, SignaturePools
    from src.video.producer import steps

    monkeypatch.setattr(
        config.llm_settings.script_templates,
        "signature",
        SignatureConfig(
            enabled=True,
            topic=SignaturePools(
                use_rate=1.0,
                openers=["Here's how to"],
                opener_templates=["topic_answer_first"],
            ),
        ),
    )
    ctx.product.topic = "a topic"
    caplog.set_level("INFO", logger=steps.logger.name)
    with (
        patch.object(steps, "_topic_step_list", AsyncMock(return_value=None)),
        patch.object(
            steps,
            "generate_ai_script",
            AsyncMock(return_value=(SCRIPT, template, "Link in bio.")),
        ),
        patch.object(steps, "_ensure_fact_checked", AsyncMock()),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
    ):
        await steps.step_generate_script(ctx)
    logged = any("Signature opener" in r.getMessage() for r in caplog.records)
    assert logged is opens
