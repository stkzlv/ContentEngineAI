"""Running bits, offered under a cap (REQ-CNT-188)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ai.script_generator import _with_running_bit
from src.utils.render_choices_store import choices_path
from src.video.config.identity_models import NarratorIdentity, RunningBit
from src.video.running_bits import select_running_bit, used_bit

pytestmark = pytest.mark.req("REQ-CNT-188")

A = RunningBit(id="a", hint="An aside about A.", cues=["alpha"])
B = RunningBit(id="b", hint="An aside about B.", cues=["beta"])
START = "2026-01-02T00:00:00+00:00"


def _identity(per_batch: int = 1, min_gap: int = 2) -> NarratorIdentity:
    return NarratorIdentity(
        running_bits=[A, B],
        running_bit_cap={"per_batch": per_batch, "min_gap": min_gap},
    )


def _rows(outputs: Path, rows: list[tuple[str, str | None, str]]) -> None:
    """Write (product_id, running_bit, recorded_at) rows of the commentary kind."""
    path = choices_path(outputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pid, bit, at in rows:
            row = {"product_id": pid, "content_kind": "commentary", "recorded_at": at}
            if bit:
                row["running_bit"] = bit
            f.write(json.dumps(row) + "\n")


def test_no_bits_offers_nothing(tmp_path):
    assert select_running_bit(NarratorIdentity(), tmp_path, "commentary") is None


def test_first_render_gets_a_bit(tmp_path):
    assert select_running_bit(_identity(), tmp_path, "commentary", START) == A


def test_the_batch_cap_holds_back_a_second_use(tmp_path):
    _rows(tmp_path, [("p1", "a", "2026-01-02T01:00:00+00:00")])
    bit = select_running_bit(
        _identity(per_batch=1, min_gap=0), tmp_path, "commentary", START
    )
    assert bit is None


def test_uses_before_this_run_do_not_count_toward_the_batch_cap(tmp_path):
    _rows(tmp_path, [("p1", "a", "2026-01-01T01:00:00+00:00")])
    bit = select_running_bit(
        _identity(per_batch=1, min_gap=0), tmp_path, "commentary", START
    )
    assert bit == B


def test_min_gap_counts_renders_of_the_kind_since_the_last_use(tmp_path):
    old = "2026-01-01T01:00:00+00:00"
    _rows(tmp_path, [("p1", "a", old), ("p2", None, old)])
    assert (
        select_running_bit(_identity(min_gap=2), tmp_path, "commentary", START) is None
    )
    _rows(tmp_path, [("p1", "a", old), ("p2", None, old), ("p3", None, old)])
    assert select_running_bit(_identity(min_gap=2), tmp_path, "commentary", START) == B


def test_other_kinds_do_not_count(tmp_path):
    path = choices_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "product_id": "x",
                "content_kind": "product",
                "running_bit": "a",
                "recorded_at": "2026-01-02T01:00:00+00:00",
            }
        )
        + "\n"
    )
    assert select_running_bit(_identity(), tmp_path, "commentary", START) == A


def test_used_bit_reads_the_cues():
    assert used_bit("Then the ALPHA joke lands.", A) == "a"
    assert used_bit("No aside here.", A) is None
    assert used_bit("alpha", None) is None


def test_the_hint_reaches_the_script_prompt_only_when_offered():
    assert _with_running_bit("Narrator.", "") == "Narrator."
    offered = _with_running_bit("Narrator.", "An aside about A.")
    assert offered.startswith("Narrator.") and "An aside about A." in offered


def test_a_bit_needs_a_cue():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RunningBit(id="c", hint="h", cues=[])


@pytest.fixture
def step_ctx(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from src.video.config import config
    from src.video.producer.context import PipelineContext
    from src.video.producer.state import get_video_run_paths

    cfg = config.model_copy(deep=True)
    cfg.global_output_root_path = tmp_path
    cfg.narrator_identity.product = NarratorIdentity(running_bits=[A])
    paths = get_video_run_paths(cfg, "B0TEST0188", "slideshow_images1")
    product = MagicMock(
        asin="B0TEST0188", topic=None, topic_kind=None, title="A mount", keyword=""
    )
    return PipelineContext(
        product=product,
        profile=cfg.video_profiles["slideshow_images1"],
        profile_name="slideshow_images1",
        config=cfg,
        secrets={},
        session=MagicMock(),
        run_paths=paths,
        debug_mode=False,
    )


async def _run_step(ctx, script: str):
    from unittest.mock import AsyncMock, patch

    from src.video.producer import steps

    seen = {}

    async def fake(product, settings, *a, **k):
        seen["hint"] = settings.script_templates.running_bit_hint
        return script, "classic_promo", "Link in bio."

    with (
        patch.object(steps, "generate_ai_script", fake),
        patch.object(steps, "_ensure_fact_checked", AsyncMock()),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
    ):
        await steps.step_generate_script(ctx)
    return seen


@pytest.mark.asyncio
async def test_the_step_offers_the_bit_and_records_a_use(step_ctx):
    from src.video.render_choices import choices_from_context

    seen = await _run_step(step_ctx, "The alpha aside lands here. Link in bio.")

    assert seen["hint"] == A.hint
    assert step_ctx.state["running_bit"] == "a"
    assert choices_from_context(step_ctx)["running_bit"] == "a"


@pytest.mark.asyncio
async def test_an_unused_bit_is_not_recorded(step_ctx):
    await _run_step(step_ctx, "No aside here at all. Link in bio.")

    assert "running_bit" not in step_ctx.state


@pytest.mark.asyncio
async def test_the_hint_is_in_the_prompt_sent_to_the_model():
    from unittest.mock import AsyncMock, patch

    from src.ai import script_generator
    from src.scraper.amazon.models import ProductData
    from src.scraper.base.models import Platform
    from src.video.config import config

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_templates.running_bit_hint = "An aside about A."
    settings.script_validation.lint.enabled = False
    cta = settings.script_templates.cta_options_for(False)[0]
    reply = "The stand tilts back for a call. " * 8 + cta
    call = AsyncMock(return_value=reply)
    product = ProductData(
        title="A Phone Stand",
        price="$20",
        url="https://www.amazon.com/dp/B0TEST0188",
        platform=Platform.AMAZON,
        asin="B0TEST0188",
        description="An adjustable phone stand.",
    )
    with patch.object(script_generator, "_call_llm_api_with_retry", call):
        await script_generator.generate_script(
            product, settings, {settings.api_key_env_var: "k"}, AsyncMock(), {}, False
        )
    assert "An aside about A." in call.await_args_list[0].args[0]
