"""The hook headline survives a resume that keeps only step entries.

A short product title (REQ-PUB-008) and the hook overlay read it in steps a
resume can run on their own.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video.config import config


@pytest.mark.req("REQ-PUB-008")
@pytest.mark.asyncio
async def test_the_script_step_entry_keeps_the_headline(monkeypatch) -> None:
    from src.video.producer import state

    ctx = SimpleNamespace(
        state={"hook_headline": "Phone stand that reaches 62 inches"},
        run_paths={"script_file": Path("script.txt")},
    )
    monkeypatch.setattr(state, "_drop_dependents", lambda ctx, step: None)

    await state._update_state_after_step(ctx, state.STEP_GENERATE_SCRIPT)

    assert ctx.state["generate_script"]["hook_headline"] == (
        "Phone stand that reaches 62 inches"
    )


@pytest.mark.req("REQ-PUB-008")
def test_a_resumed_title_uses_the_recorded_headline() -> None:
    from src.video.producer import steps

    cfg = config.model_copy(deep=True)
    cfg.description_settings.short_product_titles = True
    ctx = SimpleNamespace(
        config=cfg,
        product=SimpleNamespace(
            title="EUCOS 62 Magnetic Phone Tripod for iPhone, Selfie Stick",
            keyword="phone stand",
            topic=None,
        ),
        # Truncated state: the step entry only, no top-level key.
        state={"generate_script": {"hook_headline": "Phone stand that reaches 62"}},
    )

    assert steps._unified_title(ctx) == "Phone stand that reaches 62"
