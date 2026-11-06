"""An edited script gets a hook headline derived from it (REQ-VID-165).

The script step is skipped on a resume, so the headline recorded for the
original script would otherwise ship over the edited one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.video.producer import steps
from src.video.producer.artifact_registry import load_artifacts_for_step
from src.video.producer.state import STEP_GENERATE_SCRIPT
from src.video.producer.steps import (
    _hook_headline,
    refresh_stale_hook_headline,
    script_digest,
)

pytestmark = pytest.mark.req("REQ-VID-165")


def _ctx(script: str | None, state: dict[str, Any]) -> SimpleNamespace:
    templates = SimpleNamespace(
        narrator_for=lambda *_: None, preambles_for=lambda *_: None
    )
    config = SimpleNamespace(
        video_settings=SimpleNamespace(
            hook_overlay=SimpleNamespace(enabled=True, max_words=8)
        ),
        llm_settings=SimpleNamespace(script_templates=templates),
        api_settings=None,
    )
    return SimpleNamespace(
        script=script,
        state=state,
        product=SimpleNamespace(topic="t", pillar=None),
        config=config,
        secrets={},
        session=None,
        debug_mode=False,
    )


def test_a_headline_for_other_text_is_none():
    ctx = _ctx(
        "Edited.",
        {"hook_headline": "Old", "hook_headline_script": script_digest("Original.")},
    )
    assert _hook_headline(ctx) is None
    ctx.script = "Original."
    assert _hook_headline(ctx) == "Old"


def test_a_headline_recorded_before_the_digest_is_kept():
    ctx = _ctx("Anything.", {STEP_GENERATE_SCRIPT: {"hook_headline": "Legacy"}})
    assert _hook_headline(ctx) == "Legacy"


def test_the_step_entry_digest_counts_after_a_truncating_resume():
    entry = {"hook_headline": "Old", "hook_headline_script": script_digest("Orig.")}
    ctx = _ctx("Edited.", {STEP_GENERATE_SCRIPT: entry})
    assert _hook_headline(ctx) is None


@pytest.mark.asyncio
async def test_resume_rederives_the_headline_for_the_edited_script():
    state: dict[str, Any] = {
        "hook_headline": "Old",
        "hook_headline_script": script_digest("Original."),
        STEP_GENERATE_SCRIPT: {"hook_headline": "Old"},
    }
    ctx = _ctx("Edited.", state)
    generate = AsyncMock(return_value="New")
    with patch.object(steps, "generate_hook_headline", generate):
        await refresh_stale_hook_headline(ctx)
    assert generate.await_args is not None
    assert generate.await_args.kwargs["video_script"] == "Edited."
    assert state["hook_headline"] == "New"
    assert state["hook_headline_script"] == script_digest("Edited.")
    # Written back into the step entry, which a truncating resume keeps.
    assert state[STEP_GENERATE_SCRIPT]["hook_headline"] == "New"
    assert state[STEP_GENERATE_SCRIPT]["hook_headline_script"] == script_digest(
        "Edited."
    )
    assert _hook_headline(ctx) == "New"


@pytest.mark.asyncio
async def test_an_unchanged_script_costs_no_call():
    state = {"hook_headline": "Old", "hook_headline_script": script_digest("Same.")}
    generate = AsyncMock(return_value="New")
    with patch.object(steps, "generate_hook_headline", generate):
        await refresh_stale_hook_headline(_ctx("Same.", state))
    generate.assert_not_awaited()
    assert state["hook_headline"] == "Old"


def test_orchestration_refreshes_after_loading_a_skipped_script_step():
    from src.utils.outputs_paths import get_project_root

    source = (get_project_root() / "src/video/producer/orchestration.py").read_text()
    load = source.index("load_artifacts_for_step(step_name, ctx)")
    refresh = source.index("await refresh_stale_hook_headline(ctx)")
    assert load < refresh


def test_a_run_without_the_script_keeps_the_headline():
    # `--step assemble_video` may not load the script; that is no edit.
    ctx = _ctx(
        None,
        {"hook_headline": "Kept", "hook_headline_script": script_digest("Orig.")},
    )
    assert _hook_headline(ctx) == "Kept"


def test_a_resume_reads_the_edited_script_file(tmp_path):
    # The voiceover speaks ctx.script, which the skipped step loads from the file.
    script_file = tmp_path / "script.txt"
    script_file.write_text("Edited by hand.", encoding="utf-8")
    ctx = _ctx(None, {})
    ctx.run_paths = {"script_file": script_file}
    load_artifacts_for_step(STEP_GENERATE_SCRIPT, ctx)
    assert ctx.script == "Edited by hand."


def test_the_step_path_refreshes_too():
    from src.utils.outputs_paths import get_project_root

    source = (get_project_root() / "src/video/producer/orchestration.py").read_text()
    assert source.count("await refresh_stale_hook_headline(ctx)") == 2
