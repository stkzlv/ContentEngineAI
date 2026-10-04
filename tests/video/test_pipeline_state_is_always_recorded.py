"""`pipeline_state.json` is written on every render, with the run's choices.

`debug_settings` once carried `create_pipeline_metadata`,
`create_ffmpeg_command_logs` and `create_performance_metrics`, which no model
declared, so setting one to false changed nothing. They are gone, and the
model refuses them, so a config that still sets one fails the load instead of
looking as if it switched the file off.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.video.config import load_video_config_modular
from src.video.config.core_models import DebugSettings
from src.video.producer.state import (
    STEP_GENERATE_SCRIPT,
    _save_pipeline_state,
    _update_state_after_step,
)


@pytest.mark.parametrize(
    "key",
    [
        "create_pipeline_metadata",
        "create_ffmpeg_command_logs",
        "create_performance_metrics",
    ],
)
def test_a_removed_switch_fails_the_load(key: str) -> None:
    with pytest.raises(ValidationError, match=key):
        DebugSettings.model_validate({key: False})


@pytest.mark.req("REQ-CNT-012", "REQ-CNT-038", "REQ-CNT-120", "REQ-OPS-078")
def test_the_state_file_records_the_run_choices(tmp_path: Path) -> None:
    script = tmp_path / "script.txt"
    script.write_text("A script.")
    state_file = tmp_path / "temp" / "pipeline_state.json"
    run_paths = {"state_file": state_file, "script_file": script}
    config = load_video_config_modular()
    ctx = SimpleNamespace(
        state={
            "pillar": "how-to",
            "script_template": "problem_solution",
            "cta": "Follow for more.",
            "signoff": "See you next time.",
        },
        run_paths=run_paths,
        config=config,
        profile=config.video_profiles["slideshow_images1"],
    )

    asyncio.run(_update_state_after_step(ctx, STEP_GENERATE_SCRIPT))
    asyncio.run(_save_pipeline_state(ctx))

    saved = json.loads(state_file.read_text())
    assert saved["pillar"] == "how-to"
    step = saved[STEP_GENERATE_SCRIPT]
    assert step["status"] == "done"
    assert step["script_template"] == "problem_solution"
    assert step["cta"] == "Follow for more."
    assert step["signoff"] == "See you next time."
