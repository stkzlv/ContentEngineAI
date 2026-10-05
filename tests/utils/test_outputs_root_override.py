"""A run that names an outputs directory renders there (REQ-OPS-031, #662).

`--outputs-dir` used to pick which products a batch rendered while every
render still wrote under `global_output_directory`, so a scratch copy of a
product overwrote the real product's video and state.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from src.utils.outputs_paths import (
    OUTPUTS_ENV_VARS,
    get_project_root,
    with_outputs_root,
)


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch):
    for name in OUTPUTS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_no_directory_leaves_the_overrides_alone() -> None:
    overrides = {"voice_profile": "x"}

    assert with_outputs_root(overrides, None) is overrides
    assert with_outputs_root(None, "") is None


def test_a_relative_directory_is_taken_from_the_repository_root() -> None:
    overrides = {"voice_profile": "x"}

    out = with_outputs_root(overrides, "scratch/out")

    assert out == {
        "voice_profile": "x",
        "output_dir": str(get_project_root() / "scratch/out"),
    }
    assert "output_dir" not in overrides


@pytest.mark.req("REQ-OPS-031")
def test_the_loaded_config_renders_under_the_named_directory(tmp_path: Path) -> None:
    from src.video.config_adapter import load_video_config_modular
    from src.video.producer.state import get_video_run_paths

    config = load_video_config_modular(cli_overrides=with_outputs_root({}, tmp_path))
    paths = get_video_run_paths(config, "B0X", "slideshow_images1")

    assert config.global_output_root_path == tmp_path
    for key in ("run_root", "final_video_output", "state_file", "intermediate_base"):
        assert tmp_path in Path(paths[key]).parents, key


@pytest.mark.req("REQ-OPS-031")
def test_the_producer_and_the_batch_both_pass_it() -> None:
    """Module/batch alignment: both config loads carry the directory."""
    from src.pipeline.phases import production
    from src.video.producer import cli

    assert "with_outputs_root(cli_overrides, args.outputs_dir)" in inspect.getsource(
        cli.main
    )
    assert "batch_config.outputs_dir" in inspect.getsource(
        production.run_production_phase
    )
