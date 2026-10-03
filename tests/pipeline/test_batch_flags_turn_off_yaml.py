"""A batch flag overrides YAML in both directions, and only when passed.

The flags were `store_true`, merged as `cli or yaml`, so a YAML `true` stayed
on for every run and `--profile X` with `random_profile: true` in YAML was
refused as both modes at once. They are `--x/--no-x` pairs now.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.pipeline.cli import create_argument_parser
from src.pipeline.config import load_global_batch_config

YAML_ON = {
    "product_ids": ["B0FLAGS001"],
    "fail_fast": True,
    "debug": True,
    "skip_publish": True,
    "random_profile": True,
    "process_all_products": True,
    "platform_specific_content": True,
}


def load(tmp_path: Path, argv: list[str], **yaml_overrides):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump({"global_batch": {**YAML_ON, **yaml_overrides}}))
    return load_global_batch_config(create_argument_parser().parse_args(argv), path)


@pytest.mark.req("REQ-BAT-015")
def test_an_absent_flag_keeps_the_yaml_value(tmp_path: Path) -> None:
    config = load(tmp_path, [])
    assert config.fail_fast and config.debug and config.skip_publish
    assert config.random_profile and config.process_all_products


@pytest.mark.req("REQ-BAT-015", "REQ-OPS-002")
def test_a_no_flag_turns_a_yaml_true_off(tmp_path: Path) -> None:
    config = load(
        tmp_path,
        [
            "--no-fail-fast",
            "--no-debug",
            "--no-skip-publish",
            "--no-process-all-products",
            "--no-platform-specific",
        ],
    )
    assert not config.fail_fast
    assert not config.debug
    assert not config.skip_publish
    assert not config.process_all_products
    assert not config.platform_specific_content


@pytest.mark.req("REQ-BAT-015")
def test_a_cli_profile_beats_a_yaml_random_profile(tmp_path: Path) -> None:
    config = load(tmp_path, ["--profile", "slideshow_images1"])
    assert config.profile == "slideshow_images1"
    assert not config.random_profile


@pytest.mark.req("REQ-OPS-018")
def test_fail_fast_covers_publishing(tmp_path: Path) -> None:
    config = load(tmp_path, ["--fail-fast"], fail_fast=False)
    assert config.fail_fast_publish

    opted_out = load(
        tmp_path, ["--fail-fast", "--no-fail-fast-publish"], fail_fast=False
    )
    assert not opted_out.fail_fast_publish


@pytest.mark.req("REQ-OPS-018")
def test_a_yaml_fail_fast_publish_false_holds(tmp_path: Path) -> None:
    """Only an unset publishing switch inherits `fail_fast`."""
    config = load(tmp_path, [], fail_fast=True, fail_fast_publish=False)
    assert config.fail_fast
    assert not config.fail_fast_publish
