"""`--no-link-in-bio` skips the bio update on every publish path (REQ-PUB-066).

It existed only on `single`; `schedule` and the global batch always used
`link_in_bio.enabled` from the config.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.pipeline.cli import create_argument_parser
from src.pipeline.config import load_global_batch_config
from src.publisher.late.cli import _link_in_bio_config, build_argument_parser
from src.publisher.models import LinkInBioConfig

CONFIG = SimpleNamespace(link_in_bio_config=LinkInBioConfig(enabled=True))


@pytest.mark.req("REQ-PUB-066")
def test_schedule_takes_the_flag() -> None:
    args = build_argument_parser().parse_args(["schedule", "--no-link-in-bio"])
    assert _link_in_bio_config(args, CONFIG).enabled is False


@pytest.mark.req("REQ-PUB-066")
def test_schedule_without_the_flag_keeps_the_config() -> None:
    args = build_argument_parser().parse_args(["schedule"])
    assert _link_in_bio_config(args, CONFIG) is CONFIG.link_in_bio_config


@pytest.mark.req("REQ-PUB-066")
def test_the_global_batch_takes_the_flag(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump({"global_batch": {"product_ids": ["B0BIOFLAG1"]}}))

    off = load_global_batch_config(
        create_argument_parser().parse_args(["--no-link-in-bio"]), path
    )
    unset = load_global_batch_config(create_argument_parser().parse_args([]), path)

    assert off.link_in_bio is False
    assert unset.link_in_bio is None
