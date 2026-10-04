"""Producer flags that set a key nothing reads are refused, not accepted.

`--ass-karaoke`, `--ass-fade` and `--target-platform` were accepted and
documented, but the overrides they wrote reached no code, so passing them
changed nothing and said nothing.
"""

from __future__ import annotations

import pytest

from src.video.config.core_models import DescriptionSettings
from src.video.producer.cli import create_argument_parser

BASE = ["outputs/B0X/data.json", "slideshow_images1"]


@pytest.mark.parametrize(
    "flag", [["--ass-karaoke"], ["--ass-fade"], ["--target-platform", "youtube"]]
)
def test_a_removed_flag_is_refused(flag: list[str]) -> None:
    with pytest.raises(SystemExit):
        create_argument_parser().parse_args([*BASE, *flag])


def test_an_old_target_platform_key_is_ignored() -> None:
    settings = DescriptionSettings.model_validate({"target_platform": "youtube"})

    assert not hasattr(settings, "target_platform")
