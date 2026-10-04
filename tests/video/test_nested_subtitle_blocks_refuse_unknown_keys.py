"""The nested subtitle blocks refuse a key they do not know, naming it.

`pycaps`, `safe_zone` and `two_part_subtitles` accepted anything, so a typo
such as `pycaps.template_nme` rendered with the default template and said
nothing. A profile sets these blocks as dicts and builds the models only at
render time, so its check runs when the profile loads.
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config_manager import get_unified_config_manager
from src.video.config.subtitle_models import SubtitleSettings
from src.video.config.visual_models import VideoProfile
from src.video.config_adapter import load_video_config_modular

STALE = [
    ({"pycaps": {"template_nme": "hype"}}, "pycaps.template_nme"),
    ({"safe_zone": {"min_z": 0.1}}, "safe_zone.min_z"),
    ({"two_part_subtitles": {"enabeld": True}}, "two_part_subtitles.enabeld"),
    (
        {"two_part_subtitles": {"upper_line": {"font_scale": 1.2}}},
        "two_part_subtitles.upper_line.font_scale",
    ),
    (
        {"two_part_subtitles": {"lower_line": {"margn": 0.1}}},
        "two_part_subtitles.lower_line.margn",
    ),
]


@pytest.mark.req("REQ-VID-093")
@pytest.mark.parametrize(("block", "named"), STALE)
def test_the_global_block_refuses_an_unknown_key(block: dict, named: str) -> None:
    with pytest.raises(ValidationError, match=named.rsplit(".", 1)[-1]):
        SubtitleSettings.from_legacy_dict(block)


@pytest.mark.req("REQ-VID-093")
@pytest.mark.parametrize(("block", "named"), STALE)
def test_a_profile_refuses_an_unknown_key_at_load(block: dict, named: str) -> None:
    with pytest.raises(ValidationError, match=named):
        VideoProfile(description="A profile", subtitle_settings=block)


def test_a_profile_setting_one_known_nested_field_loads() -> None:
    profile = VideoProfile(
        description="A profile",
        subtitle_settings={"pycaps": {"template_name": "hype"}},
    )

    assert profile.subtitle_settings is not None
    assert profile.subtitle_settings.pycaps == {"template_name": "hype"}


@pytest.mark.req("REQ-VID-093")
@pytest.mark.parametrize(("block", "named"), STALE[:1] + [({"anchr": "top"}, "anchr")])
def test_the_global_config_load_refuses_an_unknown_key(block: dict, named: str) -> None:
    """The global block is a dict on `VideoConfig`; the batch never ran the
    producer's start-up validator, so a typo failed each render instead.
    """
    manager = get_unified_config_manager()
    merged = copy.deepcopy(manager.get_video_config(None))
    merged["subtitle_settings"].update(block)
    with (
        patch.object(manager, "get_video_config", return_value=merged),
        pytest.raises(ValidationError, match=named.rsplit(".", 1)[-1]),
    ):
        load_video_config_modular()
