"""`video_settings` refuses a key it does not know, naming it.

The block accepted anything, so `codec`, `crf`, `preset`, `image_duration` and
`disclosure_overlay.enabled` sat in the bundled config for releases looking
like settings while nothing read them. These drive the loader the producer
and the batch use, with the shipped config plus one stale key.
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config_manager import get_unified_config_manager
from src.video.config_adapter import load_video_config_modular


def shipped_with(update) -> dict:
    merged = copy.deepcopy(get_unified_config_manager().get_video_config(None))
    update(merged)
    return merged


@pytest.mark.req("REQ-VID-093")
def test_the_shipped_config_loads() -> None:
    assert load_video_config_modular().video_settings.frame_rate > 0


@pytest.mark.req("REQ-VID-093")
@pytest.mark.parametrize(
    ("stale", "update"),
    [
        ("codec", lambda c: c["video_settings"].update(codec="libx264")),
        (
            "enabled",
            lambda c: c["video_settings"]["disclosure_overlay"].update(enabled=True),
        ),
    ],
)
def test_a_stale_key_fails_the_load_naming_it(stale: str, update) -> None:
    manager = get_unified_config_manager()
    merged = shipped_with(update)
    with (
        patch.object(manager, "get_video_config", return_value=merged),
        pytest.raises(ValidationError, match=stale),
    ):
        load_video_config_modular()
