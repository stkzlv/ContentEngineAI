"""Held features ship switched off.

The reach-test hold ended once the test's posts were queued (decision 0014),
and its features are turned on as each is verified on a real render. Until
then each stays `held`, and this file fails if the shipped config turns one
on. Delete a check here when its feature is turned on.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.video.config import load_video_config_modular


@pytest.mark.req("REQ-VID-027")
def test_the_signature_sting_is_off() -> None:
    """No sting file ships, so the sting stays held."""
    assert load_video_config_modular().audio_settings.signature_sting is None


@pytest.mark.req("REQ-VID-123")
def test_step_visuals_are_off() -> None:
    """Step screens wait on a look at a sample render (design 0018)."""
    assert load_video_config_modular().video_settings.step_visuals.enabled is False
