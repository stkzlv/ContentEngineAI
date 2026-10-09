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


@pytest.mark.req("REQ-VID-121", "REQ-VID-122", "REQ-VID-151")
def test_topic_step_lists_are_off() -> None:
    """Step lists rewrite the topic arm's scripts (design 0017)."""
    settings = load_video_config_modular().llm_settings
    assert settings.topic_scripts.step_list.enabled is False


@pytest.mark.req("REQ-PUB-008")
def test_short_product_titles_are_off() -> None:
    """They change the published YouTube title (design 0009)."""
    settings = load_video_config_modular().description_settings
    assert settings.short_product_titles is False
