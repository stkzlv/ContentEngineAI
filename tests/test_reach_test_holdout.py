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


@pytest.mark.req("REQ-CNT-062")
def test_the_selected_voice_keeps_uniform_pauses() -> None:
    """A pause plan on the voice the pipeline selects would change the
    treatment arm's delivery. The varied profile exists to be tried by name.
    """
    tts = load_video_config_modular().tts_config
    assert tts.default_voice_profile is not None
    selected = list(tts.voice_profile_pool) or [tts.default_voice_profile]
    for name in selected:
        assert tts.voice_profiles[name].pause_plan is None, name


@pytest.mark.req("REQ-VID-027")
def test_the_signature_sting_is_off() -> None:
    """No sting file ships, so the sting stays held."""
    assert load_video_config_modular().audio_settings.signature_sting is None


@pytest.mark.req("REQ-VID-022")
def test_video_content_stays_top_aligned() -> None:
    """The code default is centre; the shipped config pins top for the
    profiles that don't set it, so their renders don't move mid-test.
    """
    from src.video.config.visual_models import VideoSettings

    assert VideoSettings.model_fields["video_vertical_align"].default == "center"
    config = load_video_config_modular()
    assert config.video_settings.video_vertical_align == "top"


@pytest.mark.req("REQ-VID-121", "REQ-VID-122", "REQ-VID-151")
def test_topic_step_lists_are_off() -> None:
    """Step lists rewrite the topic arm's scripts (design 0017)."""
    settings = load_video_config_modular().llm_settings
    assert settings.topic_scripts.step_list.enabled is False


@pytest.mark.req("REQ-VID-158")
def test_the_caption_outline_is_off() -> None:
    """The outline restyles every caption (6 px chosen, held for the readout)."""
    from src.video.config.subtitle_models import PycapsSettings

    config = load_video_config_modular()
    for profile in config.video_profiles:
        merged = config.get_profile_merged_settings(profile)
        # A profile override can arrive as a mapping.
        pycaps: Any = merged.subtitle_settings.pycaps
        if pycaps is not None and not isinstance(pycaps, PycapsSettings):
            pycaps = PycapsSettings(**pycaps)
        assert pycaps is None or pycaps.outline_px == 0, profile


@pytest.mark.req("REQ-PUB-008")
def test_short_product_titles_are_off() -> None:
    """They change the published YouTube title (design 0009)."""
    settings = load_video_config_modular().description_settings
    assert settings.short_product_titles is False
