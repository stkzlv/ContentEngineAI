"""Features that change the reach test's treatment arm ship switched off.

The format-vs-format reach comparison needs the script and the voice held
constant until its durability readout; a humanisation feature switched on
mid-test becomes a confound on the arm the comparison exists to measure. Each
such feature is built and merged off, and this file fails if the shipped
config turns one on. Delete a check here when its feature is deliberately
enabled after the readout.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.video.config import load_video_config_modular


@pytest.mark.req("REQ-CNT-043", "REQ-CNT-044")
def test_script_naturalism_is_off() -> None:
    settings = load_video_config_modular().llm_settings
    assert settings.script_templates.naturalism.intensity == 0


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


@pytest.mark.req("REQ-CNT-045", "REQ-VID-027")
def test_the_author_signature_is_off() -> None:
    config = load_video_config_modular()
    # The lines ship filled (design 0022); the switch is what holds them.
    assert config.llm_settings.script_templates.signature.enabled is False
    assert not config.llm_settings.script_templates.signature.configured
    assert config.audio_settings.signature_sting is None


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


@pytest.mark.req("REQ-CNT-053")
def test_the_script_lint_is_off() -> None:
    """The lint rejects and retries scripts, so it changes them (design 0007)."""
    lint = load_video_config_modular().llm_settings.script_validation.lint
    assert lint.enabled is False


@pytest.mark.req("REQ-CNT-054")
def test_the_hook_rules_are_off() -> None:
    """They change the script, headline and caption prompts (design 0007)."""
    settings = load_video_config_modular().llm_settings
    assert settings.script_templates.hook_rules.enabled is False
