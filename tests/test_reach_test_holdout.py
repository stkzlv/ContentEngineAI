"""Features that change the reach test's treatment arm ship switched off.

The format-vs-format reach comparison needs the script and the voice held
constant until its durability readout; a humanisation feature switched on
mid-test becomes a confound on the arm the comparison exists to measure. Each
such feature is built and merged off, and this file fails if the shipped
config turns one on. Delete a check here when its feature is deliberately
enabled after the readout.
"""

from __future__ import annotations

from src.video.config import load_video_config_modular


def test_script_naturalism_is_off() -> None:
    settings = load_video_config_modular().llm_settings
    assert settings.script_templates.naturalism.intensity == 0


def test_the_selected_voice_keeps_uniform_pauses() -> None:
    """A pause plan on the voice the pipeline selects would change the
    treatment arm's delivery. The varied profile exists to be tried by name.
    """
    tts = load_video_config_modular().tts_config
    assert tts.default_voice_profile is not None
    selected = list(tts.voice_profile_pool) or [tts.default_voice_profile]
    for name in selected:
        assert tts.voice_profiles[name].pause_plan is None, name
